import torch
from torch import nn
import torch.nn.functional as torch_f


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.mean(mins, 1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins, 1)

        return loss_1, loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = (
            xx[:, diag_ind_x, diag_ind_x]
            .unsqueeze(1)
            .expand_as(zz.transpose(2, 1))
        )
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500, use_tanh=False, out_factor=200):
        self.bottleneck_size = bottleneck_size
        self.use_tanh = use_tanh
        self.out_factor = out_factor
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            self.bottleneck_size, self.bottleneck_size, 1
        )
        self.conv2 = torch.nn.Conv1d(
            self.bottleneck_size, int(self.bottleneck_size / 2), 1
        )
        self.conv3 = torch.nn.Conv1d(
            int(self.bottleneck_size / 2), int(self.bottleneck_size / 4), 1
        )
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size / 4), 3, 1)

        if self.use_tanh:
            self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 4))

    def forward(self, x):
        x = torch_f.relu(self.bn1(self.conv1(x)))
        x = torch_f.relu(self.bn2(self.conv2(x)))
        x = torch_f.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        if self.use_tanh:
            x = self.out_factor * self.th(x)
        else:
            x = self.out_factor * x
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, bottleneck_size=2500, res_size=100, out_factor=200, residual=True
    ):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.res_size = res_size
        self.out_factor = out_factor
        self.residual = residual
        self.conv1 = torch.nn.Conv1d(bottleneck_size, res_size, 1)
        self.conv2 = torch.nn.Conv1d(res_size, res_size, 1)
        self.conv3 = torch.nn.Conv1d(res_size, 3, 1)
        self.bn1 = torch.nn.BatchNorm1d(res_size)
        self.bn2 = torch.nn.BatchNorm1d(res_size)

    def forward(self, inp):
        coords = inp[:, :3]
        x = torch_f.relu(self.bn1(self.conv1(inp)))
        x = torch_f.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        # plt.scatter(inp[0][:3][0].detach(), inp[0][:3][1].detach())
        # plt.scatter(x[0][:3][0].detach(), x[0][:3][1].detach())
        # plt.show()
        if self.residual:
            x = x + coords * self.out_factor
        return x


class PointGenConResidual(nn.Module):
    def __init__(
        self,
        bottleneck_size=2500,
        res_size=256,
        out_factor=200,
        pred_scale=False,
    ):
        self.bottleneck_size = bottleneck_size
        self.out_factor = out_factor
        super().__init__()
        self.residual1 = DecoderBlock(
            bottleneck_size, res_size=res_size, out_factor=1, residual=True
        )
        self.residual2 = DecoderBlock(
            bottleneck_size, res_size=res_size, out_factor=1, residual=True
        )
        self.residual3 = DecoderBlock(
            bottleneck_size, res_size=res_size, out_factor=1, residual=False
        )
        self.pred_scale = pred_scale
        if pred_scale:
            base_layers = []
            base_layers.append(
                nn.Linear(bottleneck_size - 3, int(bottleneck_size / 2))
            )
            base_layers.append(nn.ReLU())
            base_layers.append(nn.Linear(int(bottleneck_size / 2), 1))
            self.scale_decoder = nn.Sequential(*base_layers)

    def forward(self, x):
        features = x[:, 3:]
        x = self.residual1(x)
        interm = torch.cat((x, features), 1)
        x = self.residual2(interm)
        interm = torch.cat((x, features), 1)
        x = self.residual3(interm)
        if self.pred_scale:
            # Only pass one feature and not feature grid to scale decoder
            scale = self.scale_decoder(features[:, :, 0])
            x = (scale + (self.out_factor / 2)).unsqueeze(1) * x
        else:
            x = self.out_factor * x
        return x
