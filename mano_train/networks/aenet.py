import argparse

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f

import argutils  # Requires myana to be in PYTHONPATH
from mano_train.networks.bases import resnet
from mano_train.networks.branches.manobranch import ManoBranch, ManoLoss
from mano_train.networks.branches.atlasbranch import AtlasBranch, AtlasLoss
from mano_train.networks.branches.refinebranch import RefineBranch, RefineLoss
from mano_train.networks.branches.contactloss import compute_contact_loss
from mano_train.networks.branches.absolutebranch import AbsoluteBranch
from handobjectdatasets.queries import TransQueries, BaseQueries


class AENet(nn.Module):
    def __init__(
        self,
        atlas_lambda=None,
        atlas_loss="chamfer",
        atlas_emd_regul=0.1,
        atlas_mode="sphere",
        atlas_mesh=True,
        atlas_residual=True,
        atlas_lambda_regul_edges=0,
        atlas_lambda_laplacian=0,
        atlas_predict_trans=False,
        atlas_trans_weight=1,
        atlas_use_tanh=False,
        atlas_ico_divisions=3,
        atlas_out_factor=200,
        bottleneck_size=512,
    ):
        super().__init__()
        if atlas_ico_divisions == 3:
            points_nb = 642
        self.bottleneck_size = bottleneck_size
        self.encoder = self.encoder = nn.Sequential(
            PointNetfeat(points_nb, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU(),
        )
        self.atlas_branch = AtlasBranch(
            mode=atlas_mode,
            use_residual=atlas_residual,
            points_nb=points_nb,
            predict_trans=atlas_predict_trans,
            inference_ico_divisions=atlas_ico_divisions,
            bottleneck_size=bottleneck_size,
            use_tanh=atlas_use_tanh,
            out_factor=atlas_out_factor,
        )
        self.atlas_lambda = atlas_lambda
        self.atlas_loss = AtlasLoss(
            atlas_loss=atlas_loss,
            atlas_emd_regul=atlas_emd_regul,
            lambda_atlas=atlas_lambda,
            edge_regul_lambda=atlas_lambda_regul_edges,
            lambda_laplacian=atlas_lambda_laplacian,
            laplacian_faces=self.atlas_branch.test_faces,
            laplacian_verts=self.atlas_branch.test_verts,
        )
        self.atlas_mesh = atlas_mesh

    def decay_regul(self, gamma):
        self.atlas_loss.edge_regul_lambda = gamma * self.atlas_loss.edge_regul_lambda
        self.atlas_loss.lambda_laplacian = gamma * self.atlas_loss.lambda_laplacian

    def forward(
        self, sample, no_loss=False, return_features=False, force_objects=False
    ):
        inp_points3d = sample[TransQueries.objpoints3d]
        features = self.encoder(inp_points3d)
        total_loss = None
        results = {}
        losses = {}

        if TransQueries.objpoints3d in sample.keys() and self.atlas_lambda:
            if self.atlas_mesh:
                atlas_features = features
                atlas_results = self.atlas_branch.forward_inference(atlas_features)
            else:
                atlas_results = self.atlas_branch(features)
            for key, result in atlas_results.items():
                results[key] = result
            if not no_loss:
                atlas_total_loss, atlas_losses = self.atlas_loss.compute_loss(
                    atlas_results, sample
                )
                if total_loss is None:
                    total_loss = atlas_total_loss
                else:
                    total_loss += atlas_total_loss

                for key, val in atlas_losses.items():
                    losses[key] = val
        if total_loss is not None:
            losses["total_loss"] = total_loss
        else:
            losses["total_loss"] = None
        return total_loss, results, losses


class STN3d(nn.Module):
    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.identity = (
            torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))
            .view(1, 9)
            .cuda()
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch_f.relu(self.conv1(x))
        x = torch_f.relu(self.conv2(x))
        x = torch_f.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = torch_f.relu(self.fc1(x))
        x = torch_f.relu(self.fc2(x))
        x = self.fc3(x)

        identity = self.identity.repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(
        self, num_points=2500, global_feat=True, trans=False, feature_size=1024
    ):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.feature_size = feature_size
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_size)
        self.trans = trans

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = torch_f.relu(self.bn1(self.conv1(x.transpose(1, 2))))
        pointfeat = x
        x = torch_f.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.feature_size)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, self.feature_size, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x
