import argparse

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch import nn

import argutils  # Requires myana to be in PYTHONPATH
from mano_network.models import resnet
from mano_train.networks.branches.manobranch import ManoBranch


class ManoNet(nn.Module):
    def __init__(
        self,
        base_net,
        base_neurons=[2048, 512],
        ncomps=6,
        center_idx=9,
        use_shape=False,
        use_trans=False,
        mano_root="misc/mano",
    ):
        """
        Args:
            mano_root (path): dir containing mano pickle files
        """
        super(ManoNet, self).__init__()
        self.base_net = base_net
        self.mano_branch = ManoBranch(
            ncomps=ncomps,
            base_neurons=base_neurons,
            use_trans=use_trans,
            use_shape=use_shape,
            mano_root=mano_root,
            center_idx=center_idx,
        )

    # @profile
    def forward(self, images, sides):
        features, _ = self.base_net(images)
        results = self.mano_branch(features, sides=sides)
        return results


class HandRegNet(nn.Module):
    def __init__(
        self,
        base_net,
        expansion=4,
        joint_nb=21,
        hidden_neurons=1024,
        coord_dim=3,
        return_inter=False,
    ):
        super(HandRegNet, self).__init__()
        self.joint_nb = joint_nb
        self.base_net = base_net
        self.coord_dim = coord_dim
        self.return_inter = return_inter

        if expansion == 4:
            interm_hidden = 1024
        elif expansion == 1:
            interm_hidden = 512
        self.classifier = torch.nn.Sequential(
            nn.Linear(512 * expansion, interm_hidden),
            nn.ReLU(),
            nn.Linear(interm_hidden, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, joint_nb * coord_dim),
        )

        # @profile

    def forward(self, inp):
        if self.return_inter:
            features, inter = self.base_net(inp)
            inter["res_features"] = features.cpu()
            modulelist = list(self.classifier.modules())
            for l in modulelist[1:3]:
                features = l(features)
            inter["clas_hidden_1"] = features.cpu()
            for l in modulelist[3:5]:
                features = l(features)
            inter["clas_hidden_2"] = features.cpu()
            for l in modulelist[5:]:
                joints = l(features)
            inter["joints"] = joints.cpu()
            joints = joints.view(-1, self.joint_nb, self.coord_dim)
            return {"joints": joints, "inter": inter}
        else:
            features, _ = self.base_net(inp)
            joints = self.classifier(features).view(-1, self.joint_nb, self.coord_dim)
            return {"joints": joints}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--network", choices=["mano", "reg"], default="mano")
    parser.add_argument("--use_double", action="store_true")
    parser.add_argument("--center_idx", type=int, default=9)
    args = parser.parse_args()
    argutils.print_args(args)
    n_components = 6
    rot = 3

    base_model = resnet.resnet50(pretrained=True)
    if args.network == "mano":
        network = ManoNet(base_model, ncomps=n_components, center_idx=args.center_idx)
    elif args.network == "reg":
        network = HandRegNet(base_model)

    inputs = torch.rand(args.batch_size, 3, 100, 100)
    if args.use_double:
        network = network.double()
        inputs = inputs.double()

    if args.cuda:
        inputs = inputs.cuda()
        base_model = base_model.cuda()
        network = network.cuda()
    outputs = network(inputs, sides=["right"] * args.batch_size)
    if args.display:
        if "verts" in outputs:
            verts = outputs["verts"]
        else:
            verts = None
        Jtr = outputs["joints"]
        if args.cuda:
            verts = verts.cpu()
            Jtr = Jtr.cpu()
        joints = Jtr.data.numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if verts is not None:
            verts = verts.data.numpy()
            ax.scatter(verts[0, :, 0], verts[0, :, 1], verts[0, :, 2])
        ax.scatter(joints[0, :, 0], joints[0, :, 1], joints[0, :, 2])
        plt.show()
