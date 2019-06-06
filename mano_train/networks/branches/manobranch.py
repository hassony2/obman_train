import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f

from manopth.manolayer import ManoLayer

from handobjectdatasets.queries import TransQueries, BaseQueries


class ManoBranch(nn.Module):
    def __init__(
        self,
        ncomps=6,
        base_neurons=[1024, 512],
        center_idx=9,
        use_shape=False,
        use_trans=False,
        use_pca=True,
        mano_root="misc/mano",
        adapt_skeleton=True,
        dropout=0,
    ):
        """
        Args:
            mano_root (path): dir containing mano pickle files
        """
        super(ManoBranch, self).__init__()

        self.adapt_skeleton = adapt_skeleton
        self.use_trans = use_trans
        self.use_shape = use_shape
        self.use_pca = use_pca
        self.stereo_shape = torch.Tensor(
            [
                -0.00298099,
                -0.0013994,
                -0.00840144,
                0.00362311,
                0.00248761,
                0.00044125,
                0.00381337,
                -0.00183374,
                -0.00149655,
                0.00137479,
            ]
        ).cuda()

        if self.use_pca:
            # pca comps + 3 global axis-angle params
            mano_pose_size = ncomps + 3
        else:
            # 15 joints + 1 global rotations, 9 comps per rot
            mano_pose_size = 16 * 9
        # Base layers
        base_layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(
            zip(base_neurons[:-1], base_neurons[1:])
        ):
            if dropout:
                base_layers.append(nn.Dropout(p=dropout))
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.ReLU())
        self.base_layer = nn.Sequential(*base_layers)

        # Pose layers
        self.pose_reg = nn.Linear(base_neurons[-1], mano_pose_size)
        if not self.use_pca:
            # Initialize all nondiagonal items on rotation matrix weights to 0
            self.pose_reg.bias.data.fill_(0)
            weight_mask = (
                self.pose_reg.weight.data.new(np.identity(3))
                .view(9)
                .repeat(16)
            )
            self.pose_reg.weight.data = torch.abs(
                weight_mask.unsqueeze(1).repeat(1, 256).float()
                * self.pose_reg.weight.data
            )

        # Shape layers
        if self.use_shape:
            self.shape_reg = torch.nn.Sequential(
                nn.Linear(base_neurons[-1], 10)
            )

        # Trans layers
        if self.use_trans:
            self.trans_reg = nn.Linear(base_neurons[-1], 3)

        # Mano layers
        self.mano_layer_right = ManoLayer(
            ncomps=ncomps,
            center_idx=center_idx,
            side="right",
            mano_root=mano_root,
            use_pca=use_pca,
        )
        self.mano_layer_left = ManoLayer(
            ncomps=ncomps,
            center_idx=center_idx,
            side="left",
            mano_root=mano_root,
            use_pca=use_pca,
        )
        if self.adapt_skeleton:
            joint_nb = 21
            self.left_skeleton_reg = nn.Linear(joint_nb, joint_nb, bias=False)
            self.left_skeleton_reg.weight.data = torch.eye(joint_nb)
            self.right_skeleton_reg = nn.Linear(joint_nb, joint_nb, bias=False)
            self.right_skeleton_reg.weight.data = torch.eye(joint_nb)

        self.faces = self.mano_layer_right.th_faces

    def forward(
        self,
        inp,
        sides,
        root_palm=False,
        shape=None,
        pose=None,
        use_stereoshape=False,
    ):
        base_features = self.base_layer(inp)
        pose = self.pose_reg(base_features)
        if not self.use_pca:
            # Reshape to rotation matrixes
            mano_pose = pose.reshape(pose.shape[0], 16, 3, 3)
        else:
            mano_pose = pose

        # Prepare for splitting batch in right hands and left hands
        is_rights = inp.new_tensor([side == "right" for side in sides]).byte()
        is_lefts = 1 - is_rights
        is_rights = is_rights[: pose.shape[0]]
        is_lefts = is_lefts[: pose.shape[0]]

        # Get shape
        if use_stereoshape:
            shape = self.stereo_shape.unsqueeze(0).repeat(inp.shape[0], 1)
            shape_right = shape[is_rights]
            shape_left = shape[is_lefts]
            assert (
                is_rights.sum() == 0
            ), "When stereoshape is used only left hands expected"
        elif self.use_shape:
            shape = self.shape_reg(base_features)
            shape_right = shape[is_rights]
            shape_left = shape[is_lefts]
        else:
            shape = None
            shape_right = None
            shape_left = None

        # Get trans
        if self.use_trans:
            trans = self.trans_reg(base_features)
            trans_right = trans[is_rights]
            trans_left = trans[is_lefts]
        else:
            trans_right = torch.Tensor([0])
            trans_left = torch.Tensor([0])

        # Get pose
        pose_right = mano_pose[is_rights]
        pose_left = mano_pose[is_lefts]

        # Pass through mano_right and mano_left layers
        if pose_right.shape[0] > 0:
            verts_right, joints_right = self.mano_layer_right(
                pose_right,
                th_betas=shape_right,
                th_trans=trans_right,
                root_palm=root_palm,
            )
        if pose_left.shape[0] > 0:
            verts_left, joints_left = self.mano_layer_left(
                pose_left,
                th_betas=shape_left,
                th_trans=trans_left,
                root_palm=root_palm,
            )
        if self.adapt_skeleton:
            if len(joints_left) != 0:
                joints_left = self.left_skeleton_reg(
                    joints_left.permute(0, 2, 1)
                ).permute(0, 2, 1)
            if len(joints_right) != 0:
                joints_right = self.right_skeleton_reg(
                    joints_right.permute(0, 2, 1)
                ).permute(0, 2, 1)

        # Reassemble rights and lefts
        verts = inp.new_empty((inp.shape[0], 778, 3))
        joints = inp.new_empty((inp.shape[0], 21, 3))
        if pose_right.shape[0] > 0:
            verts[is_rights] = verts_right
            joints[is_rights] = joints_right
        if pose_left.shape[0] > 0:
            verts[is_lefts] = verts_left
            joints[is_lefts] = joints_left
        if shape is not None:
            shape = inp.new_empty((inp.shape[0], 10))
            if pose_right.shape[0] > 0:
                shape[is_rights] = shape_right
            if pose_left.shape[0] > 0:
                shape[is_lefts] = shape_left

        # Gather results
        results = {
            "verts": verts,
            "joints": joints,
            "shape": shape,
            "pose": pose,
        }
        if self.use_trans:
            results["trans"] = trans
        return results


def get_bone_ratio(pred_joints, target_joints, link=(9, 10)):
    bone_ref = torch.norm(
        target_joints[:, link[1]] - target_joints[:, link[0]], dim=1
    )
    bone_pred = torch.norm(
        pred_joints[:, link[1]] - pred_joints[:, link[0]], dim=1
    )
    bone_ratio = bone_ref / bone_pred
    return bone_ratio


class ManoLoss:
    def __init__(
        self,
        lambda_verts=None,
        lambda_joints3d=None,
        lambda_shape=None,
        lambda_pose_reg=None,
        lambda_pca=None,
        center_idx=9,
        normalize_hand=False,
    ):
        self.lambda_verts = lambda_verts
        self.lambda_joints3d = lambda_joints3d
        self.lambda_shape = lambda_shape
        self.lambda_pose_reg = lambda_pose_reg
        self.lambda_pca = lambda_pca
        self.center_idx = center_idx
        self.normalize_hand = normalize_hand

    def compute_loss(self, preds, target):
        final_loss = torch.Tensor([0]).cuda()
        mano_losses = {}

        # If needed, compute and add vertex loss
        if TransQueries.verts3d in target and self.lambda_verts:
            verts3d_loss = torch_f.mse_loss(
                preds["verts"], target[TransQueries.verts3d]
            )
            final_loss += self.lambda_verts * verts3d_loss
            verts3d_loss = verts3d_loss
        else:
            verts3d_loss = None
        mano_losses["mano_verts3d"] = verts3d_loss

        # Compute joints loss in all cases
        if TransQueries.joints3d in target:
            pred_joints = preds["joints"]
            target_joints = target[TransQueries.joints3d]
            if self.normalize_hand:
                print("=== Bone ratios ===")
                links = [
                    (0, 1, 2, 3, 4),
                    (0, 5, 6, 7, 8),
                    (0, 9, 10, 11, 12),
                    (0, 13, 14, 15, 16),
                    (0, 17, 18, 19, 20),
                ]
                for link in links:
                    for joint_idx, n_joint_idx in zip(link[:-1], link[1:]):
                        bone_ratio = get_bone_ratio(
                            pred_joints,
                            target_joints,
                            link=(joint_idx, n_joint_idx),
                        )
                        print(
                            "({}, {}) :{}".format(
                                joint_idx, n_joint_idx, torch.mean(bone_ratio)
                            )
                        )

            # Add to final_loss for backpropagation if needed
            if TransQueries.joints3d in target and self.lambda_joints3d:
                joints3d_loss = torch_f.mse_loss(pred_joints, target_joints)
                final_loss += self.lambda_joints3d * joints3d_loss
                mano_losses["mano_joints3d"] = joints3d_loss

        if self.lambda_shape:
            shape_loss = torch_f.mse_loss(
                preds["shape"], torch.zeros_like(preds["shape"])
            )
            final_loss += self.lambda_shape * shape_loss
            shape_loss = shape_loss
        else:
            shape_loss = None
        mano_losses["mano_shape"] = shape_loss
        if self.lambda_pose_reg:
            pose_reg_loss = torch_f.mse_loss(
                preds["pose"][:, 3:], torch.zeros_like(preds["pose"][:, 3:])
            )
            final_loss += self.lambda_pose_reg * pose_reg_loss
            mano_losses["pose_reg"] = pose_reg_loss

        if BaseQueries.hand_pcas in target and self.lambda_pca:
            pca_loss = torch_f.mse_loss(
                preds["pcas"], target[BaseQueries.hand_pcas]
            )
            final_loss += self.lambda_pca * pca_loss
            pca_loss = pca_loss
        else:
            pca_loss = None
        mano_losses["mano_pca"] = pca_loss
        mano_losses["mano_total_loss"] = final_loss
        return final_loss, mano_losses
