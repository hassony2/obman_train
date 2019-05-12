import argparse

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f
import trimesh

from handobjectdatasets.queries import TransQueries, BaseQueries

from mano_train.networks.branches import atlasutils, manobranch, atlasbranch
from manopth.manolayer import ManoLayer


class RefineBranch(nn.Module):
    def __init__(self,
                 img_feature_size,
                 mano_use_pca=True,
                 mano_comps=30,
                 mano_center_idx=9,
                 mano_root='misc/mano',
                 mano_use_shape=True,
                 mano_use_stereo_shape=False,
                 mano_base_neurons=[512],
                 atlas_mode='sphere'):

        super(RefineBranch, self).__init__()
        # Get size of mano input
        self.mano_use_pca = mano_use_pca
        if self.mano_use_pca:
            mano_pose_size = mano_comps + 3
        else:
            mano_pose_size = 16 * 9
        # Prepare atlas refiner
        self.atlas_decoder = atlasutils.PointGenCon(
            bottleneck_size=3 + img_feature_size + mano_pose_size,
            use_tanh=False)
        if atlas_mode == 'sphere':
            test_mesh = trimesh.creation.icosphere(subdivisions=3)

            # Initialize inference vertices and faces
            test_faces = np.array(test_mesh.faces)
            test_verts = test_mesh.vertices
        elif atlas_mode == 'disk':
            test_verts, test_faces = atlasutils.create_disk(14)
        else:
            raise ValueError('{} not in [sphere|disk]'.format(atlas_mode))
        self.test_verts = torch.Tensor(
            np.array(test_verts).astype(np.float32)).cuda()
        self.test_faces = test_faces
        self.mano_branch = manobranch.ManoBranch(
            ncomps=mano_comps,
            base_neurons=mano_base_neurons,
            use_trans=False,
            mano_root=mano_root,
            center_idx=mano_center_idx,
            refine=True,
            use_stereo_shape=mano_use_stereo_shape,
            use_shape=mano_use_shape,
            use_pca=mano_use_pca)

    def forward(self,
                img_features,
                objpoints=None,
                sides=None,
                mano_pose=None,
                root_palm=False):
        # Decode mano
        mano_results = self.mano_branch(
            img_features, sides=sides, root_palm=root_palm, pose=mano_pose)
        # Decode atlasnet
        test_grid = self.test_verts.unsqueeze(0).repeat(
            img_features.shape[0], 1, 1).transpose(2, 1)
        img_features_atlas = img_features.unsqueeze(2).repeat(
            1, 1, test_grid.size(2))
        mano_pose_atlas = mano_pose.unsqueeze(2).repeat(
            1, 1, test_grid.size(2))
        dec_img_features = torch.cat((test_grid, img_features_atlas,
                                      mano_pose_atlas), 1)
        atlas_verts = self.atlas_decoder(dec_img_features).transpose(2, 1)

        results = {
            'objpoints3d': atlas_verts,
            'objfaces': self.test_faces,
            'pose': mano_results['pose'],
            'shape': mano_results['shape'],
            'verts': mano_results['verts'],
            'joints': mano_results['joints']
        }
        return results


class RefineLoss():
    def __init__(self,
                 lambda_verts=None,
                 lambda_joints3d=None,
                 lambda_shape=None,
                 lambda_pca=None,
                 atlas_lambda=None,
                 atlas_lambda_regul_edges=None,
                 center_idx=9,
                 normalize_hand=False):
        self.lambda_verts = lambda_verts
        self.lambda_joints3d = lambda_joints3d
        self.lambda_shape = lambda_shape
        self.lambda_pca = lambda_pca
        self.center_idx = center_idx
        self.normalize_hand = normalize_hand
        self.mano_loss = manobranch.ManoLoss(
            lambda_verts=lambda_verts,
            lambda_joints3d=lambda_joints3d,
            lambda_shape=lambda_shape,
            lambda_pca=lambda_pca)
        self.atlas_loss = atlasbranch.AtlasLoss(
            lambda_atlas=atlas_lambda,
            trans_weight=False,
            edge_regul_lambda=atlas_lambda_regul_edges)

    def compute_loss(self, preds, target):
        # Get mano losses
        mano_total_loss_refine, mano_losses_refine = self.mano_loss.compute_loss(
            preds, target)
        total_loss = mano_total_loss_refine

        refine_losses = {}
        for key, val in mano_losses_refine.items():
            refine_losses[key + '_refine'] = val

        # Get atlas losses
        atlas_total_loss_refine, atlas_losses_refine = self.atlas_loss.compute_loss(
            preds, target)
        for key, val in atlas_losses_refine.items():
            refine_losses[key + '_refine'] = val
        total_loss += atlas_total_loss_refine

        return total_loss, refine_losses
