from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as torch_f

from mano_train.networks.bases import resnet
from mano_train.networks.branches.manobranch import ManoBranch, ManoLoss
from mano_train.networks.branches.atlasbranch import AtlasBranch, AtlasLoss
from mano_train.networks.branches.contactloss import (
    compute_contact_loss,
    batch_pairwise_dist,
    meshiou,
)
from mano_train.networks.branches.absolutebranch import AbsoluteBranch
from handobjectdatasets.queries import TransQueries, BaseQueries


class HandNet(nn.Module):
    def __init__(
        self,
        absolute_lambda=None,
        atlas_lambda=None,
        atlas_loss="chamfer",
        atlas_final_lambda=None,
        atlas_mesh=True,
        atlas_residual=False,
        atlas_lambda_regul_edges=0,
        atlas_lambda_laplacian=0,
        atlas_points_nb=600,
        atlas_predict_trans=False,
        atlas_trans_weight=1,
        atlas_predict_scale=False,
        atlas_scale_weight=1,
        atlas_use_tanh=False,
        atlas_ico_divisions=3,
        atlas_separate_encoder=False,
        atlas_out_factor=200,
        contact_target="all",
        contact_zones="all",
        contact_lambda=0,
        contact_thresh=25,
        contact_mode="dist_sq",
        collision_thresh=25,
        collision_mode="dist_sq",
        collision_lambda=0,
        fc_dropout=0,
        resnet_version=50,
        mano_adapt_skeleton=False,
        mano_neurons=[512],
        mano_comps=6,
        mano_use_shape=False,
        mano_lambda_pose_reg=0,
        mano_use_pca=True,
        mano_center_idx=9,
        mano_root="misc/mano",
        mano_lambda_joints3d=None,
        mano_lambda_joints2d=None,
        mano_lambda_verts=None,
        mano_lambda_shape=None,
        mano_lambda_pca=None,
        adapt_atlas_decoder=False,
    ):
        """
        Args:
            atlas_mesh (bool): Whether to get points on the mesh instead or
                randomling generating a point cloud. This allows to use
                regularizations that rely on an underlying triangulation
            atlas_ico_division: Granularity of the approximately spherical mesh
                see https://en.wikipedia.org/wiki/Geodesic_polyhedron.
                if 1, 42 vertices, if 2, 162 vertices, if 3 (default), 642
                vertices, if 4, 2562 vertices
            mano_root (path): dir containing mano pickle files
            mano_neurons: number of neurons in each layer of base mano decoder
            mano_use_pca: predict pca parameters directly instead of rotation
                angles
            mano_comps (int): number of principal components to use if
                mano_use_pca
            mano_lambda_pca: weight to supervise hand pose in PCA space
            mano_lambda_pose_reg: weight to supervise hand pose in axis-angle
                space
            mano_lambda_verts: weight to supervise vertex distances
            mano_lambda_joints3d: weight to supervise distances
            adapt_atlas_decoder: add layer between encoder and decoder, usefull
                when finetuning from separately pretrained encoder and decoder
        """
        super(HandNet, self).__init__()
        if int(resnet_version) == 18:
            img_feature_size = 512
            base_net = resnet.resnet18(pretrained=True)
        elif int(resnet_version) == 50:
            img_feature_size = 2048
            base_net = resnet.resnet50(pretrained=True)
        else:
            raise NotImplementedError(
                "Resnet {} not supported".format(resnet_version)
            )
        self.adapt_atlas_decoder = adapt_atlas_decoder
        self.atlas_separate_encoder = atlas_separate_encoder
        if self.adapt_atlas_decoder:
            self.atlas_adapter = torch.nn.Linear(
                img_feature_size, img_feature_size
            )
        mano_base_neurons = [img_feature_size] + mano_neurons
        self.contact_target = contact_target
        self.contact_zones = contact_zones
        self.contact_lambda = contact_lambda
        self.contact_thresh = contact_thresh
        self.contact_mode = contact_mode
        self.collision_lambda = collision_lambda
        self.collision_thresh = collision_thresh
        self.collision_mode = collision_mode
        if contact_lambda or collision_lambda:
            self.need_collisions = True
        else:
            self.need_collisions = False
        self.base_net = base_net
        if self.atlas_separate_encoder:
            self.atlas_base_net = deepcopy(base_net)

        self.absolute_lambda = absolute_lambda
        if mano_lambda_joints2d:
            self.scaletrans_branch = AbsoluteBranch(
                base_neurons=[img_feature_size, int(img_feature_size / 2)],
                out_dim=3,
            )

        self.mano_adapt_skeleton = mano_adapt_skeleton
        self.mano_branch = ManoBranch(
            ncomps=mano_comps,
            base_neurons=mano_base_neurons,
            adapt_skeleton=mano_adapt_skeleton,
            dropout=fc_dropout,
            use_trans=False,
            mano_root=mano_root,
            center_idx=mano_center_idx,
            use_shape=mano_use_shape,
            use_pca=mano_use_pca,
        )
        if (
            mano_lambda_verts
            or mano_lambda_joints3d
            or mano_lambda_joints2d
            or mano_lambda_pca
        ):
            self.mano_lambdas = True
        else:
            self.mano_lambdas = False
        self.mano_loss = ManoLoss(
            lambda_verts=mano_lambda_verts,
            lambda_joints3d=mano_lambda_joints3d,
            lambda_shape=mano_lambda_shape,
            lambda_pose_reg=mano_lambda_pose_reg,
            lambda_pca=mano_lambda_pca,
        )

        self.lambda_joints2d = mano_lambda_joints2d
        self.atlas_mesh = atlas_mesh
        feature_size = img_feature_size
        self.atlas_branch = AtlasBranch(
            mode="sphere",
            use_residual=atlas_residual,
            points_nb=atlas_points_nb,
            predict_trans=atlas_predict_trans,
            predict_scale=atlas_predict_scale,
            inference_ico_divisions=atlas_ico_divisions,
            bottleneck_size=feature_size,
            use_tanh=atlas_use_tanh,
            out_factor=atlas_out_factor,
            separate_encoder=self.atlas_separate_encoder,
        )
        self.atlas_lambda = atlas_lambda
        self.atlas_final_lambda = atlas_final_lambda
        self.atlas_trans_weight = atlas_trans_weight
        self.atlas_scale_weight = atlas_scale_weight
        self.atlas_loss = AtlasLoss(
            atlas_loss=atlas_loss,
            lambda_atlas=atlas_lambda,
            final_lambda_atlas=atlas_final_lambda,
            trans_weight=atlas_trans_weight,
            scale_weight=atlas_scale_weight,
            edge_regul_lambda=atlas_lambda_regul_edges,
            lambda_laplacian=atlas_lambda_laplacian,
            laplacian_faces=self.atlas_branch.test_faces,
            laplacian_verts=self.atlas_branch.test_verts,
        )

    def decay_regul(self, gamma):
        if self.atlas_loss.edge_regul_lambda is not None:
            self.atlas_loss.edge_regul_lambda = (
                gamma * self.atlas_loss.edge_regul_lambda
            )
        if self.atlas_loss.lambda_laplacian is not None:
            self.atlas_loss.lambda_laplacian = (
                gamma * self.atlas_loss.lambda_laplacian
            )

    def forward(
        self, sample, no_loss=False, return_features=False, force_objects=False
    ):
        if force_objects:
            if TransQueries.objpoints3d not in sample:
                sample[TransQueries.objpoints3d] = None
        total_loss = None
        results = {}
        losses = {}
        image = sample[TransQueries.images].cuda()
        features, _ = self.base_net(image)
        if self.atlas_separate_encoder:
            atlas_infeatures, _ = self.atlas_base_net(image)
            if return_features:
                results["atlas_features"] = atlas_infeatures
        if return_features:
            results["img_features"] = features

        if (
            self.absolute_lambda
            and TransQueries.center3d in sample
            and (TransQueries.camintrs in sample)
        ):
            predict_center = True
            supervise_center = True
        elif TransQueries.camintrs in sample and self.lambda_joints2d:
            predict_center = True
            supervise_center = False
        else:
            predict_center = False
            supervise_center = False
        if predict_center:
            focals = sample[TransQueries.camintrs][:, 0, 0]
            u_0 = sample[TransQueries.camintrs][:, 0, 2]
            v_0 = sample[TransQueries.camintrs][:, 1, 2]
            absolute_input = torch.cat(
                (
                    focals.unsqueeze(1),
                    u_0.unsqueeze(1),
                    v_0.unsqueeze(1),
                    features,
                ),
                dim=1,
            )
            pred_center3d = self.absolute_branch(absolute_input)
            results["center3d"] = pred_center3d
            if not no_loss and supervise_center:
                absolute_loss = torch_f.mse_loss(
                    pred_center3d, sample[TransQueries.center3d]
                ).view(1)
                if total_loss is None:
                    total_loss = absolute_loss
                else:
                    total_loss += self.absolute_lambda * absolute_loss
                losses["absolute_loss"] = absolute_loss
        if (
            (
                TransQueries.joints3d in sample.keys()
                or TransQueries.verts3d in sample.keys()
                or (
                    TransQueries.joints2d in sample.keys()
                    and TransQueries.camintrs in sample.keys()
                )
            )
            and BaseQueries.sides in sample.keys()
            and self.mano_lambdas
        ):
            if sample["root"] == "palm":
                root_palm = True
            else:
                root_palm = False
            mano_results = self.mano_branch(
                features,
                sides=sample[BaseQueries.sides],
                root_palm=root_palm,
                use_stereoshape=False,
            )
            if not no_loss:
                mano_total_loss, mano_losses = self.mano_loss.compute_loss(
                    mano_results, sample
                )
                if total_loss is None:
                    total_loss = mano_total_loss
                else:
                    total_loss += mano_total_loss

                for key, val in mano_losses.items():
                    losses[key] = val

            for key, result in mano_results.items():
                results[key] = result

            if self.lambda_joints2d:
                scaletrans = self.scaletrans_branch(features)
                trans = scaletrans[:, 1:]
                # Abs to make sure no inversion in scale
                scale = torch.abs(scaletrans[:, :1])

                # Trans is multiplied by 100 to make scale and trans updates
                # of same magnitude after 2d joints supervision
                # (100 is ~ the scale of the 2D joint coordinate values)
                proj_joints2d = mano_results["joints"][
                    :, :, :2
                ] * scale.unsqueeze(1) + 100 * trans.unsqueeze(1)
                results["joints2d"] = proj_joints2d
                if not no_loss:
                    gt_joints2d = sample[TransQueries.joints2d].cuda().float()
                    joints2d_loss = torch_f.mse_loss(
                        proj_joints2d, gt_joints2d
                    )
                    losses["joints2d"] = joints2d_loss
                    total_loss += self.lambda_joints2d * joints2d_loss
        predict_atlas = TransQueries.objpoints3d in sample.keys() and (
            self.atlas_lambda or self.atlas_final_lambda
        )
        if predict_atlas:
            if self.atlas_mesh:
                if self.adapt_atlas_decoder:
                    atlas_features = self.atlas_adapter(features)
                else:
                    atlas_features = features
                if self.atlas_separate_encoder:
                    atlas_results = self.atlas_branch.forward_inference(
                        atlas_features,
                        separate_encoder_features=atlas_infeatures,
                    )
                else:
                    atlas_results = self.atlas_branch.forward_inference(
                        atlas_features
                    )
            else:
                atlas_results = self.atlas_branch(features)
            if self.need_collisions:
                (
                    attr_loss,
                    penetr_loss,
                    contact_infos,
                    contact_metrics,
                ) = compute_contact_loss(
                    mano_results["verts"],
                    self.mano_branch.faces,
                    atlas_results["objpoints3d"],
                    self.atlas_branch.test_faces,
                    contact_thresh=self.contact_thresh,
                    contact_mode=self.contact_mode,
                    collision_thresh=self.collision_thresh,
                    collision_mode=self.collision_mode,
                    contact_target=self.contact_target,
                    contact_zones=self.contact_zones,
                )
                if not no_loss:
                    if (
                        TransQueries.verts3d in sample
                        and TransQueries.objpoints3d in sample
                    ):
                        h2o_dists = batch_pairwise_dist(
                            sample[TransQueries.verts3d],
                            sample[TransQueries.objpoints3d],
                        )
                        dist_h2o_gt, _ = torch.min(h2o_dists, 2)
                        contact_ious, contact_auc = meshiou(
                            dist_h2o_gt, contact_infos["min_dists"]
                        )
                        contact_infos["batch_ious"] = contact_ious
                        losses["contact_auc"] = contact_auc
                    contact_loss = (
                        self.contact_lambda * attr_loss
                        + self.collision_lambda * penetr_loss
                    )
                    total_loss += contact_loss
                    losses["penetration_loss"] = penetr_loss
                    losses["attraction_loss"] = attr_loss
                    losses["contact_loss"] = contact_loss
                    for metric_name, metric_val in contact_metrics.items():
                        losses[metric_name] = metric_val
                results["contact_info"] = contact_infos
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
