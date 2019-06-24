import pickle
import os
import traceback
import warnings

import torch

from mano_train.networks.handnet import HandNet
from mano_train.datautils import ConcatDataloader
from mano_train.netscripts.get_datasets import get_dataset
from mano_train.modelutils import modelio

from handobjectdatasets.queries import BaseQueries, TransQueries


def save_obj(filename, verticies, faces):
    with open(filename, "w") as fp:
        for v in verticies:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))


def get_opts(resume_checkpoint):
    if resume_checkpoint.endswith("tar"):
        resume_checkpoint = os.path.join(
            "/", *resume_checkpoint.split("/")[:-1]
        )
    opt_path = os.path.join(resume_checkpoint, "opt.pkl")
    with open(opt_path, "rb") as p_f:
        opts = pickle.load(p_f)
    return opts


def reload_model(
    model_path,
    checkpoint_opts,
    mano_root="misc/mano",
    ico_divisions=3,
    no_beta=False,
):
    if "absolute_lambda" not in checkpoint_opts:
        checkpoint_opts["absolute_lambda"] = 0
    if "atlas_predict_trans" not in checkpoint_opts:
        checkpoint_opts["atlas_predict_trans"] = False
    if "atlas_lambda_laplacian" not in checkpoint_opts:
        checkpoint_opts["atlas_lambda_laplacian"] = False
    if "atlas_residual" not in checkpoint_opts:
        checkpoint_opts["atlas_residual"] = False
    if "mano_lambda_joints3d" not in checkpoint_opts:
        checkpoint_opts["mano_lambda_joints3d"] = False
    if "mano_lambda_joints2d" not in checkpoint_opts:
        checkpoint_opts["mano_lambda_joints2d"] = False
    if "mano_adapt_skeleton" not in checkpoint_opts:
        checkpoint_opts["mano_adapt_skeleton"] = False
    if "contact_lambda" not in checkpoint_opts:
        checkpoint_opts["contact_lambda"] = 0
    if "collision_lambda" not in checkpoint_opts:
        checkpoint_opts["collision_lambda"] = 0
    if "mano_use_pca" not in checkpoint_opts:
        checkpoint_opts["mano_use_pca"] = True
    if "atlas_separate_encoder" not in checkpoint_opts:
        checkpoint_opts["atlas_separate_encoder"] = False
    if "atlas_final_lambda" not in checkpoint_opts:
        checkpoint_opts["atlas_final_lambda"] = 0

    if no_beta:
        mano_use_shape = False
    else:
        mano_use_shape = checkpoint_opts["use_shape"]
    if "atlas_predict_scale" not in checkpoint_opts:
        checkpoint_opts["atlas_predict_scale"] = False

    model = HandNet(
        resnet_version=18,
        absolute_lambda=checkpoint_opts["absolute_lambda"],
        atlas_mesh=True,
        atlas_points_nb=642,
        atlas_lambda_regul_edges=checkpoint_opts["atlas_lambda_regul_edges"],
        atlas_lambda_laplacian=checkpoint_opts["atlas_lambda_laplacian"],
        atlas_predict_trans=checkpoint_opts["atlas_predict_trans"],
        atlas_predict_scale=checkpoint_opts["atlas_predict_scale"],
        atlas_residual=checkpoint_opts["atlas_residual"],
        atlas_lambda=checkpoint_opts["atlas_lambda"],
        atlas_final_lambda=checkpoint_opts["atlas_final_lambda"],
        atlas_ico_divisions=ico_divisions,
        atlas_separate_encoder=checkpoint_opts["atlas_separate_encoder"],
        contact_lambda=checkpoint_opts["contact_lambda"],
        collision_lambda=checkpoint_opts["collision_lambda"],
        mano_adapt_skeleton=checkpoint_opts["mano_adapt_skeleton"],
        mano_root=mano_root,
        mano_center_idx=checkpoint_opts["center_idx"],
        mano_comps=30,
        mano_neurons=checkpoint_opts["hidden_neurons"],
        mano_use_shape=mano_use_shape,
        mano_use_pca=checkpoint_opts["mano_use_pca"],
        mano_lambda_verts=checkpoint_opts["mano_lambda_verts"],
        mano_lambda_joints3d=checkpoint_opts["mano_lambda_joints3d"],
        mano_lambda_joints2d=checkpoint_opts["mano_lambda_joints2d"],
    )
    model = torch.nn.DataParallel(model)
    model.eval()
    try:
        modelio.load_checkpoint(model, resume_path=model_path, strict=True)
    except RuntimeError:
        traceback.print_exc()
        warnings.warn(
            "Couldn' load model in strict mode, trying without strict"
        )
        modelio.load_checkpoint(model, resume_path=model_path, strict=False)
    return model


def get_loader(
    dataset_names,
    metas,
    checkpoint_opts,
    max_queries=[
        TransQueries.affinetrans,
        TransQueries.images,  # TransQueries.segms,
        TransQueries.verts3d,
        TransQueries.center3d,
        TransQueries.joints3d,
        TransQueries.objpoints3d,
        TransQueries.camintrs,
        # BaseQueries.objverts3d,
        # BaseQueries.objfaces,
        BaseQueries.sides,
    ],
    shuffle=False,
    mini_factor=0.01,
):
    loaders = []
    for dat, meta in zip(dataset_names, metas):
        dataset = get_dataset(
            dat,
            split=meta["split"],
            max_queries=max_queries,
            mini_factor=mini_factor,
            meta=meta,
            center_idx=checkpoint_opts["center_idx"],
            train_it=False,
            point_nb=642,
            sides="left",
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=shuffle, num_workers=0
        )
        loaders.append(loader)
    concat_loader = ConcatDataloader(loaders)
    return concat_loader
