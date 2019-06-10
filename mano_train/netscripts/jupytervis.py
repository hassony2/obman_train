from collections import defaultdict, OrderedDict
import math
import os
import pickle

import pythreejs as p3js
from matplotlib import pyplot as plt
import numpy as np
import scipy
import torch
import trimesh
from tqdm import tqdm

from mano_train.visualize import visualizemeshes
from mano_train.networks.branches.contactloss import compute_contact_loss
from mano_train.netscripts.reload import save_obj

from handobjectdatasets.queries import BaseQueries, TransQueries


def get_samples_score_sorted(loader, model, max_samples=500, loss_name=None):
    all_samples = []
    total_losses = []
    all_losses = defaultdict(list)
    for sample_idx, sample in enumerate(tqdm(loader)):
        with torch.no_grad():
            total_loss, results, losses = model.forward(sample)
            if loss_name is None:
                total_losses.append(total_loss.item())
            else:
                if loss_name not in losses:
                    raise ValueError(
                        "{} not in {}".format(loss_name, list(losses.keys()))
                    )
                total_losses.append(losses[loss_name].item())
            for loss, loss_val in losses.items():
                if loss_val is not None:
                    all_losses[loss].append(loss_val.item())
            all_samples.append(sample)
            if sample_idx > max_samples:
                break
    total_losses = np.array(total_losses)
    sorted_idxs = np.argsort(total_losses)
    sorted_samples = [all_samples[idx] for idx in sorted_idxs]
    sorted_losses = [total_losses[idx] for idx in sorted_idxs]
    return sorted_samples, sorted_losses, all_losses


def get_samples_score_interval(
    sorted_samples, sorted_losses, interval=[0.95, 1], reverse=False
):
    assert (
        interval[0] >= 0 and interval[0] <= 1
    ), "bounds of interval should be in [0, 1], got lower bound {}".format(
        interval[0]
    )
    assert (
        interval[1] >= 0 and interval[1] <= 1
    ), "bounds of interval should be in [0, 1], got upper bound {}".format(
        interval[1]
    )
    assert (
        interval[0] < interval[1]
    ), "Lower bound {} should be lower then upper bound {}".format(
        interval[0], interval[1]
    )
    lower_idx = math.floor(interval[0] * len(sorted_losses))
    upper_idx = math.ceil(interval[1] * len(sorted_losses))
    selected_losses = sorted_losses[lower_idx:upper_idx]

    selected_samples = [
        sorted_samples[idx] for idx in range(lower_idx, upper_idx)
    ]
    if reverse:
        selected_losses = list(reversed(selected_losses))
        selected_samples = list(reversed(selected_samples))
    return selected_samples, selected_losses


def display_top_middle_worse(
    loader,
    model,
    max_samples=500,
    max_displays=5,
    mano_faces=None,
    loss_name=None,
    force_objects=False,
    display=True,
    top=True,
    mid=True,
    bottom=True,
    reverse_bottom=True,
):
    sorted_samples, sorted_losses, all_losses = get_samples_score_sorted(
        loader, model, max_samples=max_samples, loss_name=loss_name
    )
    if display:
        if top:
            top_samples, top_scores = get_samples_score_interval(
                sorted_samples, sorted_losses, interval=[0, 0.05]
            )
            print("Top 5% samples with scores {}".format(top_scores))
            show_meshes(
                top_samples,
                model,
                max_displays=max_displays,
                force_objects=force_objects,
                mano_faces=mano_faces,
            )
        if mid:
            mid_samples, mid_scores = get_samples_score_interval(
                sorted_samples, sorted_losses, interval=[0.45, 0.5]
            )
            print("45%-50% samples with scores {}".format(mid_scores))
            show_meshes(
                mid_samples,
                model,
                max_displays=max_displays,
                force_objects=force_objects,
                mano_faces=mano_faces,
            )
        if bottom:
            bottom_samples, bottom_scores = get_samples_score_interval(
                sorted_samples, sorted_losses, interval=[0.95, 1]
            )
            if reverse_bottom:
                bottom_samples = reversed(bottom_samples)
                bottom_scores = list(reversed(bottom_scores))
            print("95%-100% samples with scores {}".format(bottom_scores))
            show_meshes(
                bottom_samples,
                model,
                max_displays=max_displays,
                force_objects=force_objects,
                mano_faces=mano_faces,
            )
    return all_losses


def show_meshes(
    loader,
    model,
    mano_faces,
    max_displays=10,
    skip=0,
    force_objects=False,
    render=True,
    save=True,
    save_root="/sequoia/data2/yhasson/code/mano_train/data/results",
    show_contacts=False,
    show_gt=True,
    show_losses=[
        "mano_verts3d",
        "mano_joints3d",
        "atlas_objpoints3d",
        "atlas_scale3d",
        "atlas_trans3d",
        "final_chamfer_loss",
    ],
):
    """
    Can only be called from jupyter notebook because of 'display' function !
    loader(ConcatDataloader): dataloader
    model: trained neural network
    """
    renderers = []
    if isinstance(model, (list, tuple)):
        models = model
    else:
        models = [model]
    for i in range(2000):
        save_folder = os.path.join(save_root, "tmp_{:06d}".format(i))
        if not os.path.exists(save_folder):
            break

    all_filter_losses = []
    for sample_idx, sample in enumerate(loader):
        if sample_idx < skip:
            continue
        for model_idx, model in enumerate(models):
            with torch.no_grad():
                _, results, losses = model.forward(
                    sample, no_loss=False, force_objects=force_objects
                )
                if model_idx == 0:
                    filter_losses = OrderedDict(
                        (loss_name, [loss_val.item()])
                        for loss_name, loss_val in sorted(losses.items())
                        if loss_name in show_losses and loss_val is not None
                    )
                else:
                    for loss_name, loss_val in losses.items():
                        if loss_name in show_losses and loss_val is not None:
                            filter_losses[loss_name].append(loss_val.item())
                show_img = True if model_idx == 0 else False
                renderer = render_mesh(
                    sample,
                    results,
                    mano_faces=mano_faces,
                    sample_idx=sample_idx,
                    model_idx=model_idx,
                    save=save,
                    save_root=save_folder,
                    show_gt=show_gt,
                    show_img=show_img,
                    render=render,
                    show_contacts=show_contacts,
                )
                renderers.append(renderer)
                all_filter_losses.append(filter_losses)
        for loss_name, loss_vals in filter_losses.items():
            print_str = ""
            if len(loss_vals) == 1:
                print_str = "{}: {}".format(loss_name, loss_vals[0])
            else:
                for loss_val in loss_vals[1:]:
                    print_str = print_str + "{}: {} --> {} ({:.2f}%)".format(
                        loss_name,
                        loss_vals[0],
                        loss_val,
                        100 * (loss_val - loss_vals[0]) / loss_vals[0],
                    )
            print(print_str)
        if max_displays is not None and sample_idx >= max_displays:
            break
    return all_filter_losses


def save_meshes_dict(path, obj_verts, obj_faces, hand_verts, hand_faces):
    save_dict = {
        "hand_verts": hand_verts,
        "hand_faces": hand_faces,
        "obj_verts": obj_verts,
        "obj_faces": obj_faces,
    }

    with open(path, "wb") as p_f:
        pickle.dump(save_dict, p_f)
    print("Saved mesh dict to {}".format(path))


def render_mesh(
    sample,
    results,
    mano_faces=None,
    save=True,
    save_root="/sequoia/data2/yhasson/code/mano_train/data/results",
    sample_idx=0,
    model_idx=0,
    scale=0.001,
    show_gt=True,
    show_img=True,
    show_contacts=True,
    render=True,
):
    if show_img:
        inp_img = sample[TransQueries.images][0]
        inp_img = inp_img.permute(1, 2, 0).numpy() + 0.5

        # Display image
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.imshow(inp_img)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
    if "verts" in results:
        pred_verts = results["verts"][0].cpu().detach().numpy()
    else:
        pred_verts = None

    if "objpoints3d" in results:
        pred_faces = results["objfaces"]
        pred_points = results["objpoints3d"][0].cpu().detach().numpy()
        if pred_verts is None:
            # pred_points = pred_points * 100
            pred_points = pred_points * 1
    else:
        pred_faces = None
        pred_points = None

    if BaseQueries.objfaces in sample and BaseQueries.objverts3d in sample:
        gt_points = sample[BaseQueries.objverts3d][0]
        gt_faces = sample[BaseQueries.objfaces][0]
    # elif TransQueries.objpoints3d in sample:
    #     gt_points = sample[TransQueries.objpoints3d][0]
    #     gt_faces = None
    else:
        gt_points = None
        gt_faces = None

    if save:
        save_model_root = os.path.join(save_root, "model_{}".format(model_idx))
        os.makedirs(save_model_root, exist_ok=True)
        save_obj_folder = os.path.join(save_model_root, "objs")
        os.makedirs(save_obj_folder, exist_ok=True)
        save_hand_path = os.path.join(
            save_obj_folder, "{:08d}_hand.obj".format(sample_idx)
        )
        save_img_folder = os.path.join(save_model_root, "images")
        os.makedirs(save_img_folder, exist_ok=True)
        save_img_path = os.path.join(
            save_img_folder, "{:08d}.png".format(sample_idx)
        )
        img = (sample[TransQueries.images][0] + 0.5).permute(1, 2, 0)
        scipy.misc.toimage(img, cmin=0, cmax=1).save(save_img_path)
        scale = 0.001
        if pred_points is not None:
            save_pkl_folder = os.path.join(save_model_root, "pkls")
            os.makedirs(save_pkl_folder, exist_ok=True)
            save_path = os.path.join(
                save_pkl_folder, "mesh_penetr_{:04d}.pkl".format(sample_idx)
            )
            save_meshes_dict(
                save_path, pred_points, pred_faces, pred_verts, mano_faces
            )
            save_obj_path = os.path.join(
                save_obj_folder, "{:08d}_obj.obj".format(sample_idx)
            )
            obj_mesh = trimesh.load(
                {"vertices": pred_points, "faces": pred_faces}
            )
            trimesh.repair.fix_normals(obj_mesh)
            obj_verts = np.array(obj_mesh.vertices)
            obj_faces = np.array(obj_mesh.faces)
            save_obj(save_obj_path, obj_verts * scale, obj_faces)
        if pred_verts is not None:
            save_obj(save_hand_path, pred_verts * scale, mano_faces)
    if render:
        hand_obj_children = visualizemeshes.hand_obj_children(
            obj_verts=pred_points,
            obj_faces=pred_faces,
            gt_obj_verts=gt_points,
            gt_obj_faces=gt_faces,
            hand_verts=pred_verts,
            mano_faces_left=mano_faces,
        )

        point_light = p3js.DirectionalLight(
            intensity=0.6, position=[3, 5, 1], color="white"
        )
        c = p3js.PerspectiveCamera(
            position=[0, 0, -400], up=[0, 0, 1], children=[point_light]
        )
        scene_children_base = [c, p3js.AmbientLight(intensity=0.4)]
        scene_children = scene_children_base + hand_obj_children
        if show_contacts:
            miss_loss, pen_loss, contact_infos, metrics = compute_contact_loss(
                torch.Tensor(pred_verts).unsqueeze(0).cuda(),
                mano_faces,
                torch.Tensor(pred_points).unsqueeze(0).cuda(),
                pred_faces,
                contact_thresh=100,
                contact_zones="zones",
            )
            all_penetr_masks = contact_infos["repulsion_masks"]
            all_missed_masks = contact_infos["attraction_masks"]
            all_close_matches = contact_infos["contact_points"]
            penetrating_verts = (
                torch.Tensor(pred_verts)[all_penetr_masks[0]].cpu().numpy()
            )
            penetrating_close_verts = (
                all_close_matches[0][all_penetr_masks[0]].cpu().numpy()
            )
            missed_verts = (
                torch.Tensor(pred_verts)[all_missed_masks[0]].cpu().numpy()
            )
            missed_close_verts = (
                all_close_matches[0][all_missed_masks[0]].cpu().numpy()
            )

            attraction_lines_children = visualizemeshes.lines_children(
                missed_verts, missed_close_verts, color="green"
            )
            repulsion_lines_children = visualizemeshes.lines_children(
                penetrating_verts, penetrating_close_verts, color="orange"
            )
            scene_children = (
                scene_children
                + attraction_lines_children
                + repulsion_lines_children
            )
        if show_gt:
            if TransQueries.joints3d in sample:
                joint_children = visualizemeshes.joint_children(
                    sample[TransQueries.joints3d][0].numpy()
                )
                scene_children = scene_children + joint_children
            if TransQueries.objpoints3d in sample:
                obj_children = visualizemeshes.scatter_children(
                    sample[TransQueries.objpoints3d][0].numpy()
                )
                scene_children = scene_children + obj_children
        scene = p3js.Scene(children=scene_children)
        controls = p3js.OrbitControls(controlling=c)
        renderer = p3js.Renderer(
            camera=c, scene=scene, controls=[controls], width=400, height=400
        )
        # display(renderer)
    else:
        renderer = None
    return renderer
