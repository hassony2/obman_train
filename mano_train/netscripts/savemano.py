import pickle
from enum import Enum

import numpy as np
import trimesh


def load_batch_info(
    save_path, faces_right, faces_left, scale=0.001, get_depth=True
):
    with open(save_path, "rb") as p_f:
        batch_data = pickle.load(p_f)
    sample, results = batch_data["sample"], batch_data["results"]
    obj_faces = results["objfaces"]
    hand_verts = results["verts"]
    obj_verts = results["objpoints3d"]
    penetr_masks = results["contact_info"]["repulsion_masks"]

    hand_faces = []
    for side in sample["sides"]:
        if side == "right":
            hand_faces.append(faces_right)
        else:
            hand_faces.append(faces_left)
    hand_faces = np.stack(hand_faces)
    sample_infos = []
    for hand_vert, hand_face, obj_vert, penetr_mask in zip(
        hand_verts, hand_faces, obj_verts, penetr_masks
    ):
        obj_mesh = trimesh.load({"vertices": obj_vert, "faces": obj_faces})
        trimesh.repair.fix_normals(obj_mesh)

        sample_info = {
            "hand_verts": hand_vert * scale,
            "hand_faces": hand_face,
            "obj_verts": np.array(obj_mesh.vertices) * scale,
            "obj_faces": np.array(obj_mesh.faces),
        }
        if get_depth:
            if penetr_mask.sum() == 0:
                max_depth = 0
            else:
                (
                    result_close,
                    result_distance,
                    _,
                ) = trimesh.proximity.closest_point(
                    obj_mesh, hand_vert[penetr_mask == 1]
                )
                max_depth = result_distance.max()
            sample_info["max_depth"] = max_depth
        sample_infos.append(sample_info)

    return sample_infos


def save_batch_info(save_path, results, sample):
    untensor_results = untensor(results)
    untensor_sample = untensor(sample)

    with open(save_path, "wb") as p_f:
        pickle.dump(
            {"sample": untensor_sample, "results": untensor_results}, p_f
        )


def untensor(results):
    import torch

    new_results = {}
    for key, value in results.items():
        if isinstance(key, Enum):
            save_key = key.value
        else:
            save_key = key
        if isinstance(results[key], torch.Tensor):
            new_results[save_key] = value.detach().cpu().numpy()
        elif isinstance(value, dict):
            new_results[save_key] = untensor(value)
        else:
            new_results[save_key] = value
    return new_results
