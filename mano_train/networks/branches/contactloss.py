import numpy as np
import trimesh
import torch

from mano_train.networks.branches.contactutils import (
    batch_mesh_contains_points,
)
from handobjectdatasets import contactutils


def batch_index_select(inp, dim, index):
    views = [inp.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(inp.shape))
    ]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def thresh_ious(gt_dists, pred_dists, thresh):
    """
    Computes the contact intersection over union for a given threshold
    """
    gt_contacts = gt_dists <= thresh
    pred_contacts = pred_dists <= thresh
    inter = (gt_contacts * pred_contacts).sum(1).float()
    union = union = (gt_contacts | pred_contacts).sum(1).float()
    iou = torch.zeros_like(union)
    iou[union != 0] = inter[union != 0] / union[union != 0]
    return iou


def meshiou(gt_dists, pred_dists, threshs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """
    For each thresholds, computes thresh_ious and averages accross batch dim
    """
    all_ious = []
    for thresh in threshs:
        ious = thresh_ious(gt_dists, pred_dists, thresh)
        all_ious.append(ious)
    iou_auc = np.mean(
        np.trapz(torch.stack(all_ious).cpu().numpy(), axis=0, x=threshs)
    )
    batch_ious = torch.stack(all_ious).mean(1)
    return batch_ious, iou_auc


def masked_mean_loss(dists, mask):
    mask = mask.float()
    valid_vals = mask.sum()
    if valid_vals > 0:
        loss = (mask * dists).sum() / valid_vals
    else:
        loss = torch.Tensor([0]).cuda()
    return loss


def batch_pairwise_dist(x, y, use_cuda=True):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
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


def thres_loss(vals, thres=25):
    """
    Args:
        vals: positive values !
    """
    thres_mask = (vals < thres).float()
    loss = masked_mean_loss(vals, thres_mask)
    return loss


def compute_naive_contact_loss(points_1, points_2, contact_threshold=25):
    dists = batch_pairwise_dist(points_1, points_2)
    mins12, _ = torch.min(dists, 1)
    mins21, _ = torch.min(dists, 2)
    loss_1 = thres_loss(mins12, contact_threshold)
    loss_2 = thres_loss(mins21, contact_threshold)
    loss = torch.mean((loss_1 + loss_2) / 2)
    return loss


def mesh_vert_int_exts(obj1_mesh, obj2_verts, result_distance, tol=0.1):
    nonzero = result_distance > tol
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    penetrating = [sign == 1][0] & nonzero
    exterior = [sign == -1][0] & nonzero
    return penetrating, exterior


def get_contact_info(
    hand_vert,
    hand_faces,
    obj_vert,
    obj_faces,
    result_close=None,
    result_distance=None,
    contact_thresh=25,
):
    obj_mesh_dict = {"vertices": obj_vert, "faces": obj_faces}
    obj_mesh = trimesh.load(obj_mesh_dict)
    trimesh.repair.fix_normals(obj_mesh)
    # hand_mesh_dict = {'vertices': hand_vert, 'faces': hand_faces}
    # hand_mesh = trimesh.load(hand_mesh_dict)
    # trimesh.repair.fix_normals(hand_mesh)
    if result_close is None or result_distance is None:
        result_close, result_distance, _ = trimesh.proximity.closest_point(
            obj_mesh, hand_vert
        )
    penetrating, exterior = mesh_vert_int_exts(
        obj_mesh, hand_vert, result_distance
    )

    below_dist = result_distance < contact_thresh
    missed_mask = below_dist & exterior
    return result_close, missed_mask, penetrating


def get_depth_info(obj_mesh, hand_verts):
    result_close, result_distance, _ = trimesh.proximity.closest_point(
        obj_mesh, hand_verts
    )
    penetrating, exterior = mesh_vert_int_exts(
        obj_mesh, hand_verts, result_distance
    )
    return result_close, result_distance, penetrating


def compute_contact_loss(
    hand_verts_pt,
    hand_faces,
    obj_verts_pt,
    obj_faces,
    contact_thresh=5,
    contact_mode="dist_sq",
    collision_thresh=10,
    collision_mode="dist_sq",
    contact_target="all",
    contact_sym=False,
    contact_zones="all",
):
    # obj_verts_pt = obj_verts_pt.detach()
    # hand_verts_pt = hand_verts_pt.detach()
    dists = batch_pairwise_dist(hand_verts_pt, obj_verts_pt)
    mins12, min12idxs = torch.min(dists, 1)
    mins21, min21idxs = torch.min(dists, 2)

    # Get obj triangle positions
    obj_triangles = obj_verts_pt[:, obj_faces]
    exterior = batch_mesh_contains_points(
        hand_verts_pt.detach(), obj_triangles.detach()
    )
    penetr_mask = ~exterior
    results_close = batch_index_select(obj_verts_pt, 1, min21idxs)

    if contact_target == "all":
        anchor_dists = torch.norm(results_close - hand_verts_pt, 2, 2)
    elif contact_target == "obj":
        anchor_dists = torch.norm(results_close - hand_verts_pt.detach(), 2, 2)
    elif contact_target == "hand":
        anchor_dists = torch.norm(results_close.detach() - hand_verts_pt, 2, 2)
    else:
        raise ValueError(
            "contact_target {} not in [all|obj|hand]".format(contact_target)
        )
    if contact_mode == "dist_sq":
        # Use squared distances to penalize contact
        if contact_target == "all":
            contact_vals = ((results_close - hand_verts_pt) ** 2).sum(2)
        elif contact_target == "obj":
            contact_vals = ((results_close - hand_verts_pt.detach()) ** 2).sum(
                2
            )
        elif contact_target == "hand":
            contact_vals = ((results_close.detach() - hand_verts_pt) ** 2).sum(
                2
            )
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(
                    contact_target
                )
            )
        below_dist = mins21 < (contact_thresh ** 2)
    elif contact_mode == "dist":
        # Use distance to penalize contact
        contact_vals = anchor_dists
        below_dist = mins21 < contact_thresh
    elif contact_mode == "dist_tanh":
        # Use thresh * (dist / thresh) distances to penalize contact
        # (max derivative is 1 at 0)
        contact_vals = contact_thresh * torch.tanh(
            anchor_dists / contact_thresh
        )
        # All points are taken into account
        below_dist = torch.ones_like(mins21).byte()
    else:
        raise ValueError(
            "contact_mode {} not in [dist_sq|dist|dist_tanh]".format(
                contact_mode
            )
        )
    if collision_mode == "dist_sq":
        # Use squared distances to penalize contact
        if contact_target == "all":
            collision_vals = ((results_close - hand_verts_pt) ** 2).sum(2)
        elif contact_target == "obj":
            collision_vals = (
                (results_close - hand_verts_pt.detach()) ** 2
            ).sum(2)
        elif contact_target == "hand":
            collision_vals = (
                (results_close.detach() - hand_verts_pt) ** 2
            ).sum(2)
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(
                    contact_target
                )
            )
    elif collision_mode == "dist":
        # Use distance to penalize collision
        collision_vals = anchor_dists
    elif collision_mode == "dist_tanh":
        # Use thresh * (dist / thresh) distances to penalize contact
        # (max derivative is 1 at 0)
        collision_vals = collision_thresh * torch.tanh(
            anchor_dists / collision_thresh
        )
    else:
        raise ValueError(
            "collision_mode {} not in "
            "[dist_sq|dist|dist_tanh]".format(collision_mode)
        )

    missed_mask = below_dist & exterior
    if contact_zones == "tips":
        tip_idxs = [745, 317, 444, 556, 673]
        tips = torch.zeros_like(missed_mask)
        tips[:, tip_idxs] = 1
        missed_mask = missed_mask & tips
    elif contact_zones == "zones":
        _, contact_zones = contactutils.load_contacts(
            "assets/contact_zones.pkl"
        )
        contact_matching = torch.zeros_like(missed_mask)
        for zone_idx, zone_idxs in contact_zones.items():
            min_zone_vals, min_zone_idxs = mins21[:, zone_idxs].min(1)
            cont_idxs = mins12.new(zone_idxs)[min_zone_idxs]
            # For each batch keep the closest point from the contact zone
            contact_matching[
                [torch.range(0, len(cont_idxs) - 1).long(), cont_idxs.long()]
            ] = 1
        missed_mask = missed_mask & contact_matching
    elif contact_zones == "all":
        missed_mask = missed_mask
    else:
        raise ValueError(
            "contact_zones {} not in [tips|zones|all]".format(contact_zones)
        )

    # Apply losses with correct mask
    missed_loss = masked_mean_loss(contact_vals, missed_mask)
    penetr_loss = masked_mean_loss(collision_vals, penetr_mask)
    if contact_sym:
        obj2hand_dists = torch.sqrt(mins12)
        sym_below_dist = mins12 < contact_thresh
        sym_loss = masked_mean_loss(obj2hand_dists, sym_below_dist)
        missed_loss = missed_loss + sym_loss
    # print('penetr_nb: {}'.format(penetr_mask.sum()))
    # print('missed_nb: {}'.format(missed_mask.sum()))
    max_penetr_depth = (
        (anchor_dists.detach() * penetr_mask.float()).max(1)[0].mean()
    )
    mean_penetr_depth = (
        (anchor_dists.detach() * penetr_mask.float()).mean(1).mean()
    )
    contact_info = {
        "attraction_masks": missed_mask,
        "repulsion_masks": penetr_mask,
        "contact_points": results_close,
        "min_dists": mins21,
    }
    metrics = {
        "max_penetr": max_penetr_depth,
        "mean_penetr": mean_penetr_depth,
    }
    return missed_loss, penetr_loss, contact_info, metrics
