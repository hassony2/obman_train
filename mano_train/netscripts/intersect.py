import json
import os
import traceback
import pickle

from joblib import Parallel, delayed
import numpy as np
import trimesh
from matplotlib import pyplot as plt

from mano_train.objectutils.objectio import load_obj
from mano_train.netscripts.savemano import load_batch_info

closed_hand = 'assets/mano/closed_mano.obj'
_, hand_faces = load_obj(closed_hand, normalization=False)

@profile
def intersect_vox(obj_mesh, hand_mesh, pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


@profile
def intersect(obj_mesh, hand_mesh, engine='auto'):
    trimesh.repair.fix_normals(obj_mesh)
    inter_mesh = obj_mesh.intersection(hand_mesh, engine=engine)
    return inter_mesh

def get_obj_intersect_volume(hand_path, obj_path):
    hand_verts, _ = load_obj(hand_obj, normalization=False)
    obj_mesh = trimesh.load(obj_path)
    closed_hand = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)
    intersection = intersect(obj_mesh, closed_hand)
    volume = intersection.volume
    return volume

def get_all_volumes(exp_id, batch_step=1, workers=8):
    save_pickles = sorted([
        os.path.join(exp_id, filename) for filename in os.listdir(exp_id)
        if '.pkl' in filename
        ])
    batch_infos = Parallel(
            n_jobs=workers,
            verbose=5)(delayed(load_batch_info)(
                save_pickle, faces_right=hand_faces, faces_left=hand_faces)
                for save_pickle in save_pickles[::batch_step])
            # Prepare simulation storing results
    sample_infos = [
            sample_info for batch_info in batch_infos for sample_info in batch_info
            ]
    volumes = get_volumes_from_samples(sample_infos, workers=workers)
    volumes_clean = [volume for volume in volumes if volume is not None]
    skipped = len(volumes) - len(volumes_clean)
    simulation_results_path = os.path.join(
            exp_id.replace('save_results', 'simulation_results'), 'results_volume_voxels_0_005.json')
    with open(simulation_results_path, 'w') as j_f:
        json.dump({
            'mean_volume': np.mean(volumes_clean),
            'volumes': volumes_clean,
            'median_volume': np.median(volumes_clean),
            'std_volume': np.std(volumes_clean),
            'min_volume': np.min(volumes_clean),
            'max_volume': np.max(volumes_clean),
            'skipped': skipped,
            'computed': len(volumes_clean),
            }, j_f)
        print('Skipped {}, kept {}'.format(skipped, len(volumes_clean)))

def get_volumes_from_samples(sample_infos, workers=8):
    volumes = Parallel(
            n_jobs=workers,
            verbose=5)(delayed(get_sample_intersect_volume)(sample_info)
                    for sample_info in sample_infos)
    return volumes

def get_sample_intersect_volume(sample_info, mode='voxels'):
    hand_mesh = trimesh.Trimesh(vertices=sample_info['hand_verts'], faces=sample_info['hand_faces'])
    obj_mesh = trimesh.Trimesh(vertices=sample_info['obj_verts'], faces=sample_info['obj_faces'])
    if mode == 'engines':
        try:
            intersection = intersect(obj_mesh, hand_mesh, engine='scad')
            if intersection.is_watertight:
                volume = intersection.volume
            else:
                intersection = intersect(obj_mesh, hand_mesh, engine='blender')
                # traceback.print_exc()
                if intersection.vertices.shape[0] == 0:
                    volume = 0
                elif intersection.is_watertight:
                    volume = intersection.volume
                else:
                    volume = None
        except Exception:
            intersection = intersect(obj_mesh, hand_mesh, engine='blender')
            # traceback.print_exc()
            if intersection.vertices.shape[0] == 0:
                volume = 0
            elif intersection.is_watertight:
                volume = intersection.volume
            else:
                volume = None
        if volume > 0.0003:
            for save_idx in range(100):
                save_hand_path = 'misc/hand_2{:04d}.obj'.format(save_idx)
                if not os.path.exists(save_hand_path):
                    hand_mesh.export(save_hand_path)
                    save_obj_path = 'misc/obj_2{:04d}.obj'.format(save_idx)
                    obj_mesh.export(save_obj_path)
                    save_inter_path = 'misc/inter_2{:04d}.obj'.format(save_idx)
                    intersection.export(save_inter_path)
                    print('saved to {}'.format(save_inter_path))
    elif mode == 'voxels':
        volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
    return volume



if __name__ == "__main__":
    for seed in range(0, 3):
       # for conts in [['0.000', '0.625']]:
        # for conts in [['0.000', '0.625'], ['0.312', '0.312'], ['0.625', '0.000']]:
        # Synthgrasps
        # exp_template = 'checkpoints2/test/synthgrasps_33/manonet/2019_03_31/2019_03_29_run24_resume_withlapl_contact/_mini_1_seed{}_resnet18_lr_1e-05_mom0.9_bs32_dec0.5_step300_scale_0.0625_trans_0.0625__frzbatch_neur_1024-256_mano_comps_30_lv0.0625_lj0.0625_ls0.0625/contactall_zonezones_lcont{}th10modedist_tanh_lcol{}th20modedist_tanh_atlas_0.0000_fin0.0625_chamfer_sepencfreeze_atmode_spherefreez_mode_all/epoch_635/synthgrasps_33_testmode_all/save_results/val/epoch_635/'

        # FHB
        #    exp_template = 'checkpoints2/test/fhbhands_obj/manonet/2019_02_11/2019_02_03_run10_no_regul_contact/__seed{}_resnet18_lr_1e-05_mom0.9_bs32_dec0.5_step300_scale_0.062_trans_0.062__frzbatch_neur_1024-256_mano_comps_30_lv0.062_lj0.062_ls0.062/contactall_zonezones_lcont{}th10modedist_tanh_lcol{}th20modedist_tanh_atlas_0.000_fin0.062_chamfer_sepenc_atmode_sphere_mode_all/epoch_500/fhbhands_obj_testmode_all/save_results/val/epoch_500'
        #     exp_id = exp_template.format(seed, conts[0], conts[1])
        exp_template = 'checkpoints2/test/fhbhands_obj/manonet/2019_02_11/2019_02_03_run10_no_regul_contact/__seed{}_resnet18_lr_1e-05_mom0.9_bs32_dec0.5_step300_scale_0.167_trans_0.167__frzbatch_neur_1024-256_mano_comps_30_lv0.167_lj0.167_ls0.167_atlas_0.000_fin0.167_chamfer_sepenc_atmode_sphere_mode_all/epoch_500/fhbhands_obj_testmode_all/save_results/val/epoch_500'
        exp_id = exp_template.format(seed)
        print(exp_id)
        try:
            get_all_volumes(exp_id, batch_step=1, workers=8)
        except Exception:
            traceback.print_exc()
            print('errored on exp_id {}'.format(exp_id))
    # exp_id = exp_template.format(conts[0], conts[1])
#     exp_id = 'checkpoints2/test/synthgrasps_33/manonet/2019_03_31/2019_03_29_run24_resume_withlapl_contact/_mini_1_seed0_resnet18_lr_1e-05_mom0.9_bs32_dec0.5_step300_scale_0.1667_trans_0.1667__frzbatch_neur_1024-256_mano_comps_30_lv0.1667_lj0.1667_ls0.1667_atlas_0.0000_fin0.1667_chamfer_sepencfreeze_atmode_spherefreez_mode_all/epoch_635/synthgrasps_33_testmode_all/save_results/val/epoch_635'
#     try:
#         get_all_volumes(exp_id, batch_step=1, workers=8)
#     except Exception:
#         traceback.print_exc()
#         print('errored on exp_id {}'.format(exp_id))



















# hand_template = 'checkpoints2/test/fhbhands_obj/manonet/2019_01_28/2019_01_24_run9_asinsubmission_fixed_regul_contact_top0_nointeratlaslambda/__seed4_resnet18_lr_1e-05_mom0.9_bs32_dec0.5_step300_scale_0.056_trans_0.056__frzbatch_neur_1024-256_mano_comps_30_lv0.056_lj0.056_ls0.056/contactall_zonezones_lcont0.556th10modedist_tanh_lcol0.000th20modedist_tanh_atlas_0.000_fin0.056_chamfer_sepenc_regul0.056_lapl0.056_atmode_sphere_mode_all/epoch_600/fhbhands_obj_testmode_all/save_objs/val/epoch_600/000{:05d}_hand.obj'
# obj_template = 'checkpoints2/test/fhbhands_obj/manonet/2019_01_28/2019_01_24_run9_asinsubmission_fixed_regul_contact_top0_nointeratlaslambda/__seed4_resnet18_lr_1e-05_mom0.9_bs32_dec0.5_step300_scale_0.056_trans_0.056__frzbatch_neur_1024-256_mano_comps_30_lv0.056_lj0.056_ls0.056/contactall_zonezones_lcont0.556th10modedist_tanh_lcol0.000th20modedist_tanh_atlas_0.000_fin0.056_chamfer_sepenc_regul0.056_lapl0.056_atmode_sphere_mode_all/epoch_600/fhbhands_obj_testmode_all/save_objs/val/epoch_600/000{:05d}_obj.obj'
# for idx in range(0, 5600, 100):
#     hand_obj = hand_template.format(idx)
#     obj_obj = obj_template.format(idx)
#     inter_volume = get_obj_intersect_volume(hand_obj, obj_obj)
#     print(inter_volume)
