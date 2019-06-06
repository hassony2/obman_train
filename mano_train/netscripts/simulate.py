import argparse
import json
import os
import pickle

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from mano_train.options import expopts, simulopts
from mano_train.simulation import simulate
from mano_train.objectutils.objectio import load_obj
from mano_train.netscripts.savemano import load_batch_info


def full_simul(
    exp_id,
    batch_step=1,
    wait_time=0,
    sample_vis_freq=100,
    use_gui=False,
    sample_step=1,
    workers=8,
    cluster=False,
    intersect_volumes=True,
):
    if cluster:
        vhacd_exe = "/sequoia/data1/yhasson/tools/v-hacd/build/linux/test/testVHACD"
    else:
        vhacd_exe = "/sequoia/data1/yhasson/code/pose_3d/mano_train/thirdparty/v-hacd/build/linux/test/testVHACD"
    assert os.path.exists(exp_id), "{} does not exists!".format(exp_id)
    save_pickles = sorted(
        [
            os.path.join(exp_id, filename)
            for filename in os.listdir(exp_id)
            if ".pkl" in filename
        ]
    )

    # Load mano faces
    _, faces_right = load_obj("misc/mano/mano_right.obj")
    _, faces_left = load_obj("misc/mano/mano_left.obj")

    batch_infos = Parallel(n_jobs=workers, verbose=5)(
        delayed(load_batch_info)(
            save_pickle, faces_right=faces_right, faces_left=faces_left
        )
        for save_pickle in save_pickles[::batch_step]
    )
    # Prepare simulation storing results
    sample_infos = [
        sample_info for batch_info in batch_infos for sample_info in batch_info
    ]
    max_depths = [sample_info["max_depth"] for sample_info in sample_infos]
    max_depth = np.mean(max_depths)
    print("Got all samples !")

    save_gif_folder = exp_id.replace("save_results", "save_gifs")
    save_obj_folder = exp_id.replace("save_results", "save_objs")
    os.makedirs(save_gif_folder, exist_ok=True)
    os.makedirs(save_obj_folder, exist_ok=True)
    distances = Parallel(n_jobs=workers)(
        delayed(simulate.process_sample)(
            sample_idx,
            sample_info,
            save_gif_folder=save_gif_folder,
            save_obj_folder=save_obj_folder,
            use_gui=use_gui,
            wait_time=wait_time,
            sample_vis_freq=sample_vis_freq,
            vhacd_exe=vhacd_exe,
        )
        for sample_idx, sample_info in enumerate(sample_infos[::sample_step])
    )
    simulation_results_path = os.path.join(
        exp_id.replace("save_results", "simulation_results"), "results.json"
    )
    os.makedirs(os.path.dirname(simulation_results_path), exist_ok=True)
    with open(simulation_results_path, "w") as j_f:
        json.dump(
            {
                "mean_dist": np.mean(distances),
                "std": np.std(distances),
                "max_depth": max_depth,
                "sample_dists": distances,
                "max_depths": max_depths,
            },
            j_f,
        )
        print("Wrote results to {}".format(simulation_results_path))
