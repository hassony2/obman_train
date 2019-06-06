import os
import pickle
import time

import numpy as np
from progress.bar import Bar as Bar
import torch

from handobjectdatasets.queries import TransQueries

from mano_train.evaluation.evalutils import AverageMeters
from mano_train.evaluation.zimeval import EvalUtil
from mano_train.visualize import displaymano
from mano_train.netscripts import savemano


def epoch_pass(
    loader,
    model,
    epoch,
    optimizer=None,
    debug=True,
    freeze_batchnorm=False,
    display=True,
    display_freq=10,
    save_path="checkpoints/debug",
    idxs=None,
    train=True,
    inspect_weights=False,
    fig=None,
    save_results=False,
):
    avg_meters = AverageMeters()
    time_meters = AverageMeters()
    print("epoch: {}".format(epoch))

    idxs = list(range(21))  # Joints to use for evaluation
    evaluator = EvalUtil()

    with open("misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces_right = mano_right_data["f"]
    with open("misc/mano/MANO_LEFT.pkl", "rb") as p_f:
        mano_left_data = pickle.load(p_f, encoding="latin1")
        faces_left = mano_left_data["f"]

    # Switch to correct model mode
    if train:
        if freeze_batchnorm:
            model.eval()
        else:
            model.train()

        save_img_folder = os.path.join(
            save_path, "images", "train", "epoch_{}".format(epoch)
        )
    else:
        model.eval()
        save_img_folder = os.path.join(
            save_path, "images", "val", "epoch_{}".format(epoch)
        )
        if save_results:
            save_results_folder = os.path.join(
                save_path, "save_results", "val", "epoch_{}".format(epoch)
            )
            os.makedirs(save_results_folder, exist_ok=True)
    os.makedirs(save_img_folder, exist_ok=True)

    end = time.time()
    bar = Bar("Processing", max=len(loader))
    for batch_idx, (sample) in enumerate(loader):
        if "vis" in sample:
            visibilities = sample["vis"].numpy()
        else:
            visibilities = None
        # measure data loading time
        time_meters.add_loss_value("data_time", time.time() - end)

        # Compute output
        model_loss, results, model_losses = model.forward(
            sample, return_features=inspect_weights
        )

        # compute gradient and do SGD step
        if train:
            optimizer.zero_grad()
            if inspect_weights:
                model_loss.backward(retain_graph=True)
            else:
                model_loss.backward()
            optimizer.step()
            if inspect_weights:
                inspect_loss_names = [
                    "atlas_trans3d",
                    "atlas_objpoints3d",
                    "mano_verts3d",
                    "mano_shape",
                    "atlas_edge_regul",
                ]
                features = results["img_features"]
                features.retain_grad()
                for inspect_loss_name in inspect_loss_names:
                    features.grad = None
                    if inspect_loss_name in model_losses:
                        loss_val = model_losses[inspect_loss_name]
                        if loss_val is not None:
                            loss_val.backward(retain_graph=True)
                            print(inspect_loss_name, torch.norm(features.grad).item())

        # Get values out of tensors
        for loss in model_losses:
            if model_losses[loss] is not None:
                tensor = model_losses[loss]
                value = model_losses[loss].item()
                model_losses[loss] = value
                if value > 100000:
                    print(loss, tensor, model_losses[loss])

        for key, val in model_losses.items():
            if val is not None:
                avg_meters.add_loss_value(key, val)

        save_img_path = os.path.join(
            save_img_folder, "img_{:06d}.png".format(batch_idx)
        )
        if (batch_idx % display_freq == 0) and display:
            displaymano.visualize_batch(
                save_img_path,
                fig=fig,
                sample=sample,
                results=results,
                faces_right=faces_right,
                faces_left=faces_left,
            )
        if save_results:
            save_batch_path = os.path.join(
                save_results_folder, "batch_{:06d}.pkl".format(batch_idx)
            )
            savemano.save_batch_info(save_batch_path, sample=sample, results=results)

        if "joints" in results and TransQueries.joints3d in sample:
            preds = results["joints"].detach().cpu()
            # Keep only evaluation joints
            preds = preds[:, idxs]
            gt = sample[TransQueries.joints3d][:, idxs]

            # Feed predictions to evaluator
            if visibilities is None:
                visibilities = [None] * len(gt)
            for gt_kp, pred_kp, visibility in zip(gt, preds, visibilities):
                evaluator.feed(gt_kp, pred_kp, keypoint_vis=visibility)

        # measure elapsed time
        time_meters.add_loss_value("batch_time", time.time() - end)

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}".format(
            batch=batch_idx + 1,
            size=len(loader),
            data=time_meters.average_meters["data_time"].val,
            bt=time_meters.average_meters["batch_time"].avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=avg_meters.average_meters["total_loss"].avg,
        )
        bar.next()

    (
        epe_mean_all,
        _,
        epe_median_all,
        auc_all,
        pck_curve_all,
        thresholds,
    ) = evaluator.get_measures(0, 50, 20)
    if "joints" in results:
        if train:
            pck_folder = os.path.join(save_path, "pcks/train")
        else:
            pck_folder = os.path.join(save_path, "pcks/val")
        os.makedirs(pck_folder, exist_ok=True)
        pck_info = {
            "auc": auc_all,
            "thres": thresholds,
            "pck_curve": pck_curve_all,
            "epe_mean": epe_mean_all,
            "epe_median": epe_median_all,
            "evaluator": evaluator,
        }

        save_pck_file = os.path.join(pck_folder, "epoch_{}.eps".format(epoch))
        if sample["dataset"] == "stereohands" and (sample["split"] == "test"):
            overlay = "stereo_test"
        elif sample["dataset"] == "stereohands" and (sample["split"] == "all"):
            overlay = "stereo_all"
        else:
            overlay = None

        if np.isnan(auc_all):
            print(
                "Not saving pck info, normal in case of only 2D info supervision, abnormal otherwise"
            )
        else:
            displaymano.save_pck_img(
                thresholds, pck_curve_all, auc_all, save_pck_file, overlay=overlay
            )
        save_pck_pkl = os.path.join(pck_folder, "epoch_{}.pkl".format(epoch))
        with open(save_pck_pkl, "wb") as p_f:
            pickle.dump(pck_info, p_f)

    else:
        pck_info = {}

    bar.finish()
    return avg_meters, pck_info
