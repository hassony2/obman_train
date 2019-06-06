from copy import deepcopy
import os
import shutil
import traceback
import warnings

import torch


def load_checkpoints(model, resume_paths, strict=True):
    # Load models
    all_state_dicts = []
    all_epochs = []
    for resume_path in resume_paths:
        checkpoint = torch.load(resume_path)
        state_dict = checkpoint["state_dict"]
        all_state_dicts.append(state_dict)
        all_epochs.append(checkpoint["epoch"])
    mean_state_dict = {}
    for state_key in state_dict.keys():
        if isinstance(state_dict[state_key], torch.cuda.LongTensor):
            mean_state_dict[state_key] = state_dict[state_key]
        else:
            params = [state_dict[state_key] for state_dict in all_state_dicts]
            mean_state_dict[state_key] = torch.stack(params).mean(0)

    model.load_state_dict(mean_state_dict, strict=strict)
    return max(all_epochs), None


def load_checkpoint(model, resume_path, optimizer=None, strict=True, load_atlas=False):
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = {
                "module.{}".format(key): item
                for key, item in checkpoint["state_dict"].items()
            }
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    resume_path, checkpoint["epoch"]
                )
            )
        if load_atlas:
            # Load encoder to separate encoder branch
            atlas_state_dict = {}
            for key, val in state_dict.items():
                if "base_net" in key:
                    atlas_state_dict[key.replace("base_net", "atlas_base_net")] = val
                else:
                    atlas_state_dict[key] = val
            state_dict = atlas_state_dict

        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)
        if optimizer is not None:
            try:
                missing_states = set(optimizer.state_dict().keys()) - set(
                    checkpoint["optimizer"].keys()
                )
                if len(missing_states) > 0:
                    warnings.warn(
                        "Missing keys in optimizer ! : {}".format(missing_states)
                    )
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError:
                traceback.print_exc()
                warnings.warn("Couldn' load optimizer from {}".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))
    if "best_auc" in checkpoint:
        warnings.warn("Using deprecated best_acc instead of best_auc")
        best = checkpoint["best_auc"]
    elif "best_acc" in checkpoint:
        warnings.warn("Using deprecated best_acc instead of best_auc")
        best = checkpoint["best_acc"]
    else:
        best = checkpoint["best_score"]
    return checkpoint["epoch"], best


def save_checkpoint(
    state,
    is_best,
    checkpoint="checkpoint",
    filename="checkpoint.pth.tar",
    snapshot=None,
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state["epoch"] % snapshot == 0:
        shutil.copyfile(
            filepath,
            os.path.join(checkpoint, "checkpoint_{}.pth.tar".format(state["epoch"])),
        )

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))
