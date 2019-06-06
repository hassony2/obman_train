import warnings

import numpy as np
from torch.utils.data import Subset

from handobjectdatasets import core50, fhbhands, stereohands, obman, yanademo
from handobjectdatasets.handataset import HandDataset
from handobjectdatasets.queries import TransQueries, BaseQueries


def get_dataset(
    dat_name,
    split,
    train_it=True,
    mini_factor=None,
    black_padding=False,
    center_idx=9,
    point_nb=600,
    sides="both",
    meta={},
    max_queries=[
        TransQueries.affinetrans,
        TransQueries.images,
        TransQueries.verts3d,
        TransQueries.center3d,
        TransQueries.joints3d,
        TransQueries.objpoints3d,
        TransQueries.camintrs,
        BaseQueries.sides,
    ],
    use_cache=True,
    limit_size=None,
):
    if dat_name == "obman":
        pose_dataset = obman.ObMan(
            mini_factor=mini_factor,
            mode=meta["mode"],
            override_scale=meta["override_scale"],
            segment=False,
            split=split,
            use_cache=use_cache,
            use_external_points=True,
        )
    elif dat_name == "core50":
        if "class_name" not in meta:
            meta["class_name"] = "can"
        pose_dataset = core50.Core50(
            use_cache=False, mini_factor=mini_factor, class_name=meta["class_name"]
        )
    elif dat_name == "yanademo":
        pose_dataset = yanademo.YanaDemo(version=meta["version"], side=meta["side"])
    elif "fhbhands" in dat_name:
        suffix = dat_name.split("_")[-1]
        if suffix == "obj":
            pose_dataset = fhbhands.FHBHands(
                mini_factor=mini_factor,
                split=split,
                use_cache=use_cache,
                use_objects=True,
                split_type=meta["fhbhands_split_type"],
                test_object=meta["fhbhands_split_choice"],
                topology=meta["fhbhands_topology"],
            )
        elif suffix == "hand":
            pose_dataset = fhbhands.FHBHands(
                mini_factor=mini_factor,
                split=split,
                use_cache=use_cache,
                use_objects=False,
                split_type=meta["fhbhands_split_type"],
                test_object=meta["fhbhands_split_choice"],
            )
        else:
            raise ValueError(
                "suffix in {} after _ should be in [obj|hand], got {}".format(
                    dat_name, suffix
                )
            )
    elif dat_name == "stereohands":
        pose_dataset = stereohands.StereoHands(
            split=split, use_cache=use_cache, gt_detections=True
        )
    else:
        raise ValueError("Unrecognized dataset name {}".format(dat_name))

    # Find maximal dataset-compatible queries
    queries = set(max_queries).intersection(set(pose_dataset.all_queries))
    if dat_name == "stereohands":
        max_rot = np.pi
        scale_jittering = 0.2
        center_jittering = 0.2
    else:
        max_rot = np.pi
        scale_jittering = 0.3
        center_jittering = 0.2

    if "override_scale" not in meta:
        meta["override_scale"] = False
    dataset = HandDataset(
        pose_dataset,
        black_padding=black_padding,
        block_rot=False,
        sides=sides,
        train=train_it,
        max_rot=max_rot,
        normalize_img=False,
        center_idx=center_idx,
        point_nb=point_nb,
        scale_jittering=scale_jittering,
        center_jittering=center_jittering,
        queries=queries,
        as_obj_only=meta["override_scale"],
    )
    if limit_size is not None:
        if len(dataset) < limit_size:
            warnings.warn(
                "limit size {} < dataset size {}, working with full dataset".format(
                    limit_size, len(dataset)
                )
            )
        else:
            warnings.warn(
                "Working wth subset of {} of size {}".format(dat_name, limit_size)
            )
            dataset = Subset(dataset, list(range(limit_size)))
    return dataset
