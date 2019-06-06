from copy import deepcopy
from collections import OrderedDict
import os
import random
import traceback

from dominate import tags as dtags
import numpy as np

import logutils


def get_array_info(
    array, higher_better=False, k_top_avg=1, show_best_idx=True, use_idx=None
):
    array = np.array(array)
    order = np.argsort(array)
    if higher_better:
        order = order[::-1]
    if use_idx is None:
        best_idx = order[0]
    else:
        best_idx = use_idx
    best = array[best_idx]
    best_idxs = order[:k_top_avg]
    k_best = np.mean(array[best_idxs])

    info = OrderedDict({"best_val": best})
    if show_best_idx:
        info["best_idx"] = best_idx

    if k_top_avg > 1:
        info["{}_best".format(k_top_avg)] = k_best
    return info, best_idx


def get_split_info(
    checkpoint,
    split="train",
    epoch=None,
    metrics={},
    show_best_idx=True,
    show_compare_epoch=True,
):
    log_file = os.path.join(checkpoint, "{}.txt".format(split))

    logs = logutils.get_logs(log_file)
    all_infos = OrderedDict()
    all_infos["epoch"] = {split: logs["epoch"]}
    print(all_infos["epoch"])

    for metric_idx, (metric, higher_better) in enumerate(metrics.items()):
        if logs["epoch"] == []:
            return all_infos

        try:
            if metric_idx == 0:
                array_info, best_idx = get_array_info(
                    logs["{}".format(metric)],
                    higher_better=higher_better,
                    show_best_idx=show_best_idx,
                    use_idx=epoch,
                )
            else:
                array_info, _ = get_array_info(
                    logs["{}".format(metric)],
                    higher_better=higher_better,
                    show_best_idx=show_best_idx,
                    use_idx=best_idx,
                )
            all_infos[metric] = array_info
        except IndexError:
            traceback.print_exc()
            print(
                "Encountered error in checkpoint {} for metric {}".format(
                    checkpoint, metric
                )
            )
            all_infos[metric] = {"best_val": None, "best_idx": None}
    return all_infos


def append_info(exp_info, split_infos):
    split_exp = deepcopy(exp_info)
    if split_infos is None:
        return split_exp
    else:
        for metric, split_info in split_infos.items():
            for key, val in split_info.items():
                split_exp["{} {}".format(metric, key)] = val
    return split_exp


def make_table(exp_list):
    table = []
    headers = list(exp_list[0].keys())
    table.append(headers)
    for exp_info in exp_list:
        exp_vals = []
        for header in headers:
            if header in exp_info:
                exp_vals.append(exp_info[header])
            else:
                exp_vals.append("Not found...")
        table.append(exp_vals)
    return table


def add_table(table):
    with dtags.table().add(dtags.tbody()):
        for row in table:
            with dtags.tr():
                for col_idx, col_val in enumerate(row[1:]):
                    if col_idx == 0:
                        dtags.td().add(dtags.a("{}".format(col_val), href=row[0]))
                    else:
                        if isinstance(col_val, float):
                            col_val = "{0:.5f}".format(col_val)
                        dtags.td().add("{}".format(col_val))


def make_image_table(doc, img_root, img_folders, shuffle=False, max_imgs=20):
    # Get all images for each folder
    all_images = []
    for img_folder in img_folders:
        img_names = [
            os.path.join(img_folder, name)
            for name in sorted(os.listdir(os.path.join(img_root, img_folder)))
        ]
        if shuffle:
            random.shuffle(img_names)
        all_images.append(img_names[:max_imgs])

    # Arrange as list [{0: img_1_folder_0, 1:img_1_folder_1, ..}, ]
    max_len = max([len(images) for images in all_images])
    all_arranged_imgs = []

    # Generate for each row dictionary of folder_idx: img_path
    for idx in range(max_len):
        idx_dic = {}
        for folder_idx, img_names in enumerate(all_images):
            if idx < len(img_names):
                idx_dic[folder_idx] = img_names[idx]
        all_arranged_imgs.append(idx_dic)

    num_folders = len(img_folders)

    with doc:
        with dtags.article(cls="markdown-body"):
            with dtags.table().add(dtags.tbody()):
                for arranged_imgs in all_arranged_imgs:
                    with dtags.tr():
                        for folder_idx in range(num_folders):
                            if folder_idx in arranged_imgs:
                                img_path = arranged_imgs[folder_idx]
                                dtags.td().add(dtags.img(src=img_path))
                            else:
                                dtags.td()
