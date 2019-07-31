import os
import pickle

import numpy as np
from PIL import Image
from scipy.io import loadmat

from handobjectdatasets.queries import BaseQueries, get_trans_queries
from handobjectdatasets import handutils


class StereoHands:
    def __init__(
        self,
        split="train",
        root="/sequoia/data2/dataset/handatasets/stereohands",
        joint_nb=21,
        use_cache=False,
        gt_detections=False,
    ):
        # Set cache path
        self.split = split
        self.use_cache = use_cache
        self.cache_folder = os.path.join("data", "cache", "stereohands")
        os.makedirs(self.cache_folder, exist_ok=True)
        self.gt_detections = gt_detections
        self.root = root
        self.joint_nb = joint_nb
        self.all_queries = [
            BaseQueries.manoidxs,
            BaseQueries.images,
            BaseQueries.joints2d,
            BaseQueries.joints3d,
            BaseQueries.sides,
        ]
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)
        self.name = "stereohands"

        self.manoidxs = list(range(1, 21))

        # Get file prefixes for images and annotations
        self.intr = np.array(
            [[822.79041, 0, 318.47345], [0, 822.79041, 250.31296], [0, 0, 1]]
        )
        self.rgb_folder = os.path.join(root, "images")
        self.label_folder = os.path.join(root, "labels")
        self.right_template = "BB_right_{}.png"
        self.left_template = "BB_left_{}.png"

        # get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
        if split == "train":
            self.sequences = [
                "B2Counting",
                "B2Random",
                "B3Counting",
                "B3Random",
                "B4Counting",
                "B4Random",
                "B5Counting",
                "B5Random",
                "B6Counting",
                "B6Random",
            ]
        elif split == "test":
            self.sequences = ["B1Counting", "B1Random"]
        elif split == "val":
            self.sequences = ["B2Counting", "B2Random"]
        elif split == "train_val":
            self.sequences = [
                "B3Counting",
                "B3Random",
                "B4Counting",
                "B4Random",
                "B5Counting",
                "B5Random",
                "B6Counting",
                "B6Random",
            ]
        elif split == "all":
            self.sequences = [
                "B1Counting",
                "B1Random",
                "B2Counting",
                "B2Random",
                "B3Counting",
                "B3Random",
                "B4Counting",
                "B4Random",
                "B5Counting",
                "B5Random",
                "B6Counting",
                "B6Random",
            ]
        else:
            raise ValueError("split {} not in [train|test|val|train_val|all]")
        self.split = split
        self.center_path = os.path.join(
            root, "detections", "centers_{}.txt".format(self.split)
        )
        self.scale_path = os.path.join(
            root, "detections", "scales_{}.txt".format(self.split)
        )
        self.bbox_path = os.path.join(
            root, "detections", "bboxes_{}.txt".format(self.split)
        )
        self.load_dataset()

    def _get_image_path(self, prefix):
        image_path = os.path.join(self.rgb_folder, "{}.png".format(prefix))
        return image_path

    def _get_json(self, prefix):
        coord2d_path = os.path.join(
            self.coord2d_folder, "{}.txt".format(prefix)
        )
        annots = np.loadtxt(coord2d_path)
        # annots = json.load(open(coord2d_path))
        return annots

    # Detection methods
    def load_dataset(self):
        # Use cache if relevant
        cache_path = os.path.join(
            self.cache_folder, "{}.pkl".format(self.split)
        )
        if os.path.exists(cache_path) and self.use_cache:
            with open(cache_path, "rb") as fid:
                annotations = pickle.load(fid)
                self.image_names = annotations["image_names"]
                self.joints_3d = annotations["joints_3d"]
                self.joints_2d = annotations["joints_2d"]
                if not self.gt_detections:
                    self.detected_centers = annotations["detected_centers"]
                    self.detected_scales = annotations["detected_scales"]
                    self.annotations = annotations["detected_bboxes"]

            print("{} gt roidb loaded from {}".format(self.name, cache_path))

        else:
            reorder_idx = np.array(
                [
                    0,
                    17,
                    18,
                    19,
                    20,
                    13,
                    14,
                    15,
                    16,
                    9,
                    10,
                    11,
                    12,
                    5,
                    6,
                    7,
                    8,
                    1,
                    2,
                    3,
                    4,
                ]
            )
            if not self.gt_detections:
                all_centers = np.loadtxt(self.center_path)
                all_scales = np.loadtxt(self.scale_path)
                all_bboxes = np.loadtxt(self.bbox_path)
                self.detected_centers = all_centers
                self.detected_scales = all_scales
                self.annotations = all_bboxes
            image_names = []
            joints_3d = []
            joints_2d = []
            for sequence in sorted(self.sequences):
                # Read annotations
                label_path = os.path.join(
                    self.label_folder, "{}_BB.mat".format(sequence)
                )
                rawmat = loadmat(label_path)
                annots = rawmat["handPara"].transpose(2, 1, 0)
                for i in range(1500):
                    left_img_path = os.path.join(
                        self.rgb_folder, sequence, self.left_template.format(i)
                    )
                    image_names.append(left_img_path)
                    joint_3d = annots[i][reorder_idx]
                    joints_3d.append(joint_3d)
                    hom_2d = self.intr.dot(joint_3d.T).T
                    joint_2d = hom_2d / hom_2d[:, 2:3]
                    joints_2d.append(joint_2d[:, :2])

            self.image_names = image_names
            self.joints_3d = joints_3d
            self.joints_2d = joints_2d
            full_info = {
                "image_names": image_names,
                "joints_2d": joints_2d,
                "joints_3d": joints_3d,
            }
            if not self.gt_detections:
                full_info["detected_centers"] = all_centers
                full_info["detected_scales"] = all_scales
                full_info["detected_bboxes"] = all_bboxes
            with open(cache_path, "wb") as fid:
                pickle.dump(full_info, fid)
            print(
                "Wrote cache for dataset {} to {}".format(
                    self.name, cache_path
                )
            )

    def get_image(self, idx):
        image_path = self.image_names[idx]
        img = Image.open(image_path).convert("RGB")
        return img

    def get_joints3d(self, idx):
        joints = self.joints_3d[idx].astype(np.float32)
        return joints

    def get_joints2d(self, idx):
        joints = self.joints_2d[idx].astype(np.float32)
        return joints

    def get_sides(self, idx):
        return "left"

    def get_manoidxs(self, idx):
        return self.manoidxs

    def get_center_scale(self, idx, scale_factor=2.2):
        if self.gt_detections:
            joints2d = self.get_joints2d(idx)
            center = handutils.get_annot_center(joints2d)
            scale = handutils.get_annot_scale(
                joints2d, scale_factor=scale_factor
            )
        else:
            center = self.detected_centers[idx]
            scale = self.detected_scales[idx] * scale_factor / 2.2
        return center, scale

    def __len__(self):
        return len(self.image_names)
