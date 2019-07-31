from collections import defaultdict
import os
import pickle
import random

import numpy as np
from tqdm import tqdm

from handobjectdatasets.queries import BaseQueries, get_trans_queries
from handobjectdatasets import handutils, fhbutils

try:
    from PIL import Image, ImageFile
    from scipy.spatial.distance import cdist

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    print("Could not load PIL")


class FHBHands:
    def __init__(
        self,
        split="train",
        split_type="subjects",
        original_subject_split=True,
        joint_nb=21,
        use_cache=False,
        mini_factor=None,
        use_objects=True,
        remove_objects=None,  # !! overriden for now
        test_object="juice_bottle",
        filter_no_contact=True,
        filter_thresh=10,
        topology=None,
        filter_object=None,
        override_scale=False,
    ):
        """
        Args:
            topology: if 0, juice_bottle, salt, liquid_soap, if 1 milk
        """
        super().__init__()
        self.all_queries = [
            BaseQueries.images,
            BaseQueries.joints2d,
            BaseQueries.joints3d,
            BaseQueries.sides,
            BaseQueries.camintrs,
            BaseQueries.meta,
        ]
        self.use_objects = use_objects
        self.filter_no_contact = filter_no_contact
        self.filter_thresh = filter_thresh
        self.override_scale = override_scale

        if self.use_objects:
            self.all_queries.append(BaseQueries.objverts3d)
            self.all_queries.append(BaseQueries.objpoints2d)
            self.all_queries.append(BaseQueries.objfaces)
        if self.use_objects:  # Overriding
            self.remove_objects = False
        else:
            self.remove_objects = False

        self.topology = topology
        self.test_object = test_object
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)
        # Set cache path
        self.use_cache = use_cache
        self.cache_folder = os.path.join("data", "cache", "fhb")
        os.makedirs(self.cache_folder, exist_ok=True)
        self.cam_extr = np.array(
            [
                [0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                [0, 0, 0, 1],
            ]
        )
        self.cam_intr = np.array(
            [
                [1395.749023, 0, 935.732544],
                [0, 1395.749268, 540.681030],
                [0, 0, 1],
            ]
        )

        self.reorder_idx = np.array(
            [
                0,
                1,
                6,
                7,
                8,
                2,
                9,
                10,
                11,
                3,
                12,
                13,
                14,
                4,
                15,
                16,
                17,
                5,
                18,
                19,
                20,
            ]
        )
        self.name = "fhb"
        self.joint_nb = joint_nb
        self.mini_factor = mini_factor
        split_opts = ["action", "objects", "subjects"]
        self.subjects = [
            "Subject_1",
            "Subject_2",
            "Subject_3",
            "Subject_4",
            "Subject_5",
            "Subject_6",
        ]
        if split_type not in split_opts:
            raise ValueError(
                "Split for dataset {} should be in {}, got {}".format(
                    self.name, split_opts, split_type
                )
            )

        self.split_type = split_type
        self.original_subject_split = original_subject_split

        self.root = "/sequoia/data2/dataset/handatasets/fhb"
        self.info_root = os.path.join(self.root, "Subjects_info")
        self.info_split = os.path.join(
            self.root, "data_split_action_recognition.txt"
        )
        self.rgb_root = os.path.join(self.root, "process_yana", "videos_480")
        self.skeleton_root = os.path.join(self.root, "Hand_pose_annotation_v1")
        self.filter_object = filter_object
        # Get file prefixes for images and annotations
        self.split = split
        self.rgb_template = "color_{:04d}.jpeg"
        # Joints are numbered from tip to base, we want opposite
        self.idxs = [
            0,
            4,
            3,
            2,
            1,
            8,
            7,
            6,
            5,
            12,
            11,
            10,
            9,
            16,
            15,
            14,
            13,
            20,
            19,
        ]
        self.load_dataset()

        print(
            "Got {} samples for split {}".format(
                len(self.image_names), self.split
            )
        )

        # get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

    def load_dataset(self):
        suffix = ""
        if self.use_objects:
            if self.filter_no_contact:
                suffix = "{}filter_dist_{}".format(suffix, self.filter_thresh)
            else:
                suffix = "{}no_filter".format(suffix)
        if self.split_type == "objects" and self.use_objects:
            suffix = "{}_obj_{}".format(suffix, self.test_object)
        if not self.use_objects and self.split_type == "subjects":
            if self.remove_objects:
                suffix = "{}_hand_without_annot_objs".format(suffix)
            else:
                suffix = "{}_hand_all".format(suffix)
        if self.split_type == "subjects":
            if self.original_subject_split:
                suffix = suffix + "_or_subjects"
            else:
                suffix = suffix + "_my_subjects"
        cache_path = os.path.join(
            self.cache_folder,
            "{}_{}_{}_top{}_filt{}.pkl".format(
                self.split,
                self.mini_factor,
                suffix,
                self.topology,
                self.filter_object,
            ),
        )
        if os.path.exists(cache_path) and self.use_cache:
            with open(cache_path, "rb") as cache_f:
                annotations = pickle.load(cache_f)
            print(
                "Cached information for dataset {} loaded from {}".format(
                    self.name, cache_path
                )
            )

        else:
            subjects_infos = {}
            for subject in self.subjects:
                subject_info_path = os.path.join(
                    self.info_root, "{}_info.txt".format(subject)
                )
                subjects_infos[subject] = {}
                with open(subject_info_path, "r") as subject_f:
                    raw_lines = subject_f.readlines()
                    for line in raw_lines[3:]:
                        line = " ".join(line.split())
                        action, action_idx, length = line.strip().split(" ")
                        subjects_infos[subject][(action, action_idx)] = length
            skel_info = get_skeletons(self.skeleton_root, subjects_infos)

            with open(self.info_split, "r") as annot_f:
                lines_raw = annot_f.readlines()
            train_list, test_list, all_infos = fhbutils.get_action_train_test(
                lines_raw, subjects_infos
            )
            if self.topology is None:
                all_objects = ["juice_bottle", "liquid_soap", "milk", "salt"]
            elif int(self.topology) == 0:
                all_objects = ["juice_bottle", "liquid_soap", "salt"]
            elif int(self.topology) == 1:
                all_objects = ["milk"]
            if self.filter_object:
                all_objects = [self.filter_object]

            if self.use_objects is True:
                self.fhb_objects = fhbutils.load_objects(
                    object_names=all_objects
                )
                obj_infos = fhbutils.load_object_infos()

            if self.split_type == "action":
                if self.split == "train":
                    sample_list = train_list
                elif self.split == "test":
                    sample_list = test_list
                elif self.split == "all":
                    sample_list = train_list + test_list
                else:
                    raise ValueError(
                        "Split {} not in [train|test|all]".format(self.split)
                    )
            elif self.split_type == "subjects":
                if self.original_subject_split:
                    if self.split == "train":
                        subjects = ["Subject_1", "Subject_3", "Subject_4"]
                    elif self.split == "test":
                        subjects = ["Subject_2", "Subject_5", "Subject_6"]
                    else:
                        raise ValueError(
                            "Split {} not in [train|test]".format(self.split)
                        )
                else:
                    if self.split == "train":
                        subjects = [
                            "Subject_1",
                            "Subject_2",
                            "Subject_3",
                            "Subject_4",
                        ]
                    elif self.split == "val":
                        subjects = ["Subject_5"]
                    elif self.split == "test":
                        subjects = ["Subject_6"]
                    else:
                        raise ValueError(
                            "Split {} not in [train|val|test]".format(
                                self.split
                            )
                        )
                self.subjects = subjects
                print(subjects)
                sample_list = all_infos
            elif self.split_type == "objects":
                if self.use_objects:
                    test_objects = {
                        self.test_object: self.fhb_objects.pop(
                            self.test_object
                        )
                    }
                    train_objects = self.fhb_objects
                    if self.split == "train":
                        self.split_objects = train_objects
                    elif self.split == "test":
                        self.split_objects = test_objects
                        pass
                    elif self.split == "all":
                        self.split_objects = {**train_objects, **test_objects}
                    else:
                        raise ValueError("Split {} not in [train|test]")
                    print(self.split_objects.keys())
                sample_list = all_infos
            else:
                raise ValueError(
                    "split_type {} not in [action|objects|subjects]".format(
                        self.split_type
                    )
                )
            if self.split_type != "subjects":
                self.subjects = [
                    "Subject_1",
                    "Subject_2",
                    "Subject_3",
                    "Subject_4",
                    "Subject_5",
                    "Subject_6",
                ]
            if self.use_objects and self.split_type != "objects":
                self.split_objects = self.fhb_objects

            image_names = []
            joints2d = []
            joints3d = []
            hand_sides = []
            clips = []
            sample_infos = []
            if self.use_objects:
                objnames = []
                objtransforms = []
            for subject, action_name, seq_idx, frame_idx in sample_list:
                img_path = os.path.join(
                    self.rgb_root,
                    subject,
                    action_name,
                    seq_idx,
                    "color",
                    self.rgb_template.format(frame_idx),
                )
                skel = skel_info[subject][(action_name, seq_idx)][frame_idx]
                skel = skel[self.reorder_idx]

                skel_hom = np.concatenate(
                    [skel, np.ones([skel.shape[0], 1])], 1
                )
                skel_camcoords = (
                    self.cam_extr.dot(skel_hom.transpose())
                    .transpose()[:, :3]
                    .astype(np.float32)
                )
                if subject in self.subjects:
                    if self.use_objects:
                        if (
                            subject in obj_infos
                            and (action_name, seq_idx, frame_idx)
                            in obj_infos[subject]
                        ):
                            obj, trans = obj_infos[subject][
                                (action_name, seq_idx, frame_idx)
                            ]
                            if obj in self.split_objects:
                                if self.filter_no_contact:
                                    verts = self.split_objects[obj]["verts"]
                                    trans_verts = fhbutils.transform_obj_verts(
                                        verts, trans, self.cam_extr
                                    )
                                    all_dists = cdist(
                                        trans_verts, skel_camcoords
                                    )
                                    if all_dists.min() > self.filter_thresh:
                                        continue
                                clips.append((subject, action_name, seq_idx))
                                objtransforms.append(trans)
                                objnames.append(obj)
                            else:
                                continue
                        else:
                            # Skip samples without objects if object mode
                            continue
                    else:
                        if self.remove_objects:
                            # Remove samples with object annoations
                            wrong_object = False
                            for obj in all_objects:
                                if obj in action_name:
                                    wrong_object = True
                            if wrong_object:
                                continue
                else:
                    continue

                joints3d.append(skel_camcoords)
                image_names.append(img_path)
                sample_infos.append(
                    {
                        "subject": subject,
                        "action_name": action_name,
                        "seq_idx": seq_idx,
                        "frame_idx": frame_idx,
                    }
                )
                hom_2d = (
                    np.array(self.cam_intr)
                    .dot(skel_camcoords.transpose())
                    .transpose()
                )
                skel2d = (hom_2d / hom_2d[:, 2:])[:, :2]
                joints2d.append(skel2d.astype(np.float32))
                hand_sides.append("right")

            # Extract labels
            if self.mini_factor:
                idxs = list(range(len(image_names)))
                mini_nb = int(len(image_names) * self.mini_factor)
                random.Random(1).shuffle(idxs)
                idxs = idxs[:mini_nb]
                image_names = [image_names[idx] for idx in idxs]
                joints2d = [joints2d[idx] for idx in idxs]
                joints3d = [joints3d[idx] for idx in idxs]
                hand_sides = [hand_sides[idx] for idx in idxs]
                sample_infos = [sample_infos[idx] for idx in idxs]

                if self.use_objects:
                    objnames = [objnames[idx] for idx in idxs]
                    objtransforms = [objtransforms[idx] for idx in idxs]
            annotations = {
                "image_names": image_names,
                "joints2d": joints2d,
                "joints3d": joints3d,
                "hand_sides": hand_sides,
                "sample_infos": sample_infos,
            }
            if self.use_objects:
                annotations["objnames"] = objnames
                annotations["objtransforms"] = objtransforms
                annotations["split_objects"] = self.split_objects
                print("clip_nb: {}".format(len(set(clips))))
            with open(cache_path, "wb") as fid:
                pickle.dump(annotations, fid)
            print(
                "Wrote cache for dataset {} to {}".format(
                    self.name, cache_path
                )
            )

            # Get image paths
        self.image_names = annotations["image_names"]
        self.joints2d = annotations["joints2d"]
        self.joints3d = annotations["joints3d"]
        self.hand_sides = annotations["hand_sides"]
        self.sample_infos = annotations["sample_infos"]
        if self.use_objects:
            self.objnames = annotations["objnames"]
            self.objtransforms = annotations["objtransforms"]
            self.split_objects = annotations["split_objects"]

    def get_image(self, idx):
        img_path = self.image_names[idx]
        img = Image.open(img_path).convert("RGB")
        return img

    def get_obj_verts_faces(self, idx):
        obj = self.objnames[idx]
        trans = self.objtransforms[idx]
        verts = self.split_objects[obj]["verts"]
        trans_verts = fhbutils.transform_obj_verts(verts, trans, self.cam_extr)
        objfaces = self.split_objects[obj]["faces"]
        if self.override_scale:
            trans_verts = trans_verts - trans_verts.mean(0)
            # Inscribe in sphere of scale 0.1 (10 cm)
            trans_verts = (
                100 * trans_verts / np.linalg.norm(trans_verts, axis=1).max()
            )
        return (
            np.array(trans_verts).astype(np.float32),
            np.array(objfaces).astype(np.int16),
        )

    def get_objpoints2d(self, idx):
        objpoints3d, _ = self.get_obj_verts_faces(idx)
        hom_2d = (
            np.array(self.cam_intr).dot(objpoints3d.transpose()).transpose()
        )
        objpoints2d = (hom_2d / hom_2d[:, 2:])[:, :2] / 4
        return objpoints2d

    def get_joints3d(self, idx):
        joints = self.joints3d[idx]
        return joints

    def get_joints2d(self, idx):
        # Images are downscaled by factor 4
        joints = self.joints2d[idx] / 4
        return joints

    def get_camintr(self, idx):
        camintr = self.cam_intr
        return camintr.astype(np.float32)

    def get_sides(self, idx):
        side = self.hand_sides[idx]
        return side

    def get_meta(self, idx):
        meta = {"objname": self.objnames[idx]}
        return meta

    def get_center_scale(self, idx):
        joints2d = self.get_joints2d(idx)
        center = handutils.get_annot_center(joints2d)
        scale = handutils.get_annot_scale(joints2d)
        return center, scale

    def __len__(self):
        return len(self.image_names)


def get_skeletons(skeleton_root, subjects_info):
    skelet_dict = defaultdict(dict)
    for subject, samples in tqdm(subjects_info.items(), desc="subj"):
        for (action, seq_idx) in tqdm(samples, desc="sample"):
            skeleton_path = os.path.join(
                skeleton_root, subject, action, seq_idx, "skeleton.txt"
            )
            skeleton_vals = np.loadtxt(skeleton_path)
            if len(skeleton_vals):
                assert np.all(
                    skeleton_vals[:, 0] == list(range(skeleton_vals.shape[0]))
                ), "row idxs should match frame idx failed at {}".format(
                    skeleton_path
                )
                skelet_dict[subject][(action, seq_idx)] = skeleton_vals[
                    :, 1:
                ].reshape(skeleton_vals.shape[0], 21, -1)
            else:
                # Handle sequences of size 0
                skelet_dict[subject, action, seq_idx] = skeleton_vals
    return skelet_dict
