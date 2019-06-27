import os
import pickle

import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

from handobjectdatasets.queries import BaseQueries, get_trans_queries
from handobjectdatasets import handutils
from handobjectdatasets.loadutils import fast_load_obj

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ObMan:
    def __init__(
        self,
        split="train",
        root=None,
        joint_nb=21,
        mini_factor=None,
        use_cache=False,
        root_palm=False,
        mode="obj",
        segment=False,
        override_scale=False,
        use_external_points=True,
        apply_obj_transform=True,
        segmented_depth=True,
        shapenet_root="datasymlinks/ShapeNetCore.v2",
        obman_root="datasymlinks/obman",
    ):
        # Set cache path
        self.split = split
        obman_root = os.path.join(obman_root, split)
        self.override_scale = override_scale  # Use fixed scale
        self.root_palm = root_palm
        self.mode = mode
        self.segment = segment
        self.apply_obj_transform = apply_obj_transform
        self.segmented_depth = segmented_depth

        self.use_external_points = use_external_points
        if mode == "all" and not self.override_scale:
            self.all_queries = [
                BaseQueries.images,
                BaseQueries.joints2d,
                BaseQueries.joints3d,
                BaseQueries.sides,
                BaseQueries.segms,
                BaseQueries.verts3d,
                BaseQueries.hand_pcas,
                BaseQueries.hand_poses,
                BaseQueries.camintrs,
                BaseQueries.depth,
            ]
            if use_external_points:
                self.all_queries.append(BaseQueries.objpoints3d)
            else:
                self.all_queries.append(BaseQueries.objverts3d)
                self.all_queries.append(BaseQueries.objfaces)
            self.rgb_folder = os.path.join(obman_root, "rgb")
        elif mode == "obj" or (self.mode == "all" and self.override_scale):
            self.all_queries = [BaseQueries.images, BaseQueries.camintrs]
            if use_external_points:
                self.all_queries.append(BaseQueries.objpoints3d)
            else:
                self.all_queries.append(BaseQueries.objpoints3d)
                self.all_queries.append(BaseQueries.objverts3d)
                self.all_queries.append(BaseQueries.objfaces)
            if mode == "obj":
                self.rgb_folder = os.path.join(obman_root, "rgb_obj")
            else:
                self.rgb_folder = os.path.join(obman_root, "rgb")
        elif mode == "hand":
            self.all_queries = [
                BaseQueries.images,
                BaseQueries.joints2d,
                BaseQueries.joints3d,
                BaseQueries.sides,
                BaseQueries.segms,
                BaseQueries.verts3d,
                BaseQueries.hand_pcas,
                BaseQueries.hand_poses,
                BaseQueries.camintrs,
                BaseQueries.depth,
            ]
            self.rgb_folder = os.path.join(obman_root, "rgb_hand")
        else:
            raise ValueError(
                "Mode should be in [all|obj|hand], got {}".format(mode)
            )

        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)

        # Cache information
        self.use_cache = use_cache
        self.name = "obman"
        self.cache_folder = os.path.join("data", "cache", self.name)
        os.makedirs(self.cache_folder, exist_ok=True)
        self.mini_factor = mini_factor
        self.cam_intr = np.array(
            [[480.0, 0.0, 128.0], [0.0, 480.0, 128.0], [0.0, 0.0, 1.0]]
        ).astype(np.float32)

        self.cam_extr = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
            ]
        ).astype(np.float32)

        self.joint_nb = joint_nb
        self.segm_folder = os.path.join(obman_root, "segm")

        self.prefix_template = "{:08d}"
        self.meta_folder = os.path.join(obman_root, "meta")
        self.coord2d_folder = os.path.join(obman_root, "coords2d")

        # Define links on skeleton
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

        # Object info
        self.shapenet_template = os.path.join(
            shapenet_root, "{}/{}/models/model_normalized.pkl"
        )
        self.load_dataset()

    def _get_image_path(self, prefix):
        image_path = os.path.join(self.rgb_folder, "{}.jpg".format(prefix))

        return image_path

    def load_dataset(self):
        pkl_path = "/sequoia/data1/yhasson/code/\
                    pose_3d/mano_render/mano/models/MANO_RIGHT_v1.pkl"
        if not os.path.exists(pkl_path):
            pkl_path = "../" + pkl_path

        cache_path = os.path.join(
            self.cache_folder,
            "{}_{}_mode_{}.pkl".format(
                self.split, self.mini_factor, self.mode
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
            idxs = [
                int(imgname.split(".")[0])
                for imgname in sorted(os.listdir(self.meta_folder))
            ]

            if self.mini_factor:
                mini_nb = int(len(idxs) * self.mini_factor)
                idxs = idxs[:mini_nb]

            prefixes = [self.prefix_template.format(idx) for idx in idxs]
            print(
                "Got {} samples for split {}, generating cache !".format(
                    len(idxs), self.split
                )
            )

            image_names = []
            all_joints2d = []
            all_joints3d = []
            hand_sides = []
            hand_poses = []
            hand_pcas = []
            hand_verts3d = []
            obj_paths = []
            obj_transforms = []
            meta_infos = []
            depth_infos = []
            for idx, prefix in enumerate(tqdm(prefixes)):
                meta_path = os.path.join(
                    self.meta_folder, "{}.pkl".format(prefix)
                )
                with open(meta_path, "rb") as meta_f:
                    meta_info = pickle.load(meta_f)
                image_path = self._get_image_path(prefix)
                image_names.append(image_path)
                all_joints2d.append(meta_info["coords_2d"])
                all_joints3d.append(meta_info["coords_3d"])
                hand_verts3d.append(meta_info["verts_3d"])
                hand_sides.append(meta_info["side"])
                hand_poses.append(meta_info["hand_pose"])
                hand_pcas.append(meta_info["pca_pose"])
                depth_infos.append(
                    {
                        "depth_min": meta_info["depth_min"],
                        "depth_max": meta_info["depth_max"],
                        "hand_depth_min": meta_info["hand_depth_min"],
                        "hand_depth_max": meta_info["hand_depth_max"],
                        "obj_depth_min": meta_info["obj_depth_min"],
                        "obj_depth_max": meta_info["obj_depth_max"],
                    }
                )
                obj_path = self._get_obj_path(
                    meta_info["class_id"], meta_info["sample_id"]
                )

                obj_paths.append(obj_path)
                obj_transforms.append(meta_info["affine_transform"])
                meta_info_full = {
                    "obj_scale": meta_info["obj_scale"],
                    "obj_class_id": meta_info["class_id"],
                    "obj_sample_id": meta_info["sample_id"],
                }
                if "grasp_quality" in meta_info:
                    meta_info_full["grasp_quality"] = meta_info[
                        "grasp_quality"
                    ]
                    meta_info_full["grasp_epsilon"] = meta_info[
                        "grasp_epsilon"
                    ]
                    meta_info_full["grasp_volume"] = meta_info["grasp_volume"]
                meta_infos.append(meta_info_full)

            annotations = {
                "depth_infos": depth_infos,
                "image_names": image_names,
                "joints2d": all_joints2d,
                "joints3d": all_joints3d,
                "hand_sides": hand_sides,
                "hand_poses": hand_poses,
                "hand_pcas": hand_pcas,
                "hand_verts3d": hand_verts3d,
                "obj_paths": obj_paths,
                "obj_transforms": obj_transforms,
                "meta_infos": meta_infos,
            }
            print(
                "class_nb: {}".format(
                    np.unique(
                        [
                            (meta_info["obj_class_id"])
                            for meta_info in meta_infos
                        ],
                        axis=0,
                    ).shape
                )
            )
            print(
                "sample_nb : {}".format(
                    np.unique(
                        [
                            (
                                meta_info["obj_class_id"],
                                meta_info["obj_sample_id"],
                            )
                            for meta_info in meta_infos
                        ],
                        axis=0,
                    ).shape
                )
            )
            with open(cache_path, "wb") as fid:
                pickle.dump(annotations, fid)
            print(
                "Wrote cache for dataset {} to {}".format(
                    self.name, cache_path
                )
            )

        # Set dataset attributes
        all_objects = [
            obj[:-7].split("/")[-1].split("_")[0]
            for obj in annotations["obj_paths"]
        ]
        selected_idxs = list(range(len(all_objects)))
        obj_paths = [annotations["obj_paths"][idx] for idx in selected_idxs]
        image_names = [
            annotations["image_names"][idx] for idx in selected_idxs
        ]
        joints3d = [annotations["joints3d"][idx] for idx in selected_idxs]
        joints2d = [annotations["joints2d"][idx] for idx in selected_idxs]
        hand_sides = [annotations["hand_sides"][idx] for idx in selected_idxs]
        hand_pcas = [annotations["hand_pcas"][idx] for idx in selected_idxs]
        hand_verts3d = [
            annotations["hand_verts3d"][idx] for idx in selected_idxs
        ]
        obj_transforms = [
            annotations["obj_transforms"][idx] for idx in selected_idxs
        ]
        meta_infos = [annotations["meta_infos"][idx] for idx in selected_idxs]
        if "depth_infos" in annotations:
            has_depth_info = True
            depth_infos = [
                annotations["depth_infos"][idx] for idx in selected_idxs
            ]
        else:
            has_depth_info = False
        if has_depth_info:
            self.depth_infos = depth_infos
        self.image_names = image_names
        self.joints2d = joints2d
        self.joints3d = joints3d
        self.hand_sides = hand_sides
        self.hand_pcas = hand_pcas
        self.hand_verts3d = hand_verts3d
        self.obj_paths = obj_paths
        self.obj_transforms = obj_transforms
        self.meta_infos = meta_infos
        # Initialize cache for center and scale in case objects are used
        self.center_scale_cache = {}

    def get_image(self, idx):
        image_path = self.image_names[idx]
        side = self.get_sides(idx)
        if self.segment:
            if self.mode == "all":
                segm_path = image_path.replace("rgb", "segm").replace(
                    "jpg", "png"
                )
            elif self.mode == "hand":
                segm_path = image_path.replace("rgb_hand", "segm").replace(
                    "jpg", "png"
                )
            elif self.mode == "obj":
                segm_path = image_path.replace("rgb_obj", "segm").replace(
                    "jpg", "png"
                )

            img = cv2.imread(image_path, 1)
            if img is None:
                raise ValueError("cv2 could not open {}".format(image_path))
            segm_img = cv2.imread(segm_path, 1)
            if segm_img is None:
                raise ValueError("cv2 could not open {}".format(segm_path))
            if self.mode == "all":
                segm_img = segm_img[:, :, 0]
            elif self.mode == "hand":
                segm_img = segm_img[:, :, 1]
            elif self.mode == "obj":
                segm_img = segm_img[:, :, 2]
            segm_img = _get_segm(segm_img, side=side)
            segm_img = segm_img.sum(2)[:, :, np.newaxis]
            # blacken not segmented
            img[~segm_img.astype(bool).repeat(3, 2)] = 0
            img = Image.fromarray(img[:, :, ::-1])
        else:
            img = Image.open(image_path)
            img = img.convert("RGB")
        return img

    def get_segm(self, idx, pil_image=True):
        side = self.get_sides(idx)
        image_path = self.image_names[idx]
        if self.mode == "all":
            image_path = image_path.replace("rgb", "segm").replace(
                "jpg", "png"
            )
        elif self.mode == "hand":
            image_path = image_path.replace("rgb_hand", "segm").replace(
                "jpg", "png"
            )
        elif self.mode == "obj":
            image_path = image_path.replace("rgb_obj", "segm").replace(
                "jpg", "png"
            )

        img = cv2.imread(image_path, 1)
        if img is None:
            raise ValueError("cv2 could not open {}".format(image_path))
        if self.mode == "all":
            segm_img = _get_segm(img[:, :, 0], side=side)
        elif self.mode == "hand":
            segm_img = _get_segm(img[:, :, 1], side=side)
        elif self.mode == "obj":
            segm_img = _get_segm(img[:, :, 2], side=side)
        if pil_image:
            segm_img = Image.fromarray((255 * segm_img).astype(np.uint8))
        return segm_img

    def get_joints2d(self, idx):
        return self.joints2d[idx].astype(np.float32)

    def get_joints3d(self, idx):
        joints3d = self.joints3d[idx]
        if self.root_palm:
            # Replace wrist with palm
            verts3d = self.hand_verts3d[idx]
            palm = (verts3d[95] + verts3d[218]) / 2
            joints3d = np.concatenate([palm[np.newaxis, :], joints3d[1:]])
        # No hom coordinates needed because no translation
        assert (
            np.linalg.norm(self.cam_extr[:, 3]) == 0
        ), "extr camera should have no translation"

        joints3d = self.cam_extr[:3, :3].dot(joints3d.transpose()).transpose()
        return 1000 * joints3d

    def get_verts3d(self, idx):
        verts3d = self.hand_verts3d[idx]
        verts3d = self.cam_extr[:3, :3].dot(verts3d.transpose()).transpose()
        return 1000 * verts3d

    def get_obj_verts_faces(self, idx):
        model_path = self.obj_paths[idx]
        model_path_obj = model_path.replace(".pkl", ".obj")
        if os.path.exists(model_path):
            with open(model_path, "rb") as obj_f:
                mesh = pickle.load(obj_f)
        elif os.path.exists(model_path_obj):
            with open(model_path_obj, "r") as m_f:
                mesh = fast_load_obj(m_f)[0]
        else:
            raise ValueError(
                "Could not find model pkl or obj file at {}".format(
                    model_path.split(".")[-2]
                )
            )

        obj_scale = self.meta_infos[idx]["obj_scale"]
        if self.mode == "obj" or self.override_scale:
            verts = mesh["vertices"] * 0.18
        else:
            verts = mesh["vertices"] * obj_scale

        # Apply transforms
        if self.apply_obj_transform:
            obj_transform = self.obj_transforms[idx]
            hom_verts = np.concatenate(
                [verts, np.ones([verts.shape[0], 1])], axis=1
            )
            trans_verts = obj_transform.dot(hom_verts.T).T[:, :3]
            trans_verts = (
                self.cam_extr[:3, :3].dot(trans_verts.transpose()).transpose()
            )
        else:
            trans_verts = verts
        return (
            np.array(trans_verts).astype(np.float32) * 1000,
            np.array(mesh["faces"]).astype(np.int16),
        )

    def get_objpoints3d(self, idx, point_nb=600):
        model_path = self.obj_paths[idx].replace(
            "model_normalized.pkl", "surface_points.pkl"
        )
        with open(model_path, "rb") as obj_f:
            points = pickle.load(obj_f)

        # Apply scaling
        if self.mode == "obj" or self.override_scale:
            points = points * 0.18
        else:
            points = points

        # Filter very far outlier points from modelnet/shapenet !!
        point_nb_or = points.shape[0]
        points = points[
            np.linalg.norm(points, 2, 1)
            < 20 * np.median(np.linalg.norm(points, 2, 1))
        ]
        if points.shape[0] < point_nb_or:
            print(
                "Filtering {} points out of {} "
                "for sample {} from split {}".format(
                    point_nb_or - points.shape[0],
                    point_nb_or,
                    self.image_names[idx],
                    self.split,
                )
            )
        # Sample points
        idxs = np.random.choice(points.shape[0], point_nb)
        points = points[idxs]
        # Apply transforms
        if self.apply_obj_transform:
            obj_transform = self.obj_transforms[idx]
            hom_points = np.concatenate(
                [points, np.ones([points.shape[0], 1])], axis=1
            )
            trans_points = obj_transform.dot(hom_points.T).T[:, :3]
            trans_points = (
                self.cam_extr[:3, :3].dot(trans_points.transpose()).transpose()
            )
        else:
            trans_points = points
        return trans_points.astype(np.float32) * 1000

    def get_sides(self, idx):
        return self.hand_sides[idx]

    def get_camintr(self, idx):
        return self.cam_intr

    def get_depth(self, idx):
        image_path = self.image_names[idx]
        if self.mode == "all":
            image_path = image_path.replace("rgb", "depth")
        elif self.mode == "hand":
            image_path = image_path.replace("rgb_hand", "depth")
        elif self.mode == "obj":
            image_path = image_path.replace("rgb_obj", "depth")
        image_path = image_path.replace("jpg", "png")

        img = cv2.imread(image_path, 1)
        if img is None:
            raise ValueError("cv2 could not open {}".format(image_path))

        depth_info = self.depth_infos[idx]
        if self.mode == "all":
            img = img[:, :, 0]
            depth_max = depth_info["depth_max"]
            depth_min = depth_info["depth_min"]
        elif self.mode == "hand":
            img = img[:, :, 1]
            depth_max = depth_info["hand_depth_max"]
            depth_min = depth_info["hand_depth_min"]
        elif self.mode == "obj":
            img = img[:, :, 2]
            depth_max = depth_info["obj_depth_max"]
            depth_min = depth_info["obj_depth_min"]
        assert (
            img.max() == 255
        ), "Max value of depth jpg should be 255, not {}".format(img.max())
        img = (img - 1) / 254 * (depth_min - depth_max) + depth_max
        if self.segmented_depth:
            obj_hand_segm = (np.asarray(self.get_segm(idx)) / 255).astype(
                np.int
            )
            segm = obj_hand_segm[:, :, 0] | obj_hand_segm[:, :, 1]
            img = img * segm
        return img

    def get_center_scale(self, idx, scale_factor=2.2):
        if self.mode == "obj" or self.override_scale:
            if idx not in self.center_scale_cache:
                segm = self.get_segm(idx, pil_image=False)
                min_y = np.nonzero(segm[:, :, 1].sum(1))[0].min()
                max_y = np.nonzero(segm[:, :, 1].sum(1))[0].max()
                min_x = np.nonzero(segm[:, :, 1].sum(0))[0].min()
                max_x = np.nonzero(segm[:, :, 1].sum(0))[0].max()
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                scale = scale_factor * np.max([max_y - min_y, max_x - min_x])
                center = np.array([center_x, center_y])
                self.center_scale_cache[idx] = (center, scale)
            else:
                center, scale = self.center_scale_cache[idx]
        else:
            joints2d = self.get_joints2d(idx)
            center = handutils.get_annot_center(joints2d)
            scale = handutils.get_annot_scale(
                joints2d, scale_factor=scale_factor
            )
        return center, scale

    def _get_obj_path(self, class_id, sample_id):
        shapenet_path = self.shapenet_template.format(class_id, sample_id)
        return shapenet_path

    def __len__(self):
        return len(self.image_names)


def _get_segm(img, side="left"):
    if side == "right":
        hand_segm_img = (img == 22).astype(float) + (img == 24).astype(float)
    elif side == "left":
        hand_segm_img = (img == 21).astype(float) + (img == 23).astype(float)
    else:
        raise ValueError("Got side {}, expected [right|left]".format(side))

    obj_segm_img = (img == 100).astype(float)
    segm_img = np.stack(
        [hand_segm_img, obj_segm_img, np.zeros_like(hand_segm_img)], axis=2
    )
    return segm_img
