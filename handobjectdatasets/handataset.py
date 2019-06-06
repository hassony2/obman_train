import random
import traceback

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms

from handobjectdatasets import imgtrans, handutils, viz2d, vertexsample
from handobjectdatasets.queries import (
    BaseQueries,
    TransQueries,
    one_query_in,
    no_query_in,
)


def bbox_from_joints(joints):
    x_min, y_min = joints.min(0)
    x_max, y_max = joints.max(0)
    bbox = [x_min, y_min, x_max, y_max]
    return bbox


class HandDataset(Dataset):
    """Class inherited by hands datasets
    hands datasets must implement the following methods:
    - get_image
    that respectively return a PIL image and a numpy array
    - the __len__ method

    and expose the following attributes:
    - the cache_folder : the path to the cache folder of the dataset
    """

    def __init__(
        self,
        pose_dataset,
        center_idx=9,
        point_nb=600,
        inp_res=256,
        max_rot=np.pi,
        normalize_img=False,
        split="train",
        scale_jittering=0.3,
        center_jittering=0.2,
        train=True,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
        queries=[
            BaseQueries.images,
            TransQueries.joints2d,
            TransQueries.verts3d,
            TransQueries.joints3d,
        ],
        sides="both",
        block_rot=False,
        black_padding=False,
        as_obj_only=False,
    ):
        """
        Args:
        center_idx: idx of joint on which to center 3d pose
        as_obj_only: apply same centering and scaling as when objects are
            not present
        sides: if both, don't flip hands, if 'right' flip all left hands to
            right hands, if 'left', do the opposite
        """
        # Dataset attributes
        self.pose_dataset = pose_dataset
        self.as_obj_only = as_obj_only
        self.inp_res = inp_res
        self.point_nb = point_nb
        self.normalize_img = normalize_img
        self.center_idx = center_idx
        self.sides = sides
        self.black_padding = black_padding

        # Color jitter attributes
        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius

        self.max_rot = max_rot
        self.block_rot = block_rot

        # Training attributes
        self.train = train
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jittering

        self.queries = queries

    def __len__(self):
        return len(self.pose_dataset)

    def get_sample(self, idx, query=None):
        if query is None:
            query = self.queries
        sample = {}

        if BaseQueries.images in query or TransQueries.images in query:
            center, scale = self.pose_dataset.get_center_scale(idx)
            needs_center_scale = True
        else:
            needs_center_scale = False

        # Get sides
        if BaseQueries.sides in query:
            hand_side = self.pose_dataset.get_sides(idx)
            # Flip if needed
            if self.sides == "right" and hand_side == "left":
                flip = True
                hand_side = "right"
            elif self.sides == "left" and hand_side == "right":
                flip = True
                hand_side = "left"
            else:
                flip = False
            sample[BaseQueries.sides] = hand_side
        else:
            flip = False

        # Get original image
        if BaseQueries.images in query or TransQueries.images in query:
            img = self.pose_dataset.get_image(idx)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if BaseQueries.images in query:
                sample[BaseQueries.images] = img

        # Flip and image 2d if needed
        if flip:
            center[0] = img.size[0] - center[0]
        # Data augmentation
        if self.train and needs_center_scale:
            # Randomly jitter center
            # Center is located in square of size 2*center_jitter_factor
            # in center of cropped image
            center_offsets = (
                self.center_jittering
                * scale
                * np.random.uniform(low=-1, high=1, size=2)
            )
            center = center + center_offsets.astype(int)

            # Scale jittering
            scale_jittering = self.scale_jittering * np.random.randn() + 1
            scale_jittering = np.clip(
                scale_jittering,
                1 - self.scale_jittering,
                1 + self.scale_jittering,
            )
            scale = scale * scale_jittering

            rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        else:
            rot = 0
        if self.block_rot:
            rot = self.max_rot
        rot_mat = np.array(
            [
                [np.cos(rot), -np.sin(rot), 0],
                [np.sin(rot), np.cos(rot), 0],
                [0, 0, 1],
            ]
        ).astype(np.float32)

        # Get 2D hand joints
        if (TransQueries.joints2d in query) or (TransQueries.images in query):
            affinetrans, post_rot_trans = handutils.get_affine_transform(
                center, scale, [self.inp_res, self.inp_res], rot=rot
            )
            if TransQueries.affinetrans in query:
                sample[TransQueries.affinetrans] = torch.from_numpy(
                    affinetrans
                )
        if BaseQueries.joints2d in query or TransQueries.joints2d in query:
            joints2d = self.pose_dataset.get_joints2d(idx)
            if flip:
                joints2d = joints2d.copy()
                joints2d[:, 0] = img.size[0] - joints2d[:, 0]
            if BaseQueries.joints2d in query:
                sample[BaseQueries.joints2d] = torch.from_numpy(joints2d)
        if TransQueries.joints2d in query:
            rows = handutils.transform_coords(joints2d, affinetrans)
            sample[TransQueries.joints2d] = torch.from_numpy(np.array(rows))

        if BaseQueries.camintrs in query or TransQueries.camintrs in query:
            camintr = self.pose_dataset.get_camintr(idx)
            if BaseQueries.camintrs in query:
                sample[BaseQueries.camintrs] = camintr
            if TransQueries.camintrs in query:
                # Rotation is applied as extr transform
                new_camintr = post_rot_trans.dot(camintr)
                sample[TransQueries.camintrs] = new_camintr

        # Get 2D object points
        if BaseQueries.objpoints2d in query or (
            TransQueries.objpoints2d in query
        ):
            objpoints2d = self.pose_dataset.get_objpoints2d(idx)
            if flip:
                objpoints2d = objpoints2d.copy()
                objpoints2d[:, 0] = img.size[0] - objpoints2d[:, 0]
            if BaseQueries.objpoints2d in query:
                sample[BaseQueries.objpoints2d] = torch.from_numpy(objpoints2d)
            if TransQueries.objpoints2d in query:
                transobjpoints2d = handutils.transform_coords(
                    objpoints2d, affinetrans
                )
                sample[TransQueries.objpoints2d] = torch.from_numpy(
                    np.array(transobjpoints2d)
                )

        # Get segmentation
        if BaseQueries.segms in query or TransQueries.segms in query:
            segm = self.pose_dataset.get_segm(idx)
            if flip:
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
            if BaseQueries.segms in query:
                sample[BaseQueries.segms] = segm
            if TransQueries.segms in query:
                segm = handutils.transform_img(
                    segm, affinetrans, [self.inp_res, self.inp_res]
                )
                segm = segm.crop((0, 0, self.inp_res, self.inp_res))
                segm = func_transforms.to_tensor(segm)
                sample[TransQueries.segms] = segm

        # Get 3D hand joints
        if (
            (BaseQueries.joints3d in query)
            or (TransQueries.joints3d in query)
            or (TransQueries.verts3d in query)
            or (TransQueries.objverts3d in query)
            or (TransQueries.objpoints3d in query)
        ):
            # Center on root joint
            center3d_queries = [
                TransQueries.joints3d,
                BaseQueries.joints3d,
                TransQueries.verts3d,
            ]
            obj_only = (
                (
                    TransQueries.objverts3d in query
                    or TransQueries.objpoints3d in query
                )
                and no_query_in(
                    center3d_queries, self.pose_dataset.all_queries
                )
                or self.as_obj_only
            )
            if not obj_only:
                if one_query_in(
                    [TransQueries.objpoints3d]
                    + [TransQueries.objverts3d]
                    + center3d_queries,
                    query,
                ):
                    joints3d = self.pose_dataset.get_joints3d(idx)
                    if flip:
                        joints3d[:, 0] = -joints3d[:, 0]

                    if BaseQueries.joints3d in query:
                        sample[BaseQueries.joints3d] = joints3d
                    if self.train:
                        joints3d = rot_mat.dot(
                            joints3d.transpose(1, 0)
                        ).transpose()
                    # Compute 3D center
                    if self.center_idx is not None:
                        if self.center_idx == -1:
                            center3d = (joints3d[9] + joints3d[0]) / 2
                        else:
                            center3d = joints3d[self.center_idx]
                    if TransQueries.joints3d in query and (
                        self.center_idx is not None
                    ):
                        joints3d = joints3d - center3d
                    if TransQueries.joints3d in query:
                        sample[TransQueries.joints3d] = torch.from_numpy(
                            joints3d
                        )

        # Get 3D hand vertices
        if TransQueries.verts3d in query:
            hand_verts3d = self.pose_dataset.get_verts3d(idx)
            if flip:
                hand_verts3d[:, 0] = -hand_verts3d[:, 0]
            hand_verts3d = rot_mat.dot(
                hand_verts3d.transpose(1, 0)
            ).transpose()
            if self.center_idx is not None:
                hand_verts3d = hand_verts3d - center3d
            sample[TransQueries.verts3d] = hand_verts3d

        # Get 3D object points
        if TransQueries.objpoints3d in query and (
            BaseQueries.objpoints3d in self.pose_dataset.all_queries
        ):
            points3d = self.pose_dataset.get_objpoints3d(
                idx, point_nb=self.point_nb
            )
            if flip:
                points3d[:, 0] = -points3d[:, 0]
            points3d = rot_mat.dot(points3d.transpose(1, 0)).transpose()
            obj_verts3d = points3d
        elif (
            TransQueries.objpoints3d in query
            or BaseQueries.objverts3d in query
            or TransQueries.objverts3d in query
        ) and (BaseQueries.objverts3d in self.pose_dataset.all_queries):
            obj_verts3d, obj_faces = self.pose_dataset.get_obj_verts_faces(idx)
            if flip:
                obj_verts3d[:, 0] = -obj_verts3d[:, 0]
            if BaseQueries.objverts3d in query:
                sample[BaseQueries.objverts3d] = obj_verts3d
            if TransQueries.objverts3d in query:
                origin_trans_mesh = rot_mat.dot(
                    obj_verts3d.transpose(1, 0)
                ).transpose()
                if self.center_idx is not None:
                    origin_trans_mesh = origin_trans_mesh - center3d
                sample[TransQueries.objverts3d] = origin_trans_mesh

            if BaseQueries.objfaces in query:
                sample[BaseQueries.objfaces] = obj_faces
            obj_verts3d = vertexsample.points_from_mesh(
                obj_faces,
                obj_verts3d,
                show_cloud=False,
                vertex_nb=self.point_nb,
            ).astype(np.float32)
            obj_verts3d = rot_mat.dot(obj_verts3d.transpose(1, 0)).transpose()

        elif TransQueries.objpoints3d in query:
            raise ValueError(
                "Requested TransQueries.objpoints3d for dataset "
                "without BaseQueries.objpoints3d and BaseQueries.objverts3d"
            )
        # Center object on hand or center of object if no hand present
        if TransQueries.objpoints3d in query:
            if obj_only:
                center3d = (obj_verts3d.max(0) + obj_verts3d.min(0)) / 2
            if self.center_idx is not None or obj_only:
                obj_verts3d = obj_verts3d - center3d
            if obj_verts3d.max() > 5000:
                print("BIIIG problem with sample")
                print(self.pose_dataset.image_names[idx])
            if obj_only:
                # Inscribe into sphere of radius 1
                radius = np.linalg.norm(obj_verts3d, 2, 1).max()
                obj_verts3d = obj_verts3d / radius
            sample[TransQueries.objpoints3d] = torch.from_numpy(obj_verts3d)

        if TransQueries.center3d in query:
            sample[TransQueries.center3d] = center3d

        if BaseQueries.manoidxs in query:
            sample[BaseQueries.manoidxs] = self.pose_dataset.get_manoidxs(idx)

        # Get rgb image
        if TransQueries.images in query:
            # Data augmentation
            if self.train:
                blur_radius = random.random() * self.blur_radius
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                img = imgtrans.color_jitter(
                    img,
                    brightness=self.brightness,
                    saturation=self.saturation,
                    hue=self.hue,
                    contrast=self.contrast,
                )
            # Transform and crop
            img = handutils.transform_img(
                img, affinetrans, [self.inp_res, self.inp_res]
            )
            img = img.crop((0, 0, self.inp_res, self.inp_res))

            # Tensorize and normalize_img
            img = func_transforms.to_tensor(img).float()
            if self.black_padding:
                padding_ratio = 0.2
                padding_size = int(self.inp_res * padding_ratio)
                img[:, 0:padding_size, :] = 0
                img[:, -padding_size:-1, :] = 0
                img[:, :, 0:padding_size] = 0
                img[:, :, -padding_size:-1] = 0

            if self.normalize_img:
                img = func_transforms.normalize(img, self.mean, self.std)
            else:
                img = func_transforms.normalize(
                    img, [0.5, 0.5, 0.5], [1, 1, 1]
                )
            if TransQueries.images in query:
                sample[TransQueries.images] = img

        # Add meta information
        if BaseQueries.meta in query:
            meta = self.pose_dataset.get_meta(idx)
            sample[BaseQueries.meta] = meta
        return sample

    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx, self.queries)
        except Exception:
            traceback.print_exc()
            print("Encountered error processing sample {}".format(idx))
            random_idx = random.randint(0, len(self))
            sample = self.get_sample(random_idx, self.queries)
        return sample

    def visualize_original(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.sides,
            BaseQueries.images,
            BaseQueries.joints2d,
            BaseQueries.objpoints2d,
            BaseQueries.camintrs,
            BaseQueries.objverts3d,
            BaseQueries.objfaces,
        ]
        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)
        sample = self.get_sample(idx, query=sample_queries)
        img = sample[BaseQueries.images]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Add title
        if BaseQueries.sides in sample:
            side = sample[BaseQueries.sides]
            ax.set_title("{} hand".format(side))
        ax.imshow(img)
        if BaseQueries.joints2d in sample:
            joints = sample[BaseQueries.joints2d]
            # Scatter hand joints on image
            viz2d.visualize_joints_2d(
                ax, joints, joint_idxs=False, links=self.pose_dataset.links
            )
            ax.axis("off")
        if BaseQueries.objpoints2d in sample:
            objpoints = sample[BaseQueries.objpoints2d]
            # Scatter hand joints on image
            ax.scatter(objpoints[:, 0], objpoints[:, 1], alpha=0.01)
        plt.show()

    def display_proj(self, ax, sample, proj="z", joint_idxs=False):

        if proj == "z":
            proj_1 = 0
            proj_2 = 1
            ax.invert_yaxis()
        elif proj == "y":
            proj_1 = 0
            proj_2 = 2
        elif proj == "x":
            proj_1 = 1
            proj_2 = 2

        if TransQueries.joints3d in sample:
            joints3d = sample[TransQueries.joints3d]
            viz2d.visualize_joints_2d(
                ax,
                np.stack([joints3d[:, proj_1], joints3d[:, proj_2]], axis=1),
                joint_idxs=joint_idxs,
                links=self.pose_dataset.links,
            )
        # Scatter  projection of 3d vertices
        if TransQueries.verts3d in sample:
            verts3d = sample[TransQueries.verts3d]
            ax.scatter(verts3d[:, proj_1], verts3d[:, proj_2], s=1)

        # Scatter projection of object 3d vertices
        if TransQueries.objpoints3d in sample:
            obj_verts3d = sample[TransQueries.objpoints3d]
            ax.scatter(obj_verts3d[:, proj_1], obj_verts3d[:, proj_2], s=1)
        ax.set_aspect("equal")  # Same axis orientation as imshow

    def visualize_3d_proj(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.sides,
            BaseQueries.images,
            TransQueries.joints3d,
            TransQueries.images,
            TransQueries.objpoints3d,
            TransQueries.verts3d,
            BaseQueries.joints2d,
        ]

        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)

        sample = self.get_sample(idx, query=sample_queries)
        fig = plt.figure()

        # Display transformed image
        ax = fig.add_subplot(121)
        img = sample[TransQueries.images].numpy().transpose(1, 2, 0)
        if not self.normalize_img:
            img += 0.5
        ax.imshow(img)

        # Add title
        if BaseQueries.sides in sample:
            side = sample[BaseQueries.sides]
            ax.set_title("{} hand".format(side))

        if TransQueries.objpoints3d in sample:
            ax = fig.add_subplot(122, projection="3d")
            objpoints3d = sample[TransQueries.objpoints3d].numpy()
            ax.scatter(objpoints3d[:, 0], objpoints3d[:, 1], objpoints3d[:, 2])
            ax.view_init(elev=90, azim=-90)
            cam_equal_aspect_3d(ax, objpoints3d)
        plt.show()

    def visualize_3d_transformed(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.images,
            BaseQueries.sides,
            TransQueries.joints3d,
            TransQueries.images,
            TransQueries.objpoints3d,
            TransQueries.verts3d,
            BaseQueries.joints2d,
        ]

        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)

        sample = self.get_sample(idx, query=sample_queries)
        fig = plt.figure()

        # Display transformed image
        ax = fig.add_subplot(141)
        img = sample[TransQueries.images].numpy().transpose(1, 2, 0)
        if not self.normalize_img:
            img += 0.5
        ax.imshow(img)
        # Add title
        if BaseQueries.sides in sample:
            side = sample[BaseQueries.sides]
            ax.set_title("{} hand".format(side))

        # Display XY projection
        ax = fig.add_subplot(142)
        self.display_proj(ax, sample, proj="z", joint_idxs=joint_idxs)

        # Display YZ projection
        ax = fig.add_subplot(143)
        self.display_proj(ax, sample, proj="x", joint_idxs=joint_idxs)

        # Display XZ projection
        ax = fig.add_subplot(144)
        self.display_proj(ax, sample, proj="y", joint_idxs=joint_idxs)
        plt.show()
        return fig

    def visualize_transformed(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.images,
            BaseQueries.joints2d,
            BaseQueries.sides,
            TransQueries.images,
            TransQueries.joints2d,
            TransQueries.objverts3d,
            BaseQueries.objfaces,
            TransQueries.camintrs,
            TransQueries.center3d,
            TransQueries.objpoints3d,
        ]
        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)
        sample = self.get_sample(idx, query=sample_queries)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = sample[TransQueries.images].numpy().transpose(1, 2, 0)
        if not self.normalize_img:
            img += 0.5
        ax.imshow(img)
        # Add title
        if BaseQueries.sides in sample:
            side = sample[BaseQueries.sides]
            ax.set_title("{} hand".format(side))

        if TransQueries.joints2d in sample:
            joints2d = sample[TransQueries.joints2d]
            viz2d.visualize_joints_2d(
                ax,
                joints2d,
                joint_idxs=joint_idxs,
                links=self.pose_dataset.links,
            )
        if (
            TransQueries.camintrs in sample
            and (TransQueries.objverts3d in sample)
            and BaseQueries.objfaces in sample
        ):
            verts = (
                torch.from_numpy(sample[TransQueries.objverts3d])
                .unsqueeze(0)
                .cuda()
            )
            center3d = (
                torch.from_numpy(sample[TransQueries.center3d])
                .cuda()
                .unsqueeze(0)
            )
            verts = center3d.unsqueeze(1) / 1000 + verts / 1000
        plt.show()


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
