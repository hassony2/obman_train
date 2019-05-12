from collections import OrderedDict, defaultdict
import json
from functools import lru_cache, wraps
import os
import pickle
import random

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from scipy.spatial.distance import cdist
from scipy.io import loadmat
import trimesh
from tqdm import tqdm

from handobjectdatasets.queries import (BaseQueries, TransQueries,
                                        get_trans_queries)
from handobjectdatasets import handutils, loadutils
from handobjectdatasets.fhbutils import hash_dict
from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model as load_mano_model

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Core50():
    def __init__(self,
                 use_cache=False,
                 mini_factor=None,
                 class_name='mobile_phone',
                 use_annots=True, scale_factor=1.2):
        """
        Args:
            filter_no_contact: remove data where hand not in contact with object
            filter_thresh: min distance between hand and object to consider contact (mm)
        """
        super().__init__()
        self.all_queries = [
            BaseQueries.images, BaseQueries.joints3d, BaseQueries.sides,
            BaseQueries.objpoints3d
        ]
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)
        self.scale_factor = scale_factor
        self.split = 'all'
        class_dict = {
            'mobile_phone': ['o{}'.format(idx) for idx in range(6, 11)],
            'ball': ['o{}'.format(idx) for idx in range(31, 36)],
            'light_bulb': ['o{}'.format(idx) for idx in range(16, 21)],
            'marker': ['o{}'.format(idx) for idx in range(36, 41)],
            'can': ['o{}'.format(idx) for idx in range(21, 26)],
            'remote_control': ['o{}'.format(idx) for idx in range(46, 51)],
            'cups': ['o{}'.format(idx) for idx in range(41, 46)]
        }
        if class_name not in class_dict:
            raise ValueError(
                '{} should be in {}'.format(class_name, class_dict.keys()))
        self.sessions = ['s{}'.format(idx) for idx in range(1, 12)]
        self.class_name = class_name
        self.class_keys = class_dict[class_name]

        # Set cache path
        self.use_cache = use_cache
        self.cache_folder = os.path.join('data', 'cache', 'core50')
        os.makedirs(self.cache_folder, exist_ok=True)
        self.name = 'core50'
        self.mini_factor = mini_factor

        self.root = '/sequoia/data2/dataset/handatasets/Core50'
        self.annot_root = os.path.join(self.root, 'core50_350x350_Annot')
        self.rgb_root = os.path.join(self.root, 'core50_350x350')
        self.depth_root = os.path.join(self.root, 'core50_350x350_DepthMap')
        self.load_dataset()

        print('Got {} samples for class {}'.format(
            len(self.image_names), class_name))

        # get paired links as neighboured joints
        self.links = [(0, 1)]
        # Values from https://github.com/OpenKinect/libfreenect2/issues/41
        self.depth_intrinsic = [[365.456, 0, 254.878], [0, 365.456, 205.395],
                                [0, 0, 1]]
        self.color_intrinsic = [[1060.707250708333, 0, 956.354471815484],
                                [1058.608326305465, 0,
                                 518.9784429882449], [0, 0, 1]]

    def load_dataset(self):
        cache_path = os.path.join(self.cache_folder, '{}_{}.pkl'.format(
            self.mini_factor, self.class_name))
        if os.path.exists(cache_path) and self.use_cache:
            with open(cache_path, 'rb') as cache_f:
                annotations = pickle.load(cache_f)
            print('Cached information for dataset {} loaded from {}'.format(
                self.name, cache_path))

        else:
            # with open(os.path.join(self.root, 'paths.pkl'), 'rb') as p_f:
            #     paths = pickle.load(p_f)
            # imgs = np.load(os.path.join(self.root, 'core50_imgs.npz'))['x']
            hand_sides = []
            centers = []
            scales = []
            img_names = []
            depth_img_names = []

            for session in self.sessions:
                sess_path = os.path.join(self.annot_root, session)
                for obj in sorted(os.listdir(sess_path)):
                    if obj in self.class_keys:
                        obj_path = os.path.join(sess_path, obj)
                        obj_annots = sorted([
                            annot for annot in os.listdir(obj_path)
                            if '.mat' in annot
                        ])
                        for obj_annot in obj_annots:
                            annot_path = os.path.join(obj_path, obj_annot)
                            annot = loadmat(annot_path)
                            hand_root2d = annot['annot']['hand'][0, 0][
                                'root2d'][0, 0]
                            hand_root_depth_png = annot['annot']['hand'][0, 0][
                                'root_depth_png'][0, 0]
                            hand_depth = 8000 * (
                                255 - hand_root_depth_png) / 1000 / 256

                            obj_root2d = annot['annot']['object'][0, 0][
                                'root2d'][0, 0]
                            obj_root_depth = annot['annot']['object'][0, 0][
                                'root_depth_png'][0, 0]
                            bbox = annot['annot']['crop'][0, 0]
                            side_code = annot['annot']['hand'][0, 0]['side'][
                                0, 0][0]
                            if side_code == 'R':
                                side = 'right'
                            elif side_code == 'L':
                                side = 'left'
                            hand_sides.append(side)
                            center = np.array([(bbox[0, 0] + bbox[0, 2]) / 2,
                                               (bbox[0, 1] + bbox[0, 3]) / 2])
                            scale = self.scale_factor * np.array([
                                bbox[0, 2] - bbox[0, 0],
                                bbox[0, 3] - bbox[0, 1]
                            ])
                            centers.append(center)
                            scales.append(scale)
                            prefix = '_'.join(
                                obj_annot.split('.')[0].split('_')[1:])
                            rgb_img_path = os.path.join(
                                self.rgb_root, session, obj,
                                'C_{}.png'.format(prefix))
                            img_names.append(rgb_img_path)
                            depth_img_path = os.path.join(
                                self.depth_root, session, obj,
                                'D_{}.png'.format(prefix))
                            depth_img_names.append(depth_img_path)

            if self.mini_factor:
                mini_nb = int(len(img_names) * self.mini_factor)
                scales = scales[:mini_nb]
                centers = centers[:mini_nb]
                hand_sides = hand_sides[:mini_nb]
            annotations = {
                'image_names': img_names,
                'hand_sides': hand_sides,
                'scales': scales,
                'centers': centers,
            }
        with open(cache_path, 'wb') as fid:
            pickle.dump(annotations, fid)
        print('Wrote cache for dataset {} to {}'.format(self.name, cache_path))

        # Get image paths
        print(cache_path)
        self.image_names = annotations['image_names']
        self.hand_sides = annotations['hand_sides']
        self.scales = annotations['scales']
        self.centers = annotations['centers']

    def get_image(self, idx):
        image_path = self.image_names[idx]
        img = Image.open(image_path)
        img = img.convert('RGB')
        return img

    def get_joints3d(self, idx):
        joints3d = np.zeros((21, 3), dtype=np.float32)
        return joints3d

    def get_objpoints3d(self, idx, point_nb=100):
        points3d = np.zeros((point_nb, 3), dtype=np.float32)
        return points3d

    def get_camintr(self, idx):
        camintr = self.cam_intr
        return camintr

    def get_sides(self, idx):
        side = self.hand_sides[idx]
        return side

    def get_center_scale(self, idx):
        center = self.centers[idx]
        scale = np.max(self.scales[idx])
        return center, scale

    def __len__(self):
        return len(self.image_names)

