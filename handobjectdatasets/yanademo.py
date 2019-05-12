from collections import OrderedDict, defaultdict
import json
from functools import lru_cache, wraps
import os
import pickle
import random

import numpy as np
from tqdm import tqdm

from handobjectdatasets.queries import (BaseQueries, TransQueries,
                                        get_trans_queries)
from handobjectdatasets import handutils, loadutils

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from scipy.spatial.distance import cdist
except Exception:
    print('Failed imports PIL and scipy in tzionas')



class YanaDemo():
    def __init__(self, split='train', joint_nb=21, scale_factor=2.2, version=1, side='right'):
        """
        Args:
            filter_no_contact: remove data where hand not in contact with object
            filter_thresh: min distance between hand and object to consider contact (mm)
        """
        super().__init__()
        super().__init__()
        self.all_queries = [
            BaseQueries.images, BaseQueries.joints3d, BaseQueries.sides,
            BaseQueries.objpoints3d
        ]
        trans_queries = get_trans_queries(self.all_queries)
        self.side = side
        self.all_queries.extend(trans_queries)
        self.scale_factor = scale_factor
        self.version = version
        self.root = os.path.join('/sequoia/data2/dataset/handatasets/yanaimages/v{}'.format(self.version))

        self.name = 'yanademo_v{}'.format(self.version)
        self.joint_nb = joint_nb
        self.split = split

        self.load_dataset()

        print('Got {} samples for split {}'.format(
            len(self.image_names), self.split))

        # get paired links as neighboured joints
        self.links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                      (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]

        # Load normalization bone values
        mean_file = 'bones_synthgrasps_root_wrist.pkl'
        with open(
                os.path.join(
                    '/sequoia/data1/yhasson/code/pose_3d/handobjectdatasets/misc',
                    'stats', mean_file), 'rb') as p_f:
            grasp_data = pickle.load(p_f)
        self.mano_means = grasp_data['means']

    def load_dataset(self):
        centers = []
        scales = []
        img_names = []
        for seq in sorted(os.listdir(self.root)):
            seq_path = os.path.join(self.root, seq)
            for img_name in sorted(os.listdir(seq_path)):
                img_name = os.path.join(seq_path, img_name)
                img_names.append(img_name)
                centers.append(np.array([1727, 1150]))
                scales.append(1000 * self.scale_factor)
 
        annotations = {
            'image_names': img_names,
            'scales': scales,
            'centers': centers,
        }
        self.image_names = annotations['image_names']
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
        return self.side

    def get_center_scale(self, idx):
        img = self.get_image(idx)
        center = np.array([img.size[0]/2, img.size[1]/2])
        scale = min(img.size)
        # center = self.centers[idx]
        # scale = np.max(self.scales[idx])
        # print('center')
        # print(center)
        return center, scale

    def __len__(self):
        return len(self.image_names)
