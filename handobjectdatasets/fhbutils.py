from collections import OrderedDict, defaultdict
from functools import lru_cache, wraps
import os

import numpy as np


def load_objects(
        obj_root='/sequoia/data2/dataset/handatasets/fhb/Object_models',
        object_names=['juice']):
    all_models = OrderedDict()
    for obj_name in object_names:
        import trimesh
        obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                '{}_model.ply'.format(obj_name))
        mesh = trimesh.load(obj_path)
        all_models[obj_name] = {
            'verts': np.array(mesh.vertices),
            'faces': np.array(mesh.faces)
        }
    return all_models


def load_object_infos(
        seq_root='/sequoia/data2/dataset/handatasets/fhb/Object_6D_pose_annotation_v1'
):
    subjects = os.listdir(seq_root)
    annots = {}
    for subject in subjects:
        subject_dict = {}
        subj_path = os.path.join(seq_root, subject)
        actions = os.listdir(subj_path)
        for action in actions:
            object_name = '_'.join(action.split('_')[1:])
            action_path = os.path.join(subj_path, action)
            seqs = os.listdir(action_path)
            for seq in seqs:
                seq_path = os.path.join(action_path, seq, 'object_pose.txt')
                with open(seq_path, 'r') as seq_f:
                    raw_lines = seq_f.readlines()
                for raw_line in raw_lines:
                    line = raw_line.strip().split(' ')
                    frame_idx = int(line[0])
                    trans_matrix = np.array(line[1:]).astype(np.float32)
                    trans_matrix = trans_matrix.reshape(4, 4).transpose()
                    subject_dict[(action, seq, frame_idx)] = (object_name,
                                                              trans_matrix)
        annots[subject] = subject_dict
    return annots


def get_action_train_test(lines_raw, subjects_info):
    """
    Returns dicts of samples where key is
        subject: name of subject
        action_name: action class
        action_seq_idx: idx of action instance
        frame_idx
    and value is the idx of the action class
    """
    all_infos = []
    test_split = False
    test_samples = {}
    train_samples = {}
    for line in lines_raw[1:]:
        if line.startswith('Test'):
            test_split = True
            continue
        subject, action_name, action_seq_idx = line.split(' ')[0].split('/')
        action_idx = line.split(' ')[1].strip()  # Action classif index
        frame_nb = int(subjects_info[subject][(action_name, action_seq_idx)])
        for frame_idx in range(frame_nb):
            sample_info = (subject, action_name, action_seq_idx, frame_idx)
            if test_split:
                test_samples[sample_info] = action_idx
            else:
                train_samples[sample_info] = action_idx
            all_infos.append(sample_info)
    test_nb = len(
        np.unique(
            list((sub, act_n, act_seq)
                 for (sub, act_n, act_seq, _) in test_samples),
            axis=0))
    assert test_nb == 575, 'Should get 575 test samples, got {}'.format(
        test_nb)
    train_nb = len(
        np.unique(
            list((sub, act_n, act_seq)
                 for (sub, act_n, act_seq, _) in train_samples),
            axis=0))
    # 600 - 1 Subject5/use_flash/6 discarded sample
    assert train_nb == 599, 'Should get 599 train samples, got {}'.format(
        train_nb)
    assert len(test_samples) + len(train_samples) == len(all_infos)
    return train_samples, test_samples, all_infos


def hash_dict(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """

    class HDict(dict):
        def __hash__(self):
            # Makes two level dictionnary hashable
            return hash(
                frozenset(
                    {key: hash(frozenset(val))
                     for key, val in self.items()}))

    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple(
            [HDict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {
            k: HDict(v) if isinstance(v, dict) else v
            for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


def transform_obj_verts(verts, trans, cam_extr):
    verts = verts * 1000
    hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
    trans_verts = trans.dot(hom_verts.T).T
    trans_verts = cam_extr.dot(trans_verts.transpose()).transpose()[:, :3]
    return trans_verts
