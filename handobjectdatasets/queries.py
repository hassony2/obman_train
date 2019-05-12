from enum import Enum


class BaseQueries(Enum):
    """
    Possible inputs from pose_dataset
    """
    camintrs = 'camintrs'
    depth = 'depth'
    hand_poses = 'hand_poses'
    hand_pcas = 'hand_pcas'
    images = 'images'
    joints2d = 'joints2d'
    joints3d = 'joints3d'
    meta = 'meta'
    objpoints2d = 'objpoints2d'
    objpoints3d = 'objpoints3d'
    objverts3d = 'objverts3d'
    objfaces = 'objfaces'
    verts3d = 'verts3d'
    sides = 'sides'
    segms = 'segms'
    manoidxs = 'manoidxs'  # Idxs of mano-compatible joints


class TransQueries(Enum):
    """
    Possible outputs from dataset
    """
    camintrs = 'camintrs'
    depth = 'depth'
    images = 'images'
    joints2d = 'joints2d '
    joints3d = 'joints3d'
    objfaces = 'objfaces'
    objpoints2d = 'objpoints2d'
    objpoints3d = 'objpoints3d'
    objverts3d = 'objverts3d'
    segms = 'segms'
    verts3d = 'verts3d'
    center3d = 'center3d'
    affinetrans = 'affinetrans'
    rotmat = 'rotmat'
    sdf = 'sdf'
    sdf_points = 'sdf_points'
    mapvals = 'mapvals'
    mapidxs = 'mapidxs'

def no_query_in(candidate_queries, base_queries):
    return not (one_query_in(candidate_queries, base_queries))


def one_query_in(candidate_queries, base_queries):
    for query in candidate_queries:
        if query in base_queries:
            return True
    return False


def get_trans_queries(base_queries):
    trans_queries = []
    add_center = False
    if BaseQueries.objverts3d in base_queries:
        trans_queries.append(TransQueries.objpoints3d)
        trans_queries.append(TransQueries.objverts3d)
        add_center = True
    if BaseQueries.objpoints3d in base_queries:
        trans_queries.append(TransQueries.objpoints3d)
        add_center = True
    if BaseQueries.images in base_queries:
        trans_queries.append(TransQueries.images)
        trans_queries.append(TransQueries.affinetrans)
        trans_queries.append(TransQueries.rotmat)
    if BaseQueries.depth in base_queries:
        trans_queries.append(TransQueries.depth)
    if BaseQueries.joints2d in base_queries:
        trans_queries.append(TransQueries.joints2d)
        trans_queries.append(TransQueries.mapvals)
        trans_queries.append(TransQueries.mapidxs)
    if BaseQueries.joints3d in base_queries:
        trans_queries.append(TransQueries.joints3d)
        add_center = True
    if BaseQueries.verts3d in base_queries:
        trans_queries.append(TransQueries.verts3d)
        add_center = True
    if BaseQueries.segms in base_queries:
        trans_queries.append(TransQueries.segms)
    if add_center:
        trans_queries.append(TransQueries.center3d)
    if BaseQueries.objpoints2d in base_queries:
        trans_queries.append(TransQueries.objpoints2d)
    if BaseQueries.camintrs in base_queries:
        trans_queries.append(TransQueries.camintrs)

    return trans_queries
