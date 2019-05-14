from mano_train.netscripts import savemano


def get_batch_infos(save_pickle, faces_right, faces_left, get_depth=True):
    batch_infos = savemano.load_batch_info(
        save_pickle, faces_right, faces_left, get_depth=get_depth)
    return batch_infos
