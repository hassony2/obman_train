import argparse
import os
import pickle

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import argutils
from handobjectdatasets.queries import TransQueries, BaseQueries

from mano_train.netscripts.reload import reload_model
from mano_train.visualize import displaymano
from mano_train.demo.attention import AttentionHook
from mano_train.demo.preprocess import prepare_input, preprocess_frame

def forward_pass_3d(model, input_image, pred_obj=True):
    sample = {}
    sample[TransQueries.images] = input_image
    sample[BaseQueries.sides] = [args.hand_side]
    sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
    sample['root'] = 'wrist'
    if pred_obj:
        sample[TransQueries.objpoints3d] = input_image.new_ones((1, 600,
                                                                 3)).float()
    _, results, _ = model.forward(sample, no_loss=True)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint',
        default='release_models/obman/checkpoint.pth.tar')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--hand_side', default='left')
    parser.add_argument('--pred_obj', action='store_true')
    parser.add_argument('--image_path', help='Path to video')
    parser.add_argument(
        '--no_beta', action='store_true', help='Force shape to average')
    parser.add_argument(
        '--flip_left_right',
        action='store_true',
        help='Force shape to average')
    args = parser.parse_args()
    argutils.print_args(args)

    checkpoint = os.path.dirname(args.resume)
    with open(os.path.join(checkpoint, 'opt.pkl'), 'rb') as opt_f:
        opts = pickle.load(opt_f)

    # Initialize network
    model = reload_model(args.resume, opts, no_beta=args.no_beta)

    model.eval()

    print('Please input image of {} hand !'.format(args.hand_side))

    # load faces of hand
    with open('misc/mano/MANO_RIGHT.pkl', 'rb') as p_f:
        mano_right_data = pickle.load(p_f, encoding='latin1')
        faces = mano_right_data['f']

    # Add attention map
    attention_hand = AttentionHook(model.module.base_net)
    if hasattr(model.module, 'atlas_base_net'):
        attention_atlas = AttentionHook(model.module.atlas_base_net)
        has_atlas_encoder = True
    else:
        has_atlas_encoder = False

    fig = plt.figure(figsize=(4, 4))
    fig.clf()
    frame = cv2.imread(args.image_path)
    frame = preprocess_frame(frame)
    input_image = prepare_input(frame)
    cv2.imshow('input', frame)
    blend_img_hand = attention_hand.blend_map(frame)
    cv2.imshow('attention hand', blend_img_hand)
    if has_atlas_encoder:
        blend_img_atlas = attention_atlas.blend_map(frame)
        cv2.imshow('attention atlas', blend_img_atlas)
    img = Image.fromarray(frame.copy())
    hand_crop = cv2.resize(np.array(img), (256, 256))

    noflip_hand_image = prepare_input(
        hand_crop, flip_left_right=False)
    flip_hand_image = prepare_input(
        hand_crop, flip_left_right=True)
    noflip_output = forward_pass_3d(model, noflip_hand_image)
    flip_output = forward_pass_3d(model, flip_hand_image)
    flip_verts = flip_output['verts'].cpu().detach().numpy()[0]
    noflip_verts = noflip_output['verts'].cpu().detach().numpy()[0]
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    displaymano.add_mesh(ax, flip_verts, faces, flip_x=True)
    if 'objpoints3d' in flip_output:
        objverts = flip_output['objpoints3d'].cpu().detach().numpy()[0]
        displaymano.add_mesh(
            ax, objverts, flip_output['objfaces'], flip_x=True, c='r')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    displaymano.add_mesh(ax, noflip_verts, faces, flip_x=True)
    if 'objpoints3d' in noflip_output:
        objverts = noflip_output['objpoints3d'].cpu().detach().numpy()[0]
        displaymano.add_mesh(
            ax, objverts, noflip_output['objfaces'], flip_x=True, c='r')
    plt.show()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # Captured right hand of user is seen as right (mirror effect)
    cv2.imshow('pose estimation', cv2.flip(frame, 1))
    cv2.imshow('mesh', buf)
    cv2.waitKey(0)
