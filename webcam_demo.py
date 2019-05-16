import argparse
import os
import pickle
import sys

import cv2
from matplotlib import pyplot as plt
import numpy as np
from torchvision.transforms import functional as func_transforms
from PIL import Image
import torch

import argutils
from handobjectdatasets.queries import TransQueries, BaseQueries

from mano_train.networks.handnet import HandNet
from mano_train.netscripts.reload import reload_model
from mano_train.modelutils import modelio
from mano_train.objectutils import objectio
from mano_train.visualize import displaymano


def preprocess_frame(frame):
    # Squarify
    frame = frame[:,
                  int(frame.shape[1] / 2 - frame.shape[0] / 2):int(
                      frame.shape[1] / 2 + frame.shape[0] / 2)]
    frame = cv2.resize(frame, (256, 256))
    return frame


def prepare_input(frame, flip_left_right=False):
    # BGR to RGB and flip frame
    input_image = np.flip(frame, axis=2).copy()
    if flip_left_right:
        input_image = np.flip(input_image, axis=1).copy()

    # Concert to shape batch_size=1, rgb, h, w
    input_image = torch.Tensor(input_image.transpose(2, 0, 1))

    # To debug what is actually fed to network
    if args.debug:
        plt.imshow(input_image.numpy().transpose(1, 2, 0) / 255)
        plt.show()
    input_image = func_transforms.normalize(input_image / 255, [0.5, 0.5, 0.5],
                                            [1, 1, 1]).unsqueeze(0)
    # Equivalently
    # input_image_1 = input_image / 255 - 0.5
    input_image = input_image.cuda()
    return input_image


def forward_pass_3d(model, input_image):
    sample = {}
    sample[TransQueries.images] = input_image
    sample[BaseQueries.sides] = [args.hand_side]
    sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
    sample['root'] = 'wrist'
    if args.pred_obj:
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
    parser.add_argument('--video_path', help='Path to video')
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

    model = torch.nn.DataParallel(model)
    model.eval()

    # Initialize stream from camera
    if args.video_path is None:
        # Read from webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)

    if cap is None:
        raise RuntimeError('OpenCV could not use webcam')

    print('Please use {} hand !'.format(args.hand_side))

    # load faces of hand
    with open('misc/mano/MANO_RIGHT.pkl', 'rb') as p_f:
        mano_right_data = pickle.load(p_f, encoding='latin1')
        faces = mano_right_data['f']

    fig = plt.figure(figsize=(4, 4))
    while (True):
        fig.clf()
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError('OpenCV could not load frame')
        frame = preprocess_frame(frame)
        input_image = prepare_input(frame)

        img = Image.fromarray(frame.copy())
        hand_crop = cv2.resize(np.array(img), (256, 256))
        hand_image = prepare_input(
            hand_crop, flip_left_right=args.flip_left_right)
        output = forward_pass_3d(model, hand_image)
        verts = output['verts'].cpu().detach().numpy()[0]
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        displaymano.add_mesh(ax, verts, faces, flip_x=True)
        if 'objpoints3d' in output:
            objverts = output['objpoints3d'].cpu().detach().numpy()[0]
            displaymano.add_mesh(
                ax, objverts, output['objfaces'], flip_x=True, c='r')
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        # Captured right hand of user is seen as right (mirror effect)
        cv2.imshow('pose estimation', cv2.flip(frame, 1))
        cv2.imshow('mesh', buf)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
