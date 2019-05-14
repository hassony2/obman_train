import argparse
import os
import pickle
import sys
import warnings

import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
sys.path.append('/sequoia/data1/yhasson/code/pytorch-pose')
from pose.utils.evaluation import get_preds
from pose.models.hourglass import hg


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


def bbox_from_joints(joint_batch, scale_factor=2.2):
    mins = joint_batch.min(1)[0]
    maxs = joint_batch.max(1)[0]
    # bboxes are tight !
    bboxes = torch.cat([mins, maxs], 1)
    centers = (mins + maxs) / 2
    scales = torch.max(maxs - mins, 1)[0] * scale_factor
    return bboxes, centers, scales


def forward_pass_2d(model, input_image, scale_factor=2.2):
    output = model(input_image)
    score_map = output[-1].data.cpu()
    conf = score_map[0].max(1)[0].max(1)[0]
    preds = get_preds(score_map) * 256 / 64
    bboxes, centers, scales = bbox_from_joints(preds, scale_factor=scale_factor)

    # Remove batch dim
    bboxes = bboxes[0]
    centers = centers[0]
    scales = scales[0]
    return bboxes, centers.numpy(), float(scales)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Path to checkpoint')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--hand_side', default='right')
    parser.add_argument('--pred_obj', action='store_true')
    parser.add_argument('--video_path', help='Path to video')
    parser.add_argument('--no_beta', action='store_true', help='Force shape to average')
    parser.add_argument('--flip_left_right', action='store_true', help='Force shape to average')
    args = parser.parse_args()
    argutils.print_args(args)

    checkpoint = os.path.dirname(args.resume)
    with open(os.path.join(checkpoint, 'opt.pkl'), 'rb') as opt_f:
        opts = pickle.load(opt_f)

    # Load 2d network
    model2d_path = 'misc/hg.pth'
    model_2d = hg(num_stacks=2, num_blocks=1, num_classes=21)
    checkpoint = torch.load(model2d_path)
    model_2d.load_state_dict(checkpoint['state_dict'])
    model_2d = torch.nn.DataParallel(model_2d)

    # Initialize network
    if 'resnet_version' in opts:
        version = opts['resnet_version']
    else:
        version = 18
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
    _, faces = objectio.load_obj(
        'misc/mano/mano_{}.obj'.format(args.hand_side))

    fig = plt.figure(figsize=(4, 4))
    while (True):
        fig.clf()
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError('OpenCV could not load frame')
        frame = preprocess_frame(frame)
        input_image = prepare_input(frame)
        
        # Get hand center and scale
        bboxes, centers, scale = forward_pass_2d(model_2d, input_image)
        # Get hand crop
        img = Image.fromarray(frame.copy())

        # Draw crop
        cv2.rectangle(frame, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), (0, 255, 0), 3)
        cv2.rectangle(frame, (int(centers[0] - scale / 2), int(centers[1] - scale / 2)),(
                        int(centers[0] + scale / 2), int(centers[1] + scale / 2)), (0, 0, 255), 3)
        print('Hand crop disabled !')
        # img = img.crop((centers[0] - scale / 2, centers[1] - scale / 2,
        #                 centers[0] + scale / 2, centers[1] + scale / 2))
        hand_crop = cv2.resize(np.array(img), (256, 256))
        hand_image = prepare_input(hand_crop, flip_left_right=args.flip_left_right)
        output = forward_pass_3d(model, hand_image)
        keypoints = output['joints'].cpu().detach().numpy()
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
