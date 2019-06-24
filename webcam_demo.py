import argparse
import os
import pickle

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from handobjectdatasets.queries import TransQueries, BaseQueries
from handobjectdatasets.viz2d import visualize_joints_2d_cv2

from mano_train.exputils import argutils
from mano_train.netscripts.reload import reload_model
from mano_train.visualize import displaymano
from mano_train.demo.attention import AttentionHook
from mano_train.demo.preprocess import prepare_input, preprocess_frame


def forward_pass_3d(model, input_image, pred_obj=True, hand_side="left"):
    sample = {}
    sample[TransQueries.images] = input_image
    sample[BaseQueries.sides] = [hand_side]
    sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
    sample["root"] = "wrist"
    if pred_obj:
        sample[TransQueries.objpoints3d] = input_image.new_ones(
            (1, 600, 3)
        ).float()
    _, results, _ = model.forward(sample, no_loss=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint",
        default="release_models/obman/checkpoint.pth.tar",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--hand_side", default="left")
    parser.add_argument("--video_path", help="Path to video")
    parser.add_argument(
        "--no_beta", action="store_true", help="Force shape to average"
    )
    parser.add_argument(
        "--flip_left_right", action="store_true", help="Force shape to average"
    )
    args = parser.parse_args()
    argutils.print_args(args)

    checkpoint = os.path.dirname(args.resume)
    with open(os.path.join(checkpoint, "opt.pkl"), "rb") as opt_f:
        opts = pickle.load(opt_f)

    # Initialize network
    model = reload_model(args.resume, opts, no_beta=args.no_beta)

    model.eval()

    # Initialize stream from camera
    if args.video_path is None:
        # Read from webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_path)

    if cap is None:
        raise RuntimeError("OpenCV could not use webcam")

    print("Please use {} hand !".format(args.hand_side))

    # load faces of hand
    with open("misc/mano/MANO_RIGHT.pkl", "rb") as p_f:
        mano_right_data = pickle.load(p_f, encoding="latin1")
        faces = mano_right_data["f"]

    # Add attention map
    attention_hand = AttentionHook(model.module.base_net)
    if hasattr(model.module, "atlas_base_net"):
        attention_atlas = AttentionHook(model.module.atlas_base_net)
        has_atlas_encoder = True
    else:
        has_atlas_encoder = False

    fig = plt.figure(figsize=(4, 4))
    while True:
        fig.clf()
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("OpenCV could not load frame")
        frame = preprocess_frame(frame)
        input_image = prepare_input(frame)
        blend_img_hand = attention_hand.blend_map(frame)
        if has_atlas_encoder:
            blend_img_atlas = attention_atlas.blend_map(frame)
            cv2.imshow("attention atlas", blend_img_atlas)
        img = Image.fromarray(frame.copy())
        hand_crop = cv2.resize(np.array(img), (256, 256))
        hand_image = prepare_input(
            hand_crop, flip_left_right=args.flip_left_right
        )
        output = forward_pass_3d(model, hand_image, hand_side=args.hand_side)

        if "joints2d" in output:
            joints2d = output["joints2d"]
            frame = visualize_joints_2d_cv2(
                frame, joints2d.cpu().detach().numpy()[0]
            )

        cv2.imshow("attention hand", blend_img_hand)

        verts = output["verts"].cpu().detach().numpy()[0]
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        displaymano.add_mesh(ax, verts, faces, flip_x=True)
        if "objpoints3d" in output:
            objverts = output["objpoints3d"].cpu().detach().numpy()[0]
            displaymano.add_mesh(
                ax, objverts, output["objfaces"], flip_x=True, c="r"
            )
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        # Captured right hand of user is seen as right (mirror effect)
        cv2.imshow("pose estimation", cv2.flip(frame, 1))
        cv2.imshow("mesh", buf)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
