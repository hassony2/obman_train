import cv2
from matplotlib import pyplot as plt


def visualize_joints_2d(
    ax, joints, joint_idxs=True, links=None, alpha=1, scatter=True, linewidth=2
):
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    if scatter:
        ax.scatter(x, y, 1, "r")

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha, linewidth=linewidth)
    ax.axis("equal")


def _draw2djoints(ax, annots, links, alpha=1, linewidth=1):
    colors = ["r", "m", "b", "c", "g"]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha,
                linewidth=linewidth,
            )


def _draw2dseg(ax, annot, idx1, idx2, c="r", alpha=1, linewidth=1):
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]],
        [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha,
        linewidth=linewidth,
    )


def visualize_joints_2d_cv2(
    img,
    joints,
    joint_idxs=True,
    links=None,
    alpha=1,
    scatter=True,
    linewidth=2,
):
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    # Scatter hand joints on image

    # Add idx labels to joints
    _draw2djoints_cv2(img, joints, links, alpha=alpha, linewidth=linewidth)
    return img


def _draw2djoints_cv2(img, annots, links, alpha=1, linewidth=1):
    colors = [
        (0, 255, 0),
        (0, 255, 255),
        (0, 0, 255),
        (255, 0, 255),
        (255, 0, 0),
        (255, 255, 0),
    ]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            img = _draw2dseg_cv2(
                img,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                col=colors[finger_idx],
                alpha=alpha,
                linewidth=linewidth,
            )
    return img


def _draw2dseg_cv2(
    img, annot, idx1, idx2, col=(0, 255, 0), alpha=1, linewidth=1
):
    cv2.line(
        img,
        (annot[idx1, 0], annot[idx1, 1]),
        (annot[idx2, 0], annot[idx2, 1]),
        col,
        linewidth,
    )
    return img
