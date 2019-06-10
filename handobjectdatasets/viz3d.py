import numpy as np


def equal_aspect_3d(ax):
    extents = np.array(
        [getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"]
    )
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def visualize_joints_3d(
    ax, joints, joint_idxs=True, alpha=1, links=None, scatter_color="r"
):
    """
    Args:
        ax = fig.add_subplot(111, projection='3d')
    """
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
    z = joints[:, 2]
    ax.scatter(x, y, z, s=1, c=scatter_color)

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            ax.text(row[0], row[1], row[2], str(row_idx))
    _draw3djoints(ax, joints, links, alpha=alpha)
    equal_aspect_3d(ax)


def _draw3djoints(ax, annots, links, alpha=1):
    colors = ["r", "m", "b", "c", "g"]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw3dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha,
            )


def _draw3dseg(ax, annot, idx1, idx2, c="r", alpha=1):
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]],
        [annot[idx1, 1], annot[idx2, 1]],
        [annot[idx1, 2], annot[idx2, 2]],
        c=c,
        alpha=alpha,
        linewidth=1,
    )
