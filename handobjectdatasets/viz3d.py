from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objs as go


def equal_aspect_3d(ax):
    extents = np.array(
        [getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def visualize_joints_3d(ax,
                        joints,
                        joint_idxs=True,
                        alpha=1,
                        links=None,
                        scatter_color='r'):
    """
    Args:
        ax = fig.add_subplot(111, projection='3d')
    """
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
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
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw3dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha)


def _draw3dseg(ax, annot, idx1, idx2, c='r', alpha=1):
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
        [annot[idx1, 2], annot[idx2, 2]],
        c=c,
        alpha=alpha,
        linewidth=1)


def pyplot_hands(joints):
    joint_nb = joints.shape[0]
    if joint_nb == 21:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    elif joint_nb == 20:
        links = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15),
                 (16, 17, 18, 19)]
    else:
        raise ValueError('{} joints not supported for pyplot visualization'.
                         format(joint_nb))
    trace1 = go.Scatter3d(
        x=joints[:, 0],
        y=joints[:, 1],
        z=joints[:, 2],
        marker=dict(size=1),
        mode='markers',
        name='markers')

    def add_trace(finger_links, joints):
        x_lines = list()
        y_lines = list()
        z_lines = list()
        # create the coordinate list for the lines
        for idx in finger_links:
            # print(idx)
            x_lines.append(joints[idx, 0])
            y_lines.append(joints[idx, 1])
            z_lines.append(joints[idx, 2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
        trace = go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines, mode='lines', name='lines')
        return trace

    traces = []
    for finger_idx, finger_links in enumerate(links):
        trace = add_trace(finger_links, joints)
        traces.append(trace)
    hand_traces = [trace1, *traces]
    return hand_traces
