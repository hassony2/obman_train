import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def tri_area(v):
    return 0.5 * np.linalg.norm(
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)


def points_from_mesh(faces, vertices, vertex_nb=600, show_cloud=False):
    areas = tri_area(vertices[faces])

    proba = areas / areas.sum()
    rand_idxs = np.random.choice(
        range(areas.shape[0]), size=vertex_nb, p=proba)

    # Randomly pick points on triangles
    u = np.random.rand(vertex_nb, 1)
    v = np.random.rand(vertex_nb, 1)

    # Force bernouilli couple to be picked on a half square
    out = u + v > 1
    u[out] = 1 - u[out]
    v[out] = 1 - v[out]

    rand_tris = vertices[faces[rand_idxs]]
    points = rand_tris[:, 0] + u * (rand_tris[:, 1] - rand_tris[:, 0]) + v * (
        rand_tris[:, 2] - rand_tris[:, 0])

    if show_cloud:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=2,
            c='b')
        ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            s=2,
            c='r')
        ax._axis3don = False
        plt.show()
    return points
