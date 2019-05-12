import ot
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pylab as pl
from matplotlib import pyplot as plt


def compute_emd(points1, points2):
    points1 = np.array(points1)
    points2 = np.array(points2)
    dists = cdist(points1, points2)
    assignments = (ot.emd(
        np.ones(points1.shape[0]), np.ones(points2.shape[0]), dists))
    emd = (assignments * dists).sum() / assignments.shape[0]
    return emd, assignments

def batch_emd(batch_points1, batch_points2):
    emds = []
    for points1, points2 in zip(batch_points1, batch_points2):
        emd, ass = compute_emd(points1.cpu().numpy(), points2.cpu().numpy())
        emds.append(emd)
    return emds


def plot2D_samples_mat(xs, xt, G, thr=1e-8, **kwargs):
    """ Plot matrix M  in 2D with  lines using alpha values
    Plot lines between source and target 2D samples with a color
    proportional to the value of the matrix G between samples.
    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    b : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    thr : float, optional
        threshold above which the line is drawn
    **kwargs : dict
        paameters given to the plot functions (default color is black if
        nothing given)
    """
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    mx = G.max()
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                pl.plot(
                    [xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                    alpha=G[i, j] / mx,
                    **kwargs)


if __name__ == '__main__':
    points1 = [[1, 0, 0], [2, 1, 0], [3, 3, 0]]
    points2 = [[1, 5, 0], [2, 6, 0], [3, 8, 0]]
    emd, assignments = compute_emd(points1, points2)
    plot2D_samples_mat(np.array(points1), np.array(points2), assignments)
    plt.show()
    print(emd, assignments)
