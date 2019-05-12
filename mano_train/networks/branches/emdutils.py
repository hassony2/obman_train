import numpy as np
import torch
from torch import nn
from matplotlib.collections import LineCollection


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def sink_bak(cost_matrix, alpha_i, beta_j, eps=.1, nits=50, tol=1e-3):
    """
    Directly following http://nbviewer.jupyter.org/github/gpeyre/numerical-tours/blob/master/python/optimaltransp_5_entropic.ipynb
    by Gabriel Peyré
    """

    if type(nits) in [list, tuple]:
        nits = nits[
            0]  # The user may give different limits for Sink and SymSink
    # Sinkh iterations
    xi = torch.exp(-cost_matrix / eps)
    b = torch.ones(xi.shape[0], xi.shape[2], 1).cuda()
    for i in range(50):
        a_basic = alpha_i / torch.bmm(xi, b)
        b = beta_j / torch.bmm(xi.transpose(2, 1), a_basic)

    # Compute transportation matrix
    P = torch.bmm(
        torch.bmm(matrix_diag(a_basic[:, :, 0]), xi), matrix_diag(b[:, :, 0]))
    cost = (P * cost_matrix).sum(1).sum(1)
    return cost, P


@torch.no_grad()
def sink(cost_matrix, alpha_i, beta_j, eps=.1, nits=50, tol=1e-3,
         verbose=False):
    """
    Based on https://github.com/jeanfeydy/global-divergences/blob/master/common/sinkhorn_balanced_simple.py
    by Jean Feydy

    Estimates earth mover distance associated with cost_matrix and histograms alpha_i and beta_j
    """

    if type(nits) in [list, tuple]:
        nits = nits[
            0]  # The user may give different limits for Sink and SymSink
    alpha_i_log, beta_j_log = alpha_i.log(), beta_j.log(
    )  # Precompute the logs of the measures' weights
    B_i, A_j = torch.zeros_like(alpha_i), torch.zeros_like(
        beta_j)  # Sampled influence fields

    # if we assume convergence, we can skip all the "save computational history" stuff
    Cxy_e = -cost_matrix / eps
    Cxy_et = Cxy_e.transpose(2, 1)

    # Sinkhorn loop with A = a/eps , B = b/eps ....................................................
    for iter_idx in range(nits):
        # Save results for future computations
        A_j_prev = A_j
        B_i_prev = B_i

        # Compute updates
        A_j_update = B_i + alpha_i_log + Cxy_e
        A_j = -torch.logsumexp(A_j_update, dim=1, keepdim=True)  # a(y)/epsilon = Smin_epsilon,x~alpha [ C(x,y) - b(x) ]  / epsilon
        A_j = A_j.transpose(2, 1)
        B_i_update = A_j + beta_j_log + Cxy_et
        B_i = -torch.logsumexp(B_i_update, dim=1, keepdim=True)  # b(x)/epsilon = Smin_epsilon,y~beta [ C(x,y) - a(y) ]  / epsilon
        B_i = B_i.transpose(2, 1)

        # Estimate update changes
        b_err = (eps * (B_i_prev - B_i).abs().mean(1)[:, 0]).max()
        a_err = (eps * (A_j_prev - A_j).abs().mean(1)[:, 0]).max()
        if b_err < tol and a_err < tol:
            if verbose:
                print(
                        'Converged after {} steps with update diff {:.2e} and {:.2e} < tol {}'.
                    format(iter_idx + 1, a_err, b_err, tol))
            break

    # Compute transport matrix
    transp = ((torch.exp(A_j.transpose(1, 2) + B_i - cost_matrix / eps)) /
              (cost_matrix.shape[1] * cost_matrix.shape[2]))
    # print(transport.sum(1)) # Should converge to alpha_i
    # print(transport.sum(2)) # Should converge to beta_j

    a_y, b_x = eps * A_j, eps * B_i
    return a_y, b_x, transp


def regularized_ot(cost_matrix,
                   alpha,
                   beta,
                   eps=.1,
                   nits=50,
                   tol=1e-3,
                   correct_matrix=None):  # OT_epsilon
    """
    Args:
        cost_matrix (batch_size, n, m): matrix of cost distances between n predicted points and m ground truth points
        a_y (batch_size, m): weights of secondset of point (each col should sum to 1)
        b_x (batch_size, n): weights of first set of point (each col should sum to 1)
        esp: Entropy regularization coefficient W(p, q) = min(P cost_matrix) - eps Entropy(P)
        correct_matrix (batch_size, n, n): matrix of cost distances between n predicted points and themselves
            if provided, W(p, q) - 1/2W(p, p) is minimized instead of W(p, q)

    Returns:
        cost: Estimation of earth mover distance between the two sets of points
    """
    a_y, b_x, transp = sink(
        cost_matrix, alpha, beta, eps=eps, nits=nits, tol=tol)
    # cost = torch.bmm(alpha.transpose(2, 1), b_x) + torch.bmm(
    #     beta.transpose(2, 1), a_y)
    cost = (transp * cost_matrix).sum(1).sum(1)
    if correct_matrix is not None:
        # Compensate entropy
        _, _, transp_correct = sink(
            correct_matrix, alpha, alpha, eps=eps, nits=nits, tol=tol)
        cost_correct = (transp_correct * correct_matrix).sum(1).sum(1)
        cost = cost - 1 / 2 * cost_correct
    return cost


class EMDLoss(nn.Module):
    def __init__(self, correct_entropy=True, iter_nb=50, tol=1e-3, eps=.1):
        super(EMDLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.correct_entropy = correct_entropy
        self.iter_nb = iter_nb
        self.tol = tol
        self.eps = eps

    def forward(self, preds, gts):
        C = self.batch_pairwise_dist(gts, preds)
        # Compute Chamfer loss
        mins, _ = torch.min(C.detach(), 1)
        loss_1 = torch.mean(mins, 1)
        mins, _ = torch.min(C.detach(), 2)
        loss_2 = torch.mean(mins, 1)
        chamfer_loss = torch.mean(loss_1 + loss_2)

        pred_weights = preds.new_full((preds.shape[0], preds.shape[1], 1), 1/ preds.shape[0])
        gt_weights = preds.new_full((gts.shape[0], gts.shape[1], 1), 1/ gts.shape[0])

        if self.correct_entropy:
            C_correct = self.batch_pairwise_dist(preds, preds)
        else:
            C_correct = None
        cost = regularized_ot(
            C,
            pred_weights,
            gt_weights,
            eps=self.eps,
            nits=self.iter_nb,
            tol=self.tol,
            correct_matrix=C_correct)
        return cost, chamfer_loss

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
