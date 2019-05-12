'''
    Author: Vasileios Choutas
    E-mail: vassilis.choutas@tuebingen.mpg.de
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import pickle

import numpy as np

import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture


class MaxMixturePrior(nn.Module):
    def __init__(self,
                 prior_folder='assets/priors',
                 use_body=True,
                 use_left_hand=False,
                 use_right_hand=False,
                 num_gaussians=8,
                 dtype=torch.float32,
                 epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        '''
            Args:
                - data_dir: The directory where the data is stored
            Keyword Arguments:
                - prior_folder: The sub-folder where the pickle files with the
                  priors are stored.
                - use_body: Use the prior for the pose of the body
                - use_left_hand: Use the prior for the pose of the left hand
                - use_right_hand: Use the prior for the pose of the right hand
                - num_gaussians: The number of mixture components
                - dtype: The data type of the tensors
                - epsilon: Constant used for numerical stability
                - use_merged: When true use an optimized function with einsum
                  for faster calculation
        '''
        super(MaxMixturePrior, self).__init__()

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        if use_body:
            gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)
        if use_left_hand:
            gmm_fn = 'gmm_left_{:02d}.pkl'.format(
                num_gaussians)
        if use_right_hand:
            gmm_fn = 'gmm_right_{:02d}.pkl'.format(
                num_gaussians)
        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np.float32)
            covs = gmm['covars'].astype(np.float32)
            weights = gmm['weights'].astype(np.float32)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np.float32)
            covs = gmm.covars_.astype(np.float32)
            weights = gmm.weights_.astype(np.float32)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(0)

        self.register_buffer('means', torch.Tensor(means))
        self.register_buffer('covs', torch.Tensor(covs))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np.float32)
        self.register_buffer('precisions', torch.Tensor(precisions))
        self.register_buffer('weights', torch.Tensor(weights).unsqueeze(dim=0))

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [
            np.log(np.linalg.det(cov.astype(np.float64)) + epsilon)
            for cov in covs
        ]
        self.register_buffer('cov_dets', torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        curr_loglikelihood = torch.einsum('mji,bmj->bmi',
                                          [self.precisions, diff_from_mean])
        curr_loglikelihood = torch.einsum('bmi,bmi->bm',
                                          [curr_loglikelihood, diff_from_mean])

        curr_loglikelihood += 0.5 * (
            self.cov_dets.unsqueeze(dim=0) + self.random_var_dim * self.pi_term
        )

        min_likelihood, min_idx = torch.min(curr_loglikelihood, dim=1)

        return min_likelihood - torch.log(self.weights[:, min_idx])

    def log_likelihood(self, pose, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum(
                'bi,bi->b', [curr_loglikelihood, diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (
                cov_term + self.random_var_dim * self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_likelihoods, min_idx = torch.min(log_likelihoods, dim=1)
        weight_component = self.weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + min_likelihoods

    def forward(self, pose):
        if self.use_merged:
            return self.log_likelihood(pose)
        else:
            return self.merged_log_likelihood(pose)
