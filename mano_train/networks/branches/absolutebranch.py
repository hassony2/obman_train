import argparse

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f


from handobjectdatasets.queries import TransQueries, BaseQueries

from mano_train.networks.branches import atlasutils


class AbsoluteBranch(nn.Module):
    def __init__(self, base_neurons=[515, 256]):
        """
        Args:
            mano_root (path): dir containing mano pickle files
        """
        super().__init__()

        layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(
                zip(base_neurons[:-1], base_neurons[1:])):
            layers.append(nn.Linear(inp_neurons, out_neurons))
            layers.append(nn.ReLU())
        self.final_layer = nn.Linear(out_neurons, 3)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)
        return out
