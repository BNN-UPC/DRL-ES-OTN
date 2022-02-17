# Copyright (c) 2022, Carlos Güemes [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: carlos.guemes@upc.edu

import numpy as np


class AdamOptimizer:
    """
    Class that implements the Adam Optimizer
    """

    # The alternative was rewritting the entire GNN from Keras to TF, and this was much simpler
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, *args):
        # Initial first moement vector
        self.m = None
        # Initial second moment vector
        self.v = None
        # Initial timestep
        self.t = 0

        # Constants
        self.alpha = -1 * alpha  # Alpha integrates the negative sign for updating the parameters
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # Parameters that change per computation
        self.beta_1_t = self.beta_2_t = 1

        # Initial previous moment vectors
        self.m = self.v = 0

    def reset(self):
        self.t = 0
        self.beta_1_t = self.beta_2_t = 1

    def optimize(self, gradients):
        # First moment vector
        self.m = (1 - self.beta_1) * gradients + self.beta_1 * self.m
        # Second moment vector
        self.v = (1 - self.beta_2) * gradients * gradients + self.beta_2 * self.v

        # Apply bias correction
        self.t += 1
        self.beta_1_t *= self.beta_1
        self.beta_2_t *= self.beta_2
        alpha_t = self.alpha * np.sqrt(1 - self.beta_2_t) / (1 - self.beta_1_t)

        # Return correction
        # Alpha integrates the negative sign for updating the parameters
        return alpha_t * self.m / (np.sqrt(self.v) + self.epsilon)


class SGDOptimizer:
    def __init__(self, alpha=0.001, *args):
        self.alpha = -1 * alpha  # Alpha integrates the negative sign for updating the parameters

    def reset(self):
        pass

    def optimize(self, gradients):
        return self.alpha * gradients


class SGDMomemtumOptimizer:
    def __init__(self, alpha=0.001, *args):
        self.alpha = -1 * alpha  # Alpha integrates the negative sign for updating the parameters

    def reset(self):
        pass

    def optimize(self, gradients):
        return self.alpha * gradients
