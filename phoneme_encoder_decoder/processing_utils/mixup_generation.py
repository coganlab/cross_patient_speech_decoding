""" Synthetic data augmentation via adaption of MixUp algorithm (Zhang et al.,
2017, https://arxiv.org/abs/1710.09412)

Author: Zac Spalding
"""

import numpy as np
from collections import defaultdict


def mixup_data(x, y, alpha=0.2):
    """MixUp algorithm for data augmentation. Applies MixUp to a single
    observation/trial.

    Args:
        x (ndarray): Feature data for a single observation/trial.
        y (ndarray): Label data for a single observation/trial.
        alpha (float): MixUp hyperparameter. Defaults to 0.2.

    Returns:
        (ndarray, ndarray): Mixed feature data and mixed label data.
    """
    # get number of features
    n_features = x.shape[0]

    # get beta distribution parameters
    lam = np.random.beta(alpha, alpha)

    # apply MixUp
    x_mixed = lam * x + (1 - lam) * x[::-1]
    y_mixed = lam * y + (1 - lam) * y[::-1]

    return x_mixed, y_mixed
