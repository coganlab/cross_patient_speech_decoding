""" Synthetic data augmentation via adaption of MixUp algorithm (Zhang et al.,
2017, https://arxiv.org/abs/1710.09412)

Author: Zac Spalding
"""

import numpy as np
from collections import defaultdict


def generate_mixup(x, prior, y, labels, alpha=0.2):
    """Generates synthetic data on batch via MixUp algorithm.

    Creates synthetic data from linear combinations of observations/trials
    that share the same label (instead of randomly sampling from entire
    dataset).

    Args:
        x (ndarray): Feature data.
        y (ndarray): One-hot encoded label data.
        labels (ndarray): Label data in original format (i.e. not one-hot
            encoded).

    Returns:
        (ndarray, ndarray): Mixed feature data and mixed label data.
    """

    # get indices of duplicate observations/trials
    dup_gen = list_duplicates(labels)

    # generate synthetic data for each non-duplicate observation/trial
    x_mixed, prior_mixed, y_mixed = [], [], []
    for (_, dup_inds) in dup_gen:  # use sequences with multiple trials
        trial_gen = trial_order_generator(np.array(dup_inds))
        for (ind1, ind2) in trial_gen:  # iterate over trial combinations
            mix_x, mix_prior, mix_y = mixup_data(x[ind1], x[ind2], prior[ind1],
                                                 prior[ind2], y[ind1], y[ind2],
                                                 alpha=alpha)
            x_mixed.append(mix_x)
            prior_mixed.append(mix_prior)
            y_mixed.append(mix_y)

    # add original data to synthetic data
    x_mixed = np.concatenate((x, np.array(x_mixed)))
    prior_mixed = np.concatenate((prior, np.array(prior_mixed)))
    y_mixed = np.concatenate((y, np.array(y_mixed)))

    return x_mixed, prior_mixed, y_mixed


def mixup_data(x1, x2, prior1, prior2, y1, y2, alpha=0.2):
    """MixUp algorithm for data augmentation. Applies MixUp to a single
    observation/trial.

    Args:
        x (ndarray): Feature data for a single observation/trial.
        y (ndarray): Label data for a single observation/trial.
        alpha (float): MixUp hyperparameter. Defaults to 0.2.

    Returns:
        (ndarray, ndarray): Mixed feature data and mixed label data.
    """
    # get beta distribution parameters
    lam = np.random.beta(alpha, alpha)

    # apply MixUp
    x_mixed = lam * x1 + (1 - lam) * x2
    prior_mixed = lam * prior1 + (1 - lam) * prior2
    y_mixed = lam * y1 + (1 - lam) * y2

    return x_mixed, prior_mixed, y_mixed


def trial_order_generator(inds):
    combs = numpy_combinations(inds)
    for comb in combs:
        yield comb


def numpy_combinations(arr):
    """Generates all pairwise combinations of elements in a numpy array.

    Faster than using itertools.combinations().
    From https://carlostgameiro.medium.com/fast-pairwise-combinations-in-numpy
    -c29b977c33e2.
    Author: Carlos Gameiro

    Args:
        arr (ndarray): Array to compute combinations of.

    Returns:
        ndarray: numpy array with all pairwise combinations. First dim is
            separate combinations, second dim is elements of each combination.
    """
    idx = np.stack(np.triu_indices(len(arr), k=1), axis=-1)
    return arr[idx]


def list_duplicates(seq):
    """Gets indices of duplicate items in a list.

    From https://stackoverflow.com/questions/5419204/index-of-duplicates-items
    -in-a-python-list
    Author: PaulMcG

    Args:
        seq (array-like): Sequence to be searched for duplicates.

    Returns:
        (Array-like type, list): Identity of duplicate items and their indices.
    """
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[np.array2string(item)].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)
