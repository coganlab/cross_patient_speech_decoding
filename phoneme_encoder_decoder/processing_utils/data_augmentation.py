"""Synthetic data augmentation via functions.

Author: Zac Spalding
"""

import numpy as np
from collections import defaultdict


def generate_mixup(x, prior, y, labels, alpha=1):
    """Generates synthetic data on batch via MixUp algorithm.

    Creates synthetic data from linear combinations of observations/trials
    that share the same label (instead of randomly sampling from entire
    dataset).

    Args:
        Args:
        x (ndarray): Feature data.
        prior (ndarray): One-hot encoded prior label data.
        y (ndarray): One-hot encoded label data.
        labels (ndarray): Label data in original format (i.e. not one-hot
            encoded).
        alpha (int|float, optional): MixUp hyperparameter for beta
            distribution selection. Defaults to 1.

    Returns:
        (ndarray, ndarray, ndarray): Combined original and mixed feature
            data, one-hot prior data, and one-hot label data.
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

    if not len(x_mixed):  # no duplicates, so no mixup data
        return x, prior, y

    # add original data to synthetic data
    x_mixed = np.concatenate((x, np.array(x_mixed)))
    prior_mixed = np.concatenate((prior, np.array(prior_mixed)))
    y_mixed = np.concatenate((y, np.array(y_mixed)))
    return x_mixed, prior_mixed, y_mixed


def generate_time_jitter(x, prior, y, jitter_vals, win_len, fs, time_axis=1):
    """Generates synthetic data on batch via time jittering.

    Creates synthetic data by shifting the center location of a
    time window of length win_len. The center of the window is shifted by
    jitter_vals such that the windows of the synthetically generated data will
    not be centered around the true onset, except for a jitter value of 0.

    Args:
        x (ndarray): Feature data.
        prior (ndarray): One-hot encoded prior label data.
        y (ndarray): One-hot encoded label data.
        jitter_vals (array-like): value(s) to shift original data window by.
        win_len (int): Length of time window in seconds.
        fs (int): Sampling rate of data.
        time_axis (int, optional): Axis of data corresponding to timepoints
            (Instead of observations, channels, etc.). Defaults to 1.

    Returns:
        (ndarray, ndarray, ndarray): Combined original and jittered feature
            data, one-hot prior data, and one-hot label data.
    """
    t_dur = x.shape[time_axis] / fs  # duration of full data
    t_range = np.array([-t_dur/2, t_dur/2])  # full data centered around 0
    reg_win = np.array([-win_len/2, win_len/2])  # non-jittered window, [-a, a]
    x_jittered, prior_jittered, y_jittered = [], [], []
    for jitter in jitter_vals:
        jitter_win = reg_win + jitter  # jittered window, [-a + j, a + j]
        jitter_x = extract_tw(x, time_axis, t_range, jitter_win, fs)
        x_jittered.append(jitter_x)
        prior_jittered.append(prior)  # same prior and labels for jitters
        y_jittered.append(y)

    x_jittered = np.concatenate((x, np.array(x_jittered)))
    prior_jittered = np.concatenate((prior, np.array(prior_jittered)))
    y_jittered = np.concatenate((y, np.array(y_jittered)))
    return x_jittered, prior_jittered, y_jittered


def mixup_data(x1, x2, prior1, prior2, y1, y2, alpha=1):
    """Applies MixUp to a single observation/trial.

    Adaptation of MixUp algorithm for data augmentation (Zhang et al., 2017,
    https://arxiv.org/abs/1710.09412).

    Args:
        x1 (ndarray): Feature data for first duplicate.
        x2 (ndarray): Feature data for first duplicate.
        prior1 (ndarray): Prior label data (one-hot) for first duplicate.
        prior2 (ndarray): Prior label data (one-hot) for first duplicate.
        y1 (ndarray): Label data (one-hot) for first duplicate.
        y2 (ndarray): Label data (one-hot) for first duplicate.
        alpha (int|float, optional): MixUp hyperparameter for beta
            distribution selection. Defaults to 1.

    Returns:
        (ndarray, ndarray, ndarray): Mixed feature data, one-hot prior data,
        and one-hot label data.
    """
    # get beta distribution parameters
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # apply MixUp
    x_mixed = lam * x1 + (1 - lam) * x2
    prior_mixed = lam * prior1 + (1 - lam) * prior2
    y_mixed = lam * y1 + (1 - lam) * y2

    return x_mixed, prior_mixed, y_mixed


def extract_tw(data, time_axis, t_range, win_range, fs):
    centered_inds = tw_inds(t_range, win_range, fs)
    return data.take(centered_inds, axis=time_axis)


def tw_inds(t_range, win_range, fs):
    t = np.linspace(t_range[0], t_range[1],
                    int((t_range[1] - t_range[0]) * fs))
    return np.array([np.where((t >= win_range[0]) & (t <= win_range[1]))[0]])


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
