""" Alignment Util Fcns

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np


def cnd_avg(data, labels):
    """Averages data trials along first axis by condition type present in
    labels.

    Args:
        data (ndarray): Data matrix with shape (n_trials, ...). The first
            dimension must be the trial dimension. Number and shape of other
            dimensions is arbitrary.
        labels (ndarray): Label array with shape (n_trials,).

    Returns:
        ndarray: Data matrix averaged within conditions with shape
        (n_conditions, ...).
    """
    data_shape = data.shape
    class_shape = (len(np.unique(labels)),) + data_shape[1:]
    data_by_class = np.zeros(class_shape)
    for i, seq in enumerate(np.unique(labels)):
        data_by_class[i] = np.mean(data[labels == seq], axis=0)
    return data_by_class


def label2str(labels):
    """Converts a 2D array of label sequences into a 1D array of label strings.

    For example, if a trial has multiple labels, such as [1, 2, 3], this
    function will convert it to a string '123'. Used to convert sequences of
    phonemes into strings of phoneme sequence labels.

    Args:
        labels (ndarray): Labels with shape (n_trials, n_labels_per_trial).

    Returns:
        ndarray: Labels with shape (n_trials,) where each label is a string.
    """
    labels_str = []
    for i in range(labels.shape[0]):
        labels_str.append(''.join(str(x) for x in labels[i, :]))
    return np.array(labels_str)
