""" Various methods for aligning microECoG datasets across patients.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
from utils import cnd_avg, label2str


def joint_PCA_decomp():
    # condition average firing rates for all datasets

    # combine all datasets into one matrix (sum channels x n_timepoints x
    # n_conditions)

    # perform PCA on channel dim of combined matrix

    # calculate condition averaged firing rates by patient

    # calculate per pt channel -> factor transformation matrices

    pass


def get_factor_matrix(n_components):
    pass


def CCA_align_by_class(X_a, X_b, y_a, y_b, return_space='b_to_a'):
    """CCA Alignment between 2 datasets with correspondence by averaging within
    class conditions.

    The number of features must be the same for datasets A and B. For example,
    if the datasets have different feature sizes, you can use PCA to reduce
    both datasets to the same number of PCs first.

    Args:
        X_a (ndarray): Data matrix for dataset A of shape (n_trials_a,
            n_timepoints, n_features)
        X_b (ndarray): Data matrix for dataset B of shape (n_trials_b,
            n_timepoints, n_features)
        y_a (ndarray): Label matrix for dataset A of shape (n_trials_a, ...)
            The first dimension must be the trial dimension. This can be a
            1D array, or a 2D array if each trial has multiple labels (e.g. a
            sequence of phonemes). Label sequences are converted to a single
            string so that only the same label sequences have correspondence
            between the datasets.
        y_b (ndarray): Label matrix for dataset B of shape (n_trials_a, ...).
            See y_a for more details.
        return_space (str, optional): How to perform alignment. Dataset B can
            be aligned to A, and vice versa ('b_to_a' and 'a_to_b',
            respectively), or both datasets can be aligned to a shared space
            ('shared'). Defaults to 'b_to_a'.

    Returns:
        (tuple): tuple containing:
            X_a (ndarray): Aligned data matrix for dataset A
            X_b (ndarray): Aligned data matrix for dataset B
    """
    parse_return_type(return_space)

    # convert labels to strings for seqeunce comparison
    y_a = label2str(y_a)
    y_b = label2str(y_b)

    # group trials by label type
    L_a = cnd_avg(X_a, y_a)
    L_b = cnd_avg(X_b, y_b)

    # find common labels between datasets for alignment
    _, y_shared_a, y_shared_b = np.intersect1d(np.unique(y_a), np.unique(y_b),
                                               assume_unique=True,
                                               return_indices=True)
    L_a = L_a[y_shared_a]
    L_b = L_b[y_shared_b]

    # fold timepoints into trials
    L_a = np.reshape(L_a, (-1, L_a.shape[-1]))
    L_b = np.reshape(L_b, (-1, L_b.shape[-1]))

    # calculate manifold directions with CCA
    M_a, M_b = CCA_align(L_a.T, L_b.T)

    # align in put data with manifold transformation matrices
    if return_space == 'b_to_a':
        return X_a, X_b @ M_b @ np.linalg.pinv(M_a)
    elif return_space == 'a_to_b':
        return X_a @ M_a @ np.linalg.pinv(M_b), X_b
    return X_a @ M_a, X_b @ M_b


def CCA_align_by_trial_subselect(X_a, X_b, y_a, y_b, return_space='b_to_a'):
    """CCA Alignment between 2 datasets with correspondence via subselection of
    trials within shared clases.

    The number of features must be the same for datasets A and B. For example,
    if the datasets have different feature sizes, you can use PCA to reduce
    both datasets to the same number of PCs first.

    Args:
        X_a (ndarray): Data matrix for dataset A of shape (n_trials_a,
            n_timepoints, n_features)
        X_b (ndarray): Data matrix for dataset B of shape (n_trials_b,
            n_timepoints, n_features)
        y_a (ndarray): Label matrix for dataset A of shape (n_trials_a, ...)
            The first dimension must be the trial dimension. This can be a
            1D array, or a 2D array if each trial has multiple labels (e.g. a
            sequence of phonemes). Label sequences are converted to a single
            string so that only the same label sequences have correspondence
            between the datasets.
        y_b (ndarray): Label matrix for dataset B of shape (n_trials_a, ...).
            See y_a for more details.
        return_space (str, optional): How to perform alignment. Dataset B can
            be aligned to A, and vice versa ('b_to_a' and 'a_to_b',
            respectively), or both datasets can be aligned to a shared space
            ('shared'). Defaults to 'b_to_a'.

    Returns:
        (tuple): tuple containing:
            X_a (ndarray): Aligned data matrix for dataset A
            X_b (ndarray): Aligned data matrix for dataset B
    """
    parse_return_type(return_space)

    y_a = label2str(y_a)
    y_b = label2str(y_b)

    L_a, L_b = [], []
    # subselect same amount of trials for each class
    for c in np.intersect1d(y_a, y_b):
        # shuffle trial order within class
        curr_a = np.random.permutation(np.where(y_a == c)[0])
        curr_b = np.random.permutation(np.where(y_b == c)[0])
        min_shared = min(curr_a.shape[0], curr_b.shape[0])

        L_a.append(X_a[curr_a[:min_shared]])
        L_b.append(X_b[curr_b[:min_shared]])

    # combine subselected trials
    L_a = np.vstack(L_a)
    L_b = np.vstack(L_b)

    # fold timepoints into trials
    L_a = np.reshape(L_a, (-1, L_a.shape[-1]))
    L_b = np.reshape(L_b, (-1, L_b.shape[-1]))

    # calculate alignment
    M_a, M_b = CCA_align(L_a.T, L_b.T)

    # align in put data with manifold transformation matrices
    if return_space == 'b_to_a':
        return X_a, X_b @ M_b @ np.linalg.pinv(M_a)
    elif return_space == 'a_to_b':
        return X_a @ M_a @ np.linalg.pinv(M_b), X_b
    return X_a @ M_a, X_b @ M_b


def parse_return_type(return_space):
    """Checks the CCA alignment return type is valid.

    Args:
        return_space (str): String detailing how to perform alignment.

    Raises:
        ValueError: Error if return_space is not 'b_to_a', 'a_to_b', or
            'shared'
    """
    if return_space not in ['b_to_a', 'a_to_b', 'shared']:
        raise ValueError('return_space must be "b_to_a" or "a_to_b" or'
                         '"shared".')


def CCA_align(L_a, L_b):
    """Canonical Correlation Analysis (CCA) alignment between 2 datasets.

    From: https://www.nature.com/articles/s41593-019-0555-4#Sec11.
    Returns manifold directions to transform L_a and L_b into a common space
    (e.g. L_a_new.T = L_a.T @ M_a, L_b_new.T = L_b.T @ M_b).
    To transform into a specific patient space, for example putting everything
    in patient A's space, use L_(b->a).T = L_b.T @ M_b @ (M_a)^-1, where L_a
    and L_(b->a) will be aligned in the same space.

    Args:
        L_a (ndarray): Latent dynamics array for dataset A of shape (m, T),
            where m is the number of latent dimensions and T is the number of
            timepoints.
        L_b (ndarray): Latent dynamics array for dataset B of shape (m, T)

    Returns:
        (tuple): tuple containing:
            M_a (ndarray): Manifold directions for dataset A of shape (m, m)
            M_b (ndarray): Manifold directions for dataset B of shape (m, m)
    """
    # QR decomposition
    Q_a, R_a = np.linalg.qr(L_a.T)
    Q_b, R_b = np.linalg.qr(L_b.T)

    # SVD on q inner product
    U, S, Vt = np.linalg.svd(Q_a.T @ Q_b)

    # calculate manifold directions
    M_a = np.linalg.pinv(R_a) @ U
    M_b = np.linalg.pinv(R_b) @ Vt.T

    return M_a, M_b
