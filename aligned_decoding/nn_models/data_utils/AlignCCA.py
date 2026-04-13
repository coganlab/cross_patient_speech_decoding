""" Class to perform CCA alignment of multiple datasets.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
from alignment_utils import label2str, cnd_avg


class AlignCCA:
    """CCA-based alignment between two neural datasets.

    Fits canonical correlation analysis to latent dynamics extracted from
    two datasets, then transforms new data into a shared or target-specific
    space.

    Args:
        type: Latent dynamics extraction method. 'class' averages trials
            within each class; 'trial' subselects matched trials.
            Defaults to 'class'.
        return_space: Transform target space. 'b_to_a' maps dataset B into
            A's space, 'a_to_b' maps A into B's space, or 'shared' maps
            both into the shared CCA space. Defaults to 'b_to_a'.

    Attributes:
        M_a: Manifold directions for dataset A (set after fit).
        M_b: Manifold directions for dataset B (set after fit).
        canon_corrs: Canonical correlation values (set after fit).
    """

    def __init__(self, type='class', return_space='b_to_a'):
        self.type = type
        self.return_space = return_space

    def fit(self, X_a, X_b, y_a, y_b):
        """Fits CCA alignment between two datasets.

        Args:
            X_a: Dataset A features of shape (n_trials, n_timepoints, n_dims).
            X_b: Dataset B features of shape (n_trials, n_timepoints, n_dims).
            y_a: Labels for dataset A.
            y_b: Labels for dataset B.
        """
        L_a, L_b = reshape_latent_dynamics(X_a, X_b, y_a, y_b, type=self.type)
        M_a, M_b, S = CCA_align(L_a.T, L_b.T)
        self.M_a = M_a
        self.M_b = M_b
        self.canon_corrs = S

    def transform(self, X):
        """Transforms data into the target space using fitted CCA directions.

        Args:
            X: Data array to transform. For 'b_to_a' or 'a_to_b', a single
                array of shape (..., n_dims). For 'shared', a tuple (X_a, X_b).

        Returns:
            ndarray or tuple: Transformed data in the target space.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self._check_fit():
            raise RuntimeError('Must call fit() before transforming data.')
        if self.return_space in ['b_to_a', 'a_to_b']:
            return self._transform_single(X)
        return self._transform_shared(X)

    def _transform_single(self, X):
        """Transforms data from one dataset's space into the other's.

        Args:
            X: Data array of shape (..., n_dims).

        Returns:
            ndarray: Transformed data in the target patient's space.
        """
        if self.return_space == 'b_to_a':
            return X @ self.M_b @ np.linalg.pinv(self.M_a)
        return X @ self.M_a @ np.linalg.pinv(self.M_b)

    def _transform_shared(self, X):
        """Transforms both datasets into the shared CCA space.

        Args:
            X: Tuple of (X_a, X_b) data arrays.

        Returns:
            tuple: (X_a_shared, X_b_shared) projected into shared space.
        """
        return X[0] @ self.M_a, X[1] @ self.M_b

    def _check_fit(self):
        """Checks if CCA Aligner has been fit to data.

        Returns:
            boolean: True if fit() has been called, False otherwise.
        """
        try:
            self.M_a
            self.M_b
        except AttributeError:
            return False
        return True
    

def reshape_latent_dynamics(X_a, X_b, y_a, y_b, type='class'):
    """Extracts and reshapes latent dynamics for CCA alignment.

    Selects the extraction method (class-average or trial-subselect), then
    folds timepoints into the trial dimension so that the latent
    dimensionality is isolated as the second axis.

    Args:
        X_a: Dataset A features of shape (n_trials, n_timepoints, n_dims).
        X_b: Dataset B features of shape (n_trials, n_timepoints, n_dims).
        y_a: Labels for dataset A.
        y_b: Labels for dataset B.
        type: Extraction method, 'class' or 'trial'. Defaults to 'class'.

    Returns:
        tuple: (L_a, L_b) reshaped latent dynamics arrays of shape
            (n_trials * n_timepoints, n_dims).

    Raises:
        ValueError: If type is not 'class' or 'trial'.
    """
    if type == 'class':
        L_a, L_b = extract_latent_dynamics_by_class(X_a, X_b, y_a, y_b)
    elif type == 'trial':
        L_a, L_b = extract_latent_dynamics_by_trial_subselect(X_a, X_b, y_a,
                                                              y_b)
    else:
        raise ValueError('type must be "class" or "trial".')

    # fold timepoints into trials (isolate latent dimensionality as 2nd dim)
    L_a, L_b = L_a.reshape(-1, L_a.shape[-1]), L_b.reshape(-1, L_b.shape[-1])
    return L_a, L_b


def extract_latent_dynamics_by_class(X_a, X_b, y_a, y_b):
    """Extracts latent dynamics by averaging trials within each shared class.

    Args:
        X_a: Dataset A features of shape (n_trials, n_timepoints, n_dims).
        X_b: Dataset B features of shape (n_trials, n_timepoints, n_dims).
        y_a: Labels for dataset A.
        y_b: Labels for dataset B.

    Returns:
        tuple: (L_a, L_b) class-averaged arrays containing only shared classes.
    """
    # process labels for easy comparison of label sequences
    y_a, y_b = label2str(y_a), label2str(y_b)
    # average trials within class
    L_a, L_b = cnd_avg(X_a, y_a), cnd_avg(X_b, y_b)

    # only align via shared classes between datasets
    _, y_shared_a, y_shared_b = np.intersect1d(np.unique(y_a), np.unique(y_b),
                                               assume_unique=True,
                                               return_indices=True)
    L_a, L_b = L_a[y_shared_a], L_b[y_shared_b]

    return L_a, L_b


def extract_latent_dynamics_by_trial_subselect(X_a, X_b, y_a, y_b):
    """Extracts latent dynamics by subselecting matched trials per class.

    Args:
        X_a: Dataset A features of shape (n_trials, n_timepoints, n_dims).
        X_b: Dataset B features of shape (n_trials, n_timepoints, n_dims).
        y_a: Labels for dataset A.
        y_b: Labels for dataset B.

    Returns:
        tuple: (L_a, L_b) trial-subselected arrays with equal counts per class.
    """
    y_a, y_b = label2str(y_a), label2str(y_b)
    L_a, L_b = shared_trial_subselect(X_a, X_b, y_a, y_b)
    return L_a, L_b


def shared_trial_subselect(X_a, X_b, y_a, y_b):
    """Subselects equal numbers of shuffled trials per shared class.

    For each shared class, randomly permutes trials and takes the minimum
    count from both datasets to ensure balanced pairing.

    Args:
        X_a: Dataset A features of shape (n_trials, n_timepoints, n_dims).
        X_b: Dataset B features of shape (n_trials, n_timepoints, n_dims).
        y_a: String labels for dataset A.
        y_b: String labels for dataset B.

    Returns:
        tuple: (L_a, L_b) arrays with matched trial counts across classes.
    """
    L_a, L_b = [], []
    # subselect same amount of trials for each class
    for c in np.intersect1d(y_a, y_b):
        # shuffle trial order within class
        curr_a = np.random.permutation(np.where(y_a == c)[0])
        curr_b = np.random.permutation(np.where(y_b == c)[0])
        min_shared = min(curr_a.shape[0], curr_b.shape[0])

        L_a.append(X_a[curr_a[:min_shared]])
        L_b.append(X_b[curr_b[:min_shared]])
    L_a, L_b = np.vstack(L_a), np.vstack(L_b)
    return L_a, L_b


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
        tuple: tuple containing:
            M_a (ndarray): Manifold directions for dataset A of shape (m, m)
            M_b (ndarray): Manifold directions for dataset B of shape (m, m)
    """
    # center data
    # L_a -= np.mean(L_a, 0)
    # L_b -= np.mean(L_b, 0)
    L_a -= np.mean(L_a, 1, keepdims=True)
    L_b -= np.mean(L_b, 1, keepdims=True)

    # determine min rank for CCA return
    rank_a = np.linalg.matrix_rank(L_a)
    rank_b = np.linalg.matrix_rank(L_b)
    d = min(rank_a, rank_b)
    # d = min(L_a.shape[0], L_b.shape[0])

    # QR decomposition
    Q_a, R_a = np.linalg.qr(L_a.T)
    Q_b, R_b = np.linalg.qr(L_b.T)

    # SVD on q inner product
    U, S, Vt = np.linalg.svd(Q_a.T @ Q_b)

    # calculate manifold directions (take only d dimensions for alignment)
    M_a = np.linalg.pinv(R_a) @ U[:,:d]
    M_b = np.linalg.pinv(R_b) @ Vt.T[:,:d]
    # M_b = np.linalg.pinv(R_b) @ Vt[:,:d]
    S = S[:d]

    # account for numerical errors
    S[S < 0] = 0
    S[S >= 1] = 1

    return M_a, M_b, S
