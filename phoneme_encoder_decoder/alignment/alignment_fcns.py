""" Various methods for aligning microECoG datasets across patients.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np


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
