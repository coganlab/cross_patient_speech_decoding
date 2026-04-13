""" Metrics for comparing quality of different cross-patient dataset alignment
methods.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
from scipy.stats import pearsonr


def pt_corr_multi(target, to_corr_list, p_vals=False):
    """Computes per-condition Pearson correlations between a target and
    multiple comparison datasets.

    Args:
        target (ndarray): Reference data of shape (n_conditions, ...).
        to_corr_list (list of ndarray): List of arrays to correlate against
            target. Each must have the same shape as target.
        p_vals (bool, optional): If True, also returns p-values. Defaults
            to False.

    Returns:
        list of ndarray: Per-condition correlation values for each
            comparison dataset. If p_vals is True, returns a tuple of
            (list of ndarray, list of ndarray) for correlations and
            p-values respectively.
    """
    pt_corrs = [0]*len(to_corr_list)
    pt_p_vals = [0]*len(to_corr_list)
    for i, to_corr in enumerate(to_corr_list):
        if p_vals:
            pt_corrs[i], pt_p_vals[i] = pt_corr(target, to_corr, p_vals=True)
        else:
            pt_corrs[i] = pt_corr(target, to_corr)
    if p_vals:
        return pt_corrs, pt_p_vals
    return pt_corrs


def pt_corr(target, to_corr, p_vals=False):
    """Computes per-condition Pearson correlation between two datasets.

    Each condition's data is flattened across time and features before
    computing the correlation.

    Args:
        target (ndarray): Reference data of shape
            (n_conditions, n_timepoints, n_features).
        to_corr (ndarray): Comparison data with the same shape as target.
        p_vals (bool, optional): If True, also returns p-values. Defaults
            to False.

    Returns:
        ndarray: Array of Pearson r values, one per condition. If p_vals
            is True, returns a tuple of (ndarray, ndarray) for r values
            and p-values respectively.
    """
    cnd_r = np.zeros(target.shape[0])
    cnd_p_vals = np.zeros(target.shape[0])
    for cnd in range(target.shape[0]):
        curr_cnd_tar = target[cnd].flatten()  # (time * PCs)
        curr_cnd_corr = to_corr[cnd].flatten()
        cnd_r[cnd] = pearsonr(curr_cnd_tar, curr_cnd_corr)[0]
        cnd_p_vals[cnd] = pearsonr(curr_cnd_tar, curr_cnd_corr)[1]
    if p_vals:
        return cnd_r, cnd_p_vals
    return cnd_r  # return average r value across conditions
