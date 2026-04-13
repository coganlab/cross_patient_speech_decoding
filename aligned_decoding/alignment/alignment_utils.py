""" Alignment Util Fcns

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
import pickle
from functools import reduce


def extract_group_conditions(Xs, ys):
    """Extracts condition-averaged data shared across multiple datasets.

    Averages trials within each condition for every dataset and retains
    only conditions present in all datasets.

    Args:
        Xs (list of ndarray): List of feature arrays, one per dataset.
        ys (list of ndarray): List of label arrays corresponding to Xs.

    Returns:
        list of ndarray: Condition-averaged arrays for each dataset,
            filtered to shared conditions only.
    """
    # process labels for easy comparison of label sequences
    ys = [label2str(labs) for labs in ys]

    # condition average firing rates for all datasets
    cnd_avg_data = [0]*len(Xs)
    for i, (feats, labs) in enumerate(zip(Xs, ys)):
        cnd_avg_data[i] = cnd_avg(feats, labs)

    # only use same conditions across datasets
    shared_lab = reduce(np.intersect1d, ys)
    cnd_avg_data = [cnd_avg_data[i][np.isin(np.unique(lab), shared_lab,
                                            assume_unique=True)] for i, lab
                    in enumerate(ys)]
    return cnd_avg_data


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
    """Converts labels to string representation.

    If labels are 2D (sequence labels), joins each row into a single
    string via label_seq2str. If 1D, casts elements to strings.

    Args:
        labels (ndarray): Label array, either 1D or 2D.

    Returns:
        ndarray: 1D array of string labels.
    """
    if len(labels.shape) > 1:
        labels = label_seq2str(labels)
    else:
        labels = labels.astype(str)
    return labels


def label_seq2str(labels):
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


def save_pkl(data, filename):
    """Saves data to a pickle file.

    Args:
        data: Object to serialize.
        filename (str): Path to the output pickle file.
    """
    with open(filename, 'wb+') as f:
        pickle.dump(data, f, protocol=-1)


def load_pkl(filename):
    """Loads data from a pickle file.

    Args:
        filename (str): Path to the pickle file to read.

    Returns:
        object: Deserialized data from the pickle file.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def decoding_data_from_dict(data_dict, pt, p_ind, lab_type='phon',
                            algn_type='phon_seq'):
    """Extracts target and pre-training patient data from a data dictionary.

    Args:
        data_dict (dict): Nested dictionary keyed by patient ID containing
            feature and label data, plus a 'pre_pts' key listing
            pre-training patient IDs.
        pt (str): Target patient ID.
        p_ind (int): Phoneme index. -1 for collapsed across all phonemes.
        lab_type (str, optional): Label type ('phon' or 'artic'). Defaults
            to 'phon'.
        algn_type (str, optional): Alignment label type. Defaults to
            'phon_seq'.

    Returns:
        tuple: ((D_tar, lab_tar, lab_tar_full), pre_data) where pre_data
            is a list of (features, labels, full_labels) tuples for each
            pre-training patient.
    """
    D_tar, lab_tar, lab_tar_full = get_features_labels(data_dict[pt], p_ind,
                                                       lab_type, algn_type)

    pre_data = []
    for p_pt in data_dict[pt]['pre_pts']:
        D_curr, lab_curr, lab_curr_full = get_features_labels(data_dict[p_pt],
                                                              p_ind, lab_type,
                                                              algn_type)
        pre_data.append((D_curr, lab_curr, lab_curr_full))

    return (D_tar, lab_tar, lab_tar_full), pre_data


def get_features_labels(data, p_ind, lab_type, algn_type):
    """Extracts features and labels for a single patient from a data dict.

    Args:
        data (dict): Patient data dictionary containing feature matrices
            and label arrays.
        p_ind (int): Phoneme index. -1 for collapsed across all phonemes.
        lab_type (str): Label type ('phon' or 'artic').
        algn_type (str): Alignment label type (e.g. 'phon_seq').

    Returns:
        tuple: (D, lab, lab_full) — feature array, label array, and full
            sequence label array.
    """
    lab_full = data['y_full_' + algn_type[:-4]]
    if p_ind == -1:  # collapsed across all phonemes
        D = data['X_collapsed']
        lab = data['y_' + lab_type + '_collapsed']
        lab_full = np.tile(lab_full, (3, 1))  # label repeat for shape match
    else:  # individual phoneme
        D = data['X' + str(p_ind)]
        lab = data['y' + str(p_ind)]
    if lab_type == 'artic':  # convert from phonemes to articulator label
        lab = phon_to_artic_seq(lab)
    return D, lab, lab_full


def phon_to_artic_seq(phon_seq):
    """Converts a phoneme label array to articulator labels.

    Args:
        phon_seq (ndarray): Array of phoneme indices (values 1-9).

    Returns:
        ndarray: Array of articulator indices with the same shape as
            phon_seq.
    """
    phon_to_artic_conv = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 4}
    flat_seq = phon_seq.flatten()
    artic_conv = np.array([phon_to_artic(phon_idx, phon_to_artic_conv) for
                           phon_idx in flat_seq])
    return np.reshape(artic_conv, phon_seq.shape)


def phon_to_artic(phon_idx, phon_to_artic_conv):
    """Maps a single phoneme index to its articulator index.

    Args:
        phon_idx (int): Phoneme index.
        phon_to_artic_conv (dict): Mapping from phoneme indices to
            articulator indices.

    Returns:
        int: Corresponding articulator index.
    """
    return phon_to_artic_conv[phon_idx]
