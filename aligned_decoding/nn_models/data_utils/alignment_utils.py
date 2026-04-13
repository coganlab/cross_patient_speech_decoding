""" Alignment Util Fcns

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
import pickle
from functools import reduce


def extract_group_conditions(Xs, ys):
    """Computes condition-averaged data for shared classes across datasets.

    Args:
        Xs: List of feature arrays, each of shape (n_trials, ...).
        ys: List of label arrays, each of shape (n_trials,) or
            (n_trials, seq_length).

    Returns:
        list[ndarray]: Condition-averaged arrays containing only classes
            shared across all datasets.
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
    """Converts numeric labels to string representation.

    For 1D labels, casts elements to strings. For 2D label sequences,
    concatenates each row into a single string via label_seq2str.

    Args:
        labels: Label array of shape (n_trials,) or
            (n_trials, n_labels_per_trial).

    Returns:
        ndarray: String labels of shape (n_trials,).
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
    """Saves data to a pickle file using the highest protocol.

    Args:
        data: Python object to serialize.
        filename: Path to the output pickle file.
    """
    with open(filename, 'wb+') as f:
        pickle.dump(data, f, protocol=-1)


def load_pkl(filename):
    """Loads data from a pickle file.

    Args:
        filename: Path to the pickle file.

    Returns:
        The deserialized Python object.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def decoding_data_from_dict(data_dict, pt, p_ind, lab_type='phon',
                            algn_type='phon_seq'):
    """Extracts target and cross-patient decoding data from a data dictionary.

    Args:
        data_dict: Nested dictionary keyed by patient ID containing feature
            arrays and labels.
        pt: Target patient ID key.
        p_ind: Phoneme position index (-1 for collapsed across positions).
        lab_type: Label type, 'phon' or 'artic'. Defaults to 'phon'.
        algn_type: Alignment label type key suffix. Defaults to 'phon_seq'.

    Returns:
        tuple: ((D_tar, lab_tar, lab_tar_full), pre_data) where pre_data is
            a list of (features, labels, full_labels) tuples for each
            cross-patient dataset.
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
    """Extracts feature matrix and labels from a single patient's data dict.

    Args:
        data: Patient data dictionary with keys like 'X0', 'y0',
            'X_collapsed', etc.
        p_ind: Phoneme position index (-1 for collapsed data).
        lab_type: Label type, 'phon' or 'artic'.
        algn_type: Alignment label key suffix (e.g. 'phon_seq').

    Returns:
        tuple: (D, lab, lab_full) — features, decoding labels, and full
            alignment labels.
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
    """Converts a phoneme label array to articulator group labels.

    Args:
        phon_seq: Phoneme label array of arbitrary shape.

    Returns:
        ndarray: Articulator labels with the same shape as phon_seq.
    """
    phon_to_artic_conv = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 4}
    flat_seq = phon_seq.flatten()
    artic_conv = np.array([phon_to_artic(phon_idx, phon_to_artic_conv) for
                           phon_idx in flat_seq])
    return np.reshape(artic_conv, phon_seq.shape)


def phon_to_artic(phon_idx, phon_to_artic_conv):
    """Maps a single phoneme index to its articulator group.

    Args:
        phon_idx: Integer phoneme index.
        phon_to_artic_conv: Dictionary mapping phoneme indices to articulator
            group indices.

    Returns:
        int: Articulator group index.
    """
    return phon_to_artic_conv[phon_idx]
