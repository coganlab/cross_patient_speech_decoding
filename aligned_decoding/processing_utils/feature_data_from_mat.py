""" Load-in feature data from .mat files for training/testing

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""

import os
import numpy as np
import scipy.io as sio


def load_subject_high_gamma(subject_id, sig_channel=False, zscore=False,
                            cluster=False, data_dir=None):
    """Loads high gamma data for a single subject.

    Args:
        subject_id (str): Subject identifier string.
        sig_channel (bool, optional): Whether to load only significant
            channels. Defaults to False.
        zscore (bool, optional): Whether to load z-scored data.
            Defaults to False.
        cluster (bool, optional): Whether to use cluster file paths.
            Defaults to False.
        data_dir (str, optional): Override directory for data files.
            Defaults to None.

    Returns:
        tuple: High gamma trace array, high gamma map array, and phoneme
            label array.
    """
    filename = process_mat_filename(subject_id, sig_channel, zscore,
                                    cluster=cluster, data_dir=data_dir)
    hg_trace, hg_map, phon_labels = get_high_gamma_data(filename)

    return hg_trace, hg_map, phon_labels


def load_subject_high_gamma_phoneme(subject_id, phons=[1, 2, 3],
                                    cluster=False, zscore=False,
                                    data_dir=None):
    """Loads high gamma data for a subject, separated by phoneme position.

    Args:
        subject_id (str): Subject identifier string.
        phons (list, optional): Phoneme positions to load. Defaults to
            [1, 2, 3].
        cluster (bool, optional): Whether to use cluster file paths.
            Defaults to False.
        zscore (bool, optional): Whether to load z-scored data.
            Defaults to False.
        data_dir (str, optional): Override directory for data files.
            Defaults to None.

    Returns:
        dict: Dictionary keyed by subject ID, with high gamma trace, map,
            and label arrays for each phoneme position.
    """
    subj_dict = dict(ID=subject_id)
    for p in phons:
        filename = process_mat_filename(subject_id, True, zscore, phon=p,
                                        cluster=cluster, data_dir=data_dir)
        hg_trace, hg_map, phon_labels = get_high_gamma_data(filename)
        subj_dict['X' + str(p)] = hg_trace
        subj_dict['X' + str(p) + '_map'] = hg_map
        subj_dict['y' + str(p)] = phon_labels[:, p-1]
    subj_dict['y_full_phon'] = phon_labels
    return subj_dict


def load_mat_data(filename):
    """Loads a .mat file using scipy.

    Args:
        filename (str): Path to the .mat file.

    Returns:
        dict: Dictionary of variable names and values from the .mat file.
    """
    return sio.loadmat(filename)


def get_feature_data(mat_data, feature_name):
    """Extracts and squeezes a named feature array from loaded .mat data.

    Args:
        mat_data (dict): Dictionary returned by ``load_mat_data``.
        feature_name (str): Key for the desired feature in the .mat file.

    Returns:
        ndarray: Squeezed numpy array of the requested feature.
    """
    return np.squeeze(np.array(mat_data[feature_name]))


def process_mat_filename(subject_id, sig_channel, zscore, phon=None,
                         cluster=False, data_dir=None):
    """Constructs a .mat filename from subject and processing parameters.

    Args:
        subject_id (str): Subject identifier string.
        sig_channel (bool): Whether to use significant-channel suffix.
        zscore (bool): Whether to use z-score suffix.
        phon (int, optional): Phoneme position number for per-phoneme files.
            Defaults to None.
        cluster (bool, optional): Whether to use cluster file paths.
            Defaults to False.
        data_dir (str, optional): Override directory for data files.
            Defaults to None.

    Returns:
        str: Full path to the constructed .mat filename.
    """
    if data_dir is None:
        if cluster:
            home_path = os.path.expanduser('~')
            data_dir = home_path + '/workspace/'
        else:
            data_dir = 'data/'
    subject_dir = subject_id + '/'
    filename_base = subject_id + '_HG'
    if sig_channel:
        chan_suffix = '_sigChannel'
    else:
        chan_suffix = '_all'
    if zscore:
        norm_suffix = '_zscore'
    else:
        norm_suffix = ''
    if phon is not None:
        phon_suffix = '_p' + str(phon)
    else:
        phon_suffix = ''

    filename = str(data_dir + subject_dir + filename_base + phon_suffix +
                   chan_suffix + norm_suffix + '_goodTrials.mat')
    # filename = str(data_dir + subject_dir + filename_base + phon_suffix +
    #                chan_suffix + norm_suffix + '.mat')
    return filename


def get_high_gamma_data(filename):
    """Loads high gamma trace, map, and phoneme labels from a .mat file.

    Args:
        filename (str): Path to the .mat file.

    Returns:
        tuple: (hg_trace, hg_map, phon_labels) where hg_trace has shape
            (trials, channel_x, channel_y, timepoints), hg_map has shape
            (trials, timepoints, channels), and phon_labels has shape
            (trials, 3).
    """
    mat_data = load_mat_data(filename)

    # shape = trials x channel_x x channel_y x timepoints
    hg_trace = get_feature_data(mat_data, 'hgTrace')
    # shape = trials x timepoints x channels
    hg_map = get_feature_data(mat_data, 'hgMap')
    # shape = trials x 3 (3 length phoneme sequence)
    phon_labels = get_feature_data(mat_data, 'phonSeqLabels')

    return hg_trace, hg_map, phon_labels


def get_high_gamma_data_spatialAvg(filename, contactSizes):
    """Loads spatially averaged high gamma data for multiple contact sizes.

    Args:
        filename (str): Path to the .mat file.
        contactSizes (list): List of contact-size strings in ``'axb'`` format
            (e.g., ``'1x1'``, ``'2x2'``).

    Returns:
        tuple: (csDictTrace, phon_labels) where csDictTrace maps each contact
            size string to its corresponding high gamma array, and phon_labels
            has shape (trials, 3).
    """
    mat_data = load_mat_data(filename)

    phon_labels = get_feature_data(mat_data, 'phonSeqLabels')

    csDictTrace = {}
    for cs in contactSizes:
        csDictTrace[cs] = get_feature_data(mat_data, 'cs_' + cs)
    
    return csDictTrace, phon_labels