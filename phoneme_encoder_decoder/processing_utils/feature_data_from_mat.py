""" Load-in feature data from .mat files for training/testing

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""

import numpy as np
import scipy.io as sio


def load_mat_data(filename):
    return sio.loadmat(filename)


def get_feature_data(mat_data, feature_name):
    return np.squeeze(np.array(mat_data[feature_name]))


def process_mat_filename(subject_id, sig_channel, zscore):
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

    filename = str(data_dir + subject_dir + filename_base + chan_suffix +
                   norm_suffix + '.mat')
    return filename


def get_high_gamma_data(filename):

    mat_data = load_mat_data(filename)

    # shape = trials x channel_x x channel_y x timepoints
    hg_trace = get_feature_data(mat_data, 'hgTrace')
    # shape = trials x timepoints x channels
    hg_map = get_feature_data(mat_data, 'hgMap')
    # shape = trials x 3 (3 length phoneme sequence)
    phon_labels = get_feature_data(mat_data, 'phonSeqLabels')

    return hg_trace, hg_map, phon_labels


def load_subject_high_gamma(subject_id, sig_channel=False, zscore=False):

    filename = process_mat_filename(subject_id, sig_channel, zscore)
    hg_trace, hg_map, phon_labels = get_high_gamma_data(filename)

    return hg_trace, hg_map, phon_labels
