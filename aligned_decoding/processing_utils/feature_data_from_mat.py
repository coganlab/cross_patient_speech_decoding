""" Load-in feature data from .mat files for training/testing

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""

import os
import numpy as np
import scipy.io as sio


def load_subject_high_gamma(subject_id, sig_channel=False, zscore=False,
                            cluster=False, data_dir=None):
    filename = process_mat_filename(subject_id, sig_channel, zscore,
                                    cluster=cluster, data_dir=data_dir)
    hg_trace, hg_map, phon_labels = get_high_gamma_data(filename)

    return hg_trace, hg_map, phon_labels


def load_subject_high_gamma_phoneme(subject_id, phons=[1, 2, 3],
                                    cluster=False, zscore=False,
                                    data_dir=None):
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
    return sio.loadmat(filename)


def get_feature_data(mat_data, feature_name):
    return np.squeeze(np.array(mat_data[feature_name]))


def process_mat_filename(subject_id, sig_channel, zscore, phon=None,
                         cluster=False, data_dir=None):
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

    # filename = str(data_dir + subject_dir + filename_base + phon_suffix +
    #                chan_suffix + norm_suffix + '_goodTrials.mat')
    filename = str(data_dir + subject_dir + filename_base + phon_suffix +
                   chan_suffix + norm_suffix + '.mat')
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
