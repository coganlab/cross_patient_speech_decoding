"""
Script to train a RNN model on the DCC.
"""

import os
import sys
import argparse
import numpy as np
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit

sys.path.insert(0, '..')

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import (pad_sequence_teacher_forcing,
                                                  decode_seq2seq)
from processing_utils.data_saving import (append_pkl_accs, dict_from_lists,
                                          save_pkl_params)
from seq2seq_models.rnn_models import (stacked_lstm_1Dcnn_model,
                                       stacked_gru_1Dcnn_model)
from train.transfer_training import (transfer_seq2seq_kfold_diff_chans,
                                     transfer_train_seq2seq_diff_chans)
from visualization.plot_model_performance import plot_loss_acc


def init_parser():
    parser = argparse.ArgumentParser(description='Transfer train RNN on DCC')
    parser.add_argument('-p', '--pretrain_patient', type=str, default='S14',
                        required=False, help='Pretrain Patient ID')
    parser.add_argument('-t', '--transfer_patient', type=str, default='S33',
                        required=False, help='Transfer Patient ID')
    parser.add_argument('-s', '--sig_channels', type=str, default='True',
                        required=False, help='Use significant channels (True)'
                        'or all channels (False)')
    parser.add_argument('-z', '--z_score', type=str, default='False',
                        required=False, help='Z-score normalization (True)'
                        'or mean-subtracted normalization (False)')
    parser.add_argument('-n', '--num_iter', type=int, default=5,
                        required=False, help='Number of times to run model')
    parser.add_argument('-k', '--k_fold', type=str, default='True',
                        required=False, help='Evaluation via k-fold (True) or'
                        'held-out test set (False)')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        required=False, help='Verbosity of model training')
    parser.add_argument('-c', '--cluster', type=str, default='True',
                        required=False,
                        help='Run on cluster (True) or local (False)')
    parser.add_argument('-f', '--filename', type=str, default='',
                        required=False,
                        help='Output filename for performance saving')
    return parser


def str2bool(s):
    return s.lower() == 'true'


def transfer_train_rnn():

    parser = init_parser()
    args = parser.parse_args()

    inputs = {}
    for key, val in vars(args).items():
        inputs[key] = val

    pretrain_pt = inputs['pretrain_patient']
    transfer_pt = inputs['transfer_patient']
    chan_ext = '_sigChannel' if str2bool(inputs['sig_channels']) else '_all'
    norm_ext = '_zscore' if str2bool(inputs['z_score']) else ''
    n_iter = inputs['num_iter']
    kfold = str2bool(inputs['k_fold'])
    verbose = inputs['verbose']
    cluster = str2bool(inputs['cluster'])

    if cluster:
        HOME_PATH = os.path.expanduser('~')
        DATA_PATH = HOME_PATH + '/workspace/'
    else:
        DATA_PATH = '../data/'

    print('==================================================================')
    print("Pretraining models on patient %s." % pretrain_pt)
    print("Transferring models to patient %s." % transfer_pt)
    print("Getting pretraining data from %s%s/." % (DATA_PATH, pretrain_pt))
    print("Getting transfer data from %s%s/." % (DATA_PATH, transfer_pt))
    print("Saving outputs to %s." % (DATA_PATH + 'outputs/'))
    print('==================================================================')

    # Load in data from workspace mat files
    pre_hg_trace, pre_hg_map, pre_phon_labels = get_high_gamma_data(
                                                    DATA_PATH +
                                                    f'{pretrain_pt}/'
                                                    f'{pretrain_pt}_HG'
                                                    f'{chan_ext}'
                                                    f'{norm_ext}'
                                                    '_goodTrials.mat')

    tar_hg_trace, tar_hg_map, tar_phon_labels = get_high_gamma_data(
                                                    DATA_PATH +
                                                    f'{transfer_pt}/'
                                                    f'{transfer_pt}_HG'
                                                    f'{chan_ext}'
                                                    f'{norm_ext}'
                                                    '_goodTrials.mat')

    n_output = 10
    X1 = pre_hg_trace  # (n_trials, n_channels, n_timepoints) for 1D CNN
    X1_prior, y1, _, _ = pad_sequence_teacher_forcing(pre_phon_labels,
                                                      n_output)

    X2 = tar_hg_trace  # (n_trials, n_channels, n_timepoints) for 1D CNN
    X2_prior, y2, _, _ = pad_sequence_teacher_forcing(tar_phon_labels,
                                                      n_output)

    # Build models
    n_input_time = X1.shape[1]
    n_input_channel_pre = X1.shape[2]
    n_input_channel_trans = X2.shape[2]
    model_type = 'lstm'
    model_fcn = stacked_gru_1Dcnn_model if model_type == 'gru' else \
        stacked_lstm_1Dcnn_model
    filter_size = 10
    n_filters = 100  # S14=100, S26=90
    n_layers = 1  # 1
    n_units = 256  # S14=800, S26=900
    reg_lambda = 1e-6  # S14=1e-6, S26=1e-5
    dropout = 0.33
    bidir = True

    # Train model
    test_size = 0.2
    num_folds = 10
    num_reps = 3
    batch_size = 200
    learning_rate = 1e-3
    kfold_rand_state = 7

    if inputs['filename'] != '':
        acc_filename = DATA_PATH + 'outputs/' + inputs['filename'] + '.pkl'
        plot_filename = DATA_PATH + ('outputs/plots/' + inputs['filename']
                                     + '_train_%d.png')
    else:
        if kfold:
            acc_filename = DATA_PATH + ('outputs/'
                                        f'{pretrain_pt}->{transfer_pt}'
                                        f'{norm_ext}_acc_'
                                        f'{num_folds}fold.pkl')
            plot_filename = DATA_PATH + ('outputs/'
                                         f'{pretrain_pt}->{transfer_pt}'
                                         f'{norm_ext}'
                                         f'{num_folds}fold_train_%d.png')
        else:
            acc_filename = DATA_PATH + ('outputs/'
                                        f'{pretrain_pt}->{transfer_pt}'
                                        f'{norm_ext}_acc_'
                                        f'{test_size}-heldout.pkl')
            plot_filename = DATA_PATH + ('outputs/'
                                         f'{pretrain_pt}->{transfer_pt}'
                                         f'{norm_ext}'
                                         f'{test_size}-heldout_train_%d.png')

    pre_epochs = 200  # 200
    conv_epochs = 60  # 60
    ft_epochs = 540  # 540
    total_epochs = pre_epochs + conv_epochs + ft_epochs

    param_keys = ['model_type', 'filter_size', 'n_filters', 'n_units',
                  'n_layers', 'reg_lambda', 'dropout', 'bidir',
                  'num_folds', 'num_reps', 'pre_epochs', 'conv_epochs',
                  'target_epochs', 'learning_rate', 'kfold_rand_state']
    param_vals = [model_type, filter_size, n_filters, n_units,
                  n_layers, reg_lambda, dropout, bidir,
                  num_folds, num_reps, pre_epochs, conv_epochs,
                  ft_epochs, learning_rate, kfold_rand_state]
    save_pkl_params(acc_filename, dict_from_lists(param_keys, param_vals))

    if not kfold:
        # Hold out test data set
        data_split = ShuffleSplit(n_splits=1, test_size=test_size,
                                  random_state=2)
        train_idx, test_idx = next(data_split.split(X2))
        X2_train, X2_test = X2[train_idx], X2[test_idx]
        X2_prior_train, X2_prior_test = X2_prior[train_idx], X2_prior[test_idx]
        y2_train, y2_test = y2[train_idx], y2[test_idx]
    else:
        X2_train = X2
        X2_prior_train = X2_prior
        y2_train = y2

    for i in range(n_iter):
        print('==============================================================')
        print('Iteration: ', i+1)
        print('==============================================================')

        pre_model, pre_enc, pre_dec = model_fcn(
                                                n_input_time,
                                                n_input_channel_pre,
                                                n_output, n_filters,
                                                filter_size, n_layers,
                                                n_units, reg_lambda,
                                                bidir=bidir,
                                                dropout=dropout)
        tar_model, tar_enc, tar_dec = model_fcn(
                                                n_input_time,
                                                n_input_channel_trans,
                                                n_output, n_filters,
                                                filter_size, n_layers,
                                                n_units, reg_lambda,
                                                bidir=bidir,
                                                dropout=dropout)

        pre_model.compile(optimizer=Adam(learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        tar_model.compile(optimizer=Adam(learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        if kfold:
            k_hist, y_pred_all, y_test_all = transfer_seq2seq_kfold_diff_chans(
                                                pre_model, pre_enc,
                                                pre_dec, tar_model,
                                                tar_enc, tar_dec,
                                                X1, X1_prior, y1, X2_train,
                                                X2_prior_train, y2_train,
                                                num_folds=num_folds,
                                                num_reps=num_reps,
                                                rand_state=kfold_rand_state,
                                                batch_size=batch_size,
                                                pretrain_epochs=pre_epochs,
                                                conv_epochs=conv_epochs,
                                                fine_tune_epochs=ft_epochs,
                                                verbose=verbose)

            # final val acc - preds from inf decoder across all folds
            acc = balanced_accuracy_score(y_test_all, y_pred_all)
            cmat = confusion_matrix(y_test_all, y_pred_all,
                                    labels=range(1, n_output))

            plot_loss_acc(k_hist, epochs=total_epochs, save_fig=True,
                          save_path=plot_filename % (i+1))
        else:
            _, _ = transfer_train_seq2seq_diff_chans(
                                pre_model, tar_model,
                                X1, X1_prior, y1,
                                X2_train,
                                X2_prior_train,
                                y2_train,
                                batch_size=batch_size,
                                pretrain_epochs=pre_epochs,
                                conv_epochs=conv_epochs,
                                fine_tune_epochs=ft_epochs,
                                verbose=verbose)

            # test acc
            y_pred_test, labels_test = decode_seq2seq(tar_enc, tar_dec,
                                                      X2_test, y2_test)
            acc = balanced_accuracy_score(labels_test, y_pred_test)
            cmat = confusion_matrix(labels_test, y_pred_test,
                                    labels=range(1, n_output))

        # save performance
        append_pkl_accs(acc_filename, acc, cmat, acc_key='val_acc' if kfold
                        else 'test_acc')


if __name__ == '__main__':
    transfer_train_rnn()
