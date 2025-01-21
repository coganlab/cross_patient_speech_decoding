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
from processing_utils.data_augmentation import (augment_mixup,
                                                augment_time_jitter)
from seq2seq_models.rnn_models import (stacked_lstm_1Dcnn_model,
                                       stacked_gru_1Dcnn_model)
from train.train import train_seq2seq_kfold, train_seq2seq
from visualization.plot_model_performance import plot_loss_acc


def init_parser():
    parser = argparse.ArgumentParser(description='Train RNN model on DCC')
    parser.add_argument('-p', '--patient', type=str, default='S14',
                        required=False, help='Patient ID')
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
    parser.add_argument('-m', '--mixup', type=str, default='False',
                        required=False,
                        help='Generate synthetic trial data via MixUp (True)'
                             'or use only original data (False)')
    parser.add_argument('-j', '--jitter', type=str, default='False',
                        required=False,
                        help='Generate synthetic trial data via time window'
                             'jittering (True) or use only original data'
                             '(False)')
    return parser


def str2bool(s):
    return s.lower() == 'true'


def train_rnn():
    parser = init_parser()
    args = parser.parse_args()

    inputs = {}
    for key, val in vars(args).items():
        inputs[key] = val

    pt = inputs['patient']
    chan_ext = '_sigChannel' if str2bool(inputs['sig_channels']) else '_all'
    norm_ext = '_zscore' if str2bool(inputs['z_score']) else ''
    n_iter = inputs['num_iter']
    kfold = str2bool(inputs['k_fold'])
    verbose = inputs['verbose']
    cluster = str2bool(inputs['cluster'])
    mixup = str2bool(inputs['mixup'])
    mixup_ext = '_mixup' if mixup else ''
    jitter = str2bool(inputs['jitter'])
    jitter_ext = '_jitter' if jitter else ''

    if cluster:
        HOME_PATH = os.path.expanduser('~')
        DATA_PATH = HOME_PATH + '/workspace/'
    else:
        DATA_PATH = '../data/'

    print('==================================================================')
    print("Training models for patient %s." % pt)
    print("Getting data from %s." % (DATA_PATH + f'{pt}/'))
    print("Saving outputs to %s." % (DATA_PATH + 'outputs/'))
    print('==================================================================')

    # Load in data from workspace mat files
    n_output = 10
    if jitter:
        hg_trace, hg_map, phon_labels = get_high_gamma_data(
                                            DATA_PATH + f'{pt}/{pt}_HG'
                                            f'{chan_ext}{norm_ext}'
                                            '_extended_goodTrials.mat')
    else:
        hg_trace, hg_map, phon_labels = get_high_gamma_data(
                                            DATA_PATH + f'{pt}/{pt}_HG'
                                            f'{chan_ext}{norm_ext}'
                                            '_goodTrials.mat')

    X = hg_trace  # use HG traces (n_trials, n_channels, n_timepoints) for CNN
    X_prior, y, _, seq_labels = pad_sequence_teacher_forcing(phon_labels,
                                                             n_output)

    # Model parameters
    win_len = 1  # 1 second decoding window
    fs = 200
    n_input_time = int(win_len * fs)
    n_input_channel = X.shape[-1]
    model_type = 'lstm'
    model_fcn = stacked_gru_1Dcnn_model if model_type == 'gru' else \
        stacked_lstm_1Dcnn_model
    filter_size = 10
    n_filters = 50  # S14=100, S26=90
    n_units = 256  # S14=800, S26=900
    n_layers = 1
    reg_lambda = 1e-6  # S14=1e-6, S26=1e-5
    dropout = 0.33  # 0.33
    bidir = True

    # Augmentation parameters
    mixup_alpha = 5
    mixup_dict = ({'alpha': mixup_alpha, 'labels': seq_labels} if mixup else
                  None)
    j_end = 0.5
    # define jitter by number of points
    n_jitter = 5
    if n_jitter % 2 == 0:
        n_jitter += 1  # +1 to include 0
    jitter_vals = np.linspace(-j_end, j_end, n_jitter)
    jitter_dict = ({'jitter_vals': jitter_vals, 'win_len': win_len, 'fs': fs}
                   if jitter else None)

    # Training parameters
    num_folds = 5
    num_reps = 3
    epochs = 800
    learning_rate = 1e-3
    kfold_rand_state = 7

    if not kfold:
        # Hold out test data set
        test_size = 0.2
        data_split = ShuffleSplit(n_splits=1, test_size=test_size,
                                  random_state=2)
        train_idx, test_idx = next(data_split.split(X))
        X_train, X_test = X[train_idx], X[test_idx]
        X_prior_train, X_prior_test = X_prior[train_idx], X_prior[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        seq_labels_train, seq_labels_test = (seq_labels[train_idx],
                                             seq_labels[test_idx])

        if mixup:
            X_train, X_prior_train, y_train = augment_mixup(
                                                X_train, X_prior_train,
                                                y_train, seq_labels_train,
                                                alpha=mixup_alpha)
        if jitter:
            X_train, X_prior_train, y_train = augment_time_jitter(
                                                X_train, X_prior_train,
                                                y_train, jitter_vals, win_len,
                                                fs)

    else:
        X_train = X
        X_prior_train = X_prior
        y_train = y
        seq_labels_train = seq_labels

    if inputs['filename'] != '':
        acc_filename = DATA_PATH + 'outputs/' + inputs['filename'] + '.pkl'
        plot_filename = DATA_PATH + ('outputs/plots/' + inputs['filename']
                                     + '_train_%d.png')
    else:
        if kfold:
            acc_filename = DATA_PATH + ('outputs/'
                                        f'{pt}{norm_ext}_acc_'
                                        f'{num_folds}fold{mixup_ext}'
                                        f'{jitter_ext}.pkl')
            plot_filename = DATA_PATH + ('outputs/'
                                         f'{pt}{norm_ext}'
                                         f'{num_folds}fold{mixup_ext}'
                                         f'{jitter_ext}_train_%d.png')
        else:
            acc_filename = DATA_PATH + ('outputs/'
                                        f'{pt}{norm_ext}_acc_'
                                        f'{test_size}-heldout{mixup_ext}'
                                        f'{jitter_ext}.pkl')
            plot_filename = DATA_PATH + ('outputs/'
                                         f'{pt}{norm_ext}'
                                         f'{test_size}-heldout{mixup_ext}'
                                         f'{jitter_ext}_train_%d.png')

    param_keys = ['model_type', 'filter_size', 'n_filters', 'n_units',
                  'n_layers', 'reg_lambda', 'dropout', 'bidir', 'mixup_alpha',
                  'num_folds', 'num_reps', 'epochs', 'learning_rate',
                  'kfold_rand_state']
    param_vals = [model_type, filter_size, n_filters, n_units, n_layers,
                  reg_lambda, dropout, bidir, mixup_alpha, num_folds, num_reps,
                  epochs, learning_rate, kfold_rand_state]
    save_pkl_params(acc_filename, dict_from_lists(param_keys, param_vals))

    for i in range(n_iter):
        print('==============================================================')
        print('Iteration: ', i+1)
        print('==============================================================')

        train_model, inf_enc, inf_dec = model_fcn(n_input_time,
                                                  n_input_channel,
                                                  n_output,
                                                  n_filters,
                                                  filter_size,
                                                  n_layers,
                                                  n_units,
                                                  reg_lambda,
                                                  bidir=bidir,
                                                  dropout=dropout)

        train_model.compile(optimizer=Adam(learning_rate),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        if kfold:
            k_hist, y_pred_all, y_test_all = train_seq2seq_kfold(
                                                train_model, inf_enc,
                                                inf_dec, X_train,
                                                X_prior_train, y_train,
                                                num_folds=num_folds,
                                                num_reps=num_reps,
                                                rand_state=kfold_rand_state,
                                                mixup_dict=mixup_dict,
                                                jitter_dict=jitter_dict,
                                                epochs=epochs,
                                                early_stop=False,
                                                verbose=verbose)

            # final val acc - preds from inf decoder across all folds
            acc = balanced_accuracy_score(y_test_all, y_pred_all)
            cmat = confusion_matrix(y_test_all, y_pred_all,
                                    labels=range(1, n_output))

            plot_loss_acc(k_hist, epochs=epochs, save_fig=True,
                          save_path=plot_filename % (i+1))
        else:
            _, _ = train_seq2seq(train_model, X_train,
                                 X_prior_train, y_train,
                                 batch_size=X_train.shape[0],
                                 epochs=epochs,
                                 verbose=verbose)

            # test acc
            y_pred_test, labels_test = decode_seq2seq(inf_enc, inf_dec, X_test,
                                                      y_test)
            acc = balanced_accuracy_score(labels_test, y_pred_test)
            cmat = confusion_matrix(labels_test, y_pred_test,
                                    labels=range(1, n_output))

        # save performance
        append_pkl_accs(acc_filename, acc, cmat, acc_key='val_acc' if kfold
                        else 'test_acc')


if __name__ == '__main__':
    train_rnn()
