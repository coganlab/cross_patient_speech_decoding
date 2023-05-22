"""
Script to train a RNN model by first freezing encoder-decoder and training
convolutional layer and then training the full network. Using as a comparison
for multi-patient transfer results.
"""

import os
import sys
import argparse
import numpy as np
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit, KFold

sys.path.insert(0, '..')

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import (pad_sequence_teacher_forcing,
                                                  decode_seq2seq)
from processing_utils.data_saving import append_pkl_accs
from seq2seq_models.rnn_models import (stacked_lstm_1Dcnn_model,
                                       stacked_gru_1Dcnn_model)
from train.train import train_seq2seq, shuffle_weights
from train.transfer_training import transfer_conv_update, concat_hists
from train.Seq2seqPredictCallback import Seq2seqPredictCallback
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
    return parser


def str2bool(s):
    return s.lower() == 'true'


def freeze_train_driver():
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
    hg_trace, hg_map, phon_labels = get_high_gamma_data(DATA_PATH +
                                                        f'{pt}/{pt}_HG'
                                                        f'{chan_ext}'
                                                        f'{norm_ext}'
                                                        '_goodTrials.mat')

    X = hg_trace  # use HG traces (n_trials, n_channels, n_timepoints) for CNN
    X_prior, y, _, _ = pad_sequence_teacher_forcing(phon_labels, n_output)

    # Build models
    n_input_time = X.shape[1]
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

    # Train model
    num_folds = 5
    num_reps = 3
    batch_size = 200
    conv_epochs = 60
    full_epochs = 540
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
    else:
        X_train = X
        X_prior_train = X_prior
        y_train = y

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
            k_hist, y_pred_all, y_test_all = freeze_train_kfold(
                                                train_model, inf_enc, inf_dec,
                                                X_train, X_prior_train,
                                                y_train, num_folds=num_folds,
                                                num_reps=num_reps,
                                                rand_state=kfold_rand_state,
                                                batch_size=batch_size,
                                                conv_epochs=conv_epochs,
                                                full_epochs=full_epochs,
                                                verbose=verbose)

            # final val acc - preds from inf decoder across all folds
            acc = balanced_accuracy_score(y_test_all, y_pred_all)
            cmat = confusion_matrix(y_test_all, y_pred_all,
                                    labels=range(1, n_output))

            plot_loss_acc(k_hist, epochs=conv_epochs + full_epochs,
                          save_fig=True,
                          save_path=DATA_PATH +
                          (f'outputs/plots/{pt}'
                           f'_{num_folds}fold_freeze_train_{i+1}.png'))
        else:
            conv_hist = transfer_conv_update(train_model, X_train,
                                             X_prior_train, y_train,
                                             epochs=conv_epochs,
                                             batch_size=batch_size)
            _, _ = train_seq2seq(train_model, X_train,
                                 X_prior_train, y_train,
                                 batch_size=batch_size,
                                 epochs=full_epochs,
                                 verbose=verbose)

            # test acc
            y_pred_test, labels_test = decode_seq2seq(inf_enc, inf_dec, X_test,
                                                      y_test)
            acc = balanced_accuracy_score(labels_test, y_pred_test)
            cmat = confusion_matrix(labels_test, y_pred_test,
                                    labels=range(1, n_output))

        if inputs['filename'] != '':
            acc_filename = DATA_PATH + 'outputs/' + inputs['filename'] \
                           + '.pkl'
        else:
            if kfold:
                acc_filename = DATA_PATH + ('outputs/'
                                            f'{pt}{norm_ext}_acc_'
                                            f'{num_folds}fold_freeze.pkl')
            else:
                acc_filename = DATA_PATH + ('outputs/'
                                            f'{pt}{norm_ext}_acc_'
                                            f'{test_size}-heldout_freeze.pkl')

        # save performance
        append_pkl_accs(acc_filename, acc, cmat, acc_key='val_acc' if kfold
                        else 'test_acc')


def freeze_train_kfold(train_model, inf_enc, inf_dec, X, X_prior, y,
                       num_folds=5, num_reps=3, rand_state=7, batch_size=200,
                       conv_epochs=60, full_epochs=540, **kwargs):
    # save initial weights to reset model for each fold
    init_train_w = train_model.get_weights()

    # define k-fold cross validation
    cv = KFold(n_splits=num_folds, shuffle=True, random_state=rand_state)

    # dictionary for history of each fold
    histories = {'accuracy': [], 'loss': []}

    y_pred_all, y_test_all = [], []
    for r in range(num_reps):  # repeat fold for stability
        print(f'======== Repetition {r + 1} ========')

        # cv training
        for f, (train_ind, test_ind) in enumerate(cv.split(X)):
            print(f'===== Fold {f + 1} =====')

            # reset model weights for current fold (also resets associated
            # inference weights)
            shuffle_weights(train_model, weights=init_train_w)

            X_train, X_test = X[train_ind], X[test_ind]
            X_prior_train, X_prior_test = X_prior[train_ind], X_prior[test_ind]
            y_train, y_test = y[train_ind], y[test_ind]

            val_data = ([X_test, X_prior_test], y_test)

            seq2seq_cb = Seq2seqPredictCallback(train_model, inf_enc, inf_dec,
                                                X_test, y_test)
            cb = [seq2seq_cb]

            conv_hist = transfer_conv_update(train_model, X_train,
                                             X_prior_train, y_train,
                                             epochs=conv_epochs, callbacks=cb,
                                             validation_data=val_data,
                                             batch_size=batch_size,
                                             **kwargs)

            _, full_hist = train_seq2seq(train_model, X_train, X_prior_train,
                                         y_train, epochs=full_epochs,
                                         validation_data=val_data,
                                         batch_size=batch_size,
                                         callbacks=cb, **kwargs)

            y_test_fold, y_pred_fold = decode_seq2seq(inf_enc, inf_dec, X_test,
                                                      y_test)

            y_pred_all.extend(y_pred_fold)
            y_test_all.extend(y_test_fold)

            concat_hists([conv_hist, full_hist])

    return histories, np.array(y_pred_all), np.array(y_test_all)


if __name__ == '__main__':
    freeze_train_driver()
