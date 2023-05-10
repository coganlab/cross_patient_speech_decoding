"""
Script to train a RNN model on the DCC.
"""

import os
import sys
import csv
import argparse
from keras.optimizers import Adam
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, '..')

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import (pad_sequence_teacher_forcing,
                                                  decode_seq2seq)
from seq2seq_models.rnn_models import (lstm_1Dcnn_model,
                                       stacked_lstm_1Dcnn_model)
from train.train import train_seq2seq_kfold, train_seq2seq
from visualization.plot_model_performance import (plot_accuracy_loss,
                                                  plot_tf_hist_loss_acc)


def init_parser():
    parser = argparse.ArgumentParser(description='Train RNN model on DCC')
    parser.add_argument('-pt', '--patient', type=str, default='S14',
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
                        required=False, help='Perform k-fold CV (True)'
                        'or not (False)')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        required=False, help='Verbosity of model training')
    parser.add_argument('-c', '--cluster', type=str, default='True',
                        required=False,
                        help='Run on cluster (True) or local (False)')
    parser.add_argument('-o', '--out_filename', type=str, default='',
                        required=False,
                        help='Output filename for accuracy csv')
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
    hg_trace, hg_map, phon_labels = get_high_gamma_data(DATA_PATH +
                                                        f'{pt}/{pt}_HG'
                                                        f'{chan_ext}'
                                                        f'{norm_ext}'
                                                        '_goodTrials.mat')

    n_output = 10
    X = hg_trace  # use HG traces (n_trials, n_channels, n_timepoints) for CNN
    X_prior, y, _, _ = pad_sequence_teacher_forcing(phon_labels, n_output)

    # Build models
    n_input_time = X.shape[1]
    n_input_channel = X.shape[2]
    filter_size = 10
    n_filters = 100  # S14=100, S26=90
    n_units = 256  # S14=800, S26=900
    n_layers = 1
    reg_lambda = 1e-6  # S14=1e-6, S26=1e-5
    bidir = True

    # Train model
    test_size = 0.2
    num_folds = 10
    num_reps = 3
    batch_size = 200
    epochs = 800
    learning_rate = 1e-3
    dropout = 0.33

    # Hold out test data set
    data_split = ShuffleSplit(n_splits=1, test_size=test_size, random_state=2)
    train_idx, test_idx = next(data_split.split(X))
    X_train, X_test = X[train_idx], X[test_idx]
    X_prior_train, X_prior_test = X_prior[train_idx], X_prior[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for i in range(n_iter):
        print('==============================================================')
        print('Iteration: ', i+1)
        print('==============================================================')

        if kfold:

            # kfold_model, kfold_enc, kfold_dec = lstm_1Dcnn_model(
            #                                         n_input_time,
            #                                         n_input_channel,
            #                                         n_output,
            #                                         n_filters,
            #                                         filter_size,
            #                                         n_units,
            #                                         reg_lambda,
            #                                         bidir=bidir,
            #                                         dropout=dropout)
            kfold_model, kfold_enc, kfold_dec = stacked_lstm_1Dcnn_model(
                                                    n_input_time,
                                                    n_input_channel,
                                                    n_output,
                                                    n_filters,
                                                    filter_size,
                                                    n_layers,
                                                    n_units,
                                                    reg_lambda,
                                                    bidir=bidir,
                                                    dropout=dropout)

            kfold_model.compile(optimizer=Adam(learning_rate),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

            k_hist, y_pred_all, y_test_all = train_seq2seq_kfold(
                                                    kfold_model, kfold_enc,
                                                    kfold_dec, X_train,
                                                    X_prior_train, y_train,
                                                    num_folds=num_folds,
                                                    num_reps=num_reps,
                                                    batch_size=batch_size,
                                                    epochs=epochs,
                                                    early_stop=False,
                                                    verbose=verbose)

            # final val acc - preds from inf decoder across all folds
            val_acc = balanced_accuracy_score(y_test_all, y_pred_all)

        # train_model, inf_enc, inf_dec = lstm_1Dcnn_model(
        #                                     n_input_time,
        #                                     n_input_channel,
        #                                     n_output, n_filters,
        #                                     filter_size, n_units,
        #                                     reg_lambda,
        #                                     bidir=bidir,
        #                                     dropout=dropout)
        train_model, inf_enc, inf_dec = stacked_lstm_1Dcnn_model(
                                                n_input_time,
                                                n_input_channel,
                                                n_output, n_filters,
                                                filter_size, n_layers,
                                                n_units, reg_lambda,
                                                bidir=bidir,
                                                dropout=dropout)

        train_model.compile(optimizer=Adam(learning_rate),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        _, hist = train_seq2seq(train_model, X_train, X_prior_train,
                                y_train, epochs=epochs,
                                verbose=verbose)

        # test acc
        y_pred_test, labels_test = decode_seq2seq(inf_enc, inf_dec, X_test,
                                                  y_test)
        test_acc = balanced_accuracy_score(labels_test, y_pred_test)

        # with open(DATA_PATH + f'outputs/{pt}_acc.txt', 'a+') as f:
        #     # f.write(f'Final validation accuracy: {val_acc}, '
        #     #         f'Final test accuracy: {test_acc}, '
        #     #         f'True labels: {labels_test}, '
        #     #         f'Predicted labels: {y_pred_test}' + '\n')
        #     f.write(f'Final test accuracy: {test_acc}, '
        #             f'True labels: {labels_test}, '
        #             f'Predicted labels: {y_pred_test}' + '\n')
        if kfold:
            field_names = ['val_acc', 'test_acc', 'labels_test',
                           'y_pred_test']
        else:
            field_names = ['test_acc', 'labels_test', 'y_pred_test']

        if inputs['out_filename'] != '':
            acc_filename = DATA_PATH + 'outputs/' + inputs['out_filename'] \
                           + '.csv'
        else:
            if kfold:
                acc_filename = DATA_PATH + (f'outputs/{pt}{norm_ext}_acc_kfold'
                                            '.csv')
            else:
                acc_filename = DATA_PATH + f'outputs/{pt}{norm_ext}_acc.csv'

        with open(acc_filename, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            # writer.writerow([test_acc] + labels_test + y_pred_test)
            # writer.writeheader()
            if kfold:
                writer.writerow({'val_acc': val_acc, 'test_acc': test_acc,
                                 'labels_test': labels_test,
                                 'y_pred_test': y_pred_test})
            else:
                writer.writerow({'test_acc': test_acc,
                                 'labels_test': labels_test,
                                 'y_pred_test': y_pred_test})

        # plot_tf_hist_loss_acc(hist, save_fig=True,
        #                       save_path=DATA_PATH +
        #                       f'outputs/plots/{pt}_reg_train_{i+1}.png')

        if kfold:
            plot_accuracy_loss(k_hist, epochs=epochs, save_fig=True,
                               save_path=DATA_PATH +
                               f'outputs/plots/{pt}_kfold_train_{i+1}.png')


if __name__ == '__main__':
    train_rnn()
