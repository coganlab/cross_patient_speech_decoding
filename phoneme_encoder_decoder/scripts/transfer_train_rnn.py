"""
Script to train a RNN model on the DCC.
"""

import os
import sys
import argparse
from keras.optimizers import Adam
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, '..')

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import (pad_sequence_teacher_forcing,
                                                  decode_seq2seq)
from seq2seq_models.rnn_models import lstm_1Dcnn_model
from train.transfer_training import transfer_seq2seq_kfold_diff_chans
from visualization.plot_model_performance import plot_accuracy_loss


def init_parser():
    parser = argparse.ArgumentParser(description='Train RNN model on DCC')
    parser.add_argument('-p', '--pretrain_patient', type=str, default='S14',
                        required=False, help='Pretrain Patient ID')
    parser.add_argument('-t', '--transfer_patient', type=str, default='S33',
                        required=False, help='Transfer Patient ID')
    parser.add_argument('-sig', '--use_sig_channels', type=str, default='True',
                        required=False, help='Use significant channels (True)'
                        'or all channels (False)')
    parser.add_argument('-n', '--num_iter', type=int, default=5,
                        required=False, help='Number of times to run model')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        required=False, help='Verbosity of model training')
    parser.add_argument('-c', '--cluster', type=str, default='True',
                        required=False,
                        help='Run on cluster (True) or local (False)')
    return parser


def transfer_train_rnn():

    parser = init_parser()
    args = parser.parse_args()

    inputs = {}
    for key, val in vars(args).items():
        inputs[key] = val

    pretrain_pt = inputs['pretrain_patient']
    transfer_pt = inputs['transfer_patient']
    chan_ext = 'sigChannel' if inputs['use_sig_channels'] else 'all'
    n_iter = inputs['num_iter']
    verbose = inputs['verbose']
    cluster = inputs['cluster']

    if cluster.lower() == 'true':
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
                                                    f'{pretrain_pt}_HG_'
                                                    f'{chan_ext}.mat')

    tar_hg_trace, tar_hg_map, tar_phon_labels = get_high_gamma_data(
                                                    DATA_PATH +
                                                    f'{transfer_pt}/'
                                                    f'{transfer_pt}_HG_'
                                                    f'{chan_ext}.mat')

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
    filter_size = 10
    n_filters = 100
    n_units = 800
    reg_lambda = 1e-6
    bidir = True

    # Train model
    test_size = 0.2
    num_folds = 10
    num_reps = 3
    batch_size = 200
    epochs = 540
    learning_rate = 1e-3

    # Hold out test data set
    data_split = ShuffleSplit(n_splits=1, test_size=test_size, random_state=2)
    train_idx, test_idx = next(data_split.split(X2))
    X2_train, X2_test = X2[train_idx], X2[test_idx]
    X2_prior_train, X2_prior_test = X2_prior[train_idx], X2_prior[test_idx]
    y2_train, y2_test = y2[train_idx], y2[test_idx]

    for i in range(n_iter):
        print('==============================================================')
        print('Iteration: ', i+1)
        print('==============================================================')

        pre_model, pre_enc, pre_dec = lstm_1Dcnn_model(n_input_time,
                                                       n_input_channel_pre,
                                                       n_output, n_filters,
                                                       filter_size, n_units,
                                                       reg_lambda, bidir=bidir)
        tar_model, tar_enc, tar_dec = lstm_1Dcnn_model(n_input_time,
                                                       n_input_channel_trans,
                                                       n_output, n_filters,
                                                       filter_size, n_units,
                                                       reg_lambda, bidir=bidir)

        pre_model.compile(optimizer=Adam(learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        tar_model.compile(optimizer=Adam(learning_rate),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        t_hist, y_pred_all, y_test_all = transfer_seq2seq_kfold_diff_chans(
                                    pre_model, tar_model, tar_enc, tar_dec,
                                    X1, X1_prior, y1, X2_train, X2_prior_train,
                                    y2_train, num_folds=num_folds,
                                    batch_size=batch_size,
                                    fine_tune_epochs=epochs,
                                    num_reps=num_reps,
                                    verbose=verbose)

        # final val acc - preds from inf decoder across all folds
        val_acc = balanced_accuracy_score(y_test_all, y_pred_all)

        # test acc
        y_pred_test, labels_test = decode_seq2seq(tar_enc, tar_dec, X2_test,
                                                  y2_test)
        test_acc = balanced_accuracy_score(labels_test, y_pred_test)

        with open(DATA_PATH + f'outputs/transfer_{pretrain_pt}-{transfer_pt}'
                  '_acc.txt', 'a+') as f:
            f.write(f'Final validation accuracy: {val_acc}, '
                    f'Final test accuracy: {test_acc}' + '\n')

        plot_accuracy_loss(t_hist, epochs=epochs, save_fig=True,
                           save_path=DATA_PATH +
                           'outputs/plots/transfer_'
                           f'{pretrain_pt}-{transfer_pt}_train_all_{i+1}.png')


if __name__ == '__main__':
    transfer_train_rnn()
