"""
Script to train a RNN model on the DCC.
"""

import os
import sys
import argparse
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, '..')

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import pad_sequence_teacher_forcing
from processing_utils.data_saving import append_pkl_accs
from seq2seq_models.rnn_models import (stacked_lstm_1Dcnn_model,
                                       stacked_gru_1Dcnn_model)
from train.train import train_seq2seq_kfold
from visualization.plot_model_performance import plot_accuracy_loss


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
    epochs = 800
    learning_rate = 1e-3
    kfold_rand_state = 7

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

        k_hist, y_pred_all, y_test_all = train_seq2seq_kfold(
                                                train_model, inf_enc,
                                                inf_dec, X,
                                                X_prior, y,
                                                num_folds=num_folds,
                                                num_reps=num_reps,
                                                rand_state=kfold_rand_state,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                early_stop=False,
                                                verbose=verbose)

        # final val acc - preds from inf decoder across all folds
        val_acc = balanced_accuracy_score(y_test_all, y_pred_all)
        cmat = confusion_matrix(y_test_all, y_pred_all,
                                labels=range(1, n_output))

        if inputs['filename'] != '':
            acc_filename = DATA_PATH + 'outputs/' + inputs['filename'] \
                           + '.pkl'
        else:
            acc_filename = DATA_PATH + ('outputs/transfer_'
                                        f'{pt}{norm_ext}_acc_'
                                        f'{num_folds}fold.pkl')

        # save performance
        append_pkl_accs(acc_filename, val_acc, cmat)
        plot_accuracy_loss(k_hist, epochs=epochs, save_fig=True,
                           save_path=DATA_PATH +
                           (f'outputs/plots/{pt}'
                               f'_{num_folds}fold_train_{i+1}.png'))


if __name__ == '__main__':
    train_rnn()
