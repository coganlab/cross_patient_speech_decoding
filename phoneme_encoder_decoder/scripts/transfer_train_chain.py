"""
Script to cross-patient transfer train a RNN model on the DCC.
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
from train.transfer_training import transfer_chain_kfold
from visualization.plot_model_performance import plot_accuracy_loss


def init_parser():
    parser = argparse.ArgumentParser(description='Transfer train RNN on DCC')
    parser.add_argument('-p', '--pretrain_patients', nargs='*', type=str,
                        default='S14', required=False,
                        help='Pretrain Patient ID')
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


def transfer_train_chain():

    parser = init_parser()
    args = parser.parse_args()

    inputs = {}
    for key, val in vars(args).items():
        inputs[key] = val

    pretrain_list = inputs['pretrain_patients']
    target_pt = inputs['transfer_patient']
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
    print(f"Pretraining models on patient(s) {pretrain_list}.")
    print(f"Transferring models to patient {target_pt}.")
    print(f"Getting pretraining data from {DATA_PATH}{pretrain_list}/.")
    print(f"Getting transfer data from {DATA_PATH}{target_pt}/.")
    print(f"Saving outputs to {DATA_PATH + 'outputs/'}.")
    print('==================================================================')

    # Load in data from workspace mat files
    n_output = 10
    chain_X_pre, chain_X_prior_pre, chain_y_pre = [], [], []
    for curr_pt in pretrain_list:
        pre_hg_trace, pre_hg_map, pre_phon_labels = get_high_gamma_data(
                                                        DATA_PATH +
                                                        f'{curr_pt}/'
                                                        f'{curr_pt}_HG'
                                                        f'{chan_ext}'
                                                        f'{norm_ext}'
                                                        '_goodTrials.mat')
        X = pre_hg_trace  # (n_trials, n_channels, n_timepoints) for 1D CNN
        X_prior, y, _, _ = pad_sequence_teacher_forcing(pre_phon_labels,
                                                        n_output)
        chain_X_pre.append(X)
        chain_X_prior_pre.append(X_prior)
        chain_y_pre.append(y)

    tar_hg_trace, tar_hg_map, tar_phon_labels = get_high_gamma_data(
                                                    DATA_PATH +
                                                    f'{target_pt}/'
                                                    f'{target_pt}_HG'
                                                    f'{chan_ext}'
                                                    f'{norm_ext}'
                                                    '_goodTrials.mat')

    chain_X_tar = tar_hg_trace  # (n_trials, n_channels, n_timepoints)
    chain_X_prior_tar, chain_y_tar, _, _ = pad_sequence_teacher_forcing(
                                                tar_phon_labels,
                                                n_output)

    # Build models
    n_input_time = chain_X_tar.shape[1]
    n_input_channel_pre = chain_X_pre[0].shape[-1]
    model_type = 'lstm'
    model_fcn = stacked_gru_1Dcnn_model if model_type == 'gru' else \
        stacked_lstm_1Dcnn_model
    filter_size = 10
    n_filters = 50  # S14=100, S26=90
    n_layers = 1  # 1
    n_units = 256  # S14=800, S26=900
    reg_lambda = 1e-6  # S14=1e-6, S26=1e-5
    dropout = 0.33
    bidir = True

    # Train model
    num_folds = 5  # 5
    num_reps = 3  # 3
    batch_size = 200
    learning_rate = 1e-3

    pre_epochs = 200  # 200
    conv_epochs = 60  # 60
    tar_epochs = 540  # 540
    total_epochs = len(chain_X_pre) * (pre_epochs + conv_epochs) + tar_epochs

    for i in range(n_iter):
        print('==============================================================')
        print('Iteration: ', i+1)
        print('==============================================================')

        train_model, inf_enc, inf_dec = model_fcn(n_input_time,
                                                  n_input_channel_pre,
                                                  n_output, n_filters,
                                                  filter_size, n_layers,
                                                  n_units, reg_lambda,
                                                  bidir=bidir,
                                                  dropout=dropout)

        train_model.compile(optimizer=Adam(learning_rate),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        k_hist, y_pred_all, y_test_all = transfer_chain_kfold(
                                            train_model, inf_enc, inf_dec,
                                            chain_X_pre, chain_X_prior_pre,
                                            chain_y_pre, chain_X_tar,
                                            chain_X_prior_tar,
                                            chain_y_tar,
                                            num_folds=num_folds,
                                            num_reps=num_reps,
                                            batch_size=batch_size,
                                            pretrain_epochs=pre_epochs,
                                            conv_epochs=conv_epochs,
                                            target_epochs=tar_epochs,
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
                                        f'[{"-".join(pretrain_list)}]'
                                        f'-{target_pt}'
                                        f'{norm_ext}_acc_{num_folds}fold.pkl')

        # save performance
        append_pkl_accs(acc_filename, val_acc, cmat)
        plot_accuracy_loss(k_hist, epochs=total_epochs, save_fig=True,
                           save_path=DATA_PATH +
                           (f'outputs/plots/transfer_'
                            f'[{"-".join(pretrain_list)}]-'
                            f'{target_pt}_{num_folds}fold_train_{i+1}.png'))


if __name__ == '__main__':
    transfer_train_chain()
