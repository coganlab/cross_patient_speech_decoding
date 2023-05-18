"""
Script to cross-patient transfer train a RNN model on the DCC.
"""

import os
import sys
import argparse
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import ShuffleSplit

sys.path.insert(0, '..')

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import (pad_sequence_teacher_forcing,
                                                  decode_seq2seq)
from processing_utils.data_saving import append_pkl_accs
from seq2seq_models.rnn_models import (stacked_lstm_1Dcnn_model,
                                       stacked_gru_1Dcnn_model)
from train.transfer_training import transfer_chain_kfold, transfer_train_chain
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


def transfer_chain():

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
    kfold = str2bool(inputs['k_fold'])
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
    kfold_rand_state = 7

    pre_epochs = 200  # 200
    conv_epochs = 60  # 60
    tar_epochs = 540  # 540
    total_epochs = len(chain_X_pre) * (pre_epochs + conv_epochs) + tar_epochs

    # TODO modify for multiple pretraining patients
    if not kfold:
        # Hold out test data set
        test_size = 0.2
        data_split = ShuffleSplit(n_splits=1, test_size=test_size,
                                  random_state=2)
        pre_splits = [data_split.split(x) for x in chain_X_pre]
        pre_inds = [next(s) for s in pre_splits]
        train_idx_pre, test_idx_pre = zip(*pre_inds)
        X_train_pre, X_test_pre = (
            [x[train_idx_pre[i]] for i, x in enumerate(chain_X_pre)],
            [x[test_idx_pre[i]] for i, x in enumerate(chain_X_pre)])
        X_prior_train_pre, X_prior_test_pre = (
            [xp[train_idx_pre[i]] for i, xp in enumerate(chain_X_prior_pre)],
            [xp[test_idx_pre[i]] for i, xp in enumerate(chain_X_prior_pre)])
        y_train_pre, y_test_pre = (
            [y[train_idx_pre[i]] for i, y in enumerate(chain_y_pre)],
            [y[test_idx_pre[i]] for i, y in enumerate(chain_y_pre)])

        train_idx, test_idx = next(data_split.split(chain_X_tar))
        X_train_tar, X_test_tar = chain_X_tar[train_idx], chain_X_tar[test_idx]
        X_prior_train_tar, X_prior_test_tar = (chain_X_prior_tar[train_idx],
                                               chain_X_prior_tar[test_idx])
        y_train_tar, y_test_tar = chain_y_tar[train_idx], chain_y_tar[test_idx]

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

        if kfold:
            k_hist, y_pred_all, y_test_all = transfer_chain_kfold(
                                                train_model, inf_enc, inf_dec,
                                                chain_X_pre, chain_X_prior_pre,
                                                chain_y_pre, chain_X_tar,
                                                chain_X_prior_tar,
                                                chain_y_tar,
                                                num_folds=num_folds,
                                                num_reps=num_reps,
                                                rand_state=kfold_rand_state,
                                                pretrain_epochs=pre_epochs,
                                                conv_epochs=conv_epochs,
                                                target_epochs=tar_epochs,
                                                batch_size=batch_size,
                                                verbose=verbose)

            # final val acc - preds from inf decoder across all folds
            acc = balanced_accuracy_score(y_test_all, y_pred_all)
            cmat = confusion_matrix(y_test_all, y_pred_all,
                                    labels=range(1, n_output))

            plot_accuracy_loss(k_hist, epochs=total_epochs, save_fig=True,
                               save_path=DATA_PATH +
                               (f'outputs/plots/transfer_'
                                f'[{"-".join(pretrain_list)}]-'
                                f'{target_pt}_'
                                f'{num_folds}fold_train_{i+1}.png'))

        else:
            train_model, inf_enc, _ = transfer_train_chain(
                                            train_model, inf_enc, inf_dec,
                                            X_train_pre, X_prior_train_pre,
                                            y_train_pre, X_train_tar,
                                            X_prior_train_tar, y_train_tar,
                                            pretrain_epochs=pre_epochs,
                                            conv_epochs=conv_epochs,
                                            target_epochs=tar_epochs,
                                            batch_size=batch_size,
                                            verbose=verbose)

            # test acc
            y_pred_test, labels_test = decode_seq2seq(inf_enc, inf_dec,
                                                      X_test_tar, y_test_tar)
            acc = balanced_accuracy_score(labels_test, y_pred_test)
            cmat = confusion_matrix(labels_test, y_pred_test,
                                    labels=range(1, n_output))

        if inputs['filename'] != '':
            acc_filename = DATA_PATH + 'outputs/' + inputs['filename'] \
                           + '.pkl'
        else:
            if kfold:
                acc_filename = DATA_PATH + ('outputs/transfer_'
                                            f'[{"-".join(pretrain_list)}]'
                                            f'-{target_pt}'
                                            f'{norm_ext}_acc_'
                                            f'{num_folds}fold.pkl')
            else:
                acc_filename = DATA_PATH + ('outputs/transfer_'
                                            f'[{"-".join(pretrain_list)}]'
                                            f'-{target_pt}'
                                            f'{norm_ext}_acc_'
                                            f'{test_size}-heldout.pkl')

        # save performance
        append_pkl_accs(acc_filename, acc, cmat, acc_key='val_acc' if kfold
                        else 'test_acc')


if __name__ == '__main__':
    transfer_chain()
