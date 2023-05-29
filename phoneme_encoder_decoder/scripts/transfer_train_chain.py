"""
Script to cross-patient transfer train a RNN model on the DCC.
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
from train.transfer_training import transfer_chain_kfold, transfer_train_chain
from visualization.transfer_results_vis import plot_transfer_loss_acc


def init_parser():
    parser = argparse.ArgumentParser(description='Transfer train RNN on DCC')
    parser.add_argument('-p', '--pretrain_patients', nargs='*', type=str,
                        default=['S14'], required=False,
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
    print(f"Pretraining models on patient(s) {pretrain_list}.")
    print(f"Transferring models to patient {target_pt}.")
    print(f"Getting pretraining data from {DATA_PATH}{pretrain_list}/.")
    print(f"Getting transfer data from {DATA_PATH}{target_pt}/.")
    print(f"Saving outputs to {DATA_PATH + 'outputs/'}.")
    print('==================================================================')

    # Load in data from workspace mat files
    n_output = 10
    chain_X_pre, chain_X_prior_pre, chain_y_pre, chain_lab_pre = [], [], [], []
    for curr_pt in pretrain_list:
        if jitter:
            pre_hg_trace, pre_hg_map, pre_phon_labels = get_high_gamma_data(
                                                DATA_PATH + f'{curr_pt}/'
                                                f'{curr_pt}_HG'
                                                f'{chan_ext}{norm_ext}'
                                                '_extended_goodTrials.mat')
        else:
            pre_hg_trace, pre_hg_map, pre_phon_labels = get_high_gamma_data(
                                                DATA_PATH + f'{curr_pt}/'
                                                f'{curr_pt}_HG'
                                                f'{chan_ext}{norm_ext}'
                                                '_goodTrials.mat')

        X = pre_hg_trace  # (n_trials, n_channels, n_timepoints) for 1D CNN
        X_prior, y, _, seq_labels = pad_sequence_teacher_forcing(
                                                        pre_phon_labels,
                                                        n_output)
        chain_X_pre.append(X)
        chain_X_prior_pre.append(X_prior)
        chain_y_pre.append(y)
        chain_lab_pre.append(seq_labels)

    if jitter:
        tar_hg_trace, tar_hg_map, tar_phon_labels = get_high_gamma_data(
                                            DATA_PATH +
                                            f'{target_pt}/'
                                            f'{target_pt}_HG'
                                            f'{chan_ext}'
                                            f'{norm_ext}'
                                            '_extended_goodTrials.mat')
    else:
        tar_hg_trace, tar_hg_map, tar_phon_labels = get_high_gamma_data(
                                            DATA_PATH +
                                            f'{target_pt}/'
                                            f'{target_pt}_HG'
                                            f'{chan_ext}'
                                            f'{norm_ext}'
                                            '_goodTrials.mat')

    chain_X_tar = tar_hg_trace  # (n_trials, n_channels, n_timepoints)
    chain_X_prior_tar, chain_y_tar, _, tar_seq_labels = (
                                        pad_sequence_teacher_forcing(
                                                tar_phon_labels,
                                                n_output))

    # Model parameters
    win_len = 1  # 1 second decoding window
    fs = 200
    n_input_time = int(win_len * fs)
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
    pre_split = True

    # Augmentation parameters
    mixup_alpha = 5
    mixup_dicts = [] if mixup else None
    if mixup:
        for i in range(len(chain_X_pre)):
            mixup_dicts.append({'alpha': mixup_alpha,
                                'labels': chain_lab_pre[i]})
        mixup_dicts.append({'alpha': mixup_alpha,
                            'labels': tar_seq_labels})
    j_end = 0.5
    # define jitter by number of points
    n_jitter = 5
    if n_jitter % 2 == 0:
        n_jitter += 1  # +1 to include 0
    jitter_vals = np.linspace(-j_end, j_end, n_jitter)
    jitter_dict = ({'jitter_vals': jitter_vals, 'win_len': win_len, 'fs': fs}
                   if jitter else None)

    # Training parameters
    num_folds = 5  # 5
    num_reps = 3  # 3
    batch_size = 200
    learning_rate = 1e-3
    kfold_rand_state = 7

    # Transfer parameters
    early_stop = True
    n_iter_pre = 1
    pre_epochs = 200  # 200
    conv_epochs = 60  # 60
    tar_epochs = 540  # 540
    total_epochs = len(chain_X_pre) * (pre_epochs + conv_epochs) + tar_epochs

    if inputs['filename'] != '':
        acc_filename = DATA_PATH + 'outputs/' + inputs['filename'] + '.pkl'
        plot_filename = DATA_PATH + ('outputs/plots/' + inputs['filename']
                                     + '_train_%d.png')
    else:
        if kfold:
            acc_filename = DATA_PATH + ('outputs/'
                                        f'[{"-".join(pretrain_list)}]'
                                        f'-{target_pt}'
                                        f'{norm_ext}_acc_'
                                        f'{num_folds}fold.pkl')
            plot_filename = DATA_PATH + ('outputs/'
                                         f'[{"-".join(pretrain_list)}]'
                                         f'-{target_pt}'
                                         f'{norm_ext}'
                                         f'{num_folds}fold_train_%d.png')
        else:
            acc_filename = DATA_PATH + ('outputs/'
                                        f'[{"-".join(pretrain_list)}]'
                                        f'-{target_pt}'
                                        f'{norm_ext}_acc_'
                                        f'{test_size}-heldout.pkl')
            plot_filename = DATA_PATH + ('outputs/'
                                         f'[{"-".join(pretrain_list)}]'
                                         f'-{target_pt}'
                                         f'{norm_ext}'
                                         f'{test_size}-heldout_train_%d.png')

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
        lab_train_pre, lab_test_pre = (
            [l[train_idx_pre[i]] for i, l in enumerate(chain_lab_pre)],
            [l[test_idx_pre[i]] for i, l in enumerate(chain_lab_pre)]
        )

        for i in range(len(chain_X_pre)):
            X_train = X_train_pre[i]
            X_prior_train = X_prior_train_pre[i]
            y_train = y_train_pre[i]
            lab_train = lab_train_pre[i]
            if mixup:
                X_train, X_prior_train, y_train = augment_mixup(
                                                    X_train, X_prior_train,
                                                    y_train, lab_train,
                                                    alpha=mixup_alpha)
            if jitter:
                X_train, X_prior_train, y_train = augment_time_jitter(
                                                    X_train, X_prior_train,
                                                    y_train, jitter_vals,
                                                    win_len,
                                                    fs)
            X_train_pre[i] = X_train
            X_prior_train_pre[i] = X_prior_train
            y_train_pre[i] = y_train

        train_idx, test_idx = next(data_split.split(chain_X_tar))
        X_train_tar, X_test_tar = chain_X_tar[train_idx], chain_X_tar[test_idx]
        X_prior_train_tar, X_prior_test_tar = (chain_X_prior_tar[train_idx],
                                               chain_X_prior_tar[test_idx])
        y_train_tar, y_test_tar = chain_y_tar[train_idx], chain_y_tar[test_idx]
        lab_train_tar, lab_test_tar = (tar_seq_labels[train_idx],
                                       tar_seq_labels[test_idx])
        if mixup:
            X_train_tar, X_prior_train_tar, y_train_tar = augment_mixup(
                                                    X_train_tar,
                                                    X_prior_train_tar,
                                                    y_train_tar, lab_train_tar,
                                                    alpha=mixup_alpha)
        if jitter:
            X_train_tar, X_prior_train_tar, y_train_tar = augment_time_jitter(
                                                            X_train_tar,
                                                            X_prior_train_tar,
                                                            y_train_tar,
                                                            jitter_vals,
                                                            win_len,
                                                            fs)

    param_keys = ['model_type', 'filter_size', 'n_filters', 'n_units',
                  'n_layers', 'reg_lambda', 'dropout', 'bidir', 'mixup_alpha',
                  'n_jitter', 'num_folds', 'num_reps', 'pre_epochs',
                  'conv_epochs', 'tar_epochs', 'learning_rate',
                  'kfold_rand_state']
    param_vals = [model_type, filter_size, n_filters, n_units, n_layers,
                  reg_lambda, dropout, bidir, mixup_alpha, n_jitter, num_folds,
                  num_reps, pre_epochs, conv_epochs, tar_epochs, learning_rate,
                  kfold_rand_state]
    save_pkl_params(acc_filename, dict_from_lists(param_keys, param_vals))

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
                                                pre_split=pre_split,
                                                rand_state=kfold_rand_state,
                                                mixup_data=mixup_dicts,
                                                jitter_data=jitter_dict,
                                                pretrain_epochs=pre_epochs,
                                                conv_epochs=conv_epochs,
                                                target_epochs=tar_epochs,
                                                early_stop=early_stop,
                                                n_iter_pre=n_iter_pre,
                                                batch_size=batch_size,
                                                verbose=verbose)

            # final val acc - preds from inf decoder across all folds
            acc = balanced_accuracy_score(y_test_all, y_pred_all)
            cmat = confusion_matrix(y_test_all, y_pred_all,
                                    labels=range(1, n_output))

            plot_transfer_loss_acc(k_hist, pre_epochs, conv_epochs,
                                   tar_epochs, len(chain_X_pre),
                                   n_iter_pre*pretrain_list+[target_pt],
                                   save_fig=True,
                                   save_path=plot_filename % (i+1))

        else:
            train_model, inf_enc, _ = transfer_train_chain(
                                            train_model, inf_enc, inf_dec,
                                            X_train_pre, X_prior_train_pre,
                                            y_train_pre, X_train_tar,
                                            X_prior_train_tar, y_train_tar,
                                            pretrain_epochs=pre_epochs,
                                            conv_epochs=conv_epochs,
                                            target_epochs=tar_epochs,
                                            early_stop=early_stop,
                                            n_iter_pre=n_iter_pre,
                                            batch_size=batch_size,
                                            verbose=verbose)

            # test acc
            y_pred_test, labels_test = decode_seq2seq(inf_enc, inf_dec,
                                                      X_test_tar, y_test_tar)
            acc = balanced_accuracy_score(labels_test, y_pred_test)
            cmat = confusion_matrix(labels_test, y_pred_test,
                                    labels=range(1, n_output))

        # save performance
        append_pkl_accs(acc_filename, acc, cmat, acc_key='val_acc' if kfold
                        else 'test_acc')


if __name__ == '__main__':
    transfer_chain()
