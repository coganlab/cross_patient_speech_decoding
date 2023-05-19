"""
Script to optimize hyperparameters for RNN model on the DCC cluster.
"""

import os
import sys
import argparse
from contextlib import redirect_stdout
import keras_tuner as kt
from sklearn.model_selection import ShuffleSplit

# too dumb to figure out python packages/importing between folders at the same
# level so need this to import functions below RIP pycodestyle error :)
sys.path.insert(0, '..')

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import pad_sequence_teacher_forcing
from train.optimize import encDecHyperModel, encDecTuner


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
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        required=False, help='Verbosity of model training')
    parser.add_argument('-or', '--oracle', type=str, default='random',
                        required=False, help='Random search (random) or '
                        'Bayesian optimization (bayesian)')
    parser.add_argument('-c', '--cluster', type=str, default='True',
                        required=False,
                        help='Run on cluster (True) or local (False)')
    return parser


def str2bool(s):
    return s.lower() == 'true'


def hyperparam_optim():
    parser = init_parser()
    args = parser.parse_args()

    inputs = {}
    for key, val in vars(args).items():
        inputs[key] = val

    pt = inputs['patient']
    chan_ext = '_sigChannel' if str2bool(inputs['sig_channels']) else '_all'
    norm_ext = '_zscore' if str2bool(inputs['z_score']) else ''
    verbose = inputs['verbose']
    cluster = str2bool(inputs['cluster'])
    oracle_type = inputs['oracle']

    if oracle_type not in ['random', 'bayesian']:
        raise ValueError('Tuner oracle input must be "random" or "bayesian"')

    if cluster:
        HOME_PATH = os.path.expanduser('~')
        DATA_PATH = HOME_PATH + '/workspace/'
    else:
        DATA_PATH = '../data/'

    print('==================================================================')
    print("Optimizing hyperparameters for patient %s." % pt)
    print("Getting data from %s." % (DATA_PATH + f'{pt}/'))
    print("Saving outputs to %s." % (DATA_PATH + ('rnn_tuning/'
                                                   f'{pt}_1Dcnn_EncDec_'
                                                   f'{oracle_type}/')))
    print('==================================================================')

    # Load in data from workspace mat files
    hg_trace, hg_map, phon_labels = get_high_gamma_data(DATA_PATH +
                                                        f'{pt}/{pt}_HG'
                                                        f'{chan_ext}'
                                                        f'{norm_ext}'
                                                        '_goodTrials.mat')

    n_output = 10
    X = hg_trace  # use HG traces (n_trials, n_channels, n_timepoints)
    X_prior, y, _, _ = pad_sequence_teacher_forcing(phon_labels, n_output)

    n_input_time = X.shape[1]
    n_input_channel = X.shape[2]

    # test set definition for reporting
    test_size = 0.2
    data_split = ShuffleSplit(n_splits=1, test_size=test_size, random_state=2)
    train_idx, test_idx = next(data_split.split(X))
    X_train, X_test = X[train_idx], X[test_idx]
    X_prior_train, X_prior_test = X_prior[train_idx], X_prior[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    dropout = 0.33
    bidir = True

    # keras tuner optimization
    max_optim_trials = 400
    tuning_dir = 'rnn_tuning'
    project_dir = f'{pt}_1Dcnn_EncDec_train_params_{oracle_type}'
    # tuning_epochs = 800
    hyper_model = encDecHyperModel(n_input_time, n_input_channel, n_output,
                                   dropout=dropout, bidir=bidir)
    obj_name = 'seq2seq_val_accuracy'

    if oracle_type == 'random':
        oracle = kt.oracles.RandomSearchOracle(
                    objective=kt.Objective(obj_name, direction='max'),
                    max_trials=max_optim_trials)
    elif oracle_type == 'bayesian':
        oracle = kt.oracles.BayesianOptimizationOracle(
                    objective=kt.Objective(obj_name, direction='max'),
                    max_trials=max_optim_trials)

    rnn_optimizer = encDecTuner(hypermodel=hyper_model, oracle=oracle,
                                directory=DATA_PATH + tuning_dir,
                                project_name=project_dir)
    # rnn_optimizer.search(X_train, X_prior_train, y_train, epochs=tuning_epochs)
    rnn_optimizer.search(X_train, X_prior_train, y_train)

    # save trial data to text file
    summary_path = DATA_PATH + tuning_dir + '/' + project_dir +\
        f'/optim_summary_{max_optim_trials}.txt'
    with open(summary_path, 'w+') as f:
        with redirect_stdout(f):
            rnn_optimizer.results_summary(num_trials=50)


if __name__ == '__main__':
    hyperparam_optim()
