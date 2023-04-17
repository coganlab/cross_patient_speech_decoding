"""
Script to optimize hyperparameters for RNN model on the DCC cluster.
"""

import os
import sys
from contextlib import redirect_stdout
from keras.optimizers import Adam
import keras_tuner as kt
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ShuffleSplit

# too dumb to figure out python packages/importing between folders at the same
# level so need this to import functions below RIP pycodestyle error :)
sys.path.insert(0, '..')

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import pad_sequence_teacher_forcing
from seq2seq_models.rnn_models import lstm_1Dcnn_model
from train.optimize import encDecHyperModel, encDecTuner
from visualization.plot_model_performance import plot_accuracy_loss

HOME_PATH = os.path.expanduser('~')
DATA_PATH = HOME_PATH + '/workspace/'
# DATA_PATH = '../data/'

# Load in data from workspace mat files
hg_trace, hg_map, phon_labels = get_high_gamma_data(DATA_PATH +
                                                    'S14/S14_HG_sigChannel'
                                                    '.mat')
n_output = 10
X = hg_trace  # use HG traces (n_trials, n_channels, n_timepoints) for 1D CNN
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

# keras tuner optimization
optim_trials = 5
tuning_dir = 'rnn_tuning'
project_dir = 'S14_1Dcnn_LSTM_v6'
tuning_epochs = 800
hyper_model = encDecHyperModel(lstm_1Dcnn_model, n_input_time, n_input_channel,
                               n_output)

# oracle = kt.oracles.RandomSearchOracle(
#              objective=kt.Objective('val_accuracy', direction='max'),
#              max_trials=optim_trials)
oracle = kt.oracles.BayesianOptimizationOracle(
             objective=kt.Objective('val_accuracy', direction='max'),
             max_trials=optim_trials)

rnn_optimizer = encDecTuner(hypermodel=hyper_model, oracle=oracle,
                            directory=DATA_PATH + tuning_dir,
                            project_name=project_dir)
rnn_optimizer.search(X_train, X_prior_train, y_train, epochs=tuning_epochs)

# save trial data to text file
summary_path = DATA_PATH + tuning_dir + '/' + project_dir +\
               '/optim_summary.txt'
with open(summary_path, 'w+') as f:
    with redirect_stdout(f):
        rnn_optimizer.results_summary()