"""
Script to train a RNN model on the DCC cluster.
"""

import os
import sys
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, '..')

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import pad_sequence_teacher_forcing
from seq2seq_models.rnn_models import lstm_1Dcnn_model
from train.transfer_training import (transfer_seq2seq_kfold, 
                                     transfer_seq2seq_kfold_diff_chans)
from visualization.plot_model_performance import plot_accuracy_loss

HOME_PATH = os.path.expanduser('~')
DATA_PATH = HOME_PATH + '/workspace/'
# DATA_PATH = '../data/'

pretrain_pt = 'S14'
transfer_pt = 'S33'

# Load in data from workspace mat files
pre_hg_trace, pre_hg_map, pre_phon_labels = get_high_gamma_data(DATA_PATH +
                                                    pretrain_pt + '/' +
                                                    pretrain_pt +
                                                    '_HG_all'
                                                    '.mat')

tar_hg_trace, tar_hg_map, tar_phon_labels = get_high_gamma_data(DATA_PATH +
                                                    transfer_pt + '/' +
                                                    transfer_pt +
                                                    '_HG_all'
                                                    '.mat')

n_output = 10
X1 = pre_hg_trace  # use HG traces (n_trials, n_channels, n_timepoints) for 1D CNN
X1_prior, y1, _, _ = pad_sequence_teacher_forcing(pre_phon_labels, n_output)

X2 = tar_hg_trace  # use HG traces (n_trials, n_channels, n_timepoints) for 1D CNN
X2_prior, y2, _, _ = pad_sequence_teacher_forcing(tar_phon_labels, n_output)

# Build models
n_input_time = X1.shape[1]
n_input_channel = X1.shape[2]
n_input_channel_trans = X2.shape[2]
filter_size = 10
n_filters = 100
n_units = 800
reg_lambda = 1e-6
bidir = True

pre_model, pre_enc, pre_dec = lstm_1Dcnn_model(n_input_time, n_input_channel,
                                                 n_output, n_filters,
                                                 filter_size, n_units,
                                                 reg_lambda, bidir=bidir)
tar_model, tar_enc, tar_dec = lstm_1Dcnn_model(n_input_time,
                                               n_input_channel_trans,
                                               n_output, n_filters,
                                               filter_size, n_units,
                                               reg_lambda, bidir=bidir)

# Train model
num_folds = 10
num_reps = 3
batch_size = 200
epochs = 540
learning_rate = 5e-5

pre_model.compile(optimizer=Adam(learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
tar_model.compile(optimizer=Adam(learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

n_iter = 5 # for accuracy distribution
accs = []
for i in range(n_iter):
    print('Iteration: ', i+1)
    t_hist, y_pred, y_test = transfer_seq2seq_kfold_diff_chans(
                                pre_model, tar_model, tar_enc, tar_dec,
                                X1, X1_prior, y1, X2, X2_prior, y2,
                                num_folds=num_folds, num_reps=num_reps)
    b_acc = balanced_accuracy_score(y_test, y_pred)
    accs.append(b_acc)

    plot_accuracy_loss(t_hist, epochs=epochs, save_fig=True,
                       save_path=DATA_PATH +
                       f'outputs/plots/transfer_train_S14-S33_{i+1}.png')

# Save outputs
with open(DATA_PATH + 'outputs/transfer_S14-S33_accs.txt', 'w+') as f:
    for acc in accs:
        f.write(str(acc) + '\n')
