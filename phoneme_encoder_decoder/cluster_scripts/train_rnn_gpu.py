"""
Script to train a RNN model on the DCC cluster.
"""

from ..processing_utils.feature_data_from_mat import get_high_gamma_data
from ..processing_utils.sequence_processing import pad_sequence_teacher_forcing
from ..seq2seq_models.rnn_models import lstm_1Dcnn_model

DATA_PATH = '~/workspace/'

# Load in data from workspace mat files
hg_trace, hg_map, phon_labels = get_high_gamma_data(DATA_PATH +
                                                    'S14/S14_HG_sigChannel')
n_output = 10
X = hg_trace  # use HG traces (n_trials, n_channels, n_timepoints) for 1D CNN
X_prior, y, _, _ = pad_sequence_teacher_forcing(phon_labels, n_output)

# Build models
n_input_time = X.shape[1]
n_input_channel = X.shape[2]
filter_size = 10
n_filters = 100
n_units = 800
reg_lambda = 1e-6
bidir = False
train_model, inf_enc, inf_dec = lstm_1Dcnn_model(n_input_time, n_input_channel,
                                                 n_output, n_filters,
                                                 filter_size, n_units,
                                                 reg_lambda, bidir=bidir)

# Train models

# Save outputs
