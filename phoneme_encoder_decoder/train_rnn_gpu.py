"""
Script to train a RNN model on the DCC cluster.
"""

from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import pad_sequence_teacher_forcing
from seq2seq_models.rnn_models import lstm_1Dcnn_model
from train.train import train_seq2seq_kfold
from visualization.plot_model_performance import plot_accuracy_loss

# DATA_PATH = '~/workspace/'
DATA_PATH = 'data/'

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

# Train model
num_folds = 2
batch_size = 200
epochs = 1
learning_rate = 5e-4

train_model.compile(optimizer=Adam(learning_rate),
                    loss='categorical_crossentropy', metrics=['accuracy'])
histories, y_pred_all, y_test_all = train_seq2seq_kfold(train_model, inf_enc,
                                                        inf_dec, X, X_prior, y,
                                                        num_folds=num_folds,
                                                        batch_size=batch_size,
                                                        epochs=epochs,
                                                        early_stop=False)

# Save outputs
b_acc = balanced_accuracy_score(y_test_all, y_pred_all)
with open(DATA_PATH + 'outputs/train1_b_acc.txt', 'w+') as f:
    f.write(str(b_acc))

plot_accuracy_loss(histories, epochs=epochs, save_fig=True,
                   save_path=DATA_PATH + 'outputs/plots/train1_plot.png')
