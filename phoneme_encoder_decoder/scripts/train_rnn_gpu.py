"""
Script to train a RNN model on the DCC.
"""

import os
import sys
import argparse
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, '..')

from processing_utils.feature_data_from_mat import get_high_gamma_data
from processing_utils.sequence_processing import pad_sequence_teacher_forcing
from seq2seq_models.rnn_models import lstm_1Dcnn_model
from train.train import train_seq2seq_kfold
from visualization.plot_model_performance import plot_accuracy_loss


def init_parser():
    parser = argparse.ArgumentParser(description='Train RNN model on DCC')
    parser.add_argument('-pt', '--patient', type=str, default='S14',
                        required=False, help='Patient ID')
    parser.add_argument('-sig', '--use_sig_channels', type=bool, default=True,
                        required=False, help='Use significant channels (True)'
                        'or all channels (False)')
    parser.add_argument('-n', '--num_iter', type=int, default=5,
                        required=False, help='Number of times to run model')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        required=False, help='Verbosity of model training')
    return parser

def train_rnn():
    parser = init_parser()
    args = parser.parse_args()

    HOME_PATH = os.path.expanduser('~')
    DATA_PATH = HOME_PATH + '/workspace/'
    # DATA_PATH = '../data/'

    inputs = {}
    for key, val in vars(args).items():
        inputs[key] = val

    pt = inputs['patient']
    chan_ext = 'sigChannel' if inputs['use_sig_channels'] else 'all'
    n_iter = inputs['num_iter']
    verbose = inputs['verbose']

    print('==================================================================')
    print("Training models for patient %s." % inputs['patient'])
    print("Saving outputs to %s." % (DATA_PATH + 'outputs/'))
    print('==================================================================')

    # Load in data from workspace mat files
    hg_trace, hg_map, phon_labels = get_high_gamma_data(DATA_PATH +
                                                        f'{pt}/{pt}_HG_'
                                                        f'{chan_ext}.mat')

    n_output = 10
    X = hg_trace  # use HG traces (n_trials, n_channels, n_timepoints) for CNN
    X_prior, y, _, _ = pad_sequence_teacher_forcing(phon_labels, n_output)

    # Build models
    n_input_time = X.shape[1]
    n_input_channel = X.shape[2]
    filter_size = 10
    n_filters = 100
    n_units = 800
    reg_lambda = 1e-6
    bidir = True
    train_model, inf_enc, inf_dec = lstm_1Dcnn_model(n_input_time,
                                                     n_input_channel,
                                                     n_output, n_filters,
                                                     filter_size, n_units,
                                                     reg_lambda, bidir=bidir)

    # Train model
    num_folds = 10
    num_reps = 3
    batch_size = 200
    epochs = 800
    learning_rate = 1e-3

    train_model.compile(optimizer=Adam(learning_rate),
                        loss='categorical_crossentropy', metrics=['accuracy'])


    for i in range(n_iter):
        print('==============================================================')
        print('Iteration: ', i+1)
        print('==============================================================')
        
        histories, y_pred_all, y_test_all = train_seq2seq_kfold(
                                                train_model, inf_enc, inf_dec,
                                                X, X_prior, y,
                                                num_folds=num_folds,
                                                num_reps=num_reps,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                early_stop=False,
                                                verbose=verbose)
        
        b_acc = balanced_accuracy_score(y_test_all, y_pred_all)
        with open(DATA_PATH + f'outputs/{pt}_acc.txt', 'a+') as f:
            f.write(str(b_acc) + '\n')

        plot_accuracy_loss(histories, epochs=epochs, save_fig=True,
                        save_path=DATA_PATH +
                                    f'outputs/plots/{pt}_train_all_{i+1}.png')
        

if __name__ == '__main__':
    train_rnn()
