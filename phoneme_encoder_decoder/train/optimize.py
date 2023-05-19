"""
Hyperparameter optimization for the phoneme encoder-decoder model. Implemented
via KerasTuner

Author: Zac Spalding. Adapted from code by Kumar Duraivel.
"""

import sys
import keras_tuner as kt
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, '..')

from seq2seq_models.rnn_models import (stacked_lstm_1Dcnn_model,
                                       stacked_gru_1Dcnn_model)
from train.train import train_seq2seq_kfold
from train.transfer_training import transfer_chain_kfold


class encDecHyperModel(kt.HyperModel):

    def __init__(self, *args, **kwargs):
        super().__init__()
        # self.model = model  # model defining function from rnn_models.py
        self.model_args = args  # model defining function args
        self.model_kwargs = kwargs  # model defining function kwargs

    def build(self, hp):
        # define hyperparameters
        model_list = ['lstm', 'gru']
        unit_vals = [64, 128, 256, 512, 800, 1024]
        reg_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        filter_size = 10
        l_rates = [1e-3, 1e-4, 1e-5, 1e-6]
        # learning_rate = 1e-3

        models = hp.Choice('model_type', values=model_list)
        n_filters = hp.Int('num_filts', min_value=10, max_value=200, step=10)
        n_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
        # filter_size = hp.Int('filt_size', min_value=3, max_value=10, step=1)
        n_units = hp.Choice('rnn_units', values=unit_vals)
        reg_lambda = hp.Choice('reg_lambda', values=reg_vals)
        learning_rate = hp.Choice('learning_rate', values=l_rates)

        # build model (args define static model parameters like output dim)
        train, inf_enc, inf_dec = self.select_model(models, *self.model_args,
                                                    n_filters, filter_size,
                                                    n_layers, n_units,
                                                    reg_lambda,
                                                    **self.model_kwargs)
        # train, inf_enc, inf_dec = self.model(*self.model_args, n_filters,
        #                                      filter_size, rnn_units,
        #                                      reg_lambda, **self.model_kwargs)
        train.compile(optimizer=Adam(learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return train, inf_enc, inf_dec

    def select_model(self, model_type, *args, **kwargs):
        if model_type == 'lstm':
            return stacked_lstm_1Dcnn_model(*args, **kwargs)
        elif model_type == 'gru':
            return stacked_gru_1Dcnn_model(*args, **kwargs)

    def fit(self, model, *args, **kwargs):
        return train_seq2seq_kfold(model, *args, verbose=0, **kwargs)

    def fit_transfer(self, model, *args, **kwargs):
        return transfer_chain_kfold(model, *args, verbose=0, **kwargs)


class encDecTuner(kt.Tuner):
    def run_trial(self, trial, X, X_prior, y, **kwargs):
        hp = trial.hyperparameters
        batch_size = X.shape[0]

        epochs = hp.Int('epochs', min_value=100, max_value=800, step=100)
        # epochs = 800

        # create model with hyperparameter space
        model, inf_enc, inf_dec = self.hypermodel.build(hp)

        # train model over hyperparemters
        _, y_pred, y_test = self.hypermodel.fit(model, inf_enc, inf_dec,
                                                X, X_prior, y,
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                **kwargs)
        # evaluate through validation accuracy
        val_accuracy = balanced_accuracy_score(y_test, y_pred)
        self.oracle.update_trial(trial.trial_id,
                                 {'seq2seq_val_accuracy': val_accuracy})


class encDecTransferTuner(kt.Tuner):
    def run_trial(self, trial, X_pre, X_prior_pre, y_pre, X_tar, X_prior_tar,
                  y_tar, **kwargs):
        hp = trial.hyperparameters

        # optimize stage epochs as percents of reg epochs or absolute values?
        pre_epochs = hp.Int('pretrain_epochs', min_value=50, max_value=300,
                            step=50)
        conv_epochs = hp.Int('conv_epochs', min_value=50, max_value=200,
                             step=50)
        tar_epochs = hp.Int('target_epochs', min_value=100, max_value=800,
                            step=100)
        # epochs = 800

        # create model with hyperparameter space
        model, inf_enc, inf_dec = self.hypermodel.build(hp)

        # train model over hyperparemters
        _, y_pred, y_test = self.hypermodel.fit_transfer(
                                                model, inf_enc, inf_dec, X_pre,
                                                X_prior_pre, y_pre, X_tar,
                                                X_prior_tar, y_tar,
                                                pretrain_epochs=pre_epochs,
                                                conv_epochs=conv_epochs,
                                                target_epochs=tar_epochs,
                                                **kwargs)
        # evaluate through validation accuracy
        val_accuracy = balanced_accuracy_score(y_test, y_pred)
        self.oracle.update_trial(trial.trial_id,
                                 {'seq2seq_val_accuracy': val_accuracy})
