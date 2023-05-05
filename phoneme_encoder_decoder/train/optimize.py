"""
Hyperparameter optimization for the phoneme encoder-decoder model. Implemented
via KerasTuner

Author: Zac Spalding. Adapted from code by Kumar Duraivel.
"""

import keras_tuner as kt
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score

from ..seq2seq_models.rnn_models import (stacked_lstm_1Dcnn_model,
                                         stacked_gru_1Dcnn_model)
from train.train import train_seq2seq_kfold


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
        learning_rate = 1e-3

        # l_rates = [1e-3, 1e-4, 1e-5, 1e-6]
        models = hp.Choice('model_type', values=model_list)
        n_filters = hp.Int('num_filts', min_value=10, max_value=200, step=10)
        n_layers = hp.Int('num_layers', min_value=1, max_value=3, step=1)
        # filter_size = hp.Int('filt_size', min_value=3, max_value=10, step=1)
        n_units = hp.Choice('rnn_units', values=unit_vals)
        reg_lambda = hp.Choice('reg_lambda', values=reg_vals)
        # learning_rate = hp.Choice('learning_rate', values=l_rates)

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


class encDecTuner(kt.Tuner):
    def run_trial(self, trial, X, X_prior, y, *args, **kwargs):
        hp = trial.hyperparameters
        batch_size = X.shape[0]
        # create model with hyperparameter space
        model, inf_enc, inf_dec = self.hypermodel.build(hp)

        # train model over hyperparemters
        _, y_pred, y_test = self.hypermodel.fit(model, inf_enc, inf_dec,
                                                X, X_prior, y, *args,
                                                batch_size=batch_size,
                                                **kwargs)
        # evaluate through validation accuracy
        val_accuracy = balanced_accuracy_score(y_test, y_pred)
        self.oracle.update_trial(trial.trial_id,
                                 {'seq2seq_val_accuracy': val_accuracy})
