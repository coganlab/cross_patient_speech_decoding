"""
Hyperparameter optimization for the phoneme encoder-decoder model. Implemented
via KerasTuner

Author: Zac Spalding. Adapted from code by Kumar Duraivel.
"""

import keras_tuner as kt
from keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score

from train.train import train_seq2seq_kfold


class encDecHyperModel(kt.HyperModel):

    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = model  # model defining function from rnn_models.py
        self.model_args = args  # model defining function args
        self.model_kwargs = kwargs  # model defining function kwargs

    def build(self, hp):
        # define hyperparameters
        reg_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        l_rates = [1e-3, 1e-4, 1e-5, 1e-6]
        n_filters = hp.Int('num_filts', min_value=10, max_value=200, step=10)
        filter_size = hp.Int('filt_size', min_value=3, max_value=10, step=1)
        rnn_units = hp.Int('rnn_units', min_value=100, max_value=800, step=100)
        reg_lambda = hp.Choice('reg_lambda', values=reg_vals)
        learning_rate = hp.Choice('learning_rate', values=l_rates)

        # build model (args define static model parameters like output dim)
        train, inf_enc, inf_dec = self.model(*self.model_args, n_filters,
                                             filter_size, rnn_units,
                                             reg_lambda, **self.model_kwargs)
        train.compile(optimizer=Adam(learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return train, inf_enc, inf_dec

    def fit(self, hp, model, *args, **kwargs):
        # return train_seq2seq_kfold(model, *args,
        #                            batch_size=hp.Choice('batch_size',
        #                                                 values=[32, 64, 128,
        #                                                         160]),
        #                            **kwargs)
        return train_seq2seq_kfold(model, *args, **kwargs)


class encDecTuner(kt.Tuner):
    def run_trial(self, trial, X, X_prior, y, *args, **kwargs):
        hp = trial.hyperparameters
        batch_hp = hp.Choice('batch_size', values=[32, 64, 128, 160])
        # create model with hyperparameter space
        model, inf_enc, inf_dec = self.hypermodel.build(hp)

        # train model over hyperparemters
        _, _, y_pred, y_test = self.hypermodel.fit(hp, model, inf_enc, inf_dec,
                                                   X, X_prior, y, *args,
                                                   batch_size=batch_hp,
                                                   **kwargs)
        # evaluate through validation accuracy
        val_accuracy = balanced_accuracy_score(y_test, y_pred)
        self.oracle.update_trial(trial.trial_id,
                                 {'val_accuracy': val_accuracy})
