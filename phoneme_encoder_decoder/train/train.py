"""
Training functions for the phoneme encoder decoder model.

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from processing_utils.sequence_processing import phoneme_padding, pre


def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the
    weight tensors (i.e., the weights have the same distribution along each
    dimension).

    TAKEN FROM: jkleint's (https://gist.github.com/jkleint) answer on Github
    (https://github.com/keras-team/keras/issues/341)

    Args:
        model (Model): Model whose weights will be shuffled.
        weights (list(ndarray), optional):  The model's weights will be
            replaced by a random permutation of these weights.
            Defaults to None.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)


def train_seq2seq_kfold(model, infenc, infdec, X, y, num_folds=10,
                        batch_size=32, epochs=800, early_stop=True):

    # save initial weights to reset models for each fold
    init_train_w = model.get_weights()

    # create padded sequences for teacher forcing
    seq_prior, _, _, _ = phoneme_padding(y, 10)

    # define k-fold cross validation
    cv = StratifiedKFold(n_splits=num_folds, shuffle=True)

    # create callbacks for early stopping (model checkpoint included to get
    # best model weights since early stopping returns model after patience
    # epochs, where performance may have started to decline)
    cb = None
    if early_stop:
        es = EarlyStopping(monitor='val_loss', patience=50)
        mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max',
                             save_best_only=True)
        cb = [es, mc]

    # dictionary for tracking history of each fold
    histories = {'accuracy': [], 'loss': [], 'val_accracy': [], 'val_loss': []}

    # cv training
    y_pred_all, y_test_all = [], []
    for train_ind, test_ind in cv.split(X, y):
        X_train, X_test = X[train_ind], X[test_ind]
        X_prior_train, X_prior_test = seq_prior[train_ind], seq_prior[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        # reset model weights for current fold (also resets associated
        # inference weights)
        shuffle_weights(model, weights=init_train_w)

        history = model.fit([X_train, X_prior_train], y_train,
                            batch_size=batch_size, epochs=epochs,
                            validation_data=([X_test, X_prior_test], y_test),
                            callbacks=cb)
        saved_model = load_model('best_model.h5')
        # TODO - figure out how to get inference weights from saved model

        histories['accuracy'].append(history.history['accuracy'])
        histories['loss'].append(history.history['loss'])
        histories['val_accuracy'].append(history.history['val_accuracy'])
        histories['val_loss'].append(history.history['val_loss'])

    return model, histories
