"""
Training functions for the phoneme encoder decoder model.

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""

from sklearn.model_selection import StratifiedKFold
from processing_utils.sequence_processing import phoneme_padding
from keras.callbacks import EarlyStopping


def train_kfold(model, X, y, num_folds=10, batch_size=32, epochs=100):

    seq_prior, _, _, _ = phoneme_padding(y, 10)

    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2)

    for train_ind, test_ind in cv.split(X, y):
        X_train, X_test = X[train_ind], X[test_ind]
        X_prior_train, X_prior_test = seq_prior[train_ind], seq_prior[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        model.fit([X_train, X_prior_train], y_train, batch_size=batch_size,
                  epochs=epochs, validation_data=([X_test, X_prior_test],
                                                  y_test),
                  callbacks=[EarlyStopping(monitor='val_loss', patience=50)])

    return model
