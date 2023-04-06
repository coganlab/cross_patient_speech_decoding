"""
Training functions for the phoneme encoder decoder model.

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping


from processing_utils.sequence_processing import (seq2seq_predict_batch,
                                                  one_hot_decode_batch)


def shuffle_weights(model, weights=None, layer_idx=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the
    weight tensors (i.e., the weights have the same distribution along each
    dimension). MODIFICATION: Added layer_idx argument to allow selection of
    specific layer to shuffle weights for.

    TAKEN FROM: jkleint's (https://gist.github.com/jkleint) answer on Github
    (https://github.com/keras-team/keras/issues/341)

    Args:
        model (Model): Model whose weights will be shuffled.
        weights (list(ndarray), optional):  The model's weights will be
            replaced by a random permutation of these weights.
            Defaults to None.
        layer_idx (int, optional): Index of layer to shuffle weights for if
            targeting a specific layer instead of whole model. Defaults to
            None.
    """
    if weights is None:
        if layer_idx is None:
            weights = model.get_weights()
        else:
            weights = model.layers[layer_idx].get_weights()

    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]

    if layer_idx is None:
        model.set_weights(weights)
    else:
        model.layers[layer_idx].set_weights(weights)


def train_seq2seq_kfold(train_model, inf_enc, inf_dec, X, X_prior, y,
                        num_folds=10, num_reps=3, batch_size=200, epochs=800,
                        early_stop=False, **kwargs):
    """Trains a seq2seq encoder-decoder model using k-fold cross validation.

    Uses k-fold cross validation to train a seq2seq encoder-decoder
    model. Each fold is repeated multiple times for stability in predictions.
    Requires a training model, as well as inference encoder and decoder
    for predicting sequences. Model is trained with teacher forcing from padded
    versions of the target sequences.

    Args:
        train_model (Functional): Full encoder-decoder model for training.
        inf_enc (Functional): Inference encoder model.
        inf_dec (Functional): Inference decoder model.
        X (ndarray): Feature data. First dimension should be number of
            observations. Dimensions should be compatible with the input to the
            provided models.
        X_prior (ndarray): Shifted labels for teacher forcing. Dimensions
            should be the same as `y`.
        y (ndarray): Labels. First dimension should be number of observations.
            Final dimension should be length of output sequence.
        num_folds (int, optional): Number of CV folds. Defaults to 10.
        batch_size (int, optional): Training batch size. Defaults to 32.
        epochs (int, optional): Number of training epochs. Defaults to 800.
        early_stop (bool, optional): Whether to stop training early based on
            validation loss performance. Defaults to True.

    Returns:
        (Dict, ndarray, ndarray): Dictionary containing trained models
            by fold, dictionary containing training performance history for
            each fold, predicted labels across folds, and true labels across
            folds.
            Dictionary structure is:
            histories = {'accuracy': [fold1rep1_acc, ..., fold1repn_acc,
                                      fold2rep1_acc, ...],
                          'loss': [fold1rep1_loss, ..., fold1repn_loss,
                                   fold2rep1_loss, ...],
                          'val_accuracy': [fold1rep1_acc, ..., fold1repn_acc,
                                           fold2rep1_acc, ...],
                          'val_loss': [fold1rep1_loss, ..., fold1repn_loss,
                                       fold2rep1_loss, ...],}
    """
    # save initial weights to reset model for each fold
    init_train_w = train_model.get_weights()

    # define k-fold cross validation
    cv = KFold(n_splits=num_folds, shuffle=True)

    cb = None
    # create callback for early stopping
    if early_stop:
        # early stopping with patience = 1/10 of total epochs
        es = EarlyStopping(monitor='val_loss', patience=int(epochs / 10),
                           restore_best_weights=True)
        cb = [es]

    # dictionary for history of each fold
    histories = {'accuracy': [], 'loss': [], 'val_accuracy': [],
                 'val_loss': []}

    # cv training
    y_pred_all, y_test_all = [], []
    for train_ind, test_ind in cv.split(X):
        fold = int((len(histories["accuracy"]) / num_reps) + 1)
        print(f'===== Fold {fold} =====')
        for _ in range(num_reps):  # repeat fold for stability
            # reset model weights for current fold (also resets associated
            # inference weights)
            shuffle_weights(train_model, weights=init_train_w)

            history, y_pred_fold, y_test_fold = train_seq2seq_single_fold(
                                        train_model, inf_enc, inf_dec, X,
                                        X_prior, y, train_ind, test_ind,
                                        batch_size=batch_size, epochs=epochs,
                                        callbacks=cb, **kwargs)

            y_pred_all.extend(y_pred_fold)
            y_test_all.extend(y_test_fold)

            track_model_history(histories, history)  # track history in-place

    return histories, np.array(y_pred_all), np.array(y_test_all)


def train_seq2seq_single_fold(train_model, inf_enc, inf_dec, X, X_prior, y,
                              train_ind, test_ind, batch_size=200, epochs=800,
                              **kwargs):
    """Implements single fold of cross-validation for seq2seq models.

    Args:
        train_model (Functional): Full encoder-decoder model for training.
        inf_enc (Functional): Inference encoder model.
        inf_dec (Functional): Inference decoder model.
        X (ndarray): Feature data. First dimension should be number of
            observations. Dimensions should be compatible with the input to the
            provided models.
        X_prior (ndarray): Shifted labels for teacher forcing. Dimensions
            should be the same as `y`.
        y (ndarray): Labels. First dimension should be number of observations.
            Final dimension should be length of output sequence.
        train_ind (ndarray): Indices of training data as returned from split
            method of sklearn cross-validation objects.
        test_ind (ndarray): Indices of test data as returned from split
            method of sklearn cross-validation objects.
        batch_size (int, optional): Training batch size. Defaults to 200.
        epochs (int, optional): Number of training epochs. Defaults to 800.

    Returns:
        (Callback, ndarray, ndarray): Model training history, predicted labels,
            and true labels.
    """
    X_train, X_test = X[train_ind], X[test_ind]
    X_prior_train, X_prior_test = X_prior[train_ind], X_prior[test_ind]
    y_train, y_test = y[train_ind], y[test_ind]

    _, history = train_seq2seq(train_model, X_train, X_prior_train, y_train,
                               batch_size=batch_size, epochs=epochs,
                               validation_data=([X_test, X_prior_test],
                                                y_test), **kwargs)

    y_test_fold, y_pred_fold = decode_seq2seq(inf_enc, inf_dec, X_test,
                                              y_test)

    return history, y_test_fold, y_pred_fold


def decode_seq2seq(inf_enc, inf_dec, X_test, y_test):
    """Uses trained inference encoder and decoder to predict sequences.

    Args:
        inf_enc (Functional): Inference encoder model.
        inf_dec (Functional): Inference decoder model.
        X_test (ndarray): Test feature data. First dimension should be number
            of observations. Dimensions should be compatible with the input to
            the inference encoder.
        y_test (ndarray): One-hot encoded test labels. First dimension should
            be number of observations. Second dimension should be length of
            output sequence.

    Returns:
        (ndarray, ndarray): 1D array of predicted labels and 1D array of true
            labels. Length of each array is number of observations times the
            sequence length.
    """
    n_output = inf_dec.output_shape[0][-1]  # number of output classes
    seq_len = y_test.shape[1]  # length of output sequence

    target = seq2seq_predict_batch(inf_enc, inf_dec, X_test, seq_len,
                                   n_output)
    y_pred_dec = np.ravel(one_hot_decode_batch(target))
    y_test_dec = np.ravel(one_hot_decode_batch(y_test))
    return y_pred_dec, y_test_dec


def train_seq2seq(model, X, X_prior, y, batch_size=200, epochs=800, **kwargs):
    """Trains a seq2seq encoder-decoder model.

    Trains a seq2seq encoder-decoder model. Model is trained with teacher
    forcing from padded versions of the target sequences.

    Args:
        model (Functional): Full encoder-decoder model for training.
        X (ndarray): Feature data. First dimension should be number of
            observations. Dimensions should be compatible with the input to the
            provided models.
        X_prior (ndarray): Shifted labels for teacher forcing. Dimensions
            should be the same as `y`.
        y (ndarray): Labels. First dimension should be number of observations.
            Final dimension should be length of output sequence.
        batch_size (int, optional): Training batch size. Defaults to 32.
        epochs (int, optional): Number of training epochs. Defaults to 800.
        early_stop (bool, optional): Whether to stop training early based on
            validation loss performance. Defaults to True.

    Returns:
        (Functional, Callback): Trained model, training performance history.
    """
    # with tf.device('/device:GPU:0'):
    history = model.fit([X, X_prior], y, batch_size=batch_size,
                        epochs=epochs, **kwargs)

    return model, history


def track_model_history(hist_dict, history):
    """Appends model training history to a dictionary in place.

    Args:
        hist_dict (Dict): Dictionary to append history to.
        history (Callback): Model training history from keras model fit method.
    """
    for key in history.history.keys():
        hist_dict[key].append(history.history[key])
