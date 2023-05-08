"""
Functions for training seq2seq models across multiple subjects.

Author: Zac Spalding
"""

import numpy as np
from sklearn.model_selection import KFold
from keras.optimizers import Adam

from processing_utils.sequence_processing import decode_seq2seq
from .train import train_seq2seq, shuffle_weights, track_model_history
from .seq2seq_predict_callback import seq2seq_predict_callback


def transfer_seq2seq_kfold(train_model, inf_enc, inf_dec, X1, X1_prior, y1,
                           X2, X2_prior, y2, num_folds=10, num_reps=3,
                           **kwargs):
    # save initial weights to reset model for each fold
    init_train_w = train_model.get_weights()

    n_output = inf_dec.output_shape[0][-1]  # number of output classes
    seq_len = y2.shape[1]  # length of output sequence

    # define k-fold cross validation
    cv = KFold(n_splits=num_folds, shuffle=True)

    # dictionary for tracking history of each fold
    histories = {'accuracy': [], 'loss': [], 'val_accuracy': [],
                 'val_loss': []}

    # cv training
    y_pred_all, y_test_all = [], []
    for train_ind, test_ind in cv.split(X2):
        fold = int((len(histories["accuracy"]) / num_reps) + 1)
        print(f'===== Fold {fold} =====')
        for _ in range(num_reps):  # repeat fold for stability

            # reset model weights for current fold (also resets associated
            # inference weights)
            shuffle_weights(train_model, weights=init_train_w)

            transfer_hist, y_pred_fold, y_test_fold = \
                transfer_train_seq2seq_single_fold(train_model, inf_enc,
                                                   inf_dec, X1, X1_prior, y1,
                                                   X2, X2_prior, y2, train_ind,
                                                   test_ind, **kwargs)

            # track history in-place
            track_model_history(histories, transfer_hist)

            y_pred_all.extend(y_pred_fold)
            y_test_all.extend(y_test_fold)

    return histories, np.array(y_pred_all), np.array(y_test_all)


def transfer_train_seq2seq_single_fold(train_model, inf_enc, inf_dec, X1,
                                       X1_prior, y1, X2, X2_prior, y2,
                                       train_ind, test_ind, batch_size=200,
                                       **kwargs):
    train_ind = train_ind.astype(int)
    test_ind = test_ind.astype(int)
    X2_train, X2_test = X2[train_ind], X2[test_ind]
    X2_prior_train, X2_prior_test = X2_prior[train_ind], X2_prior[test_ind]
    y2_train, y2_test = y2[train_ind], y2[test_ind]

    _, transfer_hist = transfer_train_seq2seq(X1, X1_prior, y1, X2_train,
                                              X2_prior_train, y2_train,
                                              X2_test, X2_prior_test,
                                              y2_test, train_model,
                                              batch_size=batch_size,
                                              **kwargs)

    y_test_fold, y_pred_fold = decode_seq2seq(inf_enc, inf_dec, X2_test,
                                              y2_test)

    return transfer_hist, y_test_fold, y_pred_fold


def transfer_train_seq2seq(X1, X1_prior, y1, X2_train, X2_prior_train,
                           y2_train, X2_test, X2_prior_test, y2_test,
                           train_model, pretrain_epochs=200,
                           conv_epochs=60, fine_tune_epochs=540,
                           cnn_layer_idx=1, enc_dec_layer_idx=-1, **kwargs):
    init_cnn_weights = train_model.layers[1].get_weights()
    lr = train_model.optimizer.get_config()['learning_rate']

    # pretrain on first subject
    pretrained_model, pretrain_hist = train_seq2seq(train_model, X1, X1_prior,
                                                    y1, epochs=pretrain_epochs,
                                                    **kwargs)

    # reset convolutional weights -- fix to make fully random
    shuffle_weights(pretrained_model.layers[cnn_layer_idx],
                    weights=init_cnn_weights)

    # freeze encoder decoder weights
    freeze_layer(pretrained_model, enc_dec_layer_idx,
                 optimizer=Adam(lr),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # train convolutional layer on second subject split 1
    updated_cnn_model, conv_hist = train_seq2seq(pretrained_model, X2_train,
                                                 X2_prior_train, y2_train,
                                                 epochs=conv_epochs, **kwargs)

    # unfreeze encoder decoder weights
    unfreeze_layer(updated_cnn_model, enc_dec_layer_idx,
                   optimizer=Adam(lr),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    # train on second subject split 2
    fine_tune_model, fine_tune_hist = train_seq2seq(updated_cnn_model,
                                                    X2_train,
                                                    X2_prior_train,
                                                    y2_train,
                                                    epochs=fine_tune_epochs,
                                                    validation_data=(
                                                        [X2_test,
                                                         X2_prior_test],
                                                        y2_test), **kwargs)

    total_hist = concat_hists([pretrain_hist, conv_hist, fine_tune_hist])

    return fine_tune_model, total_hist


def transfer_seq2seq_kfold_diff_chans(train_model, pre_enc, pre_dec,
                                      tar_model, tar_enc, tar_dec,
                                      X1, X1_prior, y1, X2, X2_prior, y2,
                                      num_folds=10, num_reps=3, **kwargs):
    # save initial weights to reset model for each fold
    init_train_w = train_model.get_weights()
    init_tar_w = tar_model.get_weights()

    # define k-fold cross validation
    cv = KFold(n_splits=num_folds, shuffle=True)
    splits_1 = cv.split(X1)
    splits_2 = cv.split(X2)

    # dictionary for tracking history of each fold
    histories = {'accuracy': [], 'loss': [], 'val_accuracy': [],
                 'val_loss': []}

    # cv training
    y_pred_all, y_test_all = [], []
    # for train_ind, test_ind in cv.split(X2):
    for f in range(num_folds):
        train_ind1, test_ind1 = next(splits_1)
        train_ind2, test_ind2 = next(splits_2)
        # fold = int((len(histories["accuracy"]) / num_reps) + 1)
        fold = f + 1
        print(f'===== Fold {fold} =====')
        for _ in range(num_reps):  # repeat fold for stability

            # reset model weights for current fold (also resets associated
            # inference weights)
            shuffle_weights(train_model, weights=init_train_w)
            shuffle_weights(tar_model, weights=init_tar_w)

            transfer_hist, y_pred_fold, y_test_fold = \
                transfer_train_seq2seq_single_fold_diff_chans(
                    train_model, pre_enc, pre_dec, tar_model, tar_enc, tar_dec,
                    X1, X1_prior, y1, train_ind1, test_ind1, X2, X2_prior, y2,
                    train_ind2, test_ind2, **kwargs)

            # track history in-place
            track_model_history(histories, transfer_hist)

            y_pred_all.extend(y_pred_fold)
            y_test_all.extend(y_test_fold)

    return histories, np.array(y_pred_all), np.array(y_test_all)


def transfer_train_seq2seq_single_fold_diff_chans(train_model, pre_enc,
                                                  pre_dec, tar_model,
                                                  tar_enc, tar_dec, X1,
                                                  X1_prior, y1,
                                                  train_ind1, test_ind1,
                                                  X2, X2_prior, y2,
                                                  train_ind2, test_ind2,
                                                  batch_size=200,
                                                  callbacks=None,
                                                  **kwargs):
    X1_train, X1_test = X1[train_ind1], X1[test_ind1]
    X1_prior_train, X1_prior_test = X1_prior[train_ind1], X1_prior[test_ind1]
    y1_train, y1_test = y1[train_ind1], y1[test_ind1]

    X2_train, X2_test = X2[train_ind2], X2[test_ind2]
    X2_prior_train, X2_prior_test = X2_prior[train_ind2], X2_prior[test_ind2]
    y2_train, y2_test = y2[train_ind2], y2[test_ind2]

    seq2seq_cb_1 = seq2seq_predict_callback(train_model, pre_enc, pre_dec,
                                            X1_test, y1_test)
    seq2seq_cb_2 = seq2seq_predict_callback(train_model, tar_enc, tar_dec,
                                            X2_test, y2_test)

    pre_cb = callbacks
    transfer_cb = callbacks
    if callbacks is not None:
        pre_cb.append(seq2seq_cb_1)
        transfer_cb.append(seq2seq_cb_2)
    else:
        pre_cb = [seq2seq_cb_1]
        transfer_cb = [seq2seq_cb_2]

    _, transfer_hist = transfer_train_seq2seq_diff_chans(train_model,
                                                         tar_model,
                                                         X1_train,
                                                         X1_prior_train,
                                                         y1_train,
                                                         X2_train,
                                                         X2_prior_train,
                                                         y2_train,
                                                         pre_val=(
                                                            [X1_test,
                                                             X1_prior_test],
                                                            y1_test),
                                                         transfer_val=(
                                                            [X2_test,
                                                             X2_prior_test],
                                                            y2_test),
                                                         batch_size=batch_size,
                                                         pre_callbacks=pre_cb,
                                                         transfer_callbacks=(
                                                            transfer_cb),
                                                         **kwargs)

    y_test_fold, y_pred_fold = decode_seq2seq(tar_enc, tar_dec, X2_test,
                                              y2_test)

    return transfer_hist, y_test_fold, y_pred_fold


def transfer_train_seq2seq_diff_chans(train_model, tar_model, X1, X1_prior, y1,
                                      X2_train, X2_prior_train, y2_train,
                                      pretrain_epochs=200, conv_epochs=60,
                                      fine_tune_epochs=540,
                                      enc_dec_layer_idx=-1, pre_val=None,
                                      transfer_val=None, pre_callbacks=None,
                                      transfer_callbacks=None, **kwargs):
    lr = train_model.optimizer.get_config()['learning_rate']

    # pretrain on first subject
    pretrained_model, pretrain_hist = train_seq2seq(train_model, X1, X1_prior,
                                                    y1, epochs=pretrain_epochs,
                                                    validation_data=pre_val,
                                                    callbacks=pre_callbacks,
                                                    **kwargs)

    # create new model with modified input layer and same enc-dec weights
    copy_applicable_weights(pretrained_model, tar_model, optimizer=Adam(lr),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    # freeze encoder decoder weights
    freeze_layer(tar_model, enc_dec_layer_idx, optimizer=Adam(lr),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # train convolutional layer on second subject split 1
    updated_cnn_model, conv_hist = train_seq2seq(tar_model, X2_train,
                                                 X2_prior_train, y2_train,
                                                 epochs=conv_epochs,
                                                 validation_data=transfer_val,
                                                 callbacks=transfer_callbacks,
                                                 **kwargs)

    # unfreeze encoder decoder weights
    unfreeze_layer(updated_cnn_model, enc_dec_layer_idx,
                   optimizer=Adam(lr),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    # train on second subject split 2
    fine_tune_model, fine_tune_hist = train_seq2seq(
                                            updated_cnn_model,
                                            X2_train,
                                            X2_prior_train,
                                            y2_train,
                                            epochs=fine_tune_epochs,
                                            validation_data=transfer_val,
                                            callbacks=transfer_callbacks,
                                            **kwargs)

    total_hist = concat_hists([pretrain_hist, conv_hist, fine_tune_hist])

    return fine_tune_model, total_hist


def copy_applicable_weights(model, new_model, **kwargs):
    """From https://datascience.stackexchange.com/questions/21734/keras-
    transfer-learning-changing-input-tensor-shape
    User: gebbissimo
    """
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))
    new_model.compile(**kwargs)


def freeze_layer(model, layer_idx, **kwargs):
    model.layers[layer_idx].trainable = False
    model.compile(**kwargs)


def unfreeze_layer(model, layer_idx, **kwargs):
    model.layers[layer_idx].trainable = True
    model.compile(**kwargs)


def concat_hists(hist_list):
    # use first history as base
    new_hist = hist_list[0]

    # extend base histories with histories from other training sessions
    for hist in hist_list[1:]:
        for key in hist.history.keys():
            new_hist.history[key].extend(hist.history[key])

    return new_hist
