"""
Functions for training seq2seq models across multiple subjects.

Author: Zac Spalding
"""

import numpy as np
from sklearn.model_selection import KFold
from keras.optimizers import Adam
from keras.models import Model

from processing_utils.sequence_processing import decode_seq2seq
from seq2seq_models.rnn_model_components import linear_cnn_1D_module
from .train import train_seq2seq, shuffle_weights, track_model_history
from .Seq2seqPredictCallback import Seq2seqPredictCallback


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

    seq2seq_cb_1 = Seq2seqPredictCallback(train_model, pre_enc, pre_dec,
                                          X1_test, y1_test)
    seq2seq_cb_2 = Seq2seqPredictCallback(train_model, tar_enc, tar_dec,
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


def transfer_chain_kfold(model, inf_enc, inf_dec, X1, X1_prior, y1, X2,
                         X2_prior, y2, num_folds=10, num_reps=3, **kwargs):
    # save initial weights to reset model for each fold
    init_train_w = model.get_weights()

    # put data in list format if not (likely not list if only single pt)
    X1, X1_prior, y1 = multi_pt_compat(X1, X1_prior, y1)

    # define k-fold cross validation
    cv = KFold(n_splits=num_folds, shuffle=True)
    pre_splits = [cv.split(x) for x in X1]
    tar_splits = cv.split(X2)

    # dictionary for tracking history of each fold
    histories = {'accuracy': [], 'loss': [], 'val_accuracy': [],
                 'val_loss': []}

    # cv training
    y_pred_all, y_test_all = [], []
    for f in range(num_folds):
        pre_inds = [next(s) for s in pre_splits]
        train_ind_pre, test_ind_pre = zip(*pre_inds)  # unpack pretrain list
        train_ind_tar, test_ind_tar = next(tar_splits)  # single target pt
        fold = f + 1
        print(f'===== Fold {fold} =====')
        for _ in range(num_reps):  # repeat fold for stability

            # reset model weights for current fold (also resets associated
            # inference weights)
            shuffle_weights(model, weights=init_train_w)

            transfer_hist, y_pred_fold, y_test_fold = \
                transfer_chain_single_fold(model, inf_enc,
                                           inf_dec, X1, X1_prior, y1,
                                           X2, X2_prior, y2, train_ind_pre,
                                           test_ind_pre, train_ind_tar,
                                           test_ind_tar, **kwargs)

            # track history in-place
            track_model_history(histories, transfer_hist)

            y_pred_all.extend(y_pred_fold)
            y_test_all.extend(y_test_fold)

    return histories, np.array(y_pred_all), np.array(y_test_all)


def transfer_chain_single_fold(model, inf_enc, inf_dec, X1,
                               X1_prior, y1, X2, X2_prior, y2,
                               train_ind_pre, test_ind_pre, train_ind_tar,
                               test_ind_tar, **kwargs):

    # split pretrain data into train and test (may be list with multi pt data)
    X1_train, X1_test = ([x[train_ind_pre[i]] for i, x in enumerate(X1)],
                         [x[test_ind_pre[i]] for i, x in enumerate(X1)])
    X1_prior_train, X1_prior_test = ([xp[train_ind_pre[i]] for i, xp in
                                      enumerate(X1_prior)],
                                     [xp[test_ind_pre[i]] for i, xp in
                                      enumerate(X1_prior)])
    y1_train, y1_test = ([y[train_ind_pre[i]] for i, y in enumerate(y1)],
                         [y[test_ind_pre[i]] for i, y in enumerate(y1)])

    # split target data into train and test
    X2_train, X2_test = X2[train_ind_tar], X2[test_ind_tar]
    X2_prior_train, X2_prior_test = (X2_prior[train_ind_tar],
                                     X2_prior[test_ind_tar])
    y2_train, y2_test = y2[train_ind_tar], y2[test_ind_tar]

    # define validation data (preserve list format for multi pt pretrain data)
    pre_val = ([([x, xp], y) for (x, xp, y) in
                zip(X1_test, X1_prior_test, y1_test)])
    tar_val = ([X2_test, X2_prior_test], y2_test)

    model, inf_enc, transfer_hist = transfer_train_chain(
                                            model, inf_enc, inf_dec, X1_train,
                                            X1_prior_train, y1_train, X2_train,
                                            X2_prior_train, y2_train,
                                            pre_val=pre_val, tar_val=tar_val,
                                            **kwargs)

    y_test_fold, y_pred_fold = decode_seq2seq(inf_enc, inf_dec, X2_test,
                                              y2_test)

    return transfer_hist, y_test_fold, y_pred_fold


def transfer_train_chain(model, inf_enc, inf_dec, X1, X1_prior, y1, X2,
                         X2_prior, y2, pretrain_epochs=200, conv_epochs=60,
                         target_epochs=540, pre_val=None, tar_val=None,
                         conv_idx=1, enc_dec_idx=-1, **kwargs):
    """Train model with cross-patient transfer learning chain.

    Function to perform cross-patient transfer learning by pretraining a model
    on one or multiple patient(s) and then fine-tuning the model on a target
    patient. This method is an extension of the method proposed by Makin et al.
    2020 (https://www.nature.com/articles/s41593-020-0608-8) designed for
    transfer from a single patient to a single patient. This function chains
    together multiple Makin et al. transfer learning steps to pretrain across
    multiple patients and transfer to a target patient. Keyword arguments are
    passed to the train_seq2seq function defined in train.py.
    *****TODO UPDATE*****

    Args:
        model (Fucntional): Full encoder-decoder model
        X1 (ndarray or list of ndarray): Array of feature data for pretrain
            patient. If pretraining on multiple patients, each patient's
            feature data should be a separate array in the list.
        X1_prior (ndarray or list of ndarray): Shifted labels for teacher
            forcing. Type should match X1. See X1 description for type info
            when pretraining on single or multiple patients.
        y1 (ndarray or list of ndarray): Labels. Type should match X1. See X1
            description for type info when pretraining on single or multiple
            patients.
        X2 (ndarray): Feature data for target patient.
        X2_prior (ndarray): Target patient shifted labels for teacher forcing.
        y2 (ndarray): Labels for target patient.
        pretrain_epochs (int, optional): Training epochs for pretraining step.
            Defaults to 200.
        conv_epochs (int, optional): Training epochs for convolutional layer
            update step. Defaults to 60.
        target_epochs (int, optional): Training epochs for full-network
            training on target patient. Defaults to 540.

    Returns:
        (Functional, Functional, Callback): Transfer model with convolutional
            layer updated for target patient shape, inference encoder with
            convolutional layer updated for target patient shape, training
            performance history across all transfer stages.
    """
    # parse pre-train input to check for multiple pts
    X1, X1_prior, y1 = multi_pt_compat(X1, X1_prior, y1)
    pre_val = val_data_to_list(pre_val)

    # check if val data is provided
    do_val = pre_val is not None and tar_val is not None
    cb = None

    # define callback to calculate seq2seq metrics during training
    if do_val:
        # val data for first pretrain patient - feature data and labels
        X1_test, y1_test = pre_val[0][0][0], pre_val[0][1]
        seq2seq_cb = Seq2seqPredictCallback(model, inf_enc, inf_dec, X1_test,
                                            y1_test)
        cb = [seq2seq_cb]

    # pretrain full model on first patient
    _, pretrain_hist = train_seq2seq(
                            model, X1[0], X1_prior[0], y1[0],
                            epochs=pretrain_epochs,
                            validation_data=pre_val[0] if do_val else None,
                            callbacks=cb,
                            **kwargs)

    curr_hist = pretrain_hist
    for i in range(1, len(X1)):
        # replace conv layer for compatibility with new channel amount
        n_channels = X1[i].shape[-1]
        model, inf_enc = replace_conv_layer_channels(
                                    model, inf_enc, n_channels,
                                    conv_idx=conv_idx,
                                    enc_dec_idx=enc_dec_idx)

        # make sure seq2seq_cb is using current pretrain pt models and data
        if do_val:
            X1_test, y1_test = pre_val[i][0][0], pre_val[i][1]
            seq2seq_cb.set_models(model, inf_enc, inf_dec)
            seq2seq_cb.set_data(X1_test, y1_test)
            cb = [seq2seq_cb]

        # update conv layer for current pretrain pt to better extract features
        conv_hist = transfer_conv_update(
                            model, X1[i], X1_prior[i], y1[i],
                            enc_dec_idx=enc_dec_idx,
                            epochs=conv_epochs,
                            validation_data=pre_val[i] if do_val else None,
                            callbacks=cb,
                            **kwargs)
        # pretraining on current pretrain pt
        _, pretrain_hist = train_seq2seq(
                            model, X1[i], X1_prior[i], y1[i],
                            epochs=pretrain_epochs,
                            validation_data=pre_val[i] if do_val else None,
                            callbacks=cb,
                            **kwargs)
        curr_hist = concat_hists([curr_hist, conv_hist, pretrain_hist])

    # replace conv layer for compatibility with target pt channel amount
    tar_channels = X2.shape[-1]
    model, inf_enc = replace_conv_layer_channels(
                                    model, inf_enc, tar_channels,
                                    conv_idx=conv_idx,
                                    enc_dec_idx=enc_dec_idx)

    # make sure seq2seq_cb is using target pt models and data
    if do_val:
        X2_test, y2_test = tar_val[0][0], tar_val[1]
        seq2seq_cb.set_models(model, inf_enc, inf_dec)
        seq2seq_cb.set_data(X2_test, y2_test)
        cb = [seq2seq_cb]

    # update conv layer weights for target pt
    conv_hist = transfer_conv_update(
                            model, X2, X2_prior, y2,
                            enc_dec_idx=enc_dec_idx,
                            epochs=conv_epochs,
                            validation_data=tar_val,
                            callbacks=cb,
                            **kwargs)

    # fine-tuning on target pt
    _, target_hist = train_seq2seq(model, X2, X2_prior, y2,
                                   epochs=target_epochs,
                                   validation_data=tar_val,
                                   callbacks=cb, **kwargs)

    total_hist = concat_hists([curr_hist, conv_hist, target_hist])

    return model, inf_enc, total_hist


def multi_pt_compat(X, X_prior, y):
    if not isinstance(X, list):
        X = list(X[np.newaxis, ...])
        X_prior = list(X_prior[np.newaxis, ...])
        y = list(y[np.newaxis, ...])
    return X, X_prior, y


def val_data_to_list(val_data):
    if not isinstance(val_data, list):
        tmp = []
        tmp.append(val_data)
        return tmp
    return val_data


def transfer_conv_update(model, X, X_prior, y, enc_dec_idx=-1, **kwargs):
    freeze_layer(model, layer_idx=enc_dec_idx)
    _, conv_hist = train_seq2seq(model, X, X_prior, y, **kwargs)
    unfreeze_layer(model, layer_idx=enc_dec_idx)
    return conv_hist


def replace_conv_layer_channels(model, inf_enc, n_channels, conv_idx=1,
                                enc_dec_idx=-1):
    input_layer = model.layers[conv_idx - 1]
    conv_layer = model.layers[conv_idx]
    reg_val = float(conv_layer.kernel_regularizer.l2)

    # define new input and conv layers for new channel amount
    new_inputs, new_conv_layer = linear_cnn_1D_module(
                                        input_layer.input_shape[0][1],
                                        n_channels, conv_layer.filters,
                                        conv_layer.kernel_size,
                                        reg_val)

    # create new model with new conv layer and old encoder-decoder module
    encoder_inputs = new_conv_layer(new_inputs)
    enc_dec_model = model.layers[enc_dec_idx]
    inf_enc_model = inf_enc.layers[enc_dec_idx]
    new_model = Model([new_inputs, enc_dec_model.input[1]],
                      enc_dec_model([encoder_inputs, enc_dec_model.input[1]]))
    new_enc = Model(new_inputs, inf_enc_model(encoder_inputs))

    # compile with properties from old model
    new_model.compile(model.optimizer, model.loss, ['accuracy'])
    return new_model, new_enc


def freeze_layer(model, layer_idx):
    model.layers[layer_idx].trainable = False
    model.compile(model.optimizer, model.loss, ['accuracy'])


def unfreeze_layer(model, layer_idx):
    model.layers[layer_idx].trainable = True
    model.compile(model.optimizer, model.loss, ['accuracy'])


def cv_split_multi_pt(cv, X):
    if isinstance(X, list):
        return [cv.split(x) for x in X]
    return cv.split(X)


def concat_hists(hist_list):
    # use first history as base
    new_hist = hist_list[0]

    # extend base histories with histories from other training sessions
    for hist in hist_list[1:]:
        for key in hist.history.keys():
            new_hist.history[key].extend(hist.history[key])

    return new_hist


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
