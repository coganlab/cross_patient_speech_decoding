"""
Functions for training seq2seq models across multiple subjects.

Author: Zac Spalding
"""

from sklearn.model_selection import KFold
from keras.optimizers import Adam
from .train import train_seq2seq, shuffle_weights
from processing_utils.sequence_processing import (seq2seq_predict_batch,
                                                  one_hot_decode_batch,
                                                  flatten_fold_preds)


def transfer_seq2seq_kfold(train_model, inf_enc, inf_dec, X1, X1_prior, y1,
                           X2, X2_prior, y2, num_folds=10):
    # save initial weights to reset model for each fold
    init_train_w = train_model.get_weights()

    n_output = inf_dec.output_shape[0][-1]  # number of output classes
    seq_len = y2.shape[1]  # length of output sequence

    # define k-fold cross validation
    cv = KFold(n_splits=num_folds, shuffle=True)

    # dictionary for tracking models and history of each fold
    models = {'train': [], 'inf_enc': [], 'inf_dec': []}
    histories = {'accuracy': [], 'loss': [], 'val_accuracy': [],
                 'val_loss': []}

    # cv training
    y_pred_all, y_test_all = [], []
    for train_ind, test_ind in cv.split(X2):
        print(f'========== Fold {len(models["train"]) + 1} ==========')
        X2_train, X2_test = X2[train_ind], X2[test_ind]
        X2_prior_train, X2_prior_test = X2_prior[train_ind], X2_prior[test_ind]
        y2_train, y2_test = y2[train_ind], y2[test_ind]

        # reset model weights for current fold (also resets associated
        # inference weights)
        shuffle_weights(train_model, weights=init_train_w)

        train_model, inf_enc, inf_dec, transfer_hist = transfer_train_seq2seq(
            X1, X1_prior, y1, X2_train, X2_prior_train, y2_train, X2_test,
            X2_prior_test, y2_test, train_model, inf_enc, inf_dec)

        models['train'].append(train_model)
        models['inf_enc'].append(inf_enc)
        models['inf_dec'].append(inf_dec)

        histories['accuracy'].append(transfer_hist.history['accuracy'])
        histories['loss'].append(transfer_hist.history['loss'])
        histories['val_accuracy'].append(transfer_hist.history['val_accuracy'])
        histories['val_loss'].append(transfer_hist.history['val_loss'])

        target = seq2seq_predict_batch(inf_enc, inf_dec, X2_test, seq_len,
                                       n_output)
        y_test_all.append(one_hot_decode_batch(y2_test))
        y_pred_all.append(one_hot_decode_batch(target))

    y_pred_all = flatten_fold_preds(y_pred_all)
    y_test_all = flatten_fold_preds(y_test_all)

    return models, histories, y_pred_all, y_test_all


def transfer_train_seq2seq(X1, X1_prior, y1, X2_train, X2_prior_train,
                           y2_train, X2_test, X2_prior_test, y2_test,
                           train_model, inf_enc, inf_dec, pretrain_epochs=200,
                           conv_epochs=60, fine_tune_epochs=540,
                           cnn_layer_idx=1, enc_dec_layer_idx=-1):
    init_cnn_weights = train_model.layers[1].get_weights()

    # pretrain on first subject
    pretrained_model, _ = train_seq2seq(train_model, X1, X1_prior, y1,
                                        epochs=pretrain_epochs)

    # reset convolutional weights -- fix to make fully random
    shuffle_weights(pretrained_model.layers[cnn_layer_idx],
                    weights=init_cnn_weights)

    # freeze encoder decoder weights
    freeze_layer(pretrained_model, enc_dec_layer_idx,
                 optimizer=Adam(1e-3),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    # train convolutional layer on second subject split 1
    updated_cnn_model, _ = train_seq2seq(pretrained_model, X2_train,
                                         X2_prior_train, y2_train,
                                         epochs=conv_epochs)

    # unfreeze encoder decoder weights
    unfreeze_layer(updated_cnn_model, enc_dec_layer_idx,
                   optimizer=Adam(1e-3),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    # train on second subject split 2
    fine_tune_model, fine_tune_history = train_seq2seq(updated_cnn_model,
                                                       X2_train,
                                                       X2_prior_train,
                                                       y2_train,
                                                       epochs=fine_tune_epochs,
                                                       validation_data=(
                                                           [X2_test,
                                                            X2_prior_test],
                                                           y2_test))

    return fine_tune_model, inf_enc, inf_dec, fine_tune_history


def freeze_layer(model, layer_idx, **kwargs):
    model.layers[layer_idx].trainable = False
    model.compile(**kwargs)


def unfreeze_layer(model, layer_idx, **kwargs):
    model.layers[layer_idx].trainable = True
    model.compile(**kwargs)
