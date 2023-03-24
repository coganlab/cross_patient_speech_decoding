"""
Functions for training seq2seq models across multiple subjects.

Author: Zac Spalding
"""

from train import train_seq2seq, shuffle_weights


def transfer_train_seq2seq(X1, X1_prior, y1, X2_s1, X2_prior_s1, y2_s1, X2_s2,
                           X2_prior_s2, y2_s2, train_model, inf_enc, inf_dec,
                           cnn_layer_idx=1, enc_dec_layer_idx=-1):
    init_cnn_weights = train_model.layers[1].get_weights()

    # pretrain on first subject
    pretrained_model, _ = train_seq2seq(train_model, X1, X1_prior, y1)

    # reset convolutional weights
    shuffle_weights(pretrained_model.layers[cnn_layer_idx],
                    weights=init_cnn_weights)

    # freeze encoder decoder weights
    freeze_layer(pretrained_model, enc_dec_layer_idx)

    # train convolutional layer on second subject split 1
    updated_cnn_model, _ = train_seq2seq(pretrained_model, X2_s1, X2_prior_s1,
                                         y2_s1)

    # unfreeze encoder decoder weights
    unfreeze_layer(updated_cnn_model, enc_dec_layer_idx)

    # train on second subject split 2
    fine_tune_model, _ = train_seq2seq(updated_cnn_model, X2_s2, X2_prior_s2,
                                       y2_s2)

    return fine_tune_model, inf_enc, inf_dec


def freeze_layer(model, layer_idx):
    model.layers[layer_idx].trainable = False


def unfreeze_layer(model, layer_idx):
    model.layers[layer_idx].trainable = True
