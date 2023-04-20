""" Class defining custom evaluator for seq2seq model training. Inherits from
    keras.callbacks.Callback. Overrides on_epoch_end method to evaluate model
    with sequence prediction instead of teacher forcing as is used in the model
    training.
"""

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score

from processing_utils.sequence_processing import (seq2seq_predict_batch,
                                                  one_hot_decode_batch_test)


class seq2seq_predict_callback(keras.callbacks.Callback):
    def __init__(self, train_model, inf_enc, inf_dec, X, y):
        self.train_model = train_model
        self.inf_enc = inf_enc
        self.inf_dec = inf_dec
        self.X = X
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        loss_fcn = keras.losses.get(self.train_model.loss)

        n_output = self.inf_dec.output_shape[0][-1]  # number of output classes
        seq_len = self.y.shape[1]  # length of output sequence

        y_pred_dist = seq2seq_predict_batch(self.inf_enc, self.inf_dec, self.X,
                                            seq_len, n_output)
        loss = tf.math.reduce_mean(loss_fcn(self.y, y_pred_dist))

        y_pred = one_hot_decode_batch_test(y_pred_dist)
        y_test = one_hot_decode_batch_test(self.y)
        b_acc = balanced_accuracy_score(y_test, y_pred)

        logs['seq2seq_val_loss'] = loss
        logs['seq2seq_val_accuracy'] = b_acc
