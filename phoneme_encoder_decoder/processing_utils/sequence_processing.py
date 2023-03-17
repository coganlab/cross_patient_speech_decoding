"""Preprocessing functions for sequence data in seq2seq model training.

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""

import numpy as np
from keras.utils import to_categorical


def pad_sequence_teacher_forcing(seq_input, n_output):

    pad_one_hot, seq_one_hot, pad_labels, seq_labels = [], [], [], []
    for seq in seq_input:  # for each observation/trial

        # shift sequence right by one and insert 0 at beginning
        # e.g. [1, 2, 3] -> [0, 1, 2]
        pad_seq = np.insert(seq[:-1], 0, 0)

        # one-hot encode regular and padded sequences
        tar_encoded = to_categorical(seq, num_classes=n_output)
        tar_pad_encoded = to_categorical(pad_seq, num_classes=n_output)

        # save sequence data for all observations/trials
        pad_one_hot.append(tar_pad_encoded)
        seq_one_hot.append(tar_encoded)
        pad_labels.append(pad_seq)
        seq_labels.append(seq)

    return (np.array(pad_one_hot), np.array(seq_one_hot), np.array(pad_labels),
            np.array(seq_labels))


def one_hot_decode(encoded_element):
    return [np.argmax(vector) for vector in encoded_element]


def one_hot_decode_sequence(encoded_seq):
    return [one_hot_decode(encoded_element) for encoded_element in encoded_seq]


def predict_lstm_sequence(inf_enc, inf_dec, source, n_steps, n_output):

    state = inf_enc.predict(source)

    # generate initial sequence (one-hot encoding corresponding to 0)
    target_seq = np.zeros((1, 1, n_output))
    target_seq[0, 0, 0] = 1

    output = []
    for _ in range(n_steps):  # iterate over time steps
        # predict next element
        y_hat, h, c = inf_dec.predict([target_seq] + state)
        output.append(y_hat[0, 0, :])
        # update decoder states
        state = [h, c]
        # update target sequence
        target_seq = y_hat

    return np.array(output)


def predict_gru_sequence(inf_enc, inf_dec, source, n_steps, n_output):

    state = inf_enc.predict(source)

    # generate initial sequence (one-hot encoding corresponding to 0)
    target_seq = np.zeros((1, 1, n_output))
    target_seq[0, 0, 0] = 1

    output = []
    for _ in range(n_steps):  # iterate over time steps
        # predict next element
        y_hat, h = inf_dec.predict([target_seq] + state)
        output.append(y_hat[0, 0, :])
        # update decoder states
        state = h
        # update target sequence
        target_seq = y_hat

    return np.array(output)


def predict_sequence(inf_enc, inf_dec, source, n_steps, n_output):
    if inf_dec.name[-4:] == 'lstm':
        return predict_lstm_sequence(inf_enc, inf_dec, source, n_steps,
                                     n_output)
    elif inf_dec.name[-3:] == 'gru':
        return predict_gru_sequence(inf_enc, inf_dec, source, n_steps,
                                    n_output)
