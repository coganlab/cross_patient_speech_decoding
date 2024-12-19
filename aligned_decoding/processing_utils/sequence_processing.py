"""
Processing functions for sequence data in seq2seq model training.

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical


def pad_sequence_teacher_forcing(seq_input, n_output):
    """Pad sequence for use in teacher forcing training of RNN. Sequence is
    shifted right by one, 0 is inserted at the beginning, and the last element.
    is removed (to preserve length of sequence with 0 token at front). The
    padded sequence and original sequence are one-hot encoded.

    Args:
        seq_input (ndarray): Sequence to be padded.
        n_output (int): Cardinality of output space (number of output classes).

    Returns:
        (ndarray, ndarray, ndarray, ndarray): One-hot encoded padded sequence,
            one-hot encoded original sequence, padded sequence, original
            sequence.
    """
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
    y_pred_dec = one_hot_decode_batch(target)
    y_test_dec = one_hot_decode_batch(y_test)
    return y_pred_dec, y_test_dec


def DEPone_hot_decode(encoded_element):
    """Decodes a sequence of one-hot encoded vectors into a sequence of
    integers. Uses argmax to determine index of 1 in each one-hot vector.

    Args:
        encoded_element (ndarray): Sequence of one-hot encoded vectors.
            (sequence length, cardinality of output space)

    Returns:
        list: Decoded sequence of integers. Shape = (sequence length).
    """
    return [np.argmax(vector) for vector in encoded_element]


def DEPone_hot_decode_batch(encoded_batch):
    """Decodes a batch of one-hot encoded sequences into a batch of sequences
    of integers.

    Args:
        encoded_batch (ndarray): Batch of one-hot encoded sequences.
            (n_trials, sequence length, cardinality of output space)

    Returns:
        list: Batch of decoded sequences of integers. Shape = (n_trials,
            sequence length)
    """
    return [DEPone_hot_decode(enc_seq) for enc_seq in encoded_batch]


def one_hot_decode_batch(encoded_batch):
    return np.ravel(np.argmax(encoded_batch, axis=-1))
    # return tf.reshape(tf.math.argmax(encoded_batch, -1), [-1])


def DEPseq2seq_predict(inf_enc, inf_dec, source, n_steps, n_output, verbose=0):
    """Predicts sequence of outputs using inference encoder and decoder models.
    Single trial feature data is passed to inference encoder to generate states
    and states are passed to inference decoder to generate output sequence.
    Agnostic to RNN models (LSTM or GRU) as long as inference decoder predict()
    method return format is "output, (states)".

    Args:
        inf_enc (Functional): Inference encoder model.
        inf_dec (Functional): Inference decoder model.
        source (ndarray): Feature data for single trial. Shape must be
            compatible with input to inference encoder.
        n_steps (int): Length of sequence to be predicted.
        n_output (int): Cardinality of output space (number of output classes).
        verbose (int, optional): Verbosity of model predict() methods.
            Defaults to 0.

    Returns:
        ndarray: Predicted sequence of output probabilities. Shape = (n_steps,
            n_output)
    """
    state = inf_enc.predict(source, verbose=verbose)
    if not isinstance(state, list):
        state = [state]  # convert to list for compatibility with decoder (gru)

    # generate initial sequence (one-hot encoding corresponding to 0)
    target_seq = np.zeros((1, 1, n_output))
    target_seq[0, 0, 0] = 1

    output = []
    for _ in range(n_steps):  # iterate over time steps
        # predict next element
        pred_tup = inf_dec.predict([target_seq] + state, verbose=verbose)
        y_hat = pred_tup[0]
        # update both states for lstm or single state for gru
        state = list(pred_tup[1:])
        output.append(y_hat[0, 0, :])
        # update target sequence
        target_seq = y_hat

    return np.array(output)


def DEPseq2seq_predict_batch(inf_enc, inf_dec, source, n_steps, n_output,
                             verbose=0):
    """Predicts batch of sequences of outputs using inference encoder and
    decoder models. Calls seq2seq_predict() for each observation in batch.

    Args:
        inf_enc (Functional): Inference encoder model.
        inf_dec (Functional): Inference decoder model.
        source (ndarray): Feature data for batch. First dimension must be batch
            size. Shape must be compatible with input to inference encoder.
        n_steps (int): Length of sequence to be predicted.
        n_output (int): Cardinality of output space (number of output classes).
        verbose (int, optional): Verbosity of model predict() methods.
            Defaults to 0.

    Returns:
        ndarray: Batch of predicted sequence of output probabilities. Shape =
            (n_trials, n_steps, n_output)
    """
    output = []
    # iterate over observations
    for i in range(source.shape[0]):
        # add batch dimension for input match
        curr_trial = np.expand_dims(source[i, :, :], axis=0)
        # appends n_steps probability distributions for each observation
        output.append(DEPseq2seq_predict(inf_enc, inf_dec, curr_trial, n_steps,
                                         n_output, verbose=verbose))
    return np.array(output)


def seq2seq_predict_batch(inf_enc, inf_dec, source, n_steps, n_output):
    batch_states = inf_enc.predict_on_batch(source)
    if isinstance(batch_states, list):  # lstm tensor conversion
        batch_states = [tf.convert_to_tensor(s) for s in batch_states]
    else:  # gru tensor conversion
        batch_states = [tf.convert_to_tensor(batch_states)]

    batch_target = tf.Variable(tf.zeros((source.shape[0], 1, n_output)))
    batch_target = batch_target[:, 0, 0].assign(1)
    # batch_target[:, 0, 0] = 1

    output = tf.Variable(tf.zeros((source.shape[0], n_steps, n_output)))
    for step in range(n_steps):
        pred_data = inf_dec.predict_on_batch([batch_target] + batch_states)
        batch_target = tf.convert_to_tensor(pred_data[0])
        output = output[:, step, :].assign(tf.squeeze(batch_target))
        batch_states = [tf.convert_to_tensor(s) for s in pred_data[1:]]

    return output


def flatten_fold_preds(preds):
    """Flattens a list of predictions from cross-validation folds into a single
    array. Accounts for inhomogeneous amount of trials across folds.

    Args:
        preds (list): List of predictions from cross-validation folds. Shape =
            (folds, trials, sequence length)

    Returns:
        ndarray: Array of predictions concatenated across folds. Shape =
            (total trials * sequence length)
    """
    return np.hstack([np.ravel(pred) for pred in preds])
