"""
Construction of seq2seq RNN encoder-decoder models from model components in
rnn_model_components.py

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""


from keras.models import Model
from keras.layers import Reshape

from rnn_model_components import (linear_cnn_1D_module,
                                  linear_cnn_3D_module,
                                  lstm_encoder_decoder_module,
                                  gru_encoder_decoder_module)


def lstm_1Dcnn_model(n_input_time, n_input_channel, n_output, n_filters,
                     filter_size, n_units, reg_lambda):
    """Creates a joint 1D CNN-LSTM  model by adding a 1D convolutional layer to
    the front of an encoder-decoder LSTM model.

    Args:
        n_input_time (int): Number of timesteps.
        n_input_channel (int): Number of channels.
        n_output (int): Cardinality of output space.
        n_filters (int): Number of convolutional filters.
        filter_size (int): Size (and stride) of convolutional filters.
        n_units (int): Number of units in LSTM layers.
        reg_lambda (float): L2 regularization parameter.

    Returns:
        (Functional, Functional, Functional): Encoder-decoder training model,
            encoder inference model, decoder inference model
    """
    cnn_inputs, cnn_layers = linear_cnn_1D_module(n_input_time,
                                                  n_input_channel, n_filters,
                                                  filter_size, reg_lambda)
    encoder_inputs = cnn_layers(cnn_inputs)
    training_model, inf_enc_model, inf_dec_model = lstm_encoder_decoder_module(
                                  encoder_inputs, n_output, n_units,
                                  reg_lambda)

    # add cnn layer to beginning of encoder-decoder LSTM
    training_model = Model([cnn_inputs, training_model.input[1]],
                           training_model([encoder_inputs,
                                           training_model.input[1]]))
    inf_enc_model = Model(cnn_inputs, inf_enc_model(encoder_inputs))

    return training_model, inf_enc_model, inf_dec_model


def lstm_3Dcnn_model(n_input_time, n_input_x, n_input_y, n_output,
                     n_filters, filter_size, n_units, reg_lambda):
    """Creates a joint 3D CNN-LSTM  model by adding a 3D convolutional layer to
    the front of an encoder-decoder LSTM model.

    Args:
        n_input_time (int): Number of timesteps.
        n_input_x (int): Width of image (size of channel map x-dimension).
        n_input_y (int): Height of image (size of channel map y-dimension).
        n_output (int): Cardinality of output space.
        n_filters (int): Number of convolutional filters.
        filter_size (int): Size (and stride) of convolutional filters.
        n_units (int): Number of units in LSTM layers.
        reg_lambda (float): L2 regularization parameter.

    Returns:
        (Functional, Functional, Functional): Encoder-decoder training model,
            encoder inference model, decoder inference model
    """
    cnn_inputs, cnn_layer = linear_cnn_3D_module(n_input_time,
                                                 n_input_x,
                                                 n_input_y,
                                                 n_filters,
                                                 filter_size,
                                                 reg_lambda)
    cnn_output = cnn_layer(cnn_inputs)
    cnn_shape = cnn_layer.output_shape
    reshape_layer = Reshape((cnn_shape[2], cnn_shape[3]),
                            input_shape=(cnn_shape[1], cnn_shape[2],
                                         cnn_shape[3]))
    encoder_inputs = reshape_layer(cnn_output)
    training_model, inf_enc_model, inf_dec_model = lstm_encoder_decoder_module(
                                  encoder_inputs, n_output, n_units,
                                  reg_lambda)

    # add cnn layer to beginning of encoder-decoder LSTM
    training_model = Model([cnn_inputs, training_model.input[1]],
                           training_model([encoder_inputs,
                                           training_model.input[1]]))
    inf_enc_model = Model(cnn_inputs, inf_enc_model(encoder_inputs))

    return training_model, inf_enc_model, inf_dec_model


def gru_1Dcnn_model(n_input_time, n_input_channel, n_output, n_filters,
                    filter_size, n_units, reg_lambda):
    """Creates a joint 1D CNN-GRU  model by adding a 1D convolutional layer to
    the front of an encoder-decoder GRU model.

    Args:
        n_input_time (int): Number of timesteps.
        n_input_channel (int): Number of channels.
        n_output (int): Cardinality of output space.
        n_filters (int): Number of convolutional filters.
        filter_size (int): Size (and stride) of convolutional filters.
        n_units (int): Number of units in LSTM layers.
        reg_lambda (float): L2 regularization parameter.

    Returns:
        (Functional, Functional, Functional): Encoder-decoder training model,
            encoder inference model, decoder inference model
    """
    cnn_inputs, cnn_layers = linear_cnn_1D_module(n_input_time,
                                                  n_input_channel, n_filters,
                                                  filter_size, reg_lambda)
    encoder_inputs = cnn_layers(cnn_inputs)
    training_model, inf_enc_model, inf_dec_model = gru_encoder_decoder_module(
                                  encoder_inputs, n_output, n_units,
                                  reg_lambda)

    # add cnn layer to beginning of encoder-decoder LSTM
    training_model = Model([cnn_inputs, training_model.input[1]],
                           training_model([encoder_inputs,
                                           training_model.input[1]]))
    inf_enc_model = Model(cnn_inputs, inf_enc_model(encoder_inputs))

    return training_model, inf_enc_model, inf_dec_model


def gru_3Dcnn_model(n_input_time, n_input_x, n_input_y, n_output,
                    n_filters, filter_size, n_units, reg_lambda):
    """Creates a joint 3D CNN-GRU  model by adding a 3D convolutional layer to
    the front of an encoder-decoder GRU model.

    Args:
        n_input_time (int): Number of timesteps.
        n_input_x (int): Width of image (size of channel map x-dimension).
        n_input_y (int): Height of image (size of channel map y-dimension).
        n_output (int): Cardinality of output space.
        n_filters (int): Number of convolutional filters.
        filter_size (int): Size (and stride) of convolutional filters.
        n_units (int): Number of units in LSTM layers.
        reg_lambda (float): L2 regularization parameter.

    Returns:
        (Functional, Functional, Functional): Encoder-decoder training model,
            encoder inference model, decoder inference model
    """
    cnn_inputs, cnn_layers = linear_cnn_3D_module(n_input_time, n_input_x,
                                                  n_input_y, n_filters,
                                                  filter_size, reg_lambda)
    encoder_inputs = cnn_layers(cnn_inputs)
    training_model, inf_enc_model, inf_dec_model = gru_encoder_decoder_module(
                                  encoder_inputs, n_output, n_units,
                                  reg_lambda)

    # add cnn layer to beginning of encoder-decoder LSTM
    training_model = Model([cnn_inputs, training_model.input[1]],
                           training_model([encoder_inputs,
                                           training_model.input[1]]))
    inf_enc_model = Model(cnn_inputs, inf_enc_model(encoder_inputs))

    return training_model, inf_enc_model, inf_dec_model
