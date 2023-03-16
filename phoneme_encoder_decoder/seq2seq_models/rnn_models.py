"""
Construction of seq2seq RNN encoder-decoder models from model components in
rnn_model_components.py

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""


from keras.models import Model
from keras.layers import Reshape, Permute

from .rnn_model_components import (linear_cnn_1D_module, linear_cnn_3D_module,
                                   lstm_enc_dec_module, gru_enc_dec_module,
                                   bi_lstm_enc_dec_module,
                                   bi_gru_enc_dec_module)


def reshape_3d_cnn(cnn_input, cnn_layer):
    """Reshapes outputs of a 3D Cnn layer to a 2D tensor for input to
    encoder-decoder RNN model. Presverses depth/timestep dimension and combines
    other dimensions with dimension permutation before reshaping.

    Args:
        cnn_input (kerasTensor): Input to 3D CNN layer.
        cnn_layer (Conv3D): 3D CNN layer (output shape:
            (batch, width, height, depth, channels))

    Returns:
        KerasTensor: Reshaped output of 3D CNN layer (output shape:
            (batch, depth, width * height * channels))
    """
    cnn_shape = cnn_layer.output_shape
    w_dim, h_dim, d_dim, c_dim = cnn_shape[1:]
    cnn_output = cnn_layer(cnn_input)

    # reorder from (batch, width, height, depth, channels) to
    # (batch, depth, width, height, channels)
    permute_layer = Permute((d_dim, w_dim, h_dim, c_dim),
                            input_shape=cnn_shape[1:])

    # reshape from (batch, depth, width, height, channels) to
    # (batch, depth, width * height * channels)
    reshape_layer = Reshape((d_dim, w_dim * h_dim * c_dim),
                            input_shape=cnn_shape[1:])
    return reshape_layer(permute_layer(cnn_output))


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
    training_model, inf_enc_model, inf_dec_model = lstm_enc_dec_module(
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
    cnn_inputs, cnn_layer = linear_cnn_3D_module(n_input_time, n_input_x,
                                                 n_input_y, n_filters,
                                                 filter_size, reg_lambda)
    encoder_inputs = reshape_3d_cnn(cnn_inputs, cnn_layer)
    training_model, inf_enc_model, inf_dec_model = lstm_enc_dec_module(
                                  encoder_inputs, n_output, n_units,
                                  reg_lambda)

    # add cnn layer to beginning of encoder-decoder LSTM
    training_model = Model([cnn_inputs, training_model.input[1]],
                           training_model([encoder_inputs,
                                           training_model.input[1]]),
                           name='training_model_final')
    inf_enc_model = Model(cnn_inputs, inf_enc_model(encoder_inputs),
                          name='inf_enc_model_final')

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
    training_model, inf_enc_model, inf_dec_model = gru_enc_dec_module(
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
    encoder_inputs = reshape_3d_cnn(cnn_inputs, cnn_layers)
    training_model, inf_enc_model, inf_dec_model = gru_enc_dec_module(
                                  encoder_inputs, n_output, n_units,
                                  reg_lambda)

    # add cnn layer to beginning of encoder-decoder LSTM
    training_model = Model([cnn_inputs, training_model.input[1]],
                           training_model([encoder_inputs,
                                           training_model.input[1]]))
    inf_enc_model = Model(cnn_inputs, inf_enc_model(encoder_inputs))

    return training_model, inf_enc_model, inf_dec_model
