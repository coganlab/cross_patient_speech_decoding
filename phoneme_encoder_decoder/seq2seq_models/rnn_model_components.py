"""
Model components for RNN-based phoneme encoder-decoder models (e.g. LSTM, GRU,
CNN feature extraction layers, etc.)

Author: Zac Spalding
Adapted from code by Kumar Duraivel
"""

from keras.models import Model
from keras.layers import (Dense, LSTM, GRU, Input, Conv1D, Conv3D,
                          Bidirectional, Average)
from keras.regularizers import L2


def linear_cnn_1D_module(n_input_time, n_input_channel, n_filters, filter_size,
                         reg_lambda):
    """Creates a linear 1D (temporal) convolutional neural network via Keras.

    Args:
        n_input_time (int): Number of timesteps
        n_input_channel (int): Number of channels
        n_filters (int): Number of convolutional filters
        filter_size (int): Size (and stride) of convolutional filters
        reg_lambda (float): L2 regularization parameter

    Returns:
        (kerasTensor, Conv1D): (Input to CNN, 1D CNN layer)
    """
    cnn_inputs = Input(shape=(n_input_time, n_input_channel))
    cnn_layer = Conv1D(n_filters, kernel_size=filter_size, strides=filter_size,
                       data_format='channels_last', activation='linear',
                       kernel_regularizer=L2(reg_lambda),
                       bias_regularizer=L2(reg_lambda))
    return cnn_inputs, cnn_layer


def linear_cnn_3D_module(n_input_depth, n_input_x, n_input_y, n_filters,
                         filter_size, reg_lambda):
    """Creates a linear 3D convolutional neural network via Keras.

    Args:
        n_input_depth (int): Depth of image (number of timesteps).
        n_input_x (int): Width of image (size of channel map x-dimension).
        n_input_y (int): Height of image (size of channel map y-dimension).
        n_filters (int): Number of convolutional filters.
        filter_size (int): Size (and stride) of convolutional filters.
        reg_lambda (float): L2 regularization parameter.

    Returns:
        (kerasTensor, Conv3D): (Input to CNN, 3D CNN layer)
    """
    cnn_inputs = Input(shape=(n_input_x, n_input_y, n_input_depth, 1))
    cnn_layer = Conv3D(n_filters, kernel_size=filter_size, strides=filter_size,
                       data_format='channels_last', activation='linear',
                       kernel_regularizer=L2(reg_lambda),
                       bias_regularizer=L2(reg_lambda))
    return cnn_inputs, cnn_layer


def lstm_enc_dec_module(encoder_inputs, n_output, n_units, reg_lambda):
    """Creates an LSTM encoder-decoder model via Keras. Designed to be used
    following another network layer (e.g. CNN)

    Args:
        encoder_inputs (KerasTensor): Input to encoder model. Should be of the
            form as returned by kears.layers.Input().
        n_output (int): Cardinality of output space.
        n_units (int): Number of units in LSTM layers.
        reg_lambda (float): L2 regularization parameter.

    Returns:
        (Functional, Functional, Functional): Encoder-decoder training model,
            encoder inference model, decoder inference model
    """
    # define training encoder
    encoder = LSTM(n_units, return_state=True,
                   kernel_regularizer=L2(reg_lambda),
                   recurrent_regularizer=L2(reg_lambda),
                   bias_regularizer=L2(reg_lambda))
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True,
                        kernel_regularizer=L2(reg_lambda),
                        recurrent_regularizer=L2(reg_lambda),
                        bias_regularizer=L2(reg_lambda))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # combine encoder and decoder into training model
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs,
                           name='training_initial')

    # define inference encoder
    inf_enc_model = Model(encoder_inputs, encoder_states,
                          name='inf_enc_initial')

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
                                        decoder_inputs,
                                        initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    inf_dec_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states,
                          name='inf_dec')

    return training_model, inf_enc_model, inf_dec_model


def gru_enc_dec_module(encoder_inputs, n_output, n_units, reg_lambda):
    """Creates a  GRU encoder-decoder model via Keras. Designed to be used
    following another network layer (e.g. CNN)

    Args:
        encoder_inputs (KerasTensor): Input to encoder model. Should be of the
            form as returned by kears.layers.Input().
        n_output (int): Cardinality of output space.
        n_units (int): Number of units in LSTM layers.
        reg_lambda (float): L2 regularization parameter.

    Returns:
        (Functional, Functional, Functional): Encoder-decoder training model,
            encoder inference model, decoder inference model
    """
    # define training encoder
    encoder = GRU(n_units, return_state=True,
                  kernel_regularizer=L2(reg_lambda),
                  recurrent_regularizer=L2(reg_lambda),
                  bias_regularizer=L2(reg_lambda))
    encoder_outputs, state_h = encoder(encoder_inputs)
    encoder_states = [state_h]

    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_gru = GRU(n_units, return_sequences=True, return_state=True,
                      kernel_regularizer=L2(reg_lambda),
                      recurrent_regularizer=L2(reg_lambda),
                      bias_regularizer=L2(reg_lambda))
    decoder_outputs, _ = decoder_gru(decoder_inputs,
                                     initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # combine encoder and decoder into training model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs,
                  name='training_initial')

    # define inference encoder
    inf_enc_model = Model(encoder_inputs, encoder_states,
                          name='inf_enc_initial')

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h]
    decoder_outputs, state_h = decoder_gru(decoder_inputs,
                                           initial_state=decoder_states_inputs)
    decoder_states = [state_h]
    decoder_outputs = decoder_dense(decoder_outputs)
    inf_dec_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states,
                          name='inf_dec')

    return model, inf_enc_model, inf_dec_model


def bi_lstm_enc_dec_module(encoder_inputs, n_output, n_units, reg_lambda):
    """Creates a biderctional LSTM encoder-decoder model via Keras. Designed to
    be used following another network layer (e.g. CNN)

    Args:
        encoder_inputs (KerasTensor): Input to encoder model. Should be of the
            form as returned by kears.layers.Input().
        n_output (int): Cardinality of output space.
        n_units (int): Number of units in LSTM layers.
        reg_lambda (float): L2 regularization parameter.

    Returns:
        (Functional, Functional, Functional): Encoder-decoder training model,
            encoder inference model, decoder inference model
    """
    # define training encoder
    encoder = Bidirectional(LSTM(n_units, return_state=True,
                                 kernel_regularizer=L2(reg_lambda),
                                 recurrent_regularizer=L2(reg_lambda),
                                 bias_regularizer=L2(reg_lambda)))
    (encoder_outputs, forward_state_h, forward_state_c, backward_state_h,
     backward_state_c) = encoder(encoder_inputs)
    state_h = Average()([forward_state_h, backward_state_h])
    state_c = Average()([forward_state_c, backward_state_c])
    encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True,
                        kernel_regularizer=L2(reg_lambda),
                        recurrent_regularizer=L2(reg_lambda),
                        bias_regularizer=L2(reg_lambda))
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # combine encoder and decoder into training model
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs,
                           name='training_initial')

    # define inference encoder
    inf_enc_model = Model(encoder_inputs, encoder_states,
                          name='inf_enc_initial')

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
                                        decoder_inputs,
                                        initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    inf_dec_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states,
                          name='inf_dec')

    return training_model, inf_enc_model, inf_dec_model


def bi_gru_enc_dec_module(encoder_inputs, n_output, n_units, reg_lambda):
    """Creates a biderctional GRU encoder-decoder model via Keras. Designed to
    be used following another network layer (e.g. CNN)

    Args:
        encoder_inputs (KerasTensor): Input to encoder model. Should be of the
            form as returned by kears.layers.Input().
        n_output (int): Cardinality of output space.
        n_units (int): Number of units in LSTM layers.
        reg_lambda (float): L2 regularization parameter.

    Returns:
        (Functional, Functional, Functional): Encoder-decoder training model,
            encoder inference model, decoder inference model
    """
    # define training encoder
    encoder = Bidirectional(GRU(n_units, return_state=True,
                                kernel_regularizer=L2(reg_lambda),
                                recurrent_regularizer=L2(reg_lambda),
                                bias_regularizer=L2(reg_lambda)))
    _, forward_state_h, backward_state_h = encoder(encoder_inputs)
    state_h = Average()([forward_state_h, backward_state_h])
    encoder_states = [state_h]

    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_gru = GRU(n_units, return_sequences=True, return_state=True,
                      kernel_regularizer=L2(reg_lambda),
                      recurrent_regularizer=L2(reg_lambda),
                      bias_regularizer=L2(reg_lambda))
    decoder_outputs, _ = decoder_gru(decoder_inputs,
                                     initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # combine encoder and decoder into training model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs,
                  name='trainingl_initial')

    # define inference encoder
    inf_enc_model = Model(encoder_inputs, encoder_states,
                          name='inf_encl_initial')

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h]
    decoder_outputs, state_h = decoder_gru(decoder_inputs,
                                           initial_state=decoder_states_inputs)
    decoder_states = [state_h]
    decoder_outputs = decoder_dense(decoder_outputs)
    inf_dec_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states,
                          name='inf_dec')

    return model, inf_enc_model, inf_dec_model
