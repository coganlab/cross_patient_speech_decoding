"""Neural network model definitions for speech decoding.

Provides Lightning-based model architectures including temporal convolution,
RNN, sequence-to-sequence, Transformer, and hybrid CNN-Transformer models
for neural speech decoding tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from torchmetrics.functional.classification import multiclass_confusion_matrix

class BaseLightningModel(L.LightningModule):
    """Base Lightning module with shared training, validation, and test logic.

    Provides common step methods and optimizer configuration for all
    classification models.

    Args:
        criterion: Loss function. Defaults to CrossEntropyLoss.
        learning_rate: Learning rate for optimizer. Defaults to 1e-3.
        l2_reg: L2 regularization weight decay. Defaults to 1e-5.
    """

    def __init__(self, criterion=nn.CrossEntropyLoss(), learning_rate=1e-3,
                 l2_reg=1e-5):
        super(BaseLightningModel, self).__init__()
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg

    def training_step(self, batch, batch_idx):
        """Computes training loss and accuracy for a single batch.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the current batch.

        Returns:
            Tensor: Training loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'train_loss': loss, 'train_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Computes validation loss and accuracy for a single batch.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the current batch.

        Returns:
            Tensor: Validation loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Computes test loss and accuracy for a single batch.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the current batch.

        Returns:
            Tensor: Test loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Runs forward pass for prediction.

        Args:
            batch: Tuple of (inputs, targets); targets are ignored.
            batch_idx: Index of the current batch.

        Returns:
            Tensor: Model predictions.
        """
        x, _ = batch
        return self(x)
    
    def configure_optimizers(self):
        """Configures the AdamW optimizer with weight decay.

        Returns:
            torch.optim.AdamW: Configured optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                weight_decay=self.l2_reg)
    

class TemporalConvRNN(BaseLightningModel):
    """Temporal convolution followed by GRU for sequence classification.

    Applies 1D convolution over the temporal dimension, then feeds the
    filtered signal through a GRU, with an optional fully-connected head.

    Args:
        in_channels: Number of input features/channels.
        n_filters: Number of convolutional filters.
        num_classes: Number of output classes.
        hidden_size: GRU hidden state dimensionality.
        n_layers: Number of GRU layers.
        kernel_size: Convolution kernel size.
        dim_fc: Fully-connected layer dimension(s). None uses GRU output
            directly; a list creates a sequential FC stack.
        stride: Convolution stride. Defaults to 1.
        padding: Convolution padding. Defaults to 0.
        cnn_dropout: Dropout rate after convolution. Defaults to 0.3.
        rnn_dropout: Dropout rate in GRU. Defaults to 0.3.
        learning_rate: Learning rate. Defaults to 1e-3.
        l2_reg: L2 regularization weight. Defaults to 1e-5.
        criterion: Loss function. Defaults to CrossEntropyLoss.
        activation: Whether to apply ReLU after convolution. Defaults to True.
        decay_iters: Number of epochs for LR linear decay. Defaults to 20.
    """

    def __init__(self, in_channels, n_filters, num_classes, hidden_size, n_layers,
                 kernel_size, dim_fc=None, stride=1, padding=0, cnn_dropout=0.3,
                 rnn_dropout=0.3, learning_rate=1e-3, l2_reg=1e-5,
                 criterion=nn.CrossEntropyLoss(), activation=True,
                 decay_iters=20):
        super(TemporalConvRNN, self).__init__(learning_rate=learning_rate,
                                              l2_reg=l2_reg, criterion=criterion)
        self.num_classes = num_classes
        self.decay_iters = decay_iters
        self.temporal_conv = TemporalConv(in_channels, n_filters, kernel_size,
                                            stride, padding, cnn_dropout,
                                            activation=activation)
        if dim_fc is None:
            self.rnn = SimpleGRU(n_filters, hidden_size, num_classes, n_layers,
                                    dropout=rnn_dropout)
            self.fc = None
        elif isinstance(dim_fc, list):
             self.rnn = SimpleGRU(n_filters, hidden_size, dim_fc[0], n_layers,
                                    dropout=rnn_dropout)
             self.fc = nn.Sequential(*[nn.Linear(dim_fc[i], dim_fc[i+1])
                                        for i in range(len(dim_fc)-1)] +
                                        [nn.Linear(dim_fc[-1], num_classes)])
        else:
            self.rnn = SimpleGRU(n_filters, hidden_size, dim_fc, n_layers,
                                    dropout=rnn_dropout)
            self.fc = nn.Linear(dim_fc, num_classes)

    def forward(self, x):
        """Forward pass through temporal convolution and GRU.

        Args:
            x: Input tensor of shape (batch_size, n_timepoints, n_features).

        Returns:
            Tensor: Class logits of shape (batch_size, num_classes).
        """
        # x is of shape (batch_size, n_timepoints, n_features) coming in
        x = x.permute(0, 2, 1)
        x = self.temporal_conv(x)
        x = x. permute(0, 2, 1) # (batch_size, n_timepoints, n_filters)
        x = self.rnn(x)
        if self.fc is not None:
            x = self.fc(x)
        return x
    
    def configure_optimizers(self):
        """Configures AdamW optimizer with linear LR decay schedule.

        Returns:
            dict: Optimizer and LR scheduler configuration.
        """
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                  weight_decay=self.l2_reg)
        # # linear increase to learning rate for first decay_iters iterations
        # lr_sch = torch.optim.lr_scheduler.LinearLR(optim, total_iters=self.decay_iters)

        # linear decay of learning rate
        lr_sch = torch.optim.lr_scheduler.LinearLR(optim,
                                                   start_factor=1.0,
                                                   end_factor=0.01,
                                                   total_iters=self.decay_iters)

        # # exponential decay of learning rate
        # lr_sch = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)
        optim_config = {'optimizer': optim,
                        'lr_scheduler': {'scheduler': lr_sch,
                                         'interval': 'epoch',
                                         'frequency': 1}}
        return optim_config
    

class Seq2SeqRNN(BaseLightningModel):
    """Sequence-to-sequence model with temporal convolution, encoder RNN, and decoder RNN.

    Encodes temporally convolved features with a bidirectional RNN, then
    autoregressively decodes an output sequence with optional teacher forcing.

    Args:
        in_channels: Number of input features/channels.
        n_filters: Number of convolutional filters.
        hidden_size: RNN hidden state dimensionality.
        num_classes: Number of output classes per sequence position.
        n_enc_layers: Number of encoder RNN layers.
        n_dec_layers: Number of decoder RNN layers.
        kernel_size: Convolution kernel size.
        stride: Convolution stride. Defaults to 1.
        padding: Convolution padding. Defaults to 0.
        cnn_dropout: Dropout rate after convolution. Defaults to 0.3.
        rnn_dropout: Dropout rate in RNNs. Defaults to 0.3.
        model_type: RNN variant, 'gru' or 'lstm'. Defaults to 'gru'.
        learning_rate: Learning rate. Defaults to 1e-3.
        l2_reg: L2 regularization weight. Defaults to 1e-5.
        criterion: Loss function. Defaults to CrossEntropyLoss.
        activation: Whether to apply ReLU after convolution. Defaults to True.
        seq_length: Output sequence length to decode. Defaults to 3.
        decay_iters: Epochs for LR linear decay. Defaults to 20.
    """

    def __init__(self, in_channels, n_filters, hidden_size, num_classes,
                 n_enc_layers, n_dec_layers, kernel_size, stride=1, padding=0,
                 cnn_dropout=0.3, rnn_dropout=0.3, model_type='gru', learning_rate=1e-3,
                 l2_reg=1e-5, criterion=nn.CrossEntropyLoss(), activation=True,
                 seq_length=3, decay_iters=20):
        super(Seq2SeqRNN, self).__init__(learning_rate=learning_rate,
                                         l2_reg=l2_reg, criterion=criterion)
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.temporal_conv = TemporalConv(in_channels, n_filters, kernel_size,
                                          stride, padding, cnn_dropout,
                                          activation=activation)
        self.encoder = EncoderRNN(n_filters, hidden_size, n_enc_layers,
                                  dropout=rnn_dropout, model_type=model_type)
        self.decoder = DecoderRNN(hidden_size, num_classes, n_dec_layers,
                                  dropout=rnn_dropout, model_type=model_type)
        self.decay_iters = decay_iters

    def forward(self, x, y=None, teacher_forcing_ratio=0.5):
        """Forward pass through encoder-decoder with optional teacher forcing.

        Args:
            x: Input tensor of shape (batch_size, n_timepoints, n_features).
            y: Target sequence of shape (batch_size, seq_length) for teacher
                forcing. None disables teacher forcing.
            teacher_forcing_ratio: Probability of using ground-truth token as
                next decoder input. Defaults to 0.5.

        Returns:
            Tensor: Predicted logits of shape
                (batch_size, seq_length, num_classes).
        """
        # x is of shape (batch_size, n_timepoints, n_features) coming in
        # y is of shape (batch_size, seq_length) coming in if not None

        x = x.permute(0, 2, 1) # (batch_size, n_features, n_timepoints)
        x = self.temporal_conv(x)  # pass data through temporal conv layer
        x = x.permute(0, 2, 1)  # (batch_size, n_timepoints, n_filteers)

        # encode temporally convolved features with RNN
        _, enc_hidden = self.encoder(x)

        # first hidden state of decoder is hidden state of encoder
        # only using last hidden state from encoder, so need to repeat for all
        # decoder layers
        dec_hidden = enc_hidden.repeat(self.decoder.rnn.num_layers, 1, 1)

        # create a tensor of start tokens for each batch - decoder will predict
        # 0 -> (num_classes-1), so we make the start token num_classes
        batch_size = x.size(0)
        start_tokens = torch.full((batch_size,), self.num_classes,
                                  dtype=torch.long, device=x.device)
        dec_input = start_tokens

        # generate output sequence predictions
        outputs = []
        for i in range(self.seq_length):
            dec_output, dec_hidden = self.decoder(dec_input, dec_hidden)
            outputs.append(dec_output)

            if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = y[:, i]  # teacher forcing
            else:
                # use predictions as input for next sequence element
                dec_input = dec_output.argmax(1)

        # (batch_size, seq_length, num_classes)
        outputs = torch.stack(outputs, dim=1)  
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Computes training loss and accuracy with teacher forcing.

        Args:
            batch: Tuple of (inputs, target_sequences).
            batch_idx: Index of the current batch.

        Returns:
            Tensor: Training loss.
        """
        x, y = batch
        # use teacher forcing during training
        y_hat = self(x, y, teacher_forcing_ratio=0.5)  # (batch_size, seq_length, num_classes)
        y_hat = y_hat.view(-1, self.num_classes)  # (batch_size*seq_length, num_classes)
        y = y.view(-1)  # (batch_size*seq_length)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'train_loss': loss, 'train_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Computes validation loss and accuracy without teacher forcing.

        Args:
            batch: Tuple of (inputs, target_sequences).
            batch_idx: Index of the current batch.

        Returns:
            Tensor: Validation loss.
        """
        x, y = batch
        # no teacher forcing during validation/testing/prediction
        y_hat = self(x, y, teacher_forcing_ratio=0)
        y_hat = y_hat.view(-1, self.num_classes)
        y = y.view(-1)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Computes test loss and accuracy without teacher forcing.

        Args:
            batch: Tuple of (inputs, target_sequences).
            batch_idx: Index of the current batch.

        Returns:
            Tensor: Test loss.
        """
        x, y = batch
        y_hat = self(x, y, teacher_forcing_ratio=0)
        y_hat = y_hat.view(-1, self.num_classes)
        y = y.view(-1)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configures AdamW optimizer with linear LR decay schedule.

        Returns:
            dict: Optimizer and LR scheduler configuration.
        """
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                  weight_decay=self.l2_reg)
        # # linear increase to learning rate for first decay_iters iterations
        # lr_sch = torch.optim.lr_scheduler.LinearLR(optim, total_iters=self.decay_iters)

        # linear decay of learning rate
        lr_sch = torch.optim.lr_scheduler.LinearLR(optim,
                                                   start_factor=1.0,
                                                   end_factor=0.01,
                                                   total_iters=self.decay_iters)

        # # exponential decay of learning rate
        # lr_sch = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)
        optim_config = {'optimizer': optim,
                        'lr_scheduler': {'scheduler': lr_sch,
                                         'interval': 'epoch',
                                         'frequency': 1}}
        return optim_config
    

class TCN_classifier(BaseLightningModel):
    """Temporal convolutional network classifier with max-pooling and FC head.

    Args:
        in_channels: Number of input features/channels.
        num_classes: Number of output classes.
        dim_fc: FC layer dimension(s). A list creates a sequential stack.
        kernel_size: Convolution kernel size.
        stride: Convolution stride. Defaults to 1.
        padding: Convolution padding. Defaults to 0.
        dropout: Dropout rate. Defaults to 0.3.
        learning_rate: Learning rate. Defaults to 1e-3.
        l2_reg: L2 regularization weight. Defaults to 1e-5.
        criterion: Loss function. Defaults to CrossEntropyLoss.
        activation: Whether to apply ReLU after convolution. Defaults to True.
    """

    def __init__(self, in_channels, num_classes, dim_fc, kernel_size, stride=1,
                 padding=0, dropout=0.3, learning_rate=1e-3, l2_reg=1e-5,
                 criterion=nn.CrossEntropyLoss(), activation=True):
        super(TCN_classifier, self).__init__(learning_rate=learning_rate,
                                              l2_reg=l2_reg, criterion=criterion)
        self.num_classes = num_classes
        self.temporal_conv = TemporalConv(in_channels, dim_fc[0], kernel_size,
                                          stride, padding, dropout,
                                          activation=activation)
        if isinstance(dim_fc, list):
            self.fc = nn.Sequential(*[nn.Linear(dim_fc[i], dim_fc[i+1])
                                      for i in range(len(dim_fc)-1)] +
                                    [nn.Linear(dim_fc[-1], num_classes)])
        else:
            self.fc = nn.Linear(dim_fc, num_classes)
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.criterion = criterion

    def forward(self, x):
        """Forward pass through temporal convolution, max-pool, and FC layers.

        Args:
            x: Input tensor of shape (batch_size, n_timepoints, n_features).

        Returns:
            Tensor: Class logits of shape (batch_size, num_classes).
        """
        # x is of shape (batch_size, n_timepoints, n_features) coming in
        x = x.permute(0, 2, 1)  # (batch_size, n_features, n_timepoints)
        x = self.temporal_conv(x)

        ##### CHOOSE FROM 1 OF THE 3 BELOW PRIOR TO FC INPUT #####
        # x = x.mean(dim=2)  # mean pooling, (batch_size, d_model)
        x, _ = torch.max(x, dim=2) # max pooling, (batch_size, d_model)
        # x = x[:, :, -1]  # get the last timepoint, (batch_size, d_model)

        x = self.fc(x)
        return x
    

class Transformer(BaseLightningModel):
    """Transformer encoder classifier with positional encoding and mean-pooling.

    Args:
        in_channels: Number of input features (must equal d_model).
        num_classes: Number of output classes.
        d_model: Transformer model dimensionality.
        kernel_size: Unused; kept for API consistency.
        stride: Unused. Defaults to 1.
        padding: Unused. Defaults to 0.
        n_head: Number of attention heads. Defaults to 8.
        num_layers: Number of Transformer encoder layers. Defaults to 3.
        dim_fc: Feed-forward dimension in each encoder layer. Defaults to 128.
        dropout: Dropout rate. Defaults to 0.3.
        learning_rate: Learning rate. Defaults to 1e-3.
        l2_reg: L2 regularization weight. Defaults to 1e-5.
        criterion: Loss function. Defaults to CrossEntropyLoss.
    """

    def __init__(self, in_channels, num_classes, d_model, kernel_size, stride=1, padding=0,
                 n_head=8, num_layers=3, dim_fc=128, dropout=0.3, learning_rate=1e-3, l2_reg=1e-5,
                 criterion=nn.CrossEntropyLoss()):
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_fc, dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.criterion = criterion

    def forward(self, x):
        """Forward pass through positional encoding, Transformer, and FC.

        Args:
            x: Input tensor of shape (batch_size, n_timepoints, d_model).

        Returns:
            Tensor: Class logits of shape (batch_size, num_classes).
        """
        # x is of shape (batch_size, n_timepoints, n_features) coming in
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        ##### CHOOSE FROM 1 OF THE 3 BELOW PRIOR TO FC INPUT #####
        x = x.mean(dim=1)  # mean pooling, (batch_size, d_model)
        # x, _ = torch.max(x, dim=1) # max pooling, (batch_size, d_model)
        # x = x[:, -1, :]  # get the last timepoint, (batch_size, d_model)

        x = self.fc(x)
        return x
    

class CNNTransformer(BaseLightningModel):
    """Hybrid CNN-Transformer encoder classifier.

    Applies temporal convolution to project features into d_model dimensions,
    adds positional encoding, then passes through a Transformer encoder
    with mean-pooling and a linear classification head.

    Args:
        in_channels: Number of input features/channels.
        num_classes: Number of output classes.
        d_model: Transformer model dimensionality.
        kernel_size: Convolution kernel size.
        stride: Convolution stride. Defaults to 1.
        padding: Convolution padding. Defaults to 0.
        n_head: Number of attention heads. Defaults to 8.
        num_layers: Number of Transformer encoder layers. Defaults to 3.
        dim_fc: Feed-forward dimension in each encoder layer. Defaults to 128.
        cnn_dropout: Dropout rate after convolution. Defaults to 0.2.
        transformer_dropout: Dropout rate in Transformer. Defaults to 0.3.
        learning_rate: Learning rate. Defaults to 1e-3.
        warmup: Warmup epochs for cosine LR schedule. Defaults to 20.
        max_epochs: Max epochs for cosine LR schedule. Defaults to 500.
        l2_reg: L2 regularization weight. Defaults to 1e-5.
        criterion: Loss function. Defaults to CrossEntropyLoss.
        activation: Whether to apply ReLU after convolution. Defaults to True.
    """

    def __init__(self, in_channels, num_classes, d_model, kernel_size, stride=1, padding=0,
                 n_head=8, num_layers=3, dim_fc=128, cnn_dropout=0.2,
                 transformer_dropout=0.3, learning_rate=1e-3, 
                 warmup=20, max_epochs=500, l2_reg=1e-5,
                 criterion=nn.CrossEntropyLoss(), activation=True):
        super(CNNTransformer, self).__init__()
        self.num_classes = num_classes
        self.temporal_conv = TemporalConv(in_channels, d_model, kernel_size,
                                          stride, padding, cnn_dropout,
                                          activation=activation)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_fc,
                                                    transformer_dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.criterion = criterion
        self.warmup = warmup
        self.max_epochs = max_epochs

    def forward(self, x):
        """Forward pass through CNN, positional encoding, Transformer, and FC.

        Args:
            x: Input tensor of shape (batch_size, n_timepoints, n_features).

        Returns:
            Tensor: Class logits of shape (batch_size, num_classes).
        """
        # x is of shape (batch_size, n_timepoints, n_features) coming in
        x = x.permute(0, 2, 1)  # (batch_size, n_features, n_timepoints)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)  # (batch_size, n_timepoints, d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        ##### CHOOSE FROM 1 OF THE 3 BELOW PRIOR TO FC INPUT #####
        x = x.mean(dim=1)  # mean pooling, (batch_size, d_model)
        # x, _ = torch.max(x, dim=1) # max pooling, (batch_size, d_model)
        # x = x[:, -1, :]  # get the last timepoint, (batch_size, d_model)

        x = self.fc(x)
        return x

    def configure_optimizers(self):
        """Configures AdamW optimizer with cosine warmup LR schedule.

        Returns:
            torch.optim.AdamW: Configured optimizer.
        """
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        # optim = torch.optim.RAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg, decoupled_weight_decay=True)
        self.lr_sch = CosineWarmupScheduler(optim, self.warmup, self.max_epochs)
        # lr_sch_conf = {'scheduler': self.lr_sch, 'interval': 'epoch'}
        # optim_dict = {'optimizer': optim, 'lr_scheduler': lr_sch_conf}
        # return optim_dict
        return optim
    
    def optimizer_step(self, *args, **kwargs):
        """Steps the optimizer and advances the cosine warmup LR scheduler."""
        super().optimizer_step(*args, **kwargs)
        self.lr_sch.step()
    

class TemporalConv(nn.Module):
    """1D temporal convolution block with batch norm, ReLU, and dropout.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Convolution kernel size.
        stride: Convolution stride. Defaults to 1.
        padding: Convolution padding. Defaults to 0.
        dropout: Dropout rate. Defaults to 0.2.
        activation: Whether to apply ReLU. Defaults to True.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dropout=0.2, activation=True):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        """Applies Conv1d, batch norm, optional ReLU, and dropout.

        Args:
            x: Input tensor of shape (batch_size, in_channels, n_timepoints).

        Returns:
            Tensor: Output of shape (batch_size, out_channels, n_timepoints').
        """
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        x = self.dropout(x)
        return x
    

class EncoderRNN(nn.Module):
    """Bidirectional RNN encoder (GRU or LSTM).

    Produces a single summary hidden state by summing forward and backward
    directions from the last layer.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Hidden state dimensionality.
        num_layers: Number of RNN layers.
        dropout: Dropout rate between RNN layers. Defaults to 0.3.
        model_type: RNN variant, 'gru' or 'lstm'. Defaults to 'gru'.

    Raises:
        ValueError: If model_type is not 'gru' or 'lstm'.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3,
                 model_type='gru'):
        super(EncoderRNN, self).__init__()
        self.model_type = model_type
        if self.model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)
        elif self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=True)
        else:
            raise ValueError('model_type must be one of "gru" or "lstm"')

    def forward(self, x):
        """Encodes input sequence and returns outputs with summary hidden state.

        Args:
            x: Input tensor of shape (batch_size, n_timepoints, n_features).

        Returns:
            tuple: (output, last_hidden) where output has shape
                (batch_size, n_timepoints, hidden_size * 2) and last_hidden
                has shape (1, batch_size, hidden_size) for GRU, or a tuple
                of (h, c) each of shape (batch_size, hidden_size) for LSTM.
        """
        # x is of shape (batch_size, n_timepoints, n_features) coming in

        # hidden = (num_layers * num_directions, batch_size, hidden_size)
        if self.model_type == 'gru':
            output, hidden = self.rnn(x)

            # hidden = (num_layers, num_directions, batch_size, hidden_size)
            hidden = hidden.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)

            # extract forward and backward hidden states from last layer
            # (batch_size, hidden_size)
            last_forward = hidden[-1, 0, :, :]
            last_backward = hidden[-1, 1, :, :]

            # sum forward and backward hidden states
            last_hidden = last_forward + last_backward # (batch_size, hidden_size)
            last_hidden = last_hidden.unsqueeze(0)  # (1, batch_size, hidden_size)

        elif self.model_type == 'lstm': # modified for lstm state tuple
            output, (h_n, c_n) = self.rnn(x)
            
            h_n = h_n.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)
            c_n = c_n.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)

            last_forward_h = h_n[-1, 0, :, :]
            last_backward_h = h_n[-1, 1, :, :]
            last_hidden_h = last_forward_h + last_backward_h
            last_forward_c = c_n[-1, 0, :, :]
            last_backward_c = c_n[-1, 1, :, :]
            last_hidden_c = last_forward_c + last_backward_c

            last_hidden = (last_hidden_h, last_hidden_c)

        return output, last_hidden
    

class DecoderRNN(nn.Module):
    """Autoregressive RNN decoder with embedding and linear output.

    Uses an embedding layer for discrete input tokens, an RNN (GRU or LSTM),
    and a linear layer to produce class logits at each decoding step.

    Args:
        hidden_size: Hidden state dimensionality (also embedding dim).
        output_size: Number of output classes.
        num_layers: Number of RNN layers.
        dropout: Dropout rate between RNN layers. Defaults to 0.3.
        model_type: RNN variant, 'gru' or 'lstm'. Defaults to 'gru'.
    """

    def __init__(self, hidden_size, output_size, num_layers, dropout=0.3,
                 model_type='gru'):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size+1, hidden_size)

        if model_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Decodes one step given the previous token and hidden state.

        Args:
            x: Input token indices of shape (batch_size,).
            hidden: Previous hidden state from the RNN.

        Returns:
            tuple: (output, hidden) where output has shape
                (batch_size, output_size) and hidden is the updated state.
        """
        # x is of shape (batch_size,) coming in
        embed = self.embedding(x).unsqueeze(1)  # (batch_size, 1, hidden_size)
        output, hidden = self.rnn(embed, hidden)  # (batch_size, 1, hidden_size)
        output = self.fc_out(output.squeeze(1))  # (batch_size, output_size)
        return output, hidden


class SimpleGRU(nn.Module):
    """Simple GRU that returns the FC-projected output from the last timestep.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: GRU hidden state dimensionality.
        out_size: Output dimensionality after the linear layer.
        num_layers: Number of GRU layers.
        dropout: Dropout rate between GRU layers. Defaults to 0.3.
        bidir: Whether to use bidirectional GRU. Defaults to False.
    """

    def __init__(self, input_size, hidden_size, out_size, num_layers,
                 dropout=0.3, bidir=False):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidir,
                          dropout=dropout)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        """Runs GRU and projects the last timestep through an FC layer.

        Args:
            x: Input tensor of shape (batch_size, n_timepoints, n_features).

        Returns:
            Tensor: Output of shape (batch_size, out_size).
        """
        # x is of shape (batch_size, n_timepoints, n_features) coming in
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x
    

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to input embeddings.

    Args:
        d_model: Model dimensionality / number of features.
        max_len: Maximum sequence length supported. Defaults to 5000.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        dim = d_model
        if dim % 2 != 0:
            dim += 1
        self.encoding = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        self.encoding = self.encoding[:, :, :d_model]
        self.register_buffer('pos_encoding', self.encoding)

    def forward(self, x):
        """Adds positional encoding to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Input with positional encoding added, same shape.
        """
        return x + self.pos_encoding[:, :x.size(1), :]


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing LR scheduler with linear warmup.

    Linearly increases the learning rate during warmup epochs, then applies
    cosine decay for the remaining epochs.

    Args:
        optimizer: Wrapped optimizer.
        warmup: Number of warmup epochs.
        max_iters: Total number of epochs for the cosine schedule.
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        """Computes current learning rates for all parameter groups.

        Returns:
            list[float]: Scaled learning rates.
        """
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        """Computes the LR scaling factor for the given epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            float: Multiplicative factor for the base learning rate.
        """
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def cmat_acc(y_hat, y, num_classes):
    """Computes accuracy from the confusion matrix diagonal.

    Args:
        y_hat: Predicted logits of shape (batch_size, num_classes).
        y: Ground-truth labels of shape (batch_size,).
        num_classes: Total number of classes.

    Returns:
        Tensor: Scalar accuracy value.
    """
    y_pred = torch.argmax(y_hat, dim=1)
    cmat = multiclass_confusion_matrix(y_pred, y, num_classes)
    acc_cmat = cmat.diag().sum() / cmat.sum()
    return acc_cmat
