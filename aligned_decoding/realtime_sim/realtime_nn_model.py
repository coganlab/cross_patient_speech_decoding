"""Lightning module and building blocks for a real-time RNN speech decoder.

Defines a stacked GRU encoder, a dense CTC classifier head, and a
LightningModule that combines them for CTC-based phoneme decoding with
sliding-window input reformatting.
"""

import torch
from torch import nn
from torchaudio.functional import edit_distance
from torchmetrics.wrappers import Running
from torchmetrics import CharErrorRate
import lightning as L

# from .ctc_decoder import greedy_decode_torch
from .ctc_decoder import greedy_decode_batch


BEAM_SIZE = 100


class StackedRNN(nn.Module):
    """Multi-layer GRU encoder.

    Wraps ``nn.GRU`` for stacking multiple recurrent layers with optional
    dropout and bidirectionality.
    """

    def __init__(self, input_size, hidden_size, n_layers, dropout=0.3,
                 bidirectional=False):
        """Initializes the stacked GRU.

        Args:
            input_size: Number of input features per time step.
            hidden_size: Number of hidden units per GRU layer.
            n_layers: Number of stacked GRU layers.
            dropout: Dropout probability between layers (ignored when
                n_layers is 1).
            bidirectional: If True, uses a bidirectional GRU.
        """
        super(StackedRNN, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, x, state=None):
        """Runs the GRU forward pass.

        Args:
            x: Input tensor of shape (B, T, input_size).
            state: Optional initial hidden state.

        Returns:
            Tuple of (output, hidden) where output has shape
            (B, T, hidden_size * num_directions).
        """
        output, hidden = self.rnn(x, state)
        return output, hidden


class DenseClassifier(nn.Module):
    """Single linear layer that maps RNN outputs to class logits per time step."""

    def __init__(self, input_size, n_classes):
        """Initializes the classifier.

        Args:
            input_size: Dimensionality of the RNN output.
            n_classes: Number of output classes (including CTC blank).
        """
        super(DenseClassifier, self).__init__()
        self.fc = nn.Linear(input_size, n_classes)

    def forward(self, x):
        """Applies the linear projection to each time step.

        Args:
            x: Input tensor of shape (B, T, input_size).

        Returns:
            Logits tensor of shape (B, T, n_classes).
        """
        # Apply the linear layer to each time step
        x = self.fc(x)
        return x


class RealtimeRNNModel(L.LightningModule):
    """CTC-based real-time RNN model for phoneme decoding.

    Combines a StackedRNN encoder, a DenseClassifier head, and
    sliding-window input reformatting. Trained with CTC loss and
    evaluated with phoneme error rate (PER).
    """

    def __init__(self, input_size, hidden_size, n_layers, n_classes,
                 dropout=0.3, win_size=14, stride=4, bidirectional=False,
                 learning_rate=1e-3, decay_steps=100, weight_decay=1e-5,
                 blank=0):
        """Initializes the model.

        Args:
            input_size: Number of input features per time step.
            hidden_size: GRU hidden-state dimensionality.
            n_layers: Number of stacked GRU layers.
            n_classes: Number of output classes (including CTC blank).
            dropout: Dropout probability for the GRU.
            win_size: Sliding-window width in time steps.
            stride: Sliding-window stride in time steps.
            bidirectional: If True, uses a bidirectional GRU.
            learning_rate: Initial learning rate for AdamW.
            decay_steps: Number of epochs over which the learning rate
                linearly decays to zero.
            weight_decay: L2 regularization strength for AdamW.
            blank: Index of the CTC blank label.
        """
        super(RealtimeRNNModel, self).__init__()
        self.save_hyperparameters()

        self.rnn = StackedRNN(input_size, hidden_size, n_layers, dropout,
                              bidirectional)
        # Orthogonal and Xavier initialization just like the reference
        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Trainable h0 (reference behavior)
        self.h0 = nn.Parameter(
            torch.zeros(n_layers * (2 if bidirectional else 1), 1, hidden_size)
        )
        nn.init.xavier_uniform_(self.h0)

        # Classifier
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = DenseClassifier(rnn_output_size, n_classes)
        with torch.no_grad():
            self.classifier.fc.bias[:] = -2.0     # suppress all phonemes initially
            self.classifier.fc.bias[blank] = 2.0  # encourage blank early in training

        self.criterion = nn.CTCLoss(blank=blank, zero_infinity=True)

        per_metric = CharErrorRate()
        self.val_PER_running = Running(per_metric, window=100)


    def forward(self, x):
        """Runs the full forward pass: window reformatting, RNN, classifier.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Logits tensor of shape (B, n_windows, n_classes).
        """
        # x: (B, T, C)
        x = self.reformat_time_windows(x)

        B = x.size(0)
        h0 = self.h0.expand(-1, B, -1).contiguous()

        out, _ = self.rnn(x, h0)
        logits = self.classifier(out)
        return logits

    def reformat_time_windows(self, x):
        """
        Inspired by https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/blob/main/model_training/rnn_model.py
        reformatting of raw timesteps into overlapping windows at desired prediction latency
        (latency defined by stride)
        Convert (B, T, C) into right-aligned sliding windows (B, n_windows, C*win).
        """
        B, T, C = x.size()
        win = self.hparams.win_size
        stride = self.hparams.stride

        # (B, C, 1, T)
        x = x.permute(0, 2, 1).unsqueeze(2)

        # Unfold on the time dimension
        x_unfold = x.unfold(dimension=3, size=win, step=stride)
        # (B, C, 1, n_windows, win)

        # Remove dummy dimension -> (B, C, n_windows, win)
        x_unfold = x_unfold.squeeze(2)

        # -> (B, n_windows, win, C)
        x_unfold = x_unfold.permute(0, 2, 3, 1)

        # Flatten window and channel dims
        x = x_unfold.reshape(B, x_unfold.shape[1], win * C)

        return x

    def training_step(self, batch, batch_idx):
        """Computes and logs CTC training loss for a single batch.

        Args:
            batch: Tuple of (inputs, targets, input_lengths, target_lengths).
            batch_idx: Index of the current batch.

        Returns:
            Scalar CTC loss.
        """
        inputs, targets, input_lengths, target_lengths = batch

        # account for sliding window adjustment in input lengths
        input_lengths_adj = ((input_lengths - self.hparams.win_size) // self.hparams.stride) + 1

        outputs = self(inputs)  # (B, T, C)
        log_probs = outputs.log_softmax(2)

        # CTCLoss expects (T, B, C)
        loss = self.criterion(
            log_probs.permute(1, 0, 2),
            targets,
            input_lengths_adj,
            target_lengths,
        )

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Computes CTC loss and phoneme error rate for a validation batch.

        Args:
            batch: Tuple of (inputs, targets, input_lengths, target_lengths).
            batch_idx: Index of the current batch.

        Returns:
            Scalar CTC loss.
        """
        inputs, targets, input_lengths, target_lengths = batch

        # account for sliding window adjustment in input lengths
        input_lengths_adj = ((input_lengths - self.hparams.win_size) // self.hparams.stride) + 1

        outputs = self(inputs)  # (B, T, C)
        log_probs = outputs.log_softmax(2)

        # CTCLoss expects (T, B, C)
        loss = self.criterion(
            log_probs.permute(1, 0, 2),
            targets,
            input_lengths_adj,
            target_lengths,
        )

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            decoded = greedy_decode_batch(log_probs)

            per = calc_PER(decoded, targets, target_lengths)
            self.val_PER_running.update(decoded, targets)

            self.log('val_PER', per, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_PER_running', self.val_PER_running, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Computes and logs CTC loss for a test batch.

        Args:
            batch: Tuple of (inputs, targets, input_lengths, target_lengths).
            batch_idx: Index of the current batch.

        Returns:
            Scalar CTC loss.
        """
        inputs, targets, input_lengths, target_lengths = batch
        outputs = self(inputs)
        outputs = outputs.log_softmax(2).permute(1, 0, 2)
        loss = self.criterion(outputs, targets, input_lengths, target_lengths)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        """Configures AdamW optimizer with linear learning-rate decay.

        Returns:
            Tuple of ([optimizer], [scheduler]).
        """
        hps = self.hparams
        optimizer = torch.optim.AdamW(self.parameters(), lr=hps.learning_rate,
                                      weight_decay=hps.weight_decay)

        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=hps.decay_steps  # Number of epochs for linear decay
        )

        return [optimizer], [scheduler]


def calc_PER(decoded, targets, target_lengths):
    """Computes the phoneme error rate (PER) for a batch.

    Args:
        decoded: List of 1-D LongTensors of decoded label sequences.
        targets: Ground-truth label tensor of shape (B, L).
        target_lengths: 1-D tensor of true target sequence lengths.

    Returns:
        PER as a percentage (0-100).
    """
    edit_dist = sum(
        edit_distance(pred, tgt[:l])
        for pred, tgt, l in zip(decoded, targets, target_lengths)
    )

    per = edit_dist / target_lengths.sum() * 100
    return per
