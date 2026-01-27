import torch
from torch import nn
from torchaudio.functional import edit_distance
import lightning as L

# from .ctc_decoder import greedy_decode_torch
from .ctc_decoder import greedy_decode_batch


BEAM_SIZE = 100


class StackedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout=0.3,
                 bidirectional=False):
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
        output, hidden = self.rnn(x, state)
        return output, hidden


class DenseClassifier(nn.Module):
    def __init__(self, input_size, n_classes):
        super(DenseClassifier, self).__init__()
        self.fc = nn.Linear(input_size, n_classes)

    def forward(self, x):
        # Apply the linear layer to each time step
        x = self.fc(x)
        return x


class RealtimeRNNModel(L.LightningModule):
    def __init__(self, input_size, hidden_size, n_layers, n_classes,
                 dropout=0.3, win_size=14, stride=4, bidirectional=False,
                 learning_rate=1e-3, decay_steps=100, weight_decay=1e-5,
                 blank=0):
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

    # def forward(self, x):
    #     # input x shape x: (batches, time_steps, channels)

    #     # reformat time_steps to overlapping prediction windows
    #     # x: (batches, n_windows, channels * win_size)
    #     x = self.reformat_time_windows(x)

    #     # generate predictions
    #     x = self.rnn(x)
    #     x = self.classifier(x)s
    #     return x
    # ----------------------------------------------------------------------
    # ✔ NEW: forward() with learned h₀ and matched behavior
    # ----------------------------------------------------------------------
    def forward(self, x):
        # x: (B, T, C)
        x = self.reformat_time_windows(x)

        B = x.size(0)
        h0 = self.h0.expand(-1, B, -1).contiguous()

        out, _ = self.rnn(x, h0)
        logits = self.classifier(out)
        return logits

    # def reformat_time_windows(self, x):
    #     """ Inspired by https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/blob/main/model_training/rnn_model.py
    #     reformatting of raw timesteps into overlapping windows at desired prediction latency
    #     (latency defined by stride)
    #     """
    #     hps = self.hparams
    #     win_size = hps.win_size
    #     stride = hps.stride

    #     # x: (batches, time_steps, channels)
    #     batch_size, time_steps, n_channels = x.size()

    #     # Breakout into overlapping windows
    #     x_unfold = x.unfold(1, win_size, stride)  # (batches, n_windows, channels, win_size))
    #     x_unfold = x_unfold.permute(0, 1, 3, 2)  # (batches, n_windows, win_size, channels)

    #     # Flatten the window dimension into the channel dimension
    #     x = x_unfold.contiguous().view(batch_size, x_unfold.size(1), -1)  # (batches, n_windows, channels * win_size)

    #     return x

    # ----------------------------------------------------------------------
    # ✔ NEW: Right-aligned sliding windows that match the reference code
    # ----------------------------------------------------------------------
    def reformat_time_windows(self, x):
        """
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

            edit_dist = sum(
                edit_distance(pred, tgt[:l])
                for pred, tgt, l in zip(decoded, targets, target_lengths)
            )

            per = edit_dist / target_lengths.sum() * 100
            self.log('val_PER', per, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, input_lengths, target_lengths = batch
        outputs = self(inputs)
        outputs = outputs.log_softmax(2).permute(1, 0, 2)
        loss = self.criterion(outputs, targets, input_lengths, target_lengths)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
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
        # return optimizer
