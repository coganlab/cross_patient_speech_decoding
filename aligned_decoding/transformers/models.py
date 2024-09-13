import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from torchmetrics.functional.classification import multiclass_confusion_matrix


class TemporalConv(nn.Module):
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
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        x = self.dropout(x)
        return x
    

class TCN_classifier(L.LightningModule):
    def __init__(self, in_channels, num_classes, dim_fc, kernel_size, stride=1,
                 padding=0, dropout=0.3, learning_rate=1e-3, l2_reg=1e-5,
                 criterion=nn.CrossEntropyLoss(), activation=True):
        super(TCN_classifier, self).__init__()
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
        # x is of shape (batch_size, n_timepoints, n_features) coming in
        x = x.permute(0, 2, 1)  # (batch_size, n_features, n_timepoints)
        x = self.temporal_conv(x)

        ##### CHOOSE FROM 1 OF THE 3 BELOW PRIOR TO FC INPUT #####
        # x = x.mean(dim=2)  # mean pooling, (batch_size, d_model)
        x, _ = torch.max(x, dim=2) # max pooling, (batch_size, d_model)
        # x = x[:, -1, :]  # get the last timepoint, (batch_size, d_model)

        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'train_loss': loss, 'train_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                weight_decay=self.l2_reg)


class PositionalEncoding(nn.Module):
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
        return x + self.pos_encoding[:, :x.size(1), :]


class CNNTransformer(L.LightningModule):
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'train_loss': loss, 'train_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        # optim = torch.optim.RAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg, decoupled_weight_decay=True)
        self.lr_sch = CosineWarmupScheduler(optim, self.warmup, self.max_epochs)
        # lr_sch_conf = {'scheduler': self.lr_sch, 'interval': 'epoch'}
        # optim_dict = {'optimizer': optim, 'lr_scheduler': lr_sch_conf}
        # return optim_dict
        return optim
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_sch.step()
        


class Transformer(L.LightningModule):
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
        # x is of shape (batch_size, n_timepoints, n_features) coming in
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        ##### CHOOSE FROM 1 OF THE 3 BELOW PRIOR TO FC INPUT #####
        x = x.mean(dim=1)  # mean pooling, (batch_size, d_model)
        # x, _ = torch.max(x, dim=1) # max pooling, (batch_size, d_model)
        # x = x[:, -1, :]  # get the last timepoint, (batch_size, d_model)

        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'train_loss': loss, 'train_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'val_loss': loss, 'val_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = cmat_acc(y_hat, y, self.num_classes)
        res = {'test_loss': loss, 'test_acc': acc}
        self.log_dict(res, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                weight_decay=self.l2_reg)
    

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def cmat_acc(y_hat, y, num_classes):
    y_pred = torch.argmax(y_hat, dim=1)
    cmat = multiclass_confusion_matrix(y_pred, y, num_classes)
    acc_cmat = cmat.diag().sum() / cmat.sum()
    return acc_cmat
