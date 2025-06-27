##### TODO update to do hyperparameter tuning with ray.tune #####


import sys

import ray.train.lightning
from ray.train.torch import TorchTrainer
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.model_summary import summarize
import torch
import numpy as np
from data_utils.datamodules import SimpleMicroDataModule, AlignedMicroDataModule
from models import CNNTransformer, Transformer, TCN_classifier, TemporalConvRNN, Seq2SeqRNN
import data_utils.augmentations as augs
import csv

import os
import sys
sys.path.append('..')
from alignment import alignment_utils as utils
from alignment.AlignCCA import AlignCCA


def load_data(pt, p_ind, lab_type, algn_type):
    data_filename = os.path.expanduser('~/data/pt_decoding_data_S62.pkl')
    pt_data = utils.load_pkl(data_filename)
    tar_data, pre_data = utils.decoding_data_from_dict(pt_data, pt, p_ind,
                                                    lab_type=lab_type,
                                                    algn_type=algn_type)
    return tar_data, pre_data


def to_datamodule(tar_data, pre_data=None, batch_size=5000, n_folds=20, val_size=0.1):
    fold_data_path = os.path.expanduser('~/workspace/transformer_data')
    # fold_data_path = '.'

    # augmentations = [augs.time_warping, augs.time_masking, augs.time_shifting, augs.noise_jitter, augs.scaling]
    augmentations = [augs.time_shifting, augs.noise_jitter, augs.scaling]
    # augmentations = None
    # data = torch.rand(n_samples, n_timepoints, n_features)
    # labels = torch.randint(0, 9, (n_samples,))
    # data = torch.Tensor(all_pt_dict['S14']['X1'])
    # labels = torch.Tensor(all_pt_dict['S14']['y1']).long() - 1
    data = torch.Tensor(tar_data[0])
    labels = torch.Tensor(tar_data[1]).long().unsqueeze(1) - 1
    align_labels = torch.Tensor(tar_data[2]).long() - 1
    # pool_data = [(torch.Tensor(p[0]), torch.Tensor(p[1]).long().unsqueeze(1) - 1, torch.Tensor(p[2]).long() - 1) for p in pre_data]
    pool_data = [(torch.Tensor(p[0]), torch.Tensor(p[2]).long() - 1, torch.Tensor(p[2]).long() - 1) for p in pre_data]  # for seq2seq RNN
    # data = torch.Tensor(all_pt_dict['S14']['X_collapsed'])
    # labels = torch.Tensor(all_pt_dict['S14']['y_phon_collapsed']).long() - 1

    # context_prefix = 'ptSpecific'
    if pre_data is None:
        dm = SimpleMicroDataModule(data, align_labels, batch_size=batch_size,
                                   folds=n_folds, val_size=val_size,
                                   augmentations=augmentations,
                                   data_path=fold_data_path + '/patient_specific')

    # context_prefix = 'pooled
    else:
        dm = AlignedMicroDataModule(data, align_labels, align_labels,
                                    pool_data, AlignCCA, batch_size=batch_size,
                                    folds=n_folds, val_size=val_size,
                                    augmentations=augmentations,
                                    data_path=fold_data_path)
    return dm




def train_func():
    pt = 'S14'
    p_ind = 1
    lab_type = 'phon'
    algn_type = 'phon_seq'
    tar_data, pre_data = load_data(pt, p_ind, lab_type, algn_type)
    
    batch_size = 5000
    n_folds = 20
    val_size = 0.1
    dm = to_datamodule(tar_data, pre_data, batch_size, n_folds, val_size)

    # model parameters
    in_channels = tar_data.shape[-1]
    num_classes = 9
    n_filters = 100
    # d_model = data.shape[-1]
    fs = 200  # Hz
    kernel_time = 50  # ms
    kernel_size = int(kernel_time * fs / 1000)  # kernel length in samples
    stride_time = 50  # ms
    stride = int(stride_time * fs / 1000)  # stride length in samples
    padding = 0
    n_enc_layers = 2
    n_dec_layers = 1
    hidden_size = 500
    cnn_dropout = 0.3
    rnn_dropout = 0.3
    learning_rate = 1e-4
    l2_reg = 1e-5
    activ = False
    model_type = 'gru'

    sum_model = Seq2SeqRNN(in_channels, n_filters, hidden_size, num_classes, n_enc_layers,
                        n_dec_layers, kernel_size, stride, padding, cnn_dropout,
                        rnn_dropout, model_type, learning_rate, l2_reg)

    print(summarize(sum_model))

    # instantiate the trainer
    max_epochs = 500
    # es_pat = max_steps // 20
    # max_steps = 500
    es_pat = 50
    warmup = 100
    # callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
    log_dir = os.path.expanduser('~/workspace/transformer_data/transformer_logs')

    dm.setup()
    fold_accs = []
    # for fold in range(n_folds):
    for fold in range(1):
        dm.set_fold(fold)
        # print(dm.current_fold)

        # instantiate the model
        in_channels = dm.get_data_shape()[-1]
        model = CNNTransformer(in_channels, num_classes, d_model, kernel_size, stride, padding,
                               n_head, num_layers, dim_fc, dropout, learning_rate)

        # model.current_fold = fold
        # callbacks = [ModelCheckpoint(monitor='val_loss'),
        #              EarlyStopping(monitor='val_loss', patience=es_pat)]
        trainer = L.Trainer(max_epochs=max_epochs,
                            gradient_clip_val=gclip_val,
                            devices='auto',
                            accelerator='auto',
                            strategy=ray.train.lightning.RayDDPStrategy(),
                            plugins=[ray.train.lightning.RayLightningEnvironment()],
                            callbacks=[EarlyStopping(monitor='val_loss', patience=es_pat),
                                       ],
                            logger=False,
                            # enable_model_summary=False,
                            # enable_progress_bar=False,
                            # enable_checkpointing=False,
                            )
        trainer = ray.train.lightning.prepare_trainer(trainer)
        trainer.fit(model, dm)
        # print(trainer.logged_metrics)
        trainer.test(model, dm)
        fold_accs.append(trainer.logged_metrics['test_acc'])

        # save loss information
        # loss_dict = trainer.logger.metrics
        # loss_dict['fold'] = fold
        # loss_dict['model'] = model
    print(f'Averaged accuracy: {sum(fold_accs) / len(fold_accs)}')

scaling_config = ray.train.ScalingConfig(num_workers=1, use_gpu=False)
run_config = ray.train.RunConfig(
    storage_path="./ray-results",
    name="transformer_test",
)

trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
)
result = trainer.fit()
print(result.metrics)
print(result.checkpoint)
print(result.path)
print(result.error)