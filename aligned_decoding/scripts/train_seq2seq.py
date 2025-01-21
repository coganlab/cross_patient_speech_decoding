import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.model_summary import summarize
import torch
import numpy as np
import os
import sys
import argparse

sys.path.append('..')
from alignment import alignment_utils as utils
from alignment.AlignCCA import AlignCCA
from nn_models.data_utils.datamodules import SimpleMicroDataModule, AlignedMicroDataModule
import nn_models.data_utils.augmentations as augs
from nn_models.models import Seq2SeqRNN
import csv

def init_parser():
    parser = argparse.ArgumentParser(description='Seq2seq on DCC')
    parser.add_argument('-pt', '--patient', type=str, required=True,
                        help='Patient ID')
    parser.add_argument('-p', '--pool_train', type=str, default='False',
                        required=False, help='Pool patient data for training')
    return parser


def str2bool(s):
    return s.lower() == 'true'


def seq2seq_decoding():

    ##### Parse arguments #####
    parser = init_parser()
    args = parser.parse_args()
    inputs = {}
    for key, val in vars(args).items():
        inputs[key] = val

    pt = inputs['patient']
    pool_train = str2bool(inputs['pool_train'])

    ##### Data module definition #####
    data_filename = os.path.expanduser('~/data/pt_decoding_data_S62.pkl')
    # data_filename = ('../data/pt_decoding_data_S62.pkl')
    pt_data = utils.load_pkl(data_filename)

    pt = 'S14'
    p_ind = 1
    lab_type = 'phon'
    algn_type = 'phon_seq'
    tar_data, pre_data = utils.decoding_data_from_dict(pt_data, pt, p_ind,
                                                    lab_type=lab_type,
                                                    algn_type=algn_type)

    fs = 200 # Hz
    # augmentations = [augs.time_warping, augs.time_masking, augs.time_shifting, augs.noise_jitter, augs.scaling]
    augmentations = [augs.time_shifting, augs.noise_jitter, augs.scaling]
    # augmentations = None

    data = torch.Tensor(tar_data[0])
    # labels = torch.Tensor(tar_data[1]).long().unsqueeze(1) - 1
    align_labels = torch.Tensor(tar_data[2]).long() - 1
    pool_data = [(torch.Tensor(p[0]), torch.Tensor(p[2]).long() - 1, torch.Tensor(p[2]).long() - 1) for p in pre_data]  # for seq2seq RNN

    # create the data module
    batch_size = 5000
    n_folds = 20
    val_size = 0.1

    fold_data_path = os.path.expanduser(f'~/workspace/nn_data/datamodules/{pt}/') + context_prefix

    if pool_train:
        context_prefix = 'pooled'
        dm = AlignedMicroDataModule(data, align_labels, align_labels, pool_data, AlignCCA,
                                    batch_size=batch_size, folds=n_folds, val_size=val_size,
                                    augmentations=augmentations, data_path=fold_data_path)
    else:
        context_prefix = 'ptSpecific'
        dm = SimpleMicroDataModule(data, align_labels, batch_size=batch_size, folds=n_folds,
                               val_size=val_size, augmentations=augmentations, data_path=fold_data_path)

    ##### Define model (seq2seq RNN) #####

    # model parameters
    gclip_val = 0.5
    in_channels = data.shape[-1]
    num_classes = 9
    n_filters = 100
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

    ##### Train model #####

    # instantiate the trainer
    max_epochs = 500
    log_dir = os.path.expanduser(f'~/workspace/nn_data/nn_logs/{pt}/')

    acc_dir = os.path.expanduser(f'~/workspace/nn_data/accs/{pt}')

    # train the model
    n_iters = 20
    iter_accs = []
    for i in range(n_iters):
        print(f'##### Setting up data module for iteration {i+1} #####')
        dm.setup()
        
        fold_accs = []
        for fold in range(n_folds):
            dm.set_fold(fold)
            
            # instantiate the model
            in_channels = dm.get_data_shape()[-1]
            model = Seq2SeqRNN(in_channels, n_filters, hidden_size, num_classes, n_enc_layers,
                            n_dec_layers, kernel_size, stride, padding, cnn_dropout, rnn_dropout, model_type,
                            learning_rate, l2_reg, activation=activ, decay_iters=max_epochs)
            
            callbacks = [
                ModelCheckpoint(monitor='val_acc', mode='max'),
                LearningRateMonitor(logging_interval='epoch'),
                ]
            trainer = L.Trainer(default_root_dir=log_dir,
                                max_epochs=max_epochs,
                                gradient_clip_val=gclip_val,
                                accelerator='auto',
                                callbacks=callbacks,
                                logger=True,
                                enable_model_summary=False,
                                enable_progress_bar=False,
                            )
            trainer.fit(model=model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
            print(trainer.logged_metrics)

            trainer.test(model=model, dataloaders=dm.test_dataloader(), ckpt_path='best')
            
            fold_accs.append(trainer.logged_metrics['test_acc'])
        
        # save accuracies fold by fold in case of interruption
        iter_accs.append(fold_accs)
        with open(os.path.join(acc_dir, f'{context_prefix}/{pt}_{context_prefix}_seq2seq_rnn_accs_iter{i+1}.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(iter_accs)
        print(np.mean(fold_accs))

    # save all accuracies together
    print(iter_accs)
    with open(os.path.join(acc_dir, f'{context_prefix}/{pt}_{context_prefix}_seq2seq_rnn_accs.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(iter_accs)
    np.save(os.path.join(acc_dir, f'{context_prefix}/{pt}_{context_prefix}_seq2seq_rnn_accs.npy'), np.array(iter_accs))
