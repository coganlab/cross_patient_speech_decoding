# %% Imports
# %% Imports
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import summarize
import torch
from repos.cross_patient_speech_decoding.aligned_decoding.nn_models.data_utils.datamodules import SimpleMicroDataModule, AlignedMicroDataModule
from models import CNNTransformer, Transformer
import repos.cross_patient_speech_decoding.aligned_decoding.nn_models.data_utils.augmentations as augs

import os
import sys
sys.path.append('..')
from alignment import alignment_utils as utils
from alignment.AlignCCA import AlignCCA


import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", "The number of training batches.*")
print('Done imports')

# %% Define data module

# data_filename = '../data/pt_decoding_data_S62.pkl'
data_filename = os.path.expanduser('~/data/pt_decoding_data_S62.pkl')
pt_data = utils.load_pkl(data_filename)

pt = 'S14'
p_ind = -1
lab_type = 'phon'
algn_type = 'phon_seq'
tar_data, pre_data = utils.decoding_data_from_dict(pt_data, pt, p_ind,
                                                   lab_type=lab_type,
                                                   algn_type=algn_type)
print([d.shape for d in tar_data])
print([[d.shape for d in p] for p in pre_data])

# dummy data
# n_samples = 144
# n_timepoints = 200
# n_features = 111
fs = 200 # Hz
augmentations = [augs.time_warping, augs.time_masking, augs.time_shifting, augs.noise_jitter, augs.scaling]
# data = torch.rand(n_samples, n_timepoints, n_features)
# labels = torch.randint(0, 9, (n_samples,))
# data = torch.Tensor(all_pt_dict['S14']['X1'])
# labels = torch.Tensor(all_pt_dict['S14']['y1']).long() - 1
data = torch.Tensor(tar_data[0])
labels = torch.Tensor(tar_data[1]).long() - 1
align_labels = torch.Tensor(tar_data[2]).long() - 1
pool_data = [(torch.Tensor(p[0]), torch.Tensor(p[1]).long() - 1, torch.Tensor(p[2]).long() - 1) for p in pre_data]
# data = torch.Tensor(all_pt_dict['S14']['X_collapsed'])
# labels = torch.Tensor(all_pt_dict['S14']['y_phon_collapsed']).long() - 1

# create the data module
batch_size = -1
n_folds = 20
val_size = 0.1
# dm = SimpleMicroDataModule(data, labels, batch_size=batch_size, folds=n_folds,
#                            val_size=val_size, augmentations=augmentations)
dm = AlignedMicroDataModule(data, labels, align_labels, pool_data, AlignCCA,
                            batch_size=batch_size, folds=n_folds, val_size=val_size,
                            augmentations=augmentations)
# dm.setup()

# %% Define model

# model parameters
in_channels = data.shape[-1]
num_classes = 9
d_model = 64
# d_model = data.shape[-1]
kernel_time = 50  # ms
kernel_size = int(kernel_time * fs / 1000)  # kernel length in samples
stride_time = 50  # ms
stride = int(stride_time * fs / 1000)  # stride length in samples
padding = 0
n_head = 2
num_layers = 2
dim_fc = 128
dropout = 0.4
learning_rate = 5e-4
l2_reg = 1e-5
gclip_val = 0.5

sum_model = CNNTransformer(in_channels, num_classes, d_model, kernel_size, stride, padding,
                           n_head, num_layers, dim_fc, dropout, learning_rate)
print(summarize(sum_model))


# %% Train model

# instantiate the trainer
max_epochs = 500
es_pat = max_epochs // 20
# callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

# train the model
n_iters = 1
iter_accs = []
for i in range(n_iters):
    dm.setup()
    
    fold_accs = []
    for fold in range(n_folds):
        dm.set_fold(fold)
        # print(dm.current_fold)
        
        # instantiate the model
        in_channels = dm.get_data_shape()[-1]
        model = CNNTransformer(in_channels, num_classes, d_model, kernel_size, stride, padding,
                               n_head, num_layers, dim_fc, dropout, learning_rate)
        
        # model.current_fold = fold
        callbacks = [ModelCheckpoint(monitor='val_loss'), EarlyStopping(monitor='val_loss', patience=es_pat)]
        trainer = L.Trainer(max_epochs=max_epochs,
                            gradient_clip_val=gclip_val,
                            accelerator='auto',
                            callbacks=callbacks,
                            logger=True,
                            enable_model_summary=False,
                            enable_progress_bar=False,
                           )
        trainer.fit(model, dm)
        print(trainer.logged_metrics)
        trainer.test(model, dm)
        fold_accs.append(trainer.logged_metrics['test_acc'])
    
        # save loss information
        # loss_dict = trainer.logger.metrics
        # loss_dict['fold'] = fold
        # loss_dict['model'] = model
    print(f'Averaged accuracy: {sum(fold_accs) / len(fold_accs)}')
    iter_accs.append(fold_accs)
print(sum(iter_accs) / len(iter_accs), iter_accs)
