# %% Imports
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import torch
from datamodules import SimpleMicroDataModule
from models import CNNTransformer

# %% Define data module
# dummy data
n_samples = 144
n_timepoints = 200
n_features = 111
fs = 200
data = torch.rand(n_samples, n_timepoints, n_features)
labels = torch.randint(0, 9, (n_samples,))

# create the data module
batch_size = 32
n_folds = 5
val_size = 0.2
data_module = SimpleMicroDataModule(data, labels, batch_size=batch_size, folds=n_folds,
                                    val_size=val_size)
data_module.setup()

# %% Create model
# model parameters
in_channels = data.shape[-1]
num_classes = 9
d_model = 64
kernel_time = 50  # ms
kernel_size = int(kernel_time * fs / 1000)  # kernel length in samples
stride_time = 10  # ms
stride = int(stride_time * fs / 1000)  # stride length in samples
padding = 0
n_head = 8
num_layers = 3
dim_fc = 128
dropout = 0.3
learning_rate = 1e-3

# instantiate the model
model = CNNTransformer(in_channels, num_classes, d_model, kernel_size, stride, padding,
                       n_head, num_layers, dim_fc, dropout, learning_rate)

# %% Train the model with kfold
# instantiate the trainer
n_folds = 5
max_epochs = 10
callbacks = [ModelCheckpoint(monitor='val_loss'), EarlyStopping(monitor='val_loss', patience=3)]
trainer = L.Trainer(max_epochs=max_epochs,
                    accelerator='gpu',
                    callbacks=callbacks,
                    logger=True,
                    )

# train the model
for fold in range(n_folds):
    data_module.set_fold(fold)
    model.current_fold = fold
    trainer.fit(model, data_module)
    trainer.test(model, data_module.test_dataloader())
