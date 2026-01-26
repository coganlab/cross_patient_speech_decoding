import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.decomposition import PCA
import h5py
import os
from pathlib import Path


class CTCHeldOutDataModule(L.LightningDataModule):
    def __init__(self, train_data, train_labels, test_data, test_labels,
                 batch_size=128, folds=20, val_size=0.2, augmentations=None,
                 data_path=None):
        super().__init__()
        self.train_data = torch.Tensor(train_data)
        self.train_labels = torch.Tensor(train_labels).long()
        self.test_data = torch.Tensor(test_data)
        self.test_labels = torch.Tensor(test_labels).long()
        self.batch_size = batch_size
        self.folds = folds
        self.val_size = val_size
        self.augmentations = augmentations if augmentations else []
        self.current_fold = 0
        self.data_path = Path(os.getcwd() if data_path is None else data_path)

    def setup(self, stage=None):
        n_classes = len(torch.unique(self.train_labels))
        if self.val_size > 0:
            if self.val_size * len(self.train_data) < n_classes:
                split_labels = None
            elif len(self.train_labels.shape) > 1:
                split_labels = self.train_labels[:, 0]
            else:
                split_labels = self.train_labels
            train_data, val_data, train_labels, val_labels = (
                train_test_split(self.train_data, self.train_labels,
                                 test_size=self.val_size,
                                 stratify=split_labels))
        else:
            train_data, train_labels = self.train_data, self.train_labels
            val_data, val_labels = None, None

        # apply augmentations to training data
        aug_data   = train_data.clone()
        aug_labels = train_labels.clone()
        for aug in self.augmentations:
            aug_data   = torch.cat((aug_data,   aug(train_data)))
            aug_labels = torch.cat((aug_labels, train_labels))

        # save fold precomputed fold data to hdf5 file to load in later
        os.makedirs(self.data_path, exist_ok=True)
        with h5py.File(self.data_path / 'rnn_realtime.h5', 'w') as f:
            f.create_dataset('train_data', data=aug_data)
            f.create_dataset('train_labels', data=aug_labels)
            f.create_dataset('val_data', data=val_data)
            f.create_dataset('val_labels', data=val_labels)
            f.create_dataset('test_data', data=self.test_data)
            f.create_dataset('test_labels', data=self.test_labels)

    def train_dataloader(self):
        # get train data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'rnn_realtime.h5', 'r') as f:
            train_data = f['train_data'][()]
            train_labels = f['train_labels'][()]

        if self.batch_size == -1:
            batch_len = len(train_data)
        else:
            batch_len = self.batch_size

        train_data = torch.Tensor(train_data)
        train_labels = torch.Tensor(train_labels).long()

        return DataLoader(CTCDataset(train_data, train_labels),
                          batch_size=batch_len, shuffle=True,
                          )

    def val_dataloader(self):
        # get val data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'rnn_realtime.h5', 'r') as f:
            val_data = f['val_data'][()]
            val_labels = f['val_labels'][()]

        if self.batch_size == -1:
            batch_len = len(val_data)
        else:
            batch_len = self.batch_size

        val_data = torch.Tensor(val_data)
        val_labels = torch.Tensor(val_labels).long()

        return DataLoader(CTCDataset(val_data, val_labels),
                          batch_size=batch_len, shuffle=False,
                          )

    def test_dataloader(self):
        # get test data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'rnn_realtime.h5', 'r') as f:
            test_data = f['test_data'][()]
            test_labels = f['test_labels'][()]

        if self.batch_size == -1:
            batch_len = len(test_data)
        else:
            batch_len = self.batch_size

        test_data = torch.Tensor(test_data)
        test_labels = torch.Tensor(test_labels).long()

        return DataLoader(CTCDataset(test_data, test_labels),
                          batch_size=batch_len, shuffle=False,
                          )

    def get_data_shape(self):
        with h5py.File(self.data_path / 'rnn_realtime.h5', 'r') as f:
            data_shape = f['train_data'][()].shape
        return data_shape
    

class CTCHeldOutTargetValDataModule(CTCHeldOutDataModule):
    """
    This shares the same dataloader return structure (from h5 files created
    in setup) as CTCHeldOutDataModule, but performs validation split on
    target patient data only and adds in cross-patient data to training data
    """

    def __init__(self, train_data_tgt, train_labels_tgt, train_data_cross,
                 train_labels_cross, test_data, test_labels, batch_size=128,
                 val_size=0.2, augmentations=None, data_path=None):
        L.LightningDataModule().__init__()
        self.train_data_tgt = torch.Tensor(train_data_tgt)
        self.train_labels_tgt = torch.Tensor(train_labels_tgt).long()
        self.train_data_cross = torch.Tensor(train_data_cross)
        self.train_labels_cross = torch.Tensor(train_labels_cross).long()
        self.test_data = torch.Tensor(test_data)
        self.test_labels = torch.Tensor(test_labels).long()
        self.batch_size = batch_size
        self.val_size = val_size
        self.augmentations = augmentations if augmentations else []
        self.current_fold = 0
        self.data_path = Path(os.getcwd() if data_path is None else data_path)

    def setup(self, stage=None):
        n_classes = len(torch.unique(self.train_labels_tgt))
        # perform validation split on target patient only
        if self.val_size > 0:
            if self.val_size * len(self.train_data_tgt) < n_classes:
                split_labels = None
            elif len(self.train_labels_tgt.shape) > 1:
                split_labels = self.train_labels_tgt[:, 0]
            else:
                split_labels = self.train_labels_tgt
            train_data_tgt, val_data, train_labels_tgt, val_labels = (
                train_test_split(self.train_data_tgt, self.train_labels_tgt,
                                 test_size=self.val_size,
                                 stratify=split_labels))
        else:
            train_data_tgt, train_labels_tgt = self.train_data_tgt, self.train_labels_tgt
            val_data, val_labels = None, None

        # add in cross-patient data to post-split target patient training data
        train_data = torch.cat((train_data_tgt, self.train_data_cross))
        train_labels = torch.cat((train_labels_tgt, self.train_labels_cross))

        # apply augmentations to training data
        aug_data   = train_data.clone()
        aug_labels = train_labels.clone()
        for aug in self.augmentations:
            aug_data   = torch.cat((aug_data,   aug(train_data)))
            aug_labels = torch.cat((aug_labels, train_labels))
        # aug_data = torch.cat((torch.Tensor([]), train_data))
        # aug_labels = torch.cat((torch.Tensor([]).long(), train_labels))
        # for aug in self.augmentations:
        #     aug_data = torch.cat((aug_data, aug(train_data)))
        #     aug_labels = torch.cat((aug_labels, train_labels))

        # save fold precomputed fold data to hdf5 file to load in later
        os.makedirs(self.data_path, exist_ok=True)
        with h5py.File(self.data_path / 'rnn_realtime.h5', 'w') as f:
            f.create_dataset('train_data', data=aug_data)
            f.create_dataset('train_labels', data=aug_labels)
            f.create_dataset('val_data', data=val_data)
            f.create_dataset('val_labels', data=val_labels)
            f.create_dataset('test_data', data=self.test_data)
            f.create_dataset('test_labels', data=self.test_labels)


class CTCDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # additionally return input lengths and target lengths for CTC loss
        return self.X[idx], self.y[idx], len(self.X[idx]), len(self.y[idx])


class CTCDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # additionally return input lengths and target lengths for CTC loss
        return self.X[idx], self.y[idx], len(self.X[idx]), len(self.y[idx])