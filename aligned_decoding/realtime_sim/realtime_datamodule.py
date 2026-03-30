import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.decomposition import PCA
from numpy.linalg import LinAlgError
import h5py
import os
from pathlib import Path

from alignment.AlignCCA import AlignCCA


class CTCHeldOutDataModule(L.LightningDataModule):
    def __init__(self, train_data, train_labels, test_data, test_labels,
                 batch_size=128, val_size=0.2, augmentations=None,
                 data_path=None):
        super().__init__()
        self.train_data = torch.Tensor(train_data)
        self.train_labels = torch.Tensor(train_labels).long()
        self.test_data = torch.Tensor(test_data)
        self.test_labels = torch.Tensor(test_labels).long()
        self.batch_size = batch_size
        self.val_size = val_size
        self.augmentations = augmentations if augmentations else []
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

        # save fold precomputed fold data to hdf5 file to load in later
        os.makedirs(self.data_path, exist_ok=True)
        with h5py.File(self.data_path / 'rnn_realtime.h5', 'w') as f:
            f.create_dataset('train_data', data=aug_data)
            f.create_dataset('train_labels', data=aug_labels)
            f.create_dataset('val_data', data=val_data)
            f.create_dataset('val_labels', data=val_labels)
            f.create_dataset('test_data', data=self.test_data)
            f.create_dataset('test_labels', data=self.test_labels)


class CTCHeldOutTargetValAlignDataModule(CTCHeldOutDataModule):
    """
    This shares the same dataloader return structure (from h5 files created
    in setup) as CTCHeldOutDataModule, but performs validation split on
    target patient data only and adds in cross-patient data to training data
    """

    def __init__(self, train_data_tgt, train_labels_tgt, train_data_cross,
                 train_labels_cross, test_data, test_labels, batch_size=128,
                 val_size=0.2, augmentations=None, data_path=None, pool=True,
                 dim_red=PCA, n_comp=30, align=True, aligner=AlignCCA):
        L.LightningDataModule().__init__()
        self.train_data_tgt = torch.Tensor(train_data_tgt)
        self.train_labels_tgt = torch.Tensor(train_labels_tgt).long()
        if train_data_cross is not None:
            self.train_data_cross = [torch.Tensor(d) for d in train_data_cross]
            self.train_labels_cross = [torch.Tensor(lab).long() for lab in train_labels_cross]
        else:
            self.train_data_cross = None
            self.train_labels_cross = None
        self.test_data = torch.Tensor(test_data)
        self.test_labels = torch.Tensor(test_labels).long()
        self.batch_size = batch_size
        self.val_size = val_size
        self.augmentations = augmentations if augmentations else []
        self.data_path = Path(os.getcwd() if data_path is None else data_path)
        self.pool = pool
        self.dim_red = dim_red
        self.n_components = n_comp
        self.align = align
        self.aligner = aligner

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
        test_data = self.test_data.clone()
        test_labels = self.test_labels.clone()
        if self.train_data_cross is not None:
            train_data_cross = self.train_data_cross.copy()
            train_labels_cross = self.train_labels_cross.copy()
        else:
            train_data_cross = None
            train_labels_cross = None

        # reduce data to latent space if pooling across patients
        if self.pool:
            train_data_tgt, tgt_pca = reduce_to_latent_space(
                                            train_data_tgt,
                                            n_components=self.n_components,
                                        )
            # apply same reduction to val and test data
            val_data, _ = reduce_to_latent_space(val_data, pca=tgt_pca)
            test_data, _ = reduce_to_latent_space(test_data, pca=tgt_pca)
            if self.train_data_cross is not None:
                for j in range(len(train_data_cross)):
                    data_red, _ = reduce_to_latent_space(
                                    train_data_cross[j],
                                    n_components=self.n_components,
                                )
                    train_data_cross[j] = data_red
            # align data if specified
            if self.align:
                # only need to align cross patients to target space,
                # no modification of target patient data
                for j in range(len(train_data_cross)):
                    train_data_cross[j] = align_to_target(
                                                    self.aligner,
                                                    train_data_tgt,
                                                    train_data_cross[j],
                                                    train_labels_tgt,
                                                    train_labels_cross[j]
                                                )
            else:
                min_dim = min([d.shape[-1] for d in [train_data_tgt] + train_data_cross])
                # truncate data to minimum dimension across patients for
                # feature compatibility
                train_data_tgt = train_data_tgt[:, :, :min_dim]
                val_data = val_data[:, :, :min_dim]
                test_data = test_data[:, :, :min_dim]
                for j in range(len(train_data_cross)):
                    train_data_cross[j] = train_data_cross[j][:, :, :min_dim]

        # add in cross-patient data to post-split target patient training data
        if train_data_cross is None:
            train_data = train_data_tgt
            train_labels = train_labels_tgt
        else:
            train_data = torch.cat([train_data_tgt] + train_data_cross)
            train_labels = torch.cat([train_labels_tgt] + train_labels_cross)

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
            f.create_dataset('test_data', data=test_data)
            f.create_dataset('test_labels', data=test_labels)


class CTCHeldOutTargetValCVDataModule(L.LightningDataModule):
    """
    Update of target val datamodule from above to form validation set via
    k-fold cross-validation instead of a simple held-out split. Intended to
    improved robustness/generalizability of hyperparameter tuning results.
    """

    def __init__(self, train_data_tgt, train_labels_tgt, train_data_cross,
                 train_labels_cross, test_data, test_labels, batch_size=128,
                 n_folds=5, augmentations=None, data_path=None):
        super().__init__()
        self.train_data_tgt = torch.Tensor(train_data_tgt)
        self.train_labels_tgt = torch.Tensor(train_labels_tgt).long()
        if train_data_cross is not None:
            self.train_data_cross = torch.Tensor(train_data_cross)
            self.train_labels_cross = torch.Tensor(train_labels_cross).long()
        else:
            self.train_data_cross = None
            self.train_labels_cross = None
        self.test_data = torch.Tensor(test_data)
        self.test_labels = torch.Tensor(test_labels).long()
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.augmentations = augmentations if augmentations else []
        self.current_fold = 0
        self.data_path = Path(os.getcwd() if data_path is None else data_path)

    def setup(self, stage=None):
        n_classes = len(torch.unique(self.train_labels_tgt))
        # perform validation fold splits on target patient only
        cv = select_cv(self.n_folds, self.train_labels_tgt)
        for i, (train_idx, val_idx) in enumerate(cv.split(self.train_data_tgt,
                                                  self.train_labels_tgt)):
            train_data_tgt = self.train_data_tgt[train_idx]
            train_labels_tgt = self.train_labels_tgt[train_idx]
            val_data = self.train_data_tgt[val_idx]
            val_labels = self.train_labels_tgt[val_idx]
        
            # add in cross-patient data to post-split target patient training data
            if self.train_data_cross is None:
                train_data = train_data_tgt
                train_labels = train_labels_tgt
            else:
                train_data = torch.cat((train_data_tgt, self.train_data_cross))
                train_labels = torch.cat((train_labels_tgt, self.train_labels_cross))

            # apply augmentations to training data
            aug_data   = train_data.clone()
            aug_labels = train_labels.clone()
            for aug in self.augmentations:
                aug_data   = torch.cat((aug_data,   aug(train_data)))
                aug_labels = torch.cat((aug_labels, train_labels))

            # save fold precomputed fold data to hdf5 file to load in later
            os.makedirs(self.data_path, exist_ok=True)
            with h5py.File(self.data_path / f'rnn_realtime_fold{i}.h5', 'w') as f:
                f.create_dataset('train_data', data=aug_data)
                f.create_dataset('train_labels', data=aug_labels)
                f.create_dataset('val_data', data=val_data)
                f.create_dataset('val_labels', data=val_labels)
                f.create_dataset('test_data', data=self.test_data)
                f.create_dataset('test_labels', data=self.test_labels)

    def train_dataloader(self):
        # get train data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / f'rnn_realtime_fold{self.current_fold}.h5', 'r') as f:
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
        with h5py.File(self.data_path / f'rnn_realtime_fold{self.current_fold}.h5', 'r') as f:
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
        with h5py.File(self.data_path / f'rnn_realtime_fold{self.current_fold}.h5', 'r') as f:
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

    def set_fold(self, fold):
        assert 0 <= fold < self.n_folds, "Fold index out of range"
        self.current_fold = fold

    def get_data_shape(self):
        with h5py.File(self.data_path / f'rnn_realtime_fold{self.current_fold}.h5', 'r') as f:
            data_shape = f['train_data'][()].shape
        return data_shape


class CTCHeldOutTargetValAlignCVDataModule(CTCHeldOutTargetValCVDataModule):
    """
    Update of CV datamodule from above to learn data alignment on provided
    training data. Previously, alignment was learned on the offline HG data
    from the whole training set, this biased validation performance during
    hyperparameter tuning.

    *** Note: This module assumes that train_data_cross and
    train_labels_cross are lists of tensors where each entry corresponds to
    a different cross-patient dataset to be aligned to the target patient
    instead of the pre-concatenated cross-patient data as in previous modules.
    This is because alignment works on a per-patient basis, so we need to keep
    that separation when calculating alignment on the fly here. ***
    """

    def __init__(self, train_data_tgt, train_labels_tgt, train_data_cross,
                 train_labels_cross, test_data, test_labels, batch_size=128,
                 n_folds=5, augmentations=None, data_path=None, pool=True,
                 dim_red=PCA, n_comp=30, align=True, aligner=AlignCCA):
        L.LightningDataModule().__init__()
        self.train_data_tgt = torch.Tensor(train_data_tgt)
        self.train_labels_tgt = torch.Tensor(train_labels_tgt).long()
        if train_data_cross is not None:
            self.train_data_cross = [torch.Tensor(d) for d in train_data_cross]
            self.train_labels_cross = [torch.Tensor(lab).long() for lab in train_labels_cross]
        else:
            self.train_data_cross = None
            self.train_labels_cross = None
        self.test_data = torch.Tensor(test_data)
        self.test_labels = torch.Tensor(test_labels).long()
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.augmentations = augmentations if augmentations else []
        self.current_fold = 0
        self.data_path = Path(os.getcwd() if data_path is None else data_path)
        self.pool = pool
        self.dim_red = dim_red
        self.n_components = n_comp
        self.align = align
        self.aligner = aligner
        
    def setup(self, stage=None):
        n_classes = len(torch.unique(self.train_labels_tgt))
        # perform validation fold splits on target patient only
        cv = select_cv(self.n_folds, self.train_labels_tgt)
        for i, (train_idx, val_idx) in enumerate(cv.split(self.train_data_tgt,
                                                  self.train_labels_tgt)):
            train_data_tgt = self.train_data_tgt[train_idx]
            train_labels_tgt = self.train_labels_tgt[train_idx]
            val_data = self.train_data_tgt[val_idx]
            val_labels = self.train_labels_tgt[val_idx]
            test_data = self.test_data.clone()
            test_labels = self.test_labels.clone()
            if self.train_data_cross is not None:
                train_data_cross = self.train_data_cross.copy()
                train_labels_cross = self.train_labels_cross.copy()
            else:
                train_data_cross = None
                train_labels_cross = None

            # reduce data to latent space if pooling across patients
            if self.pool:
                train_data_tgt, tgt_pca = reduce_to_latent_space(
                                                train_data_tgt,
                                                n_components=self.n_components,
                                            )
                # apply same reduction to val and test data
                val_data, _ = reduce_to_latent_space(val_data, pca=tgt_pca)
                test_data, _ = reduce_to_latent_space(test_data, pca=tgt_pca)
                if train_data_cross is not None:
                    for j in range(len(train_data_cross)):
                        data_red, _ = reduce_to_latent_space(
                                        train_data_cross[j],
                                        n_components=self.n_components,
                                    )
                        train_data_cross[j] = data_red
                # align data if specified
                if self.align:
                    # only need to align cross patients to target space,
                    # no modification of target patient data
                    for j in range(len(train_data_cross)):
                        train_data_cross[j] = align_to_target(
                                                        self.aligner,
                                                        train_data_tgt,
                                                        train_data_cross[j],
                                                        train_labels_tgt,
                                                        train_labels_cross[j]
                                                    )
                else:
                    min_dim = min([d.shape[-1] for d in [train_data_tgt] + train_data_cross])
                    # truncate data to minimum dimension across patients for
                    # feature compatibility
                    train_data_tgt = train_data_tgt[:, :, :min_dim]
                    val_data = val_data[:, :, :min_dim]
                    test_data = test_data[:, :, :min_dim]
                    for j in range(len(train_data_cross)):
                        train_data_cross[j] = train_data_cross[j][:, :, :min_dim]

            # add in cross-patient data to post-split target patient training data
            if self.train_data_cross is None:
                train_data = train_data_tgt
                train_labels = train_labels_tgt
            else:
                train_data = torch.cat([train_data_tgt] + train_data_cross)
                train_labels = torch.cat([train_labels_tgt] + train_labels_cross)

            # apply augmentations to training data
            aug_data   = train_data.clone()
            aug_labels = train_labels.clone()
            for aug in self.augmentations:
                aug_data   = torch.cat((aug_data,   aug(train_data)))
                aug_labels = torch.cat((aug_labels, train_labels))

            # save fold precomputed fold data to hdf5 file to load in later
            os.makedirs(self.data_path, exist_ok=True)
            with h5py.File(self.data_path / f'rnn_realtime_fold{i}.h5', 'w') as f:
                f.create_dataset('train_data', data=aug_data)
                f.create_dataset('train_labels', data=aug_labels)
                f.create_dataset('val_data', data=val_data)
                f.create_dataset('val_labels', data=val_labels)
                f.create_dataset('test_data', data=test_data)
                f.create_dataset('test_labels', data=test_labels)


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


def select_cv(folds, labels):
    if len(labels.shape) > 1:
        cv_labels = labels[:,0]
    else:
        cv_labels = labels
    class_counts = torch.bincount(cv_labels)
    if (class_counts < folds).any():
        cv = KFold(n_splits=folds, shuffle=True)
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True)
    return cv


def reduce_to_latent_space(data, pca=None, n_components=30, low_thresh=5):
    shapes = data.shape
    data_2d = data.reshape(-1, shapes[-1])

    if pca is not None:
        dr = pca
        data_r = dr.transform(data_2d)
    else:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                dr = PCA(n_components=n_components)
                data_r = dr.fit_transform(data_2d)
                break
            except LinAlgError as e:
                if ("SVD did not converge" in str(e) and 
                    attempt < max_retries - 1):
                    continue
                else:
                    raise
        
        if dr.n_components_ <= low_thresh:
            # too few components likely means component representing large
            # noise/artifact in data, so just use 30 components and remove
            # the first component as likely culprit
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    dr = PCA(n_components=30)
                    data_r = dr.fit_transform(data_2d)
                    break
                except LinAlgError as e:
                    if ("SVD did not converge" in str(e) and 
                        attempt < max_retries - 1):
                        continue
                    else:
                        raise
    data_r = torch.Tensor(data_r.reshape(shapes[0], shapes[1], -1))

    return data_r, dr


def align_to_target(aligner, target_data, source_data, target_labels, source_labels):
    source_shapes = source_data.shape

    align = aligner()
    align.fit(target_data.numpy(), source_data.numpy(), target_labels.numpy(),
              source_labels.numpy())
    source_a = align.transform(source_data)
    source_a = torch.Tensor(source_a.reshape(source_shapes[0], source_shapes[1], -1))

    return source_a
    