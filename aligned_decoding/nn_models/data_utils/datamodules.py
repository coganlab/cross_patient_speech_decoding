"""Lightning DataModules for single-patient and cross-patient speech decoding.

Provides k-fold cross-validation data modules that precompute fold splits to
HDF5 files, supporting optional data augmentation and CCA-based cross-patient
alignment.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.decomposition import PCA
import h5py
import os
from pathlib import Path

# sys.path.append('..')
# from alignment.AlignCCA import AlignCCA

class SimpleMicroDataModule(L.LightningDataModule):
    """Lightning DataModule for single-patient microelectrode data with k-fold CV.

    Precomputes stratified k-fold splits (with optional augmentations) and
    saves each fold to HDF5 for efficient reloading across training runs.

    Args:
        data: Input feature tensor of shape (n_trials, n_timepoints, n_features).
        labels: Label tensor of shape (n_trials,) or (n_trials, seq_length).
        batch_size: Batch size for data loaders. Use -1 for full-batch.
            Defaults to 128.
        folds: Number of cross-validation folds. Defaults to 20.
        val_size: Fraction of training data for validation. Defaults to 0.2.
        augmentations: List of augmentation callables applied to training data.
        data_path: Directory for saving fold HDF5 files. Defaults to cwd.
    """

    def __init__(self, data, labels, batch_size=128, folds=20, val_size=0.2,
                 augmentations=None, data_path=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.folds = folds
        self.val_size = val_size
        self.augmentations = augmentations if augmentations else []
        self.current_fold = 0
        self.data_path = Path(os.getcwd() if data_path is None else data_path)


    def setup(self, stage=None):
        """Splits data into k folds and saves each to an HDF5 file.

        Args:
            stage: Lightning stage ('fit', 'test', etc.). Unused.
        """
        cv = self.select_cv(self.folds)
        for k, (train_idx, test_idx) in enumerate(cv.split(self.data, self.labels)):
            train_data, test_data = self.data[train_idx], self.data[test_idx]
            train_labels, test_labels = (self.labels[train_idx],
                                         self.labels[test_idx])
            n_classes = len(torch.unique(train_labels))

            if self.val_size > 0:
                if self.val_size * len(train_data) < n_classes:
                    split_labels = None
                elif len(train_labels.shape) > 1:
                    split_labels = train_labels[:,0]
                else:
                    split_labels = train_labels
                train_data, val_data, train_labels, val_labels = (
                    train_test_split(train_data, train_labels,
                                     test_size=self.val_size,
                                     stratify=split_labels))
            else:
                val_data, val_labels = None, None

            aug_data = torch.cat((torch.Tensor([]), train_data))
            aug_labels = torch.cat((torch.Tensor([]).long(), train_labels))
            for aug in self.augmentations:
                aug_data = torch.cat((aug_data, aug(train_data)))
                aug_labels = torch.cat((aug_labels, train_labels))

            # save fold precomputed fold data to hdf5 file to load in later
            os.makedirs(self.data_path / 'fold_data', exist_ok=True)
            with h5py.File(self.data_path / 'fold_data' / f'fold_{k}.h5', 'w') as f:
                # f.create_dataset('train_data', data=train_data)
                # f.create_dataset('train_labels', data=train_labels)
                f.create_dataset('train_data', data=aug_data)
                f.create_dataset('train_labels', data=aug_labels)
                f.create_dataset('val_data', data=val_data)
                f.create_dataset('val_labels', data=val_labels)
                f.create_dataset('test_data', data=test_data)
                f.create_dataset('test_labels', data=test_labels)

    def train_dataloader(self):
        """Returns a DataLoader for the current fold's training data.

        Returns:
            DataLoader: Shuffled training data loader.
        """
        # get train data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            train_data = f['train_data'][()]
            train_labels = f['train_labels'][()]

        if self.batch_size == -1:
            batch_len = len(train_data)
        else:
            batch_len = self.batch_size
        
        train_data = torch.Tensor(train_data)
        train_labels = torch.Tensor(train_labels).long()
        return DataLoader(TensorDataset(train_data, train_labels),
                          batch_size=batch_len, shuffle=True,
                          # num_workers=7, persistent_workers=True,
                          )

    def val_dataloader(self):
        """Returns a DataLoader for the current fold's validation data.

        Returns:
            DataLoader: Validation data loader (no shuffling).
        """
        # get val data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            val_data = f['val_data'][()]
            val_labels = f['val_labels'][()]

        if self.batch_size == -1:
            batch_len = len(val_data)
        else:
            batch_len = self.batch_size

        val_data = torch.Tensor(val_data)
        val_labels = torch.Tensor(val_labels).long()
        return DataLoader(TensorDataset(val_data, val_labels),
                          batch_size=batch_len, shuffle=False,
                          # num_workers=7, persistent_workers=True,
                          )

    def test_dataloader(self):
        """Returns a DataLoader for the current fold's test data.

        Returns:
            DataLoader: Test data loader (no shuffling).
        """
        # get test data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            test_data = f['test_data'][()]
            test_labels = f['test_labels'][()]

        if self.batch_size == -1:
            batch_len = len(test_data)
        else:
            batch_len = self.batch_size

        test_data = torch.Tensor(test_data)
        test_labels = torch.Tensor(test_labels).long()
        return DataLoader(TensorDataset(test_data, test_labels),
                          batch_size=batch_len, shuffle=False,
                          # num_workers=7, persistent_workers=True,
                          )

    def set_fold(self, fold):
        """Sets the active fold index for data loading.

        Args:
            fold: Zero-based fold index.

        Raises:
            AssertionError: If fold is out of range.
        """
        assert 0 <= fold < self.folds, "Fold index out of range"
        self.current_fold = fold

    def select_cv(self, folds):
        """Selects a cross-validation splitter based on class distribution.

        Falls back to KFold if any class has fewer samples than folds,
        otherwise uses StratifiedKFold.

        Args:
            folds: Number of CV folds.

        Returns:
            sklearn splitter: KFold or StratifiedKFold instance.
        """
        if len(self.labels.shape) > 1:
            cv_labels = self.labels[:,0]
        else:
            cv_labels = self.labels
        class_counts = torch.bincount(cv_labels)
        if (class_counts < folds).any():
            cv = KFold(n_splits=folds, shuffle=True)
        else:
            cv = StratifiedKFold(n_splits=folds, shuffle=True)
        return cv

    def get_data_shape(self):
        """Returns the shape of the current fold's training data.

        Returns:
            tuple: Shape of the training data array.
        """
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            data_shape = f['train_data'][()].shape
        return data_shape


class AlignedMicroDataModule(L.LightningDataModule):
    """DataModule for cross-patient aligned microelectrode data with k-fold CV.

    Performs CCA-based alignment of pooled cross-patient data onto the target
    patient space during setup, applies PCA dimensionality reduction, then
    saves augmented fold data to HDF5.

    Args:
        data: Target patient feature tensor of shape
            (n_trials, n_timepoints, n_features).
        labels: Target label tensor of shape (n_trials, seq_length).
        align_labels: Alignment labels for CCA of shape
            (n_trials, seq_length).
        pool_data: List of (features, labels, align_labels) tuples for each
            cross-patient dataset.
        algner: Callable that returns an aligner instance (e.g., AlignCCA).
        batch_size: Batch size for data loaders. Use -1 for full-batch.
            Defaults to 128.
        folds: Number of cross-validation folds. Defaults to 20.
        val_size: Fraction of training data for validation. Defaults to 0.2.
        augmentations: List of augmentation callables.
        data_path: Directory for saving fold HDF5 files. Defaults to cwd.
    """

    def __init__(self, data, labels, align_labels, pool_data, algner,
                 batch_size=128, folds=20, val_size=0.2, augmentations=None,
                 data_path=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.align_labels = align_labels
        self.pool_data = pool_data
        self.algner = algner
        self.batch_size = batch_size
        self.folds = folds
        self.val_size = val_size
        self.augmentations = augmentations if augmentations else []
        self.data_path = Path(os.getcwd() if data_path is None else data_path)
        self.current_fold = 0

    def setup(self, stage=None):
        """Splits data, aligns cross-patient pools, and saves folds to HDF5.

        Alignment and PCA are fit on each fold's training data. Validation and
        test sets are projected using the fitted PCA.

        Args:
            stage: Lightning stage ('fit', 'test', etc.). Unused.
        """
        cv = self.select_cv(self.folds)
        for k, (train_idx, test_idx) in enumerate(cv.split(self.data, self.labels.squeeze(1))):
            train_data, test_data = self.data[train_idx], self.data[test_idx]
            train_labels, test_labels = (self.labels[train_idx],
                                         self.labels[test_idx])
            test_labels = test_labels.squeeze(1)
            align_labels = self.align_labels[train_idx]
            n_classes = len(torch.unique(train_labels))

            if self.val_size > 0:
                if self.val_size * len(train_data) < n_classes:
                    split_labels = None
                elif len(train_labels.shape) > 1:
                    split_labels = train_labels[:,0]
                else:
                    split_labels = train_labels
                train_data, val_data, train_labels, val_labels, align_labels, _ = \
                    (train_test_split(train_data, train_labels,
                                      align_labels,
                                     test_size=self.val_size,
                                     stratify=split_labels))
                val_labels = val_labels.squeeze(1)
            else:
                val_data, val_labels = None, None

            aug_data = torch.cat((torch.Tensor([]), train_data))
            aug_labels = torch.cat((torch.empty((0, train_labels.shape[-1])).long(), train_labels))
            aug_align_labels = torch.cat((torch.empty((0, align_labels.shape[-1])).long(), align_labels))
            aug_pool_data = [[
                torch.cat((torch.Tensor([]),x)),
                torch.cat((torch.empty((0, y.shape[-1])).long(),y)),
                torch.cat((torch.empty((0, y_a.shape[-1])).long(), y_a))
                ] for (x, y, y_a) in self.pool_data]
            for aug in self.augmentations:
                aug_data = torch.cat((aug_data, aug(train_data)))
                aug_labels = torch.cat((aug_labels, train_labels))
                aug_align_labels = torch.cat((aug_align_labels, align_labels))
                for i, (x, y, y_a) in enumerate(self.pool_data):
                    aug_pool_data[i][0] = torch.cat((aug_pool_data[i][0], aug(x)))
                    aug_pool_data[i][1] = torch.cat((aug_pool_data[i][1], y))
                    aug_pool_data[i][2] = torch.cat((aug_pool_data[i][2], y_a))
            
            # clear unnecessary data after augmentations
            del train_data, train_labels, align_labels

            # align pooled data to current data
            aug_data, aug_labels, dim_red = (
                process_aligner(aug_data, aug_labels, aug_align_labels,
                                aug_pool_data, self.algner))

            if val_data is not None:
                val_shape = val_data.shape
                val_data = dim_red.transform(val_data.reshape(-1, val_shape[-1]))
                val_data = torch.Tensor(val_data.reshape(val_shape[0], val_shape[1], -1))
            test_shape = test_data.shape
            test_data = dim_red.transform(test_data.reshape(-1, test_shape[-1]))
            test_data = torch.Tensor(test_data.reshape(test_shape[0], test_shape[1], -1))

            # save fold precomputed fold data to hdf5 file to load in later
            os.makedirs(self.data_path / 'fold_data', exist_ok=True)
            with h5py.File(self.data_path / 'fold_data' / f'fold_{k}.h5', 'w') as f:
                # f.create_dataset('train_data', data=train_data)
                # f.create_dataset('train_labels', data=train_labels)
                f.create_dataset('train_data', data=aug_data)
                f.create_dataset('train_labels', data=aug_labels)
                f.create_dataset('val_data', data=val_data)
                f.create_dataset('val_labels', data=val_labels)
                f.create_dataset('test_data', data=test_data)
                f.create_dataset('test_labels', data=test_labels)

    def train_dataloader(self):
        """Returns a DataLoader for the current fold's training data.

        Returns:
            DataLoader: Shuffled training data loader.
        """
        # get train data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            train_data = f['train_data'][()]
            train_labels = f['train_labels'][()]

        if self.batch_size == -1:
            batch_len = len(train_data)
        else:
            batch_len = self.batch_size
        
        train_data = torch.Tensor(train_data)
        train_labels = torch.Tensor(train_labels).long()
        return DataLoader(TensorDataset(train_data, train_labels),
                          batch_size=batch_len, shuffle=True,
                          # num_workers=7, persistent_workers=True,
                          )

    def val_dataloader(self):
        """Returns a DataLoader for the current fold's validation data.

        Returns:
            DataLoader: Validation data loader (no shuffling).
        """
        # get val data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            val_data = f['val_data'][()]
            val_labels = f['val_labels'][()]

        if self.batch_size == -1:
            batch_len = len(val_data)
        else:
            batch_len = self.batch_size

        val_data = torch.Tensor(val_data)
        val_labels = torch.Tensor(val_labels).long()
        return DataLoader(TensorDataset(val_data, val_labels),
                          batch_size=batch_len, shuffle=False,
                          # num_workers=7, persistent_workers=True,
                          )

    def test_dataloader(self):
        """Returns a DataLoader for the current fold's test data.

        Returns:
            DataLoader: Test data loader (no shuffling).
        """
        # get test data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            test_data = f['test_data'][()]
            test_labels = f['test_labels'][()]

        if self.batch_size == -1:
            batch_len = len(test_data)
        else:
            batch_len = self.batch_size

        test_data = torch.Tensor(test_data)
        test_labels = torch.Tensor(test_labels).long()
        return DataLoader(TensorDataset(test_data, test_labels),
                          batch_size=batch_len, shuffle=False,
                          # num_workers=7, persistent_workers=True,
                          )

    def set_fold(self, fold):
        """Sets the active fold index for data loading.

        Args:
            fold: Zero-based fold index.

        Raises:
            AssertionError: If fold is out of range.
        """
        assert 0 <= fold < self.folds, "Fold index out of range"
        self.current_fold = fold

    def select_cv(self, folds):
        """Selects a cross-validation splitter based on class distribution.

        Args:
            folds: Number of CV folds.

        Returns:
            sklearn splitter: KFold or StratifiedKFold instance.
        """
        if len(self.labels.shape) > 1:
            cv_labels = self.labels[:,0]
        else:
            cv_labels = self.labels
        class_counts = torch.bincount(cv_labels)
        if (class_counts < folds).any():
            cv = KFold(n_splits=folds, shuffle=True)
        else:
            cv = StratifiedKFold(n_splits=folds, shuffle=True)
        return cv

    def get_data_shape(self):
        """Returns the shape of the current fold's training data.

        Returns:
            tuple: Shape of the training data array.
        """
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            data_shape = f['train_data'][()].shape
        return data_shape


class AlignedMicroValDataModule(AlignedMicroDataModule):
    """Variant of AlignedMicroDataModule that aligns before train/val split.

    Overrides setup to perform cross-patient alignment on the full training
    fold before splitting into train and validation sets, giving the
    validation set aligned data.
    """

    ### OVERRIDING SETUP METHOD FOR CROSS-PATIENT DATA TO EXPAND VALIDATION SET
    def setup(self, stage=None):
        """Aligns cross-patient data, then splits into train/val and saves folds.

        Unlike the parent class, alignment is performed before the train/val
        split so that validation data is also in the aligned space.

        Args:
            stage: Lightning stage ('fit', 'test', etc.). Unused.
        """
        cv = self.select_cv(self.folds)
        for k, (train_idx, test_idx) in enumerate(cv.split(self.data, self.labels.squeeze(1))):
            train_data, test_data = self.data[train_idx], self.data[test_idx]
            train_labels, test_labels = (self.labels[train_idx],
                                         self.labels[test_idx])
            test_labels = test_labels.squeeze(1)
            align_labels = self.align_labels[train_idx]
            n_classes = len(torch.unique(train_labels))

            ### MAIN DIFFERENCE - ALIGNING DATA BEFORE SPLITTING TO TRAIN/VAL
            train_data, train_labels, dim_red = (
                process_aligner(train_data, train_labels, align_labels,
                                self.pool_data, self.algner))
            align_labels

            if self.val_size > 0:
                if self.val_size * len(train_data) < n_classes:
                    split_labels = None
                elif len(train_labels.shape) > 1:
                    split_labels = train_labels[:,0]
                else:
                    split_labels = train_labels
                train_data, val_data, train_labels, val_labels = \
                    (train_test_split(train_data, train_labels,
                                     test_size=self.val_size,
                                     stratify=split_labels))
                val_labels = val_labels.squeeze(1)
            else:
                val_data, val_labels = None, None

            aug_data = torch.cat((torch.Tensor([]), train_data))
            aug_labels = torch.cat((torch.empty((0, train_labels.shape[-1])).long(), train_labels))
            for aug in self.augmentations:
                aug_data = torch.cat((aug_data, aug(train_data)))
                aug_labels = torch.cat((aug_labels, train_labels))

            
            # clear unnecessary data after augmentations
            del train_data, train_labels, align_labels

            test_shape = test_data.shape
            test_data = dim_red.transform(test_data.reshape(-1, test_shape[-1]))
            test_data = torch.Tensor(test_data.reshape(test_shape[0], test_shape[1], -1))

            # save fold precomputed fold data to hdf5 file to load in later
            os.makedirs(self.data_path / 'fold_data', exist_ok=True)
            with h5py.File(self.data_path / 'fold_data' / f'fold_{k}.h5', 'w') as f:
                f.create_dataset('train_data', data=aug_data)
                f.create_dataset('train_labels', data=aug_labels)
                f.create_dataset('val_data', data=val_data)
                f.create_dataset('val_labels', data=val_labels)
                f.create_dataset('test_data', data=test_data)
                f.create_dataset('test_labels', data=test_labels)
    

def process_aligner(X, y, y_align, pool_data, algner, n_components=0.95):
    """PCA-reduces and CCA-aligns cross-patient data onto the target space.

    Applies PCA independently to each dataset, aligns each cross-patient
    dataset to the target via the provided aligner, and concatenates all
    data and labels.

    Args:
        X: Target patient features of shape (n_trials, n_timepoints, n_features).
        y: Target patient labels.
        y_align: Alignment labels for the target patient.
        pool_data: List of (features, labels, align_labels) tuples for
            cross-patient datasets.
        algner: Callable returning an aligner instance with fit/transform API.
        n_components: PCA variance threshold or number of components.
            Defaults to 0.95.

    Returns:
        tuple: (X_pool, y_pool, tar_dr) where X_pool is the concatenated
            aligned feature tensor, y_pool is the concatenated label tensor,
            and tar_dr is the fitted PCA object for the target patient.
    """
    cross_pt_trials = [x.shape[0] for x, _, _ in pool_data]
    X_cross_r = [x.reshape(-1, x.shape[-1]) for x, _, _ in pool_data]
    X_tar_r = X.reshape(-1, X.shape[-1])

    # reduce dimensionality of cross-patient data
    X_cross_dr = [PCA(n_components=n_components).fit_transform(x)
                  for x in X_cross_r]

    # reduce dimensionality of target data, saving dim. red. object for
    # test set
    tar_dr = PCA(n_components=n_components)
    X_tar_dr = tar_dr.fit_transform(X_tar_r)

    # reshape back to 3D
    X_cross_dr = [x.reshape(cross_pt_trials[i], -1, x.shape[-1]) for i, x
                  in enumerate(X_cross_dr)]
    X_tar_dr = X_tar_dr.reshape(X.shape[0], -1, X_tar_dr.shape[-1])

    # option for separate alignment labels
    if y_align is None:
        y_align = y
    y_align_cross = [y_a for _, _, y_a in pool_data]

    # align data to target patient
    aligns = [algner() for _ in range(len(pool_data))]
    X_algn_dr = []
    for i, algn in enumerate(aligns):
        algn.fit(X_tar_dr, X_cross_dr[i], y_align.numpy(), y_align_cross[i].numpy())
        X_algn_dr.append(algn.transform(X_cross_dr[i]))

    # concatenate cross-patient data
    X_pool = np.vstack([X_tar_dr] + X_algn_dr)
    try:
        y_pool = np.hstack([y] + [y for _, y, _ in pool_data])
    except ValueError:
        y_pool = np.vstack([y] + [y for _, y, _ in pool_data])

    return torch.Tensor(X_pool), torch.Tensor(y_pool).long().squeeze(1), tar_dr