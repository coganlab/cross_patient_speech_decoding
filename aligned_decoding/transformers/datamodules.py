import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.decomposition import PCA
import sys

# sys.path.append('..')
# from alignment.AlignCCA import AlignCCA

class SimpleMicroDataModule(L.LightningDataModule):
    def __init__(self, data, labels, batch_size=128, folds=20, val_size=0.2,
                 augmentations=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        if self.batch_size == -1:
            self.batch_size = len(self.data)
        self.folds = folds
        self.val_size = val_size
        self.augmentations = augmentations if augmentations else []
        self.current_fold = 0
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

    def setup(self, stage=None):
        cv = self.select_cv(self.folds)
        for train_idx, test_idx in cv.split(self.data, self.labels):
            train_data, test_data = self.data[train_idx], self.data[test_idx]
            train_labels, test_labels = (self.labels[train_idx],
                                         self.labels[test_idx])

            if self.val_size > 0:
                train_data, val_data, train_labels, val_labels = (
                    train_test_split(train_data, train_labels,
                                     test_size=self.val_size,
                                     stratify=train_labels))
            else:
                val_data, val_labels = None, None

            aug_data = torch.Tensor([])
            aug_labels = torch.Tensor([]).long()
            for aug in self.augmentations:
                aug_data = torch.cat((aug_data, aug(train_data)))
                aug_labels = torch.cat((aug_labels, train_labels))

            # train_dataset = TensorDataset(train_data, train_labels)
            train_dataset = TensorDataset(aug_data, aug_labels)
            val_dataset = TensorDataset(val_data, val_labels)
            test_dataset = TensorDataset(test_data, test_labels)

            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)

    def train_dataloader(self):
        # get the train dataset for the current fold
        return DataLoader(self.train_datasets[self.current_fold],
                          batch_size=self.batch_size, shuffle=True,
                          # num_workers=7, persistent_workers=True,
                          )

    def val_dataloader(self):
        # get the val dataset for the current fold
        return DataLoader(self.val_datasets[self.current_fold],
                          batch_size=self.batch_size, shuffle=False,
                          # num_workers=7, persistent_workers=True,
                          )

    def test_dataloader(self):
        # get the test dataset for the current fold
        return DataLoader(self.test_datasets[self.current_fold],
                          batch_size=self.batch_size, shuffle=False,
                          # num_workers=7, persistent_workers=True,
                          )

    def set_fold(self, fold):
        assert 0 <= fold < self.folds, "Fold index out of range"
        self.current_fold = fold

    def select_cv(self, folds):
        class_counts = torch.bincount(self.labels)
        if (class_counts < folds).any():
            cv = KFold(n_splits=folds, shuffle=True)
        else:
            cv = StratifiedKFold(n_splits=folds, shuffle=True)
        return cv

    def get_data_shape(self, type='train'):
        if type == 'train':
            return self.train_datasets[self.current_fold].tensors[0].shape
        elif type == 'val':
            return self.val_datasets[self.current_fold].tensors[0].shape
        elif type == 'test':
            return self.test_datasets[self.current_fold].tensors[0].shape
        else:
            print('Type must be one of "train", "val", or "test"')

class AlignedMicroDataModule(L.LightningDataModule):
    def __init__(self, data, labels, align_labels, pool_data, algner,
                 batch_size=128, folds=20, val_size=0.2, augmentations=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.align_labels = align_labels
        self.pool_data = pool_data
        self.algner = algner
        self.batch_size = batch_size
        if self.batch_size == -1:
            self.batch_size = len(self.data) + sum(len(x) for x, _, _ in pool_data)
        self.folds = folds
        self.val_size = val_size
        self.augmentations = augmentations if augmentations else []
        self.current_fold = 0
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

    def setup(self, stage=None):
        cv = self.select_cv(self.folds)
        for train_idx, test_idx in cv.split(self.data, self.labels):
            train_data, test_data = self.data[train_idx], self.data[test_idx]
            train_labels, test_labels = (self.labels[train_idx],
                                         self.labels[test_idx])
            align_labels = self.align_labels[train_idx]

            if self.val_size > 0:
                train_data, val_data, train_labels, val_labels, align_labels, _ = \
                    (train_test_split(train_data, train_labels,
                                      align_labels,
                                     test_size=self.val_size,
                                     stratify=train_labels))
            else:
                val_data, val_labels = None, None

            # aug_data = torch.Tensor([])
            # aug_labels = torch.Tensor([]).long()
            # aug_align_labels = torch.Tensor([]).long()
            # for aug in self.augmentations:
            #     aug_data = torch.cat((aug_data, aug(train_data)))
            #     aug_labels = torch.cat((aug_labels, train_labels))
            #     aug_align_labels = torch.cat((aug_labels, align_labels))
            #     aug_pool_data = [(aug(x), y, y_a) for x, y, y_a in self.pool_data]
            #     pool_data = [(x, y, y_a) for x, y, y_a in aug_pool_data]


            # align pooled data to current data
            train_data, train_labels, dim_red = (
                process_aligner(train_data, train_labels, align_labels,
                                self.pool_data, self.algner))
            # train_data, train_labels, dim_red = (
            #     process_aligner(aug_data, aug_labels, aug_align_labels,
            #                     self.pool_data, self.algner))

            aug_data = torch.Tensor([])
            aug_labels = torch.Tensor([]).long()
            for aug in self.augmentations:
                aug_data = torch.cat((aug_data, aug(train_data)))
                aug_labels = torch.cat((aug_labels, train_labels))

            if val_data is not None:
                val_data = dim_red.transform(val_data.reshape(-1, val_data.shape[-1]))
                val_data = torch.Tensor(val_data.reshape(-1, train_data.shape[1], val_data.shape[-1]))
            test_data = dim_red.transform(test_data.reshape(-1, test_data.shape[-1]))
            test_data = torch.Tensor(test_data.reshape(-1, train_data.shape[1], test_data.shape[-1]))

            # train_dataset = TensorDataset(train_data, train_labels)
            train_dataset = TensorDataset(aug_data, aug_labels)
            val_dataset = TensorDataset(val_data, val_labels)
            test_dataset = TensorDataset(test_data, test_labels)

            self.train_datasets.append(train_dataset)
            self.val_datasets.append(val_dataset)
            self.test_datasets.append(test_dataset)

    def train_dataloader(self):
        # get the train dataset for the current fold
        return DataLoader(self.train_datasets[self.current_fold],
                          batch_size=self.batch_size, shuffle=True,
                          # num_workers=7, persistent_workers=True,
                          )

    def val_dataloader(self):
        # get the val dataset for the current fold
        return DataLoader(self.val_datasets[self.current_fold],
                          batch_size=self.batch_size, shuffle=False,
                          # num_workers=7, persistent_workers=True,
                          )

    def test_dataloader(self):
        # get the test dataset for the current fold
        return DataLoader(self.test_datasets[self.current_fold],
                          batch_size=self.batch_size, shuffle=False,
                          # num_workers=7, persistent_workers=True,
                          )

    def set_fold(self, fold):
        assert 0 <= fold < self.folds, "Fold index out of range"
        self.current_fold = fold

    def select_cv(self, folds):
        class_counts = torch.bincount(self.labels)
        if (class_counts < folds).any():
            cv = KFold(n_splits=folds, shuffle=True)
        else:
            cv = StratifiedKFold(n_splits=folds, shuffle=True)
        return cv

    def get_data_shape(self, type='train'):
        if type == 'train':
            return self.train_datasets[self.current_fold].tensors[0].shape
        elif type == 'val':
            return self.val_datasets[self.current_fold].tensors[0].shape
        elif type == 'test':
            return self.test_datasets[self.current_fold].tensors[0].shape
        else:
            print('Type must be one of "train", "val", or "test"')


def process_aligner(X, y, y_align, pool_data, algner, n_components=0.95):
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

    # X_algn_dr = [x.reshape(x.shape[0], -1) for x in X_algn_dr]
    # X_tar_dr = X_tar_dr.reshape(X_tar_dr.shape[0], -1)

    # concatenate cross-patient data
    X_pool = np.vstack([X_tar_dr] + X_algn_dr)
    y_pool = np.hstack([y] + [y for _, y, _ in pool_data])

    return torch.Tensor(X_pool), torch.Tensor(y_pool).long(), tar_dr