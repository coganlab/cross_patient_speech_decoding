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
    def __init__(self, data, labels, batch_size=128, folds=20, val_size=0.2,
                 augmentations=None, data_path=None):
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
        self.data_path = Path(os.getcwd() if data_path is None else data_path)
        # self.data_shapes_folds = []
        # self.train_datasets = []
        # self.val_datasets = []
        # self.test_datasets = []

    def setup(self, stage=None):
        cv = self.select_cv(self.folds)
        for k, (train_idx, test_idx) in enumerate(cv.split(self.data, self.labels)):
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

            # self.data_shapes_folds.append(aug_data.shape)

            # train_dataset = TensorDataset(train_data, train_labels)
            # train_dataset = TensorDataset(aug_data, aug_labels)
            # val_dataset = TensorDataset(val_data, val_labels)
            # test_dataset = TensorDataset(test_data, test_labels)

            # self.train_datasets.append(train_dataset)
            # self.val_datasets.append(val_dataset)
            # self.test_datasets.append(test_dataset)

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
        # get the train dataset for the current fold
        # return DataLoader(self.train_datasets[self.current_fold],
        #                   batch_size=self.batch_size, shuffle=True,
        #                   # num_workers=7, persistent_workers=True,
        #                   )
        
        # get train data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            train_data = f['train_data'][()]
            train_labels = f['train_labels'][()]
        
        train_data = torch.Tensor(train_data)
        train_labels = torch.Tensor(train_labels).long()
        return DataLoader(TensorDataset(train_data, train_labels),
                          batch_size=self.batch_size, shuffle=True,
                          # num_workers=7, persistent_workers=True,
                          )

    def val_dataloader(self):
        # get the val dataset for the current fold
        # return DataLoader(self.val_datasets[self.current_fold],
        #                   batch_size=self.batch_size, shuffle=False,
        #                   # num_workers=7, persistent_workers=True,
        #                   )

        # get val data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            val_data = f['val_data'][()]
            val_labels = f['val_labels'][()]

        val_data = torch.Tensor(val_data)
        val_labels = torch.Tensor(val_labels).long()
        return DataLoader(TensorDataset(val_data, val_labels),
                          batch_size=self.batch_size, shuffle=False,
                          # num_workers=7, persistent_workers=True,
                          )

    def test_dataloader(self):
        # # get the test dataset for the current fold
        # return DataLoader(self.test_datasets[self.current_fold],
        #                   batch_size=self.batch_size, shuffle=False,
        #                   # num_workers=7, persistent_workers=True,
        #                   )

        # get test data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            test_data = f['test_data'][()]
            test_labels = f['test_labels'][()]

        test_data = torch.Tensor(test_data)
        test_labels = torch.Tensor(test_labels).long()
        return DataLoader(TensorDataset(test_data, test_labels),
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

    def get_data_shape(self):
            with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
                data_shape = f['train_data'][()].shape
            return data_shape


class AlignedMicroDataModule(L.LightningDataModule):
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
        if self.batch_size == -1:
            self.batch_size = len(self.data) + sum(len(x) for x, _, _ in pool_data)
        self.folds = folds
        self.val_size = val_size
        self.augmentations = augmentations if augmentations else []
        self.data_path = Path(os.getcwd() if data_path is None else data_path)
        self.current_fold = 0
        # self.data_shapes_folds = []
        # self.train_datasets = []
        # self.val_datasets = []
        # self.test_datasets = []

    def setup(self, stage=None):
        cv = self.select_cv(self.folds)
        for k, (train_idx, test_idx) in enumerate(cv.split(self.data, self.labels)):
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

            aug_data = torch.cat((torch.Tensor([]), train_data))
            aug_labels = torch.cat((torch.Tensor([]).long(), train_labels))
            aug_align_labels = torch.cat((torch.Tensor([[], [], []]).long().T, align_labels))
            aug_pool_data = [[
                torch.cat((torch.Tensor([]),x)),
                torch.cat((torch.Tensor([]).long(),y)),
                torch.cat((torch.Tensor([[],[],[]]).long().T, y_a))
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
            # train_data, train_labels, dim_red = (
            #     process_aligner(train_data, train_labels, align_labels,
            #                     self.pool_data, self.algner))
            aug_data, aug_labels, dim_red = (
                process_aligner(aug_data, aug_labels, aug_align_labels,
                                aug_pool_data, self.algner))

            # aug_data = torch.Tensor([])
            # aug_labels = torch.Tensor([]).long()
            # for aug in self.augmentations:
            #     aug_data = torch.cat((aug_data, aug(train_data)))
            #     aug_labels = torch.cat((aug_labels, train_labels))

            if val_data is not None:
                val_shape = val_data.shape
                val_data = dim_red.transform(val_data.reshape(-1, val_shape[-1]))
                val_data = torch.Tensor(val_data.reshape(val_shape[0], val_shape[1], -1))
            test_shape = test_data.shape
            test_data = dim_red.transform(test_data.reshape(-1, test_shape[-1]))
            test_data = torch.Tensor(test_data.reshape(test_shape[0], test_shape[1], -1))

            # train_dataset = TensorDataset(train_data, train_labels)
            # train_dataset = TensorDataset(aug_data, aug_labels)
            # val_dataset = TensorDataset(val_data, val_labels)
            # test_dataset = TensorDataset(test_data, test_labels)

            # self.train_datasets.append(train_dataset)
            # self.val_datasets.append(val_dataset)
            # self.test_datasets.append(test_dataset)

            # self.data_shapes_folds.append(aug_data.shape)

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
        # get the train dataset for the current fold
        # return DataLoader(self.train_datasets[self.current_fold],
        #                   batch_size=self.batch_size, shuffle=True,
        #                   # num_workers=7, persistent_workers=True,
        #                   )
        
        # get train data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            train_data = f['train_data'][()]
            train_labels = f['train_labels'][()]
        
        train_data = torch.Tensor(train_data)
        train_labels = torch.Tensor(train_labels).long()
        return DataLoader(TensorDataset(train_data, train_labels),
                          batch_size=self.batch_size, shuffle=True,
                          # num_workers=7, persistent_workers=True,
                          )

    def val_dataloader(self):
        # get the val dataset for the current fold
        # return DataLoader(self.val_datasets[self.current_fold],
        #                   batch_size=self.batch_size, shuffle=False,
        #                   # num_workers=7, persistent_workers=True,
        #                   )

        # get val data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            val_data = f['val_data'][()]
            val_labels = f['val_labels'][()]

        val_data = torch.Tensor(val_data)
        val_labels = torch.Tensor(val_labels).long()
        return DataLoader(TensorDataset(val_data, val_labels),
                          batch_size=self.batch_size, shuffle=False,
                          # num_workers=7, persistent_workers=True,
                          )

    def test_dataloader(self):
        # # get the test dataset for the current fold
        # return DataLoader(self.test_datasets[self.current_fold],
        #                   batch_size=self.batch_size, shuffle=False,
        #                   # num_workers=7, persistent_workers=True,
        #                   )

        # get test data from the current fold from saved hdf5 file
        with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
            test_data = f['test_data'][()]
            test_labels = f['test_labels'][()]

        test_data = torch.Tensor(test_data)
        test_labels = torch.Tensor(test_labels).long()
        return DataLoader(TensorDataset(test_data, test_labels),
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

    def get_data_shape(self):
            with h5py.File(self.data_path / 'fold_data' / f'fold_{self.current_fold}.h5', 'r') as f:
                data_shape = f['train_data'][()].shape
            return data_shape
    

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