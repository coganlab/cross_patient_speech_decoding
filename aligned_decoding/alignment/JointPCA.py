""" Class to perform joint PCA decomposition of multiple datasets to align
them to a shared space.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
from sklearn.decomposition import PCA
from .alignment_utils import extract_group_conditions


class JointPCA:

    def __init__(self, n_components=40, dim_red=PCA):
        """Initializes JointPCADecomp class with the number of latent
        components and the method for dimensionality reduction.

        Args:
            n_components (int, optional): Number of components for
                dimensionality reduction i.e. dimensionality of latent space.
                Defaults to 40.
            dim_red (Callable, optional): Dimensionality reduction function.
                Must implement sklearn-style fit_transform() function. Defaults
                to PCA.
        """
        self.n_components = n_components
        self.dim_red = dim_red

    def fit(self, X, y):
        """Learns source-specific (e.g. patient-specific) transformations to
        the shared latent space and stores transformations in self.transforms.

        Args:
            X (list of ndarray): List of features from multiple sources to
            compute shared latent space.
            y (list of ndarray): List of labels corresponding to feature
            sources. Must be the same length as features.
        """
        transforms = get_joint_PCA_transforms(X, y,
                                              n_components=self.n_components,
                                              dim_red=self.dim_red)
        self.transforms = transforms

    def transform(self, X, idx=-1):
        """Applies learned transformations to input data. Supports transforming
        a single, specified dataset or all-source datasets at once.

        Args:
            X (ndarray or list of ndarray): Features to transform. If a list,
                the length must be equal to the number of learned transforms
                (i.e. transforming all sources). If an ndarray, a
                source-specific transformation is applied to the data, with the
                source specified by the idx input.
            idx (int, optional): Index of saved transform list to apply to
                single source data, or -1 if applying to all sources. Defaults
                to -1.

        Raises:
            IndexError: Error if idx is too large to select a learned transform
                from the saved list.
            RuntimeError: Error if fit() has not been called before calling
                transform().

        Returns:
            ndarray or tuple: Transformed data from single session if idx is
                not -1, or a tuple of containing:
                    Transformed data from all sources input to the fit()
                    method. Length will be equal to the number of learned
                    transformations.
        """
        if not self._check_fit():
            raise RuntimeError('Must call fit() before transforming data.')
        if idx == -1:
            return self._transform_multiple(X)
        if idx >= len(self.transforms):
            raise IndexError('Input idx is greater than the number of learned '
                             'transforms. For transformation of data from a '
                             'specific session, provide the input idx as the '
                             'index of the session in the input list. If '
                             'transforming multiple sessions, set idx=-1 '
                             '(default).')
        return self._transform_single(X, idx)

    def fit_transform(self, X, y):
        """Fits the model with X and y and applies the learned transformations
        to X.

        Args:
            X (list of ndarray): List of features from multiple sources to
            compute shared latent space.
            y (list of ndarray): List of labels corresponding to feature
            sources. Must be the same length as features.

        Returns:
            tuple: tuple of containing:
                    Transformed ndarray data from all sources input to the
                    fit() method. Length will be equal to the number of learned
                    transformations.
        """
        self.fit(X, y)
        return self.transform(X)

    def _transform_multiple(self, X):
        """Uses learned latent space transformations to transform data from all
        sources used to fit the decomposition.

        Args:
            X (list of ndarray): List of features from multiple sources to
            compute shared latent space.

        Returns:
            tuple of ndarray: tuple of containing:
                    Transformed ndarray data from all sources input to the
                    fit() method. Length will be equal to the number of learned
                    transformations.
        """
        transformed_lst = [0]*len(X)
        for i, (feats, transform) in enumerate(zip(X, self.transforms)):
            transform_feats = feats.reshape(-1, feats.shape[-1]) @ transform
            transformed_lst[i] = transform_feats.reshape(feats.shape[:-1] +
                                                         (-1,))
        return (*transformed_lst,)

    def _transform_single(self, X, idx):
        """Applies learned latent space transformations to data from a single
        a single source specified by idx.

        Args:
            X (ndarray): Features to transform.
            idx (int): Index of learned transforms to apply to X.

        Returns:
            ndarray: Features transformed to shared latent space.
        """
        transform = self.transforms[idx]
        transform_feats = X.reshape(-1, X.shape[-1]) @ transform
        return transform_feats.reshape(X.shape[:-1] + (-1,))

    def _check_fit(self):
        """Checks if the joint PCA decomposition has been fit to data.

        Returns:
            boolean: True if fit() has been called, False otherwise.
        """
        try:
            self.transforms
        except AttributeError:
            return False
        return True
    

def get_joint_PCA_transforms(features, labels, n_components=40, dim_red=PCA):
    """Calculates a shared latent space across features from multiple patients
    or recording sessions.

    Uses the method described by Pandarinath et al. in
    https://www.nature.com/articles/s41592-018-0109-9 (2018) for pre-computing
    session specific read-in matrices (see Methods: Modifications to the LFADS
    algorithm for stitching together data from multiple recording sessions)

    Args:
        features (list): List of features from multiple sources to compute
            shared latent space.
        labels (list): List of labels corresponding to feature sources. Must
            be the same length as features.
        n_components (int, optional): Number of components for dimensionality
            reduction i.e. dimensionality of latent space. Defaults to 40.
        dim_red (Callable, optional): Dimensionality reduction function. Must
            implement sklearn-style fit_transform() function. Defaults to PCA.

    Returns:
        tuple: tuple containing:
            Transformation matrices to shared latent space for each input
            source. Length will be equal to the length of the input feature
            list.
    """
    cnd_avg_data = extract_group_conditions(features, labels)

    # combine all datasets into one matrix (n_conditions x n_timepoints x
    # sum channels)
    cross_pt_mat = np.concatenate(cnd_avg_data, axis=-1)
    # reshape to 2D with channels as final dim
    cross_pt_mat = cross_pt_mat.reshape(-1, cross_pt_mat.shape[-1])

    # perform dimensionality reduction on channel dim of combined matrix
    latent_mat = dim_red(n_components=n_components).fit_transform(cross_pt_mat)

    # calculate per pt channel -> factor transformation matrices
    pt_latent_trans = [0]*len(cnd_avg_data)
    for i, pt_ca in enumerate(cnd_avg_data):
        pt_ca = pt_ca.reshape(-1, pt_ca.shape[-1])  # isolate channel dim
        latent_trans = np.linalg.pinv(pt_ca) @ latent_mat  # lst_sq soln
        pt_latent_trans[i] = latent_trans
        # latent_trans = np.linalg.pinv(latent_mat) @ pt_ca  # lst_sq soln
        # pt_latent_trans[i] = latent_trans.T
        # pt_latent_trans[i] = np.linalg.pinv(latent_trans)

    return (*pt_latent_trans,)