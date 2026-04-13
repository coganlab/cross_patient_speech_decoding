""" Class to perform MCCA alignment of multiple datasets via mvlearn
implementation of MCCA.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
from mvlearn.embed import MCCA
from .alignment_utils import extract_group_conditions


class AlignMCCA:
    """MCCA-based alignment of multiple neural datasets into a shared space.

    Wraps mvlearn's MCCA to learn multi-view transformations from
    condition-averaged data and apply them to individual trials.

    Attributes:
        n_components (int): Number of CCA components.
        regs (float): Regularization parameter for MCCA.
        pca_var (float): Fraction of PCA variance to retain for rank
            estimation. Set to 1 to skip PCA pre-reduction.
        mcca (mvlearn.embed.MCCA): Fitted MCCA object (set after fit).
    """

    def __init__(self, n_components=10, regs=0.5, pca_var=1):
        """Initializes the AlignMCCA instance.

        Args:
            n_components (int, optional): Number of canonical components.
                Defaults to 10.
            regs (float, optional): Regularization parameter for MCCA.
                Defaults to 0.5.
            pca_var (float, optional): Fraction of variance to retain when
                estimating signal ranks via PCA. Values in (0, 1) enable
                rank estimation; 1 disables it. Defaults to 1.
        """
        self.n_components = n_components
        self.regs = regs
        self.pca_var = pca_var

    def fit(self, X, y):
        """Fits the MCCA model on condition-averaged multi-view data.

        Args:
            X (list of ndarray): List of feature arrays, one per view/session.
            y (list of ndarray): List of label arrays corresponding to each
                view in X.
        """
        mcca = get_MCCA_transforms(X, y, n_components=self.n_components,
                                        regs=self.regs,
                                        pca_var=self.pca_var)
        self.mcca = mcca

    def transform(self, X, idx=-1):
        """Transforms data using the learned MCCA alignment.

        Args:
            X (ndarray or list of ndarray): Data to transform. A list
                transforms all views; a single ndarray transforms one view
                specified by idx.
            idx (int, optional): Index of the view to transform. -1
                transforms all views. Defaults to -1.

        Returns:
            tuple of ndarray or ndarray: Transformed data for all views
                (if idx=-1) or a single transformed array.

        Raises:
            RuntimeError: If fit() has not been called.
            IndexError: If idx exceeds the number of learned transforms.
        """
        if not self._check_fit():
            raise RuntimeError('Must call fit() before transforming data.')
        if idx == -1:
            return self._transform_multiple(X)
        if idx >= len(self.mcca.loadings_):
            raise IndexError('Input idx is greater than the number of learned '
                             'transforms. For transformation of data from a '
                             'specific session, provide the input idx as the '
                             'index of the session in the input list. If '
                             'transforming multiple sessions, set idx=-1 '
                             '(default).')
        return self._transform_single(X, idx)
    
    def fit_transform(self, X, y):
        """Fits the model and transforms all views.

        Args:
            X (list of ndarray): List of feature arrays, one per view.
            y (list of ndarray): List of label arrays corresponding to X.

        Returns:
            tuple of ndarray: Transformed data for all views.
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _transform_multiple(self, X):
        """Transforms all views into the shared MCCA space.

        Args:
            X (list of ndarray): List of feature arrays, one per view.

        Returns:
            tuple of ndarray: Transformed arrays for each view, reshaped
                to match the original trial structure.
        """
        transformed_data = [self.mcca.transform_view(x.reshape(-1, x.shape[-1]), i) for i, x in enumerate(X)]
        transformed_data = [d.reshape(x.shape[:-1] + (-1,)) for d, x in zip(transformed_data, X)]
        return (*transformed_data,)
    
    def _transform_single(self, X, idx):
        """Transforms a single view into the shared MCCA space.

        Args:
            X (ndarray): Feature array for a single view.
            idx (int): Index of the view's learned transform to apply.

        Returns:
            ndarray: Transformed data reshaped to match the original
                trial structure.
        """
        transformed_data = self.mcca.transform_view(X.reshape(-1, X.shape[-1]), idx)
        return transformed_data.reshape(X.shape[:-1] + (-1,))
    
    def _check_fit(self):
        """Checks if the MCCA aligner has been fit to data.

        Returns:
            boolean: True if fit() has been called, False otherwise.
        """
        try:
            self.mcca
        except AttributeError:
            return False
        return True
    
def get_MCCA_transforms(features, labels, n_components=10, regs=0.5,
                        pca_var=1):
    """Calculates transformation matrices to multi-view shared latent space"""
    cnd_avg_data = extract_group_conditions(features, labels)
    cnd_avg_data = [d.reshape(-1, d.shape[-1]) for d in cnd_avg_data]
    
    ranks = None
    if pca_var > 0 and pca_var < 1:
        ranks = [min(n_components,
                     n_components_var(x.reshape(-1, x.shape[-1]), pca_var))
                     for x in features]

    mcca = MCCA(n_components=n_components, regs=regs, signal_ranks=ranks)
    mcca.fit(cnd_avg_data)
    return mcca

def n_components_var(X, var):
    """Determines the number of PCA components needed to explain a given
    fraction of variance.

    Args:
        X (ndarray): 2D data array of shape (n_samples, n_features).
        var (float): Target cumulative variance fraction in (0, 1).

    Returns:
        int: Minimum number of components whose cumulative explained
            variance exceeds var.
    """
    # get squared singular values from svd of X
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s = s**2
    # normalize singular values to sum to 1
    s /= np.sum(s)
    # find number of components to reach desired variance %
    return np.argmax(np.cumsum(s) > var)

