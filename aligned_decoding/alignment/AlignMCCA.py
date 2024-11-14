""" Class to perform MCCA alignment of multiple datasets via mvlearn
implementation of MCCA.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
from mvlearn.embed import MCCA
from .alignment_utils import extract_group_conditions


class AlignMCCA:

    def __init__(self, n_components=10, regs=0.5, pca_var=1):
        self.n_components = n_components
        self.regs = regs
        self.pca_var = pca_var

    def fit(self, X, y):
        mcca = get_MCCA_transforms(X, y, n_components=self.n_components,
                                        regs=self.regs,
                                        pca_var=self.pca_var)
        self.mcca = mcca

    def transform(self, X, idx=-1):
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
        self.fit(X, y)
        return self.transform(X)
    
    def _transform_multiple(self, X):
        transformed_data = [self.mcca.transform_view(x.reshape(-1, x.shape[-1]), i) for i, x in enumerate(X)]
        transformed_data = [d.reshape(x.shape[:-1] + (-1,)) for d, x in zip(transformed_data, X)]
        return (*transformed_data,)
    
    def _transform_single(self, X, idx):
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
    # get squared singular values from svd of X
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s = s**2
    # normalize singular values to sum to 1
    s /= np.sum(s)
    # find number of components to reach desired variance %
    return np.argmax(np.cumsum(s) > var)

