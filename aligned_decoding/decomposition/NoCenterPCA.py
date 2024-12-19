""" Implementation of PCA without mean-centering of the input data. Created
as a custom class due to the lack of this functionality in the sklearn PCA
implementation. Lack of centering is desired because the input data is already
normalized by the experiment structure and removing centering tends to increase
decoding performance.

Author: Zac Spalding
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NoCenterPCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=None):
        self.n_components = n_components
        self._fit = False

    def fit(self, X, y=None):
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = self._get_components(X, S)
        self.components_ = Vt[:k].T
        self.explained_variance_ = S**2
        self._fit = True
        return self

    def transform(self, X):
        self._check_fit()
        return X @ self.components_

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def _get_components(self, X, S):
        if self.n_components is None or self.n_components >= min(X.shape):
            print("n_components is None or greater than the number of features"
                  "/samples. Using n_components = min(X.shape)")
            return min(X.shape)
        elif self.n_components < 1:
            cum_var = np.cumsum(S**2) / np.sum(S**2)
            return np.argmax(cum_var >= self.n_components) + 1
        else:
            return int(self.n_components)
        
    def _check_fit(self):
        if not self._fit:
            raise ValueError("PCA must be fit before transforming data.")