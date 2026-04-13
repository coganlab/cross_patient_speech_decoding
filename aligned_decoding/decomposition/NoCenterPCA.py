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
    """PCA implementation that skips mean-centering of input data.

    Uses truncated SVD directly on the (uncentered) data matrix.
    ``n_components`` can be an integer for a fixed number of components
    or a float in (0, 1) to select components by cumulative explained
    variance.

    Attributes:
        n_components: Number or fraction of components to keep.
        components_: Projection matrix of shape ``(n_features, k)``
            (set after fit).
        explained_variance_: Squared singular values (set after fit).
    """

    def __init__(self, n_components=None):
        """Initializes NoCenterPCA.

        Args:
            n_components: Number of components to retain. If an int >= 1,
                keeps that many components. If a float in (0, 1), selects
                the minimum number of components explaining at least that
                fraction of variance. If None, keeps ``min(n_samples,
                n_features)`` components.
        """
        self.n_components = n_components
        self._fit = False

    def fit(self, X, y=None):
        """Fits the PCA model via SVD without centering.

        Args:
            X: Input array of shape ``(n_samples, n_features)``.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            self
        """
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = self._get_components(X, S)
        self.components_ = Vt[:k].T
        self.explained_variance_ = S**2
        self._fit = True
        return self

    def transform(self, X):
        """Projects data onto the fitted principal components.

        Args:
            X: Input array of shape ``(n_samples, n_features)``.

        Returns:
            Projected array of shape ``(n_samples, k)``.

        Raises:
            ValueError: If the model has not been fitted.
        """
        self._check_fit()
        return X @ self.components_

    def fit_transform(self, X, y=None):
        """Fits the model and projects data in a single step.

        Args:
            X: Input array of shape ``(n_samples, n_features)``.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            Projected array of shape ``(n_samples, k)``.
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _get_components(self, X, S):
        """Resolves ``n_components`` to a concrete integer count.

        Args:
            X: Input array used to determine the maximum possible
                number of components.
            S: Singular values from the SVD of ``X``.

        Returns:
            Integer number of components to keep.
        """
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
        """Verifies that the model has been fitted.

        Raises:
            ValueError: If ``fit`` has not been called.
        """
        if not self._fit:
            raise ValueError("PCA must be fit before transforming data.")