""" Class combining some form of dimensionality reduction with reshaping of 
input data to a 2D format. This is useful for consistency of inputs when
passing data through a pipeline where some methods work on the data in a
different format than (n_samples, n_features). 

Author: Zac Spalding
"""

from sklearn.base import BaseEstimator

class DimRedReshape(BaseEstimator):
    """Wraps a dimensionality reduction method with automatic 2D reshaping.

    Flattens input arrays to (n_samples, n_features) before applying the
    specified dimensionality reduction, allowing consistent pipeline usage
    with higher-dimensional input data.

    Attributes:
        dim_red: Dimensionality reduction class to instantiate.
        n_components: Number of components for the reducer.
        transformer: Fitted dimensionality reduction instance (set after fit).
    """

    def __init__(self, dim_red, n_components=None):
        """Initializes DimRedReshape.

        Args:
            dim_red: A dimensionality reduction class (e.g., PCA) that accepts
                an ``n_components`` keyword argument.
            n_components: Number of components to keep. Passed directly to
                ``dim_red``. Defaults to None.
        """
        self.dim_red = dim_red
        self.n_components = n_components

    def fit(self, X, y=None):
        """Fits the dimensionality reduction model on reshaped data.

        Args:
            X: Input array of shape ``(n_samples, ...)``. Dimensions beyond
                the first are flattened.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            self
        """
        X_r = X.reshape(X.shape[0], -1)
        self.transformer = self.dim_red(n_components=self.n_components)
        self.transformer.fit(X_r)
        return self

    def transform(self, X, y=None):
        """Transforms data using the fitted dimensionality reduction model.

        Args:
            X: Input array of shape ``(n_samples, ...)``. Dimensions beyond
                the first are flattened.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            Transformed array of shape ``(n_samples, n_components)``.
        """
        X_r = X.reshape(X.shape[0], -1)
        X_dr = self.transformer.transform(X_r)
        return X_dr

    def fit_transform(self, X, y=None):
        """Fits the model and transforms data in a single step.

        Args:
            X: Input array of shape ``(n_samples, ...)``. Dimensions beyond
                the first are flattened.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            Transformed array of shape ``(n_samples, n_components)``.
        """
        self.fit(X)
        return self.transform(X)