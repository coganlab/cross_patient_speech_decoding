""" Class combining some form of dimensionality reduction with reshaping of 
input data to a 2D format. This is useful for consistency of inputs when
passing data through a pipeline where some methods work on the data in a
different format than (n_samples, n_features). 

Author: Zac Spalding
"""

from sklearn.base import BaseEstimator

class DimRedReshape(BaseEstimator):

    def __init__(self, dim_red, n_components=None):
        self.dim_red = dim_red
        self.n_components = n_components

    def fit(self, X, y=None):
        # X_r = X.reshape(-1, X.shape[-1])
        X_r = X.reshape(X.shape[0], -1)
        self.transformer = self.dim_red(n_components=self.n_components)
        self.transformer.fit(X_r)
        return self

    def transform(self, X, y=None):
        # X_r = X.reshape(-1, X.shape[-1])
        X_r = X.reshape(X.shape[0], -1)
        X_dr = self.transformer.transform(X_r)
        # X_dr = X_dr.reshape(X.shape[0], -1)
        return X_dr

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)