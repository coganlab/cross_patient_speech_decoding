""" Cross-Patient Decoder Classes

Author: Zac Spalding
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

class crossPtDecoder(BaseEstimator):

    def preprocess_train(self, X, y=None):
        pass

    def preprocess_test(self, X, y=None):
        pass

    def fit(self, X, y):
        X_p, y_p = self.preprocess_train(X, y)
        return self.decoder.fit(X_p, y_p)
    
    def predict(self, X):
        X_p = self.preprocess_test(X)
        return self.decoder.predict(X_p)
    
    def score(self, X, y, **kwargs):
        X_p = self.preprocess_test(X)
        return self.decoder.score(X_p, y, **kwargs)

class crossPtDecoder_sepDimRed(crossPtDecoder):
    """ Cross-Patient Decoder with separate PCA for each patient. """
    
    def __init__(self, cross_pt_data, decoder, dim_red=PCA, n_comp=10):
        self.cross_pt_data = cross_pt_data
        self.decoder = decoder
        self.dim_red = dim_red
        self.n_comp = n_comp

    def preprocess_train(self, X, y):
        cross_pt_trials = [x.shape[0] for x, _ in self.cross_pt_data]
        tar_trials = X.shape[0]
        # reshape features to be 2D (preserve last dimension for reduction)
        X_cross_r = [x.reshape(-1, x.shape[-1]) for x, _ in self.cross_pt_data]
        X_tar_r = X.reshape(-1, X.shape[-1])
        # reduce dimensionality of cross-patient data
        X_cross_dr = [self.dim_red(n_components=self.n_comp).fit_transform(x)
                      for x in X_cross_r]

        # reduce dimensionality of target data, saving dim. red. object for 
        # test set
        tar_dr = self.dim_red(n_components=self.n_comp)
        X_tar_dr = tar_dr.fit_transform(X_tar_r)
        self.tar_dr = tar_dr

        # reshape back to 3D
        X_cross_dr = [x.reshape(cross_pt_trials[i], -1, x.shape[-1]) for i, x
                      in enumerate(X_cross_dr)]
        X_cross_dr = [x.reshape(x.shape[0], -1) for x in X_cross_dr]
        X_tar_dr = X_tar_dr.reshape(X.shape[0], -1)

        # concatenate cross-patient data
        X_dr = np.vstack([X_tar_dr] + X_cross_dr)
        y_dr = np.hstack([y] + [y for _, y in self.cross_pt_data])
        return X_dr, y_dr
    
    def preprocess_test(self, X):
        X_r = X.reshape(-1, X.shape[-1])
        X_dr =  self.tar_dr.transform(X_r)
        return X_dr.reshape(X.shape[0], -1)
