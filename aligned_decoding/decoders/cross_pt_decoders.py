""" Cross-Patient Decoder Classes

Author: Zac Spalding
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA


class crossPtDecoder(BaseEstimator):
    """Base cross-patient decoder using sklearn's BaseEstimator interface.

    Subclasses implement ``preprocess_train`` and ``preprocess_test`` to define
    how multi-patient data is pooled before fitting and predicting with the
    underlying decoder.

    Attributes:
        decoder: Sklearn-compatible estimator used for fitting and prediction.
    """

    def preprocess_train(self, X, y=None):
        """Preprocess training data by pooling cross-patient features.

        Args:
            X: Target patient feature array.
            y: Target patient labels.

        Returns:
            Tuple of (X_processed, y_processed) ready for decoder fitting.
        """
        pass

    def preprocess_test(self, X, y=None):
        """Preprocess test data for prediction.

        Args:
            X: Target patient test feature array.
            y: Unused. Present for API consistency.

        Returns:
            Preprocessed feature array.
        """
        pass

    def fit(self, X, y, **kwargs):
        """Preprocess training data and fit the underlying decoder.

        Args:
            X: Target patient feature array.
            y: Target patient labels.
            **kwargs: Additional keyword arguments passed to
                ``preprocess_train``.

        Returns:
            Fitted decoder instance.
        """
        X_p, y_p = self.preprocess_train(X, y, **kwargs)
        return self.decoder.fit(X_p, y_p)

    def predict(self, X):
        """Preprocess test data and generate predictions.

        Args:
            X: Target patient test feature array.

        Returns:
            Predicted labels from the underlying decoder.
        """
        X_p = self.preprocess_test(X)
        return self.decoder.predict(X_p)

    def score(self, X, y, **kwargs):
        """Preprocess test data and score predictions against true labels.

        Args:
            X: Target patient test feature array.
            y: True labels for scoring.
            **kwargs: Additional keyword arguments passed to
                ``decoder.score``.

        Returns:
            Score from the underlying decoder's scoring method.
        """
        X_p = self.preprocess_test(X)
        return self.decoder.score(X_p, y, **kwargs)


class crossPtDecoder_sepDimRed(crossPtDecoder):
    """ Cross-Patient Decoder with separate PCA for each patient. """

    def __init__(self, cross_pt_data, decoder, dim_red=PCA, n_comp=0.8,
                 tar_in_train=True):
        """Initialize separate-PCA cross-patient decoder.

        Args:
            cross_pt_data: List of (X, y, y_align) tuples for each
                cross-patient source.
            decoder: Sklearn-compatible estimator for classification or
                regression.
            dim_red: Dimensionality reduction class (e.g., PCA). Defaults to
                PCA.
            n_comp: Number of components or variance ratio for dimensionality
                reduction. Defaults to 0.8.
            tar_in_train: Whether to include target patient data in the pooled
                training set. Defaults to True.
        """
        self.cross_pt_data = cross_pt_data
        self.decoder = decoder
        self.dim_red = dim_red
        self.n_comp = n_comp
        self.tar_in_train = tar_in_train

    def preprocess_train(self, X, y, **kwargs):
        """Reduce each patient's features separately and pool for training.

        Fits independent dimensionality reduction models per patient, truncates
        to a common latent dimensionality, and concatenates the results.

        Args:
            X: Target patient feature array of shape
                (n_trials, n_timepoints, n_features).
            y: Target patient labels of shape (n_trials,).
            **kwargs: Unused.

        Returns:
            Tuple of (X_pool, y_pool) with pooled, reduced features and
            labels.
        """
        cross_pt_trials = [x.shape[0] for x, _, _ in self.cross_pt_data]
        # reshape features to be 2D (preserve last dimension for reduction)
        X_cross_r = [x.reshape(-1, x.shape[-1]) for x, _, _ in
                     self.cross_pt_data]
        X_tar_r = X.reshape(-1, X.shape[-1])
        # reduce dimensionality of cross-patient data
        X_cross_dr = [self.dim_red(n_components=self.n_comp).fit_transform(x)
                      for x in X_cross_r]

        # reduce dimensionality of target data, saving dim. red. object for
        # test set
        tar_dr = self.dim_red(n_components=self.n_comp)
        X_tar_dr = tar_dr.fit_transform(X_tar_r)
        self.tar_dr = tar_dr

        # use the same latent dimensionality for all patients
        lat_dims = [X_tar_dr.shape[-1]] + [x.shape[-1] for x in X_cross_dr]
        self.common_dim = min(lat_dims)
        X_tar_dr = X_tar_dr[:, :self.common_dim]
        X_cross_dr = [x[:, :self.common_dim] for x in X_cross_dr]

        # reshape for concatenation
        X_cross_dr = [x.reshape(cross_pt_trials[i], -1, x.shape[-1]) for i, x
                      in enumerate(X_cross_dr)]
        X_cross_dr = [x.reshape(x.shape[0], -1) for x in X_cross_dr]
        X_tar_dr = X_tar_dr.reshape(X.shape[0], -1)

        # concatenate cross-patient data
        if self.tar_in_train:
            X_pool = np.vstack([X_tar_dr] + X_cross_dr)
            y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        else:
            X_pool = np.vstack(X_cross_dr)
            y_pool = np.hstack([y for _, y, _ in self.cross_pt_data])
        return X_pool, y_pool

    def preprocess_test(self, X):
        """Apply the fitted target dimensionality reduction to test data.

        Args:
            X: Target patient test feature array of shape
                (n_trials, n_timepoints, n_features).

        Returns:
            Reduced feature array of shape (n_trials, n_timepoints *
            common_dim).
        """
        X_r = X.reshape(-1, X.shape[-1])
        X_dr = self.tar_dr.transform(X_r)
        X_dr = X_dr[:, :self.common_dim]
        return X_dr.reshape(X.shape[0], -1)


class crossPtDecoder_sepAlign(crossPtDecoder):
    """ Cross-Patient Decoder with CCA alignment of separate dimensionality
    reductions for different patients."""

    def __init__(self, cross_pt_data, decoder, aligner, dim_red=PCA,
                 n_comp=0.8, tar_in_train=True):
        """Initialize separate-align cross-patient decoder.

        Args:
            cross_pt_data: List of (X, y, y_align) tuples for each
                cross-patient source.
            decoder: Sklearn-compatible estimator for classification or
                regression.
            aligner: Alignment class constructor (e.g., CCA-based aligner)
                used to map each source patient into the target space.
            dim_red: Dimensionality reduction class. Defaults to PCA.
            n_comp: Number of components or variance ratio for dimensionality
                reduction. Defaults to 0.8.
            tar_in_train: Whether to include target patient data in the pooled
                training set. Defaults to True.
        """
        self.cross_pt_data = cross_pt_data
        self.decoder = decoder
        self.dim_red = dim_red
        self.n_comp = n_comp
        self.aligner = aligner
        self.tar_in_train = tar_in_train

    def preprocess_train(self, X, y, y_align=None):
        """Reduce dimensions separately, then CCA-align sources to target.

        Each patient is independently reduced via ``dim_red``, then each
        cross-patient source is aligned to the target patient using the
        provided aligner.

        Args:
            X: Target patient feature array of shape
                (n_trials, n_timepoints, n_features).
            y: Target patient labels of shape (n_trials,).
            y_align: Optional alignment labels. Falls back to ``y`` if None.

        Returns:
            Tuple of (X_pool, y_pool) with pooled, aligned features and
            labels.
        """
        cross_pt_trials = [x.shape[0] for x, _, _ in self.cross_pt_data]
        # reshape features to be 2D (preserve last dimension for reduction)
        X_cross_r = [x.reshape(-1, x.shape[-1]) for x, _, _ in
                     self.cross_pt_data]
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
        X_tar_dr = X_tar_dr.reshape(X.shape[0], -1, X_tar_dr.shape[-1])

        # option for separate alignment labels
        if y_align is None:
            y_align = y
        y_align_cross = [y_a for _, _, y_a in self.cross_pt_data]

        # align data to target patient
        self.algns = [self.aligner() for _ in range(len(self.cross_pt_data))]
        X_algn_dr = []
        for i, algn in enumerate(self.algns):
            algn.fit(X_tar_dr, X_cross_dr[i], y_align, y_align_cross[i])
            X_algn_dr.append(algn.transform(X_cross_dr[i]))

        X_algn_dr = [x.reshape(x.shape[0], -1) for x in X_algn_dr]
        X_tar_dr = X_tar_dr.reshape(X_tar_dr.shape[0], -1)

        # concatenate cross-patient data
        if self.tar_in_train:
            X_pool = np.vstack([X_tar_dr] + X_algn_dr)
            y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        else:
            X_pool = np.vstack(X_algn_dr)
            y_pool = np.hstack([y for _, y, _ in self.cross_pt_data])
        return X_pool, y_pool

    def preprocess_test(self, X):
        """Apply the fitted target dimensionality reduction to test data.

        Args:
            X: Target patient test feature array of shape
                (n_trials, n_timepoints, n_features).

        Returns:
            Reduced feature array of shape (n_trials, n_timepoints *
            n_components).
        """
        X_r = X.reshape(-1, X.shape[-1])
        X_dr = self.tar_dr.transform(X_r)
        return X_dr.reshape(X.shape[0], -1)
    

class crossPtDecoder_jointDimRed(crossPtDecoder):
    """ Cross-Patient Decoder with joint dimensionality reduction to align and
    pool patients."""

    def __init__(self, cross_pt_data, decoder, joint_dr_method, n_comp=0.8,
                 tar_in_train=True):
        """Initialize joint-dimensionality-reduction cross-patient decoder.

        Args:
            cross_pt_data: List of (X, y, y_align) tuples for each
                cross-patient source.
            decoder: Sklearn-compatible estimator for classification or
                regression.
            joint_dr_method: Joint dimensionality reduction class that accepts
                a list of patient arrays and alignment labels.
            n_comp: Number of components or variance ratio for the joint
                reduction. Defaults to 0.8.
            tar_in_train: Whether to include target patient data in the pooled
                training set. Defaults to True.
        """
        self.cross_pt_data = cross_pt_data
        self.decoder = decoder
        self.joint_dr_method = joint_dr_method
        self.n_comp = n_comp
        self.tar_in_train = tar_in_train

    def preprocess_train(self, X, y, y_align=None):
        """Jointly reduce all patients' features and pool for training.

        Args:
            X: Target patient feature array.
            y: Target patient labels.
            y_align: Optional alignment labels. Falls back to ``y`` if None.

        Returns:
            Tuple of (X_pool, y_pool) with pooled, jointly reduced features
            and labels.
        """
        # option for separate alignment labels
        if y_align is None:
            y_align = y
        y_align_cross = [y_a for _, _, y_a in self.cross_pt_data]

        # extract features from cross pt data
        X_cross = [x for x, _, _ in self.cross_pt_data]

        # joint dimensionality reduction
        self.joint_dr = self.joint_dr_method(n_components=self.n_comp)
        X_joint_dr = self.joint_dr.fit_transform(
                                                [X] + X_cross,
                                                [y_align] + y_align_cross)
        X_tar_dr, X_algn_dr = X_joint_dr[0], X_joint_dr[1:]
        
        # reshape to trialx x features
        X_algn_dr = [x.reshape(x.shape[0], -1) for x in X_algn_dr]
        X_tar_dr = X_tar_dr.reshape(X_tar_dr.shape[0], -1)
        
        # concatenate cross-patient data
        if self.tar_in_train:
            X_pool = np.vstack([X_tar_dr] + X_algn_dr)
            y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        else:
            X_pool = np.vstack(X_algn_dr)
            y_pool = np.hstack([y for _, y, _ in self.cross_pt_data])
        return X_pool, y_pool

    def preprocess_test(self, X):
        """Transform test data using the fitted joint dimensionality reduction.

        Args:
            X: Target patient test feature array.

        Returns:
            Reduced feature array of shape (n_trials, flattened_features).
        """
        X_dr = self.joint_dr.transform(X, idx=0)
        return X_dr.reshape(X.shape[0], -1)


class crossPtDecoder_mcca(crossPtDecoder):
    """ Cross-patient Decoder with MCCA to align and pool patients. """

    def __init__(self, cross_pt_data, decoder, aligner, n_comp=10, regs=0.5,
                 pca_var=1, tar_in_train=True):
        """Initialize MCCA-based cross-patient decoder.

        Args:
            cross_pt_data: List of (X, y, y_align) tuples for each
                cross-patient source.
            decoder: Sklearn-compatible estimator for classification or
                regression.
            aligner: MCCA alignment class constructor.
            n_comp: Number of MCCA components. Defaults to 10.
            regs: Regularization parameter for MCCA. Defaults to 0.5.
            pca_var: Variance ratio for the internal PCA pre-reduction step.
                Defaults to 1 (no reduction).
            tar_in_train: Whether to include target patient data in the pooled
                training set. Defaults to True.
        """
        self.cross_pt_data = cross_pt_data
        self.decoder = decoder
        self.aligner = aligner
        self.n_comp = n_comp
        self.regs = regs
        self.pca_var = pca_var
        self.tar_in_train = tar_in_train

    def preprocess_train(self, X, y, y_align=None):
        """Align all patients via MCCA and pool for training.

        Args:
            X: Target patient feature array.
            y: Target patient labels.
            y_align: Optional alignment labels. Falls back to ``y`` if None.

        Returns:
            Tuple of (X_pool, y_pool) with pooled, MCCA-aligned features and
            labels.
        """
        # option for separate alignment labels
        if y_align is None:
            y_align = y
        y_align_cross = [y_a for _, _, y_a in self.cross_pt_data]

        # extract features from cross pt data
        X_cross = [x for x, _, _ in self.cross_pt_data]

        # joint dimensionality reduction
        self.aligner = self.aligner(n_components=self.n_comp, regs=self.regs,
                                    pca_var=self.pca_var)
        X_mcca = self.aligner.fit_transform([X] + X_cross,
                                            [y_align] + y_align_cross)
        X_tar_dr, X_algn_dr = X_mcca[0], X_mcca[1:]
        
        # reshape to trialx x features
        X_algn_dr = [x.reshape(x.shape[0], -1) for x in X_algn_dr]
        X_tar_dr = X_tar_dr.reshape(X_tar_dr.shape[0], -1)
        
        # concatenate cross-patient data
        if self.tar_in_train:
            X_pool = np.vstack([X_tar_dr] + X_algn_dr)
            y_pool = np.hstack([y] + [y for _, y, _ in self.cross_pt_data])
        else:
            X_pool = np.vstack(X_algn_dr)
            y_pool = np.hstack([y for _, y, _ in self.cross_pt_data])
        return X_pool, y_pool

    def preprocess_test(self, X):
        """Transform test data using the fitted MCCA alignment.

        Args:
            X: Target patient test feature array.

        Returns:
            MCCA-aligned feature array of shape (n_trials, flattened_features).
        """
        X_mcca = self.aligner.transform(X, idx=0)
        return X_mcca.reshape(X.shape[0], -1)
