
import sys
import os
import argparse
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     train_test_split)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from skopt import BayesSearchCV

sys.path.insert(0, '..')

from aligned_decoding.alignment.JointPCA import JointPCA
from aligned_decoding.alignment.AlignCCA import AlignCCA
from aligned_decoding.alignment.AlignMCCA import AlignMCCA
from alignment.cross_pt_decoders import (crossPtDecoder_sepDimRed,
                                         crossPtDecoder_sepAlign,
                                         crossPtDecoder_jointDimRed,
                                         crossPtDecoder_mcca)
import alignment.utils as utils


def init_parser():
    parser = argparse.ArgumentParser(description='Aligned decoding SVM on DCC')
    parser.add_argument('-pt', '--patient', type=str, required=True,
                        help='Patient ID')
    parser.add_argument('-pi', '--p_ind', type=int, default=-1, required=False,
                        help='Sequence position index')
    parser.add_argument('-po', '--pool_train', type=str, default='False',
                        required=False, help='Pool patient data for training')
    parser.add_argument('-t', '--tar_in_train', type=str, default='True',
                        required=False, help='Include target data in training')
    parser.add_argument('-a', '--cca_align', type=str, default='False',
                        required=False,
                        help='Align pooled data to target data with CCA')
    parser.add_argument('-m', '--MCCA_align', type=str, default='False',
                        required=False,
                        help='Align pooled data to shared space with MCCA')
    parser.add_argument('-j', '--joint_dim_red', type=str, default='False',
                        required=False, help='Learn joint PCA decomposition')
    parser.add_argument('-r', '--random_data', type=str, default='False',
                        required=False, help='Use random data for pooling')
    parser.add_argument('-n', '--no_S23', type=str, default='False',
                        required=False, help='Exclude S23 from pooling')
    parser.add_argument('-tss', '--trial_subsample', type=float, default=1.0,
                        required=False, help='Fraction of trials to subsample'
                        'from training data')
    parser.add_argument('-c', '--cluster', type=str, default='True',
                        required=False,
                        help='Run on cluster (True) or local (False)')
    parser.add_argument('-cv', '--cross_validate', type=str, default='False',
                        required=False, help='Perform nested cross-validation')
    parser.add_argument('-f', '--filename', type=str, default='',
                        required=False,
                        help='Output filename for performance saving')
    parser.add_argument('-s', '--suffix', type=str, default='',
                        required=False, help='Filename suffix if full filename'
                        'not specified')
    return parser


def str2bool(s):
    return s.lower() == 'true'


class DimRedReshape(BaseEstimator):

    def __init__(self, dim_red, n_components=10):
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


def aligned_decoding():
    parser = init_parser()
    args = parser.parse_args()

    inputs = {}
    for key, val in vars(args).items():
        inputs[key] = val

    cluster = str2bool(inputs['cluster'])
    if cluster:
        DATA_PATH = os.path.expanduser('~') + '/data/'
        OUT_PATH = os.path.expanduser('~') + '/workspace/'
    else:
        DATA_PATH = '../data/'
        OUT_PATH = '../acc_data/'

    # patient and target params
    pt = inputs['patient']
    p_ind = inputs['p_ind']

    # experiment params
    pool_train = str2bool(inputs['pool_train'])
    tar_in_train = str2bool(inputs['tar_in_train'])
    cca_align = str2bool(inputs['cca_align'])
    mcca_align = str2bool(inputs['MCCA_align'])
    joint_dim_red = str2bool(inputs['joint_dim_red'])
    random_data = str2bool(inputs['random_data'])
    no_S23 = str2bool(inputs['no_S23'])
    tr_subsamp_r = inputs['trial_subsample']
    do_cv = str2bool(inputs['cross_validate'])

    # constant params
    n_iter = 50
    n_folds = 5

    ###### CV GRID ######
    if do_cv:
        if mcca_align:
            param_grid = {
                'n_comp': (10, 50),
                # 'regs': (1e-3, 1, 'log-uniform'),
                'pca_var': (0.1, 0.95, 'uniform'),
                'decoder__dimredreshape__n_components': (0.1, 0.95, 'uniform'),
                'decoder__baggingclassifier__estimator__C': (1e-3, 1e5, 'log-uniform'),
                'decoder__baggingclassifier__estimator__gamma': (1e-4, 1e3, 'log-uniform'),
                'decoder__baggingclassifier__n_estimators': (10, 100),
            }
        else:
            param_grid = {
                'n_comp': (10, 50),
                'decoder__dimredreshape__n_components': (0.1, 0.95, 'uniform'),
                'decoder__baggingclassifier__estimator__C': (1e-3, 1e5, 'log-uniform'),
                'decoder__baggingclassifier__estimator__gamma': (1e-4, 1e3, 'log-uniform'),
                'decoder__baggingclassifier__n_estimators': (10, 100),
            }
        param_grid_single = {
            'decoder__dimredreshape__n_components': (0.1, 0.95, 'uniform'),
            'decoder__baggingclassifier__estimator__C': (1e-3, 1e5, 'log-uniform'),
            'decoder__baggingclassifier__estimator__gamma': (1e-4, 1e3, 'log-uniform'),
            'decoder__baggingclassifier__n_estimators': (10, 100),
        }
    else:
        if mcca_align:
            param_grid = {
                'n_comp': 30,
                'regs': 0.5,
                'pca_var': 0.8,
                'decoder__dimredreshape__n_components': 0.8,
            }
        else:
            param_grid = {
                'n_comp': 30,
                'decoder__dimredreshape__n_components': 0.8,
            }
        param_grid_single = {
            'dimredreshape__n_components': 0.8,
        }
    ###################

    # alignment label type
    # algn_type = 'artic_seq'
    algn_type = 'phon_seq'
    algn_grouping = 'class'

    # decoding label type
    lab_type = 'phon'
    # lab_type = 'artic'

    # dimensionality reduction type
    red_method = 'PCA'
    dim_red = PCA

    # check alignment type
    if sum([cca_align, mcca_align, joint_dim_red]) > 1:
        print('Multiple alignment types are true. Using joint_dim_red '
              'to perform alignment.')
        cca_align = False
        mcca_align = False

    # decoding run filename
    if inputs['filename'] != '':
        filename = inputs['filename']
    else:
        filename_suffix = inputs['suffix']
        if cluster:
            out_prefix = OUT_PATH + f'outputs/alignment_accs/{pt}/'
        else:
            out_prefix = OUT_PATH + f'ncv_accs/{pt}/'
        filename = out_prefix + (f"{pt}_{'p' if lab_type == 'phon'else 'a'}"
                                 f"{'All' if p_ind == -1 else p_ind}_"
                                 f"{filename_suffix}.pkl")

    print('==================================================================')
    print("Training model for patient %s." % pt)
    print("Saving outputs to %s." % (DATA_PATH + 'outputs/'))
    print('Pool train: %s' % pool_train)
    print('Target in train: %s' % tar_in_train)
    print('CCA align: %s' % cca_align)
    print('MCCA align: %s' % mcca_align)
    print('Joint Dim Red: %s' % joint_dim_red)
    print('Random data: %s' % random_data)
    print('No S23: %s' % no_S23)
    print('Alignment type: %s' % algn_type)
    print('Alignment grouping: %s' % algn_grouping)
    print('Label type: %s' % lab_type)
    print('Reduction method: %s' % red_method)
    # print('Reduction components: %d' % n_comp)
    print('Trial subsampling ratio: %f' % inputs['trial_subsample'])
    print('Do nested CV: %s' % do_cv)
    print('Number of iterations: %d' % n_iter)
    print('Number of folds: %d' % n_folds)
    print('==================================================================')

    # load data
    pt_data = utils.load_pkl(DATA_PATH + 'pt_decoding_data.pkl')
    tar_data, pre_data = utils.decoding_data_from_dict(pt_data, pt, p_ind,
                                                       lab_type=lab_type,
                                                       algn_type=algn_type)
    D_tar, lab_tar, lab_tar_full = tar_data
    D1, lab1, lab1_full = pre_data[0]
    D2, lab2, lab2_full = pre_data[1]
    D3, lab3, lab3_full = pre_data[2]

    if random_data:
        D1 = np.random.rand(*D1.shape)
        D2 = np.random.rand(*D2.shape)
        D3 = np.random.rand(*D3.shape)

    # define classifier
    decoder = SVC(
        # kernel='linear',
        kernel='rbf',
        class_weight='balanced',
        )
    clf = make_pipeline(
                DimRedReshape(dim_red, n_components=0.8),
                BaggingClassifier(
                    estimator=decoder,
                    # n_estimators=10,
                    n_jobs=-1,
                    )
                )

    iter_accs = []
    wrong_trs_iter = []
    y_true_iter, y_pred_iter = [], []
    for _ in range(n_iter):
        y_true_all, y_pred_all = [], []
        wrong_trs_fold = []
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True)

        for i, (train_idx, test_idx) in enumerate(cv.split(D_tar, lab_tar)):
            print(f'Fold {i+1}')
            D_tar_train, D_tar_test = D_tar[train_idx], D_tar[test_idx]
            lab_tar_train, lab_tar_test = lab_tar[train_idx], lab_tar[test_idx]
            lab_tar_full_train, lab_tar_full_test = (lab_tar_full[train_idx],
                                                     lab_tar_full[test_idx])
            
            if tr_subsamp_r < 1:
                # train test split to subsample training data with stratification
                # (could also use np.random.choice for non-stratified subsampling)
                D_tar_train, _, lab_tar_train, _, lab_tar_full_train, _ = (
                                    train_test_split(D_tar_train,
                                                    lab_tar_train,
                                                    lab_tar_full_train,
                                                    train_size=tr_subsamp_r,
                                                    stratify=lab_tar_train,
                                                    shuffle=True))    

            if pool_train:
                # define data to pool across patients
                # (may exclude hihh noise S23)
                if no_S23:
                    cross_pt_data = [(D1, lab1, lab1_full),
                                     (D2, lab2, lab2_full)]
                else:
                    cross_pt_data = [(D1, lab1, lab1_full),
                                     (D2, lab2, lab2_full),
                                     (D3, lab3, lab3_full)]
                # define alignment method
                if joint_dim_red:
                    model = crossPtDecoder_jointDimRed(cross_pt_data, clf,
                                                       JointPCA)
                elif cca_align:
                    model = crossPtDecoder_sepAlign(cross_pt_data, clf,
                                                    AlignCCA, dim_red=dim_red)
                elif mcca_align:
                    model = crossPtDecoder_mcca(cross_pt_data, clf, AlignMCCA)
                else:
                    model = crossPtDecoder_sepDimRed(cross_pt_data, clf,
                                                     dim_red=dim_red)
                # nested cross-validation
                if do_cv:
                    # search = GridSearchCV(model, param_grid, cv=cv,
                    #                       verbose=5, n_jobs=-1)
                    # search = RandomizedSearchCV(model, param_grid,
                    #                             n_iter=5, cv=cv, n_jobs=-1,
                    #                             verbose=1)

                    # need to call fit with the extra kwarg, so set refit to
                    # False and call fit manually after finding params
                    search = BayesSearchCV(model, param_grid, n_iter=25, cv=cv,
                                          verbose=5, n_jobs=-1, n_points=5,
                                          refit=False)
                    search.fit(D_tar_train, lab_tar_train,
                               y_align=lab_tar_full_train)
                    print(f'Best Params: {search.best_params_},'
                          f'Best Score: {search.best_score_}')
                    best_params = search.best_params_
                else:
                    # if not doing CV
                    best_params = param_grid

                # manually fit model with best params (see note in CV section)
                model.set_params(**best_params)
                model.fit(D_tar_train, lab_tar_train,
                        y_align=lab_tar_full_train)
                y_pred = model.predict(D_tar_test)
            else:
                # nested cross-validation
                if do_cv:
                    # model = Pipeline([('dim_red', DimRedReshape(dim_red)),
                    #                   ('decoder', clf)])
                    # search = GridSearchCV(clf, param_grid_single, cv=cv,
                    #                       verbose=5, n_jobs=-1)
                    search = BayesSearchCV(clf, param_grid_single, cv=3,
                                        verbose=5, n_jobs=-1, n_iter=1,
                                        refit=False)
                    search.fit(D_tar_train, lab_tar_train)
                    print(f'Best Params: {search.best_params_},'
                        f'Best Score: {search.best_score_}')
                    best_params = search.best_params_
                else:
                    # if not doing CV
                    best_params = param_grid_single

                # manually fit model with best params (see note in CV section)
                clf.set_params(**best_params)
                clf.fit(D_tar_train, lab_tar_train)
                y_pred = clf.predict(D_tar_test)

            y_test = lab_tar_test
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            wrong_idxs = np.where(y_test != y_pred)[0]
            wrong_trs_fold.extend(test_idx[wrong_idxs])

        y_true_iter.append(y_true_all)
        y_pred_iter.append(y_pred_all)
        wrong_trs_iter.append(wrong_trs_fold)
        bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
        print(bal_acc)
        iter_accs.append(bal_acc)

    out_data = {}
    out_data['y_true'] = y_true_iter
    out_data['y_pred'] = y_pred_iter
    out_data['wrong_trs'] = wrong_trs_iter
    out_data['accs'] = iter_accs
    out_data['params'] = {'pt': pt, 'p_ind': p_ind, 'pool_train': pool_train,
                          'tar_in_train': tar_in_train, 'cca_align': cca_align,
                          'joint_dim_red': joint_dim_red, 'n_iter': n_iter,
                          'n_folds': n_folds,
                          'hyperparams': param_grid,
                          'algn_type': algn_type,
                          'algn_grouping': algn_grouping,
                          'lab_type': lab_type, 'red_method': red_method,
                          'dim_red': dim_red}
    utils.save_pkl(out_data, filename)


if __name__ == '__main__':
    aligned_decoding()
