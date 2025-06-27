
import sys
import os
import argparse
from copy import deepcopy
import numpy as np
import scipy.io as sio
from sklearn.model_selection import (StratifiedKFold, KFold, GridSearchCV,
                                     RandomizedSearchCV, train_test_split)
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from skopt import BayesSearchCV

sys.path.insert(0, '..')

from alignment.AlignCCA import AlignCCA
from decomposition.NoCenterPCA import NoCenterPCA
from decomposition.DimRedReshape import DimRedReshape
from decoders.cross_pt_decoders import (crossPtDecoder_sepDimRed,
                                        crossPtDecoder_sepAlign)
import alignment.alignment_utils as utils
from processing_utils.spatial_avg_subsampling import (spatial_avg_sig_channels,
                                                      spatial_avg_data)


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
    parser.add_argument('-cs', '--contact_size', type=int, required='True',
                        help='Contact size for spatial averaging"')
    parser.add_argument('-pp', '--pooled_patients', type=str, default='all',
                        required=False, help='Cross patient indices')
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
        OUT_PATH = '../data/outputs/'

    # patient and target params
    pt = inputs['patient']
    p_ind = inputs['p_ind']

    # experiment params
    pool_train = str2bool(inputs['pool_train'])
    tar_in_train = str2bool(inputs['tar_in_train'])
    cca_align = str2bool(inputs['cca_align'])
    contact_size = inputs['contact_size']
    do_cv = str2bool(inputs['cross_validate'])

    pooled_pts = []
    if inputs['pooled_patients'] != 'all':
        pooled_pts = inputs['pooled_patients'].split(',')

    # constant params
    # n_iter = 2
    # n_folds = 5
    n_iter = 50
    n_folds = 20

    ###### CV GRID ######
    if do_cv:
        param_grid = {
            # 'n_comp': (10, 50),
            'n_comp': (0.1, 0.95, 'uniform'),
            # 'n_comp': np.arange(0.1, 0.95, 0.05),
            'decoder__dimredreshape__n_components': (0.1, 0.95, 'uniform'),
            # 'decoder__dimredreshape__n_components': np.arange(0.1, 0.95, 0.05),
            # 'decoder__baggingclassifier__estimator__C': (1e-3, 1e5, 'log-uniform'),
            # 'decoder__baggingclassifier__estimator__gamma': (1e-4, 1e3, 'log-uniform'),
            # 'decoder__baggingclassifier__n_estimators': (10, 100),
        }
        param_grid_single = {
            # 'dimredreshape__n_components': (0.1, 0.95, 'uniform'),
            'dimredreshape__n_components': np.arange(0.1, 1, 0.1),
            'svc__C': (1e-3, 1e5, 'log-uniform'),
            'svc__gamma': (1e-4, 1e3, 'log-uniform'),
            # 'decoder__baggingclassifier__n_estimators': (10, 100),
        }
    else:
        param_grid = {
            # 'n_comp': 30, # old CCA method
            'n_comp': 0.9,
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

    # red_method = 'PCA (no centering)'
    # dim_red = NoCenterPCA

    # decoding run filename
    if inputs['filename'] != '':
        filename = inputs['filename']
    else:
        filename_suffix = inputs['suffix']
        filename_suffix = filename_suffix.replace('.', '-')
        if cluster:
            out_prefix = OUT_PATH + f'outputs/alignment_accs/{pt}/'
        else:
            out_prefix = OUT_PATH + f'grid_accs/{pt}/'
        filename = out_prefix + (f"{pt}_{'p' if lab_type == 'phon'else 'a'}"
                                 f"{'All' if p_ind == -1 else p_ind}_"
                                 f"{filename_suffix}.pkl")
        
    # load data
    data_filename = DATA_PATH + 'pt_savg_data_S62.pkl'
    
    all_pt_savg_data = utils.load_pkl(data_filename)
    pt_savg_data, pre_savg_data = utils.decoding_data_from_dict(
        all_pt_savg_data, pt, p_ind, lab_type=lab_type, algn_type=algn_type)
    pre_pts = all_pt_savg_data[pt]['pre_pts'] 
    cs_str = f'{contact_size}x{contact_size}'
    D_tar_subsamp = pt_savg_data[0][cs_str]
    lab_tar = pt_savg_data[1]
    lab_tar_full = pt_savg_data[2]

    if len(pooled_pts) > 0:
        cross_pts = pooled_pts
    else:
        cross_pts = pre_pts
    cross_pt_subsamp = []
    for p in cross_pts:
        D_cross = pre_savg_data[pre_pts.index(p)][0][cs_str]
        lab_cross = pre_savg_data[pre_pts.index(p)][1]
        lab_cross_full = pre_savg_data[pre_pts.index(p)][2]
        cross_pt_subsamp.append((D_cross, lab_cross, lab_cross_full))      

    print('==================================================================')
    print("Training model for patient %s." % pt)
    print("Saving outputs to %s." % (DATA_PATH + 'outputs/'))
    print('Pool train: %s' % pool_train)
    print('Target in train: %s' % tar_in_train)
    print('CCA align: %s' % cca_align)
    print('Subsample contact size: (%dx%d)' % (contact_size, contact_size))
    print('Alignment type: %s' % algn_type)
    print('Alignment grouping: %s' % algn_grouping)
    print('Label type: %s' % lab_type)
    print('Reduction method: %s' % red_method)
    # print('Reduction components: %d' % n_comp)
    print('Pooled patients: %s' % pooled_pts)
    print('Do nested CV: %s' % do_cv)
    print('Number of iteratiions: %d' % n_iter)
    print('Number of folds: %d' % n_folds)
    print('==================================================================')

    out_data = {}
    out_data['params'] = {'pt': pt, 'p_ind': p_ind, 'pool_train': pool_train,
                          'tar_in_train': tar_in_train, 'cca_align': cca_align,
                          'contact_size': contact_size,
                          'n_iter': n_iter, 'n_folds': n_folds,
                          'hyperparams': param_grid,
                          'algn_type': algn_type,
                          'algn_grouping': algn_grouping,
                          'lab_type': lab_type, 'red_method': red_method}

    # define classifier
    # decoder = SVC(
    #     # kernel='linear',
    #     kernel='rbf',
    #     class_weight='balanced',
    #     )
    # clf = make_pipeline(
    #             DimRedReshape(dim_red),
    #             BaggingClassifier(
    #                 estimator=decoder,
    #                 # n_estimators=10,
    #                 n_jobs=-1,
    #                 )
    #             )

    decoder = SVC(
        # kernel='linear',
        kernel='rbf',
        class_weight='balanced',
        )
    clf = make_pipeline(
                DimRedReshape(dim_red),
                decoder
                )

    # decoder = LinearDiscriminantAnalysis()
    # clf = make_pipeline(
    #             DimRedReshape(dim_red),
    #             decoder
    #             )

    iter_accs = []
    wrong_trs_iter = []
    y_true_iter, y_pred_iter = [], []
    # grid orientation is pre-calculated so this is just iteration over
    # data splitting
    for j in range(n_iter):
        y_true_all, y_pred_all = [], []
        wrong_trs_fold = []

        try:  # default to stratified split
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
             # list forces computation to check if stratified is viable
            splits = list(cv.split(D_tar_subsamp, lab_tar))
        except ValueError:  # if not enough samples in a class
            cv = KFold(n_splits=n_folds, shuffle=True)
            splits = list(cv.split(D_tar_subsamp))

        for i, (train_idx, test_idx) in enumerate(splits):
            print(f'Iteration {j+1}, Fold {i+1}')
            D_tar_train, D_tar_test = (D_tar_subsamp[train_idx],
                                       D_tar_subsamp[test_idx])
            lab_tar_train, lab_tar_test = lab_tar[train_idx], lab_tar[test_idx]
            lab_tar_full_train, lab_tar_full_test = (lab_tar_full[train_idx],
                                                     lab_tar_full[test_idx])
             

            if pool_train:
                # define alignment method
                if cca_align:
                    model = crossPtDecoder_sepAlign(cross_pt_subsamp, clf,
                                                    AlignCCA, dim_red=dim_red,
                                                    tar_in_train=tar_in_train)
                else:
                    model = crossPtDecoder_sepDimRed(cross_pt_subsamp, clf,
                                                     dim_red=dim_red,
                                                     tar_in_train=tar_in_train)
                # nested cross-validation
                if do_cv:
                    # search = GridSearchCV(model, param_grid, cv=cv,
                    #                       verbose=5, n_jobs=-1)
                    # search = RandomizedSearchCV(model, param_grid,
                    #                             n_iter=25, cv=cv, n_jobs=-1,
                    #                             verbose=5)

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
                    search = BayesSearchCV(clf, param_grid_single, cv=cv,
                                        verbose=5, n_jobs=-1, n_iter=25,
                                        n_points=5, refit=False)
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

        out_data['y_true'] = y_true_iter
        out_data['y_pred'] = y_pred_iter
        out_data['wrong_trs'] = wrong_trs_iter
        out_data['accs'] = iter_accs
        
        utils.save_pkl(out_data, filename)


if __name__ == '__main__':
    aligned_decoding()
    print('########## Done ###########')
