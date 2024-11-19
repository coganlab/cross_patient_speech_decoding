import sys
import os
import argparse
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     RandomizedSearchCV, train_test_split)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix
from skopt import BayesSearchCV

sys.path.insert(0, '..')

from alignment.JointPCA import JointPCA
from alignment.AlignCCA import AlignCCA
from alignment.AlignMCCA import AlignMCCA
from decomposition.DimRedReshape import DimRedReshape
from decomposition.NoCenterPCA import NoCenterPCA
from decoders.cross_pt_decoders import (crossPtDecoder_sepDimRed,
                                        crossPtDecoder_sepAlign,
                                        crossPtDecoder_jointDimRed,
                                        crossPtDecoder_mcca)
import alignment.alignment_utils as utils


def init_parser():
    parser = argparse.ArgumentParser(description='Cross-patient decoding with'
                                     'subsampling of pooeld data across patients.')
    parser.add_argument('-pt', '--patient', type=str, required=True,
                        help='Patient ID')
    parser.add_argument('-pi', '--p_ind', type=int, default=-1, required=False,
                        help='Sequence position index')
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


# def cmat_acc(y_true, y_pred):
#     cmat = confusion_matrix(y_true, y_pred)
#     acc_cmat = np.trace(cmat) / np.sum(cmat)
#     return acc_cmat

# def cmat_wrap(y_true_iter, y_pred_iter):
#     accs = []
#     for y_true, y_pred in zip(y_true_iter, y_pred_iter):
#         accs.append(cmat_acc(y_true, y_pred))
#     return np.array(accs)

    
def pooled_sampled_decoding():
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
    tar_in_train = str2bool(inputs['tar_in_train'])
    cca_align = str2bool(inputs['cca_align'])
    mcca_align = str2bool(inputs['MCCA_align'])
    joint_dim_red = str2bool(inputs['joint_dim_red'])
    do_cv = str2bool(inputs['cross_validate'])

    # constant params
    n_iter = 50
    n_folds = 20
    # n_iter = 2
    # n_folds = 2
    
    trial_step = 25

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
                # 'n_comp': (10, 50),
                'n_comp': (0.1, 0.95, 'uniform'),
                # 'n_comp': np.arange(0.1, 0.95, 0.05),
                'decoder__dimredreshape__n_components': (0.1, 0.95, 'uniform'),
                # 'decoder__dimredreshape__n_components': np.arange(0.1, 0.95, 0.05),
                # 'decoder__baggingclassifier__estimator__C': (1e-3, 1e5, 'log-uniform'),
                # 'decoder__baggingclassifier__estimator__gamma': (1e-4, 1e3, 'log-uniform'),
                # 'decoder__baggingclassifier__n_estimators': (10, 100),
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
                # 'n_comp': 30, # old CCA method
                'n_comp': 0.9,
                'decoder__dimredreshape__n_components': 0.8,
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
    print('Target in train: %s' % tar_in_train)
    print('CCA align: %s' % cca_align)
    print('MCCA align: %s' % mcca_align)
    print('Joint Dim Red: %s' % joint_dim_red)
    print('Alignment type: %s' % algn_type)
    print('Alignment grouping: %s' % algn_grouping)
    print('Label type: %s' % lab_type)
    print('Reduction method: %s' % red_method)
    # print('Reduction components: %d' % n_comp)
    print('Do nested CV: %s' % do_cv)
    print('Number of iterations: %d' % n_iter)
    print('Number of folds: %d' % n_folds)
    print('==================================================================')

    out_data = {}
    out_data['params'] = {'pt': pt, 'p_ind': p_ind,
                          'tar_in_train': tar_in_train, 'cca_align': cca_align,
                          'joint_dim_red': joint_dim_red, 'n_iter': n_iter,
                          'n_folds': n_folds,
                          'hyperparams': param_grid,
                          'algn_type': algn_type,
                          'algn_grouping': algn_grouping,
                          'lab_type': lab_type, 'red_method': red_method}


    # decoder = SVC(
    #     # kernel='linear',
    #     kernel='rbf',
    #     class_weight='balanced',
    #     )
    # clf = make_pipeline(
    #             DimRedReshape(dim_red, n_components=0.8),
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
                DimRedReshape(dim_red, n_components=0.8),
                decoder
                )

    # decoder = LinearDiscriminantAnalysis()
    # clf = make_pipeline(
    #             DimRedReshape(dim_red),
    #             decoder
    #             )

    # load data
    # data_filename = DATA_PATH + 'pt_decoding_data.pkl'
    # data_filename = DATA_PATH + 'pt_decoding_data_S22.pkl'
    # data_filename = DATA_PATH + 'pt_decoding_data_S39.pkl'
    # data_filename = DATA_PATH + 'pt_decoding_data_S58.pkl'
    data_filename = DATA_PATH + 'pt_decoding_data_S62.pkl'
    pt_data = utils.load_pkl(data_filename)
    tar_data, pre_data = utils.decoding_data_from_dict(pt_data, pt, p_ind,
                                                       lab_type=lab_type,
                                                       algn_type=algn_type)
    D_tar, lab_tar, lab_tar_full = tar_data

    # max_trs = max([x.shape[0] for x, _, _ in pre_data])
    max_trs = int(np.ceil(np.median([x.shape[0] for x, _, _ in pre_data]))) # only go to median amount of trials to avoid 
    # k_trials_per_pt = np.arange(1, max_trs + 1, trial_step)
    k_trials_per_pt = np.arange(5, max_trs + 1, trial_step)
    print(f'Max trials: {max_trs}')

    pool_samp_mat = np.full((len(k_trials_per_pt), n_iter), np.nan)
    trial_vec = np.full(len(k_trials_per_pt), np.nan)
    for trial_idx, k in enumerate(k_trials_per_pt):
        print(f'##### Sampling {k} trials per patient #####')
        # do n iterations of standard aligned cross-patient decoding with
        # subsampled data
        for i in range(n_iter):

            # sample k trials from each patient to make a subsampled cross-patient
            # dataset
            cross_pt_data = []
            for x, y, y_a in pre_data:
                if x.shape[0] < k:
                    # take all of the data if there are fewer than k trials
                    cross_pt_data.append((x, y, y_a))
                else:
                    # sample k trials
                    # print(x.shape)
                    samp_idx = np.random.choice(x.shape[0], k, replace=False)
                    cross_pt_data.append((x[samp_idx], y[samp_idx], y_a[samp_idx]))
                    # print(x[samp_idx].shape)

            # keep track of how many trials are sampled at each step for figure
            num_trials = np.sum([x.shape[0] for x, _, _ in cross_pt_data])
            trial_vec[trial_idx] = num_trials

            y_true_all, y_pred_all = [], []
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True)

            for j, (train_idx, test_idx) in enumerate(cv.split(D_tar, lab_tar)):
                print(f'Iteration {i+1}, Fold {j+1}')
                D_tar_train, D_tar_test = D_tar[train_idx], D_tar[test_idx]
                lab_tar_train, lab_tar_test = lab_tar[train_idx], lab_tar[test_idx]
                lab_tar_full_train, lab_tar_full_test = (lab_tar_full[train_idx],
                                                        lab_tar_full[test_idx])
                # print(lab_tar_train, np.unique(lab_tar_full_train))
                # for x, y, y_a in cross_pt_data:
                #     print(y_a, np.unique(y_a))
                
                if joint_dim_red:
                    model = crossPtDecoder_jointDimRed(cross_pt_data, clf,
                                                       JointPCA,
                                                       tar_in_train=tar_in_train)
                elif cca_align:
                    model = crossPtDecoder_sepAlign(cross_pt_data, clf,
                                                    AlignCCA, dim_red=dim_red,
                                                    tar_in_train=tar_in_train)
                elif mcca_align:
                    model = crossPtDecoder_mcca(cross_pt_data, clf, AlignMCCA,
                                                tar_in_train=tar_in_train)
                else:
                    model = crossPtDecoder_sepDimRed(cross_pt_data, clf,
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
            
                y_test = lab_tar_test
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)
        
            bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
            print(bal_acc)
            pool_samp_mat[trial_idx, i] = bal_acc

            # save after every iteration in case of unexpected interrupt
            out_data['acc_mat'] = pool_samp_mat
            out_data['trial_vec'] = trial_vec
            utils.save_pkl(out_data, filename)


if __name__ == '__main__':
    pooled_sampled_decoding()
    print('########## Done ###########')