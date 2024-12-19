
import sys
import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

sys.path.insert(0, '..')

from aligned_decoding.alignment.JointPCA import JointPCA
from aligned_decoding.alignment.AlignCCA import AlignCCA
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
    parser.add_argument('-r', '--random_data', type=str, default='False',
                        required=False, help='Use random data for pooling')
    parser.add_argument('-j', '--joint_dim_red', type=str, default='False',
                        required=False, help='Learn joint PCA decomposition')
    parser.add_argument('-n', '--no_S23', type=str, default='False',
                        required=False, help='Exclude S23 from pooling')
    parser.add_argument('-c', '--cluster', type=str, default='True',
                        required=False,
                        help='Run on cluster (True) or local (False)')
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
        DATA_PATH = os.path.expanduser('~') + '/workspace/'
    else:
        DATA_PATH = '../data/'

    # patient and target params
    pt = inputs['patient']
    p_ind = inputs['p_ind']

    # experiment params
    pool_train = str2bool(inputs['pool_train'])
    tar_in_train = str2bool(inputs['tar_in_train'])
    cca_align = str2bool(inputs['cca_align'])
    random_data = str2bool(inputs['random_data'])
    joint_dim_red = str2bool(inputs['joint_dim_red'])
    no_S23 = str2bool(inputs['no_S23'])

    # constant params
    n_iter = 50
    n_folds = 5
    n_comp = 30

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

    # decoding run filename
    if inputs['filename'] != '':
        filename = inputs['filename']
    else:
        filename_suffix = inputs['suffix']
        if cluster:
            out_prefix = DATA_PATH + f'outputs/alignment_accs/{pt}/'
        else:
            out_prefix = f'../acc_data/joint_algn_accs/{pt}/'
        filename = out_prefix + (f"{pt}_{'p' if lab_type == 'phon'else 'a'}"
                                 f"{'All' if p_ind == -1 else p_ind}_"
                                 f"{filename_suffix}.pkl")

    print('==================================================================')
    print("Training model for patient %s." % pt)
    print("Saving outputs to %s." % (DATA_PATH + 'outputs/'))
    print('Pool train: %s' % pool_train)
    print('Target in train: %s' % tar_in_train)
    print('CCA align: %s' % cca_align)
    print('Random data: %s' % random_data)
    print('Joint Dim Red: %s' % joint_dim_red)
    print('No S23: %s' % no_S23)
    print('Alignment type: %s' % algn_type)
    print('Alignment grouping: %s' % algn_grouping)
    print('Label type: %s' % lab_type)
    print('Reduction method: %s' % red_method)
    print('Reduction components: %d' % n_comp)
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

    iter_accs = []
    wrong_trs_iter = []
    y_true_iter, y_pred_iter = [], []
    for _ in range(n_iter):
        y_true_all, y_pred_all = [], []
        wrong_trs_fold = []
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
        for train_idx, test_idx in cv.split(D_tar, lab_tar):
            X1, X2, X3 = D1, D2, D3
            y1, y2, y3 = lab1, lab2, lab3
            y1_full, y2_full, y3_full = lab1_full, lab2_full, lab3_full

            # split target data into train and test
            X_tar_train, X_tar_test = D_tar[train_idx], D_tar[test_idx]
            y_tar_train, y_tar_test = lab_tar[train_idx], lab_tar[test_idx]
            y_tar_full_train, y_tar_full_test = (lab_tar_full[train_idx],
                                                 lab_tar_full[test_idx])

            # learn joint PCA decomposition from full label sequences
            if joint_dim_red:
                jointPCA = JointPCA(n_components=n_comp)
                X1, X2, X3, X_tar_train = jointPCA.fit_transform(
                                                    [X1, X2, X3,
                                                     X_tar_train],
                                                    [y1_full, y2_full,
                                                     y3_full,
                                                     y_tar_full_train])
                # apply target transformation to test data
                X_tar_test = jointPCA.transform(X_tar_test, idx=3)
            else:  # apply separate dimensionality reduction to each dataset
                X1_p, X2_p, X3_p, X_tar_train_p, X_tar_test_p = (
                                        [X.reshape(-1, X.shape[-1]) for X in
                                         (X1, X2, X3, X_tar_train,
                                          X_tar_test)])
                X1_p, X2_p, X3_p = [dim_red(n_components=n_comp).fit_transform(
                                        X) for X in [X1_p, X2_p, X3_p]]
                tar_dr = dim_red(n_components=n_comp)
                X_tar_train_p = tar_dr.fit_transform(X_tar_train_p)
                X_tar_test_p = tar_dr.transform(X_tar_test_p)
                X1, X2, X3, X_tar_train, X_tar_test = [X.reshape(Xs.shape[0],
                                                                 -1, n_comp)
                                                       for (X, Xs) in
                                                       zip((X1_p, X2_p, X3_p,
                                                            X_tar_train_p,
                                                            X_tar_test_p),
                                                           (X1, X2, X3,
                                                            X_tar_train,
                                                            X_tar_test))]

            # align each pooled patient data to target data with CCA
            if cca_align:
                cca1 = AlignCCA(type=algn_grouping)
                cca2 = AlignCCA(type=algn_grouping)
                cca3 = AlignCCA(type=algn_grouping)
                cca1.fit(X_tar_train, X1, y_tar_full_train, y1_full)
                cca2.fit(X_tar_train, X2, y_tar_full_train, y2_full)
                cca3.fit(X_tar_train, X3, y_tar_full_train, y3_full)
                X1 = cca1.transform(X1)
                X2 = cca2.transform(X2)
                X3 = cca3.transform(X3)

            # reshape to trials x features
            X_tar_train = X_tar_train.reshape(X_tar_train.shape[0], -1)
            X_tar_test = X_tar_test.reshape(X_tar_test.shape[0], -1)
            X1 = X1.reshape(X1.shape[0], -1)
            X2 = X2.reshape(X2.shape[0], -1)
            X3 = X3.reshape(X3.shape[0], -1)

            if not pool_train:
                X_train, y_train = X_tar_train, y_tar_train
            else:
                if not tar_in_train:
                    X_train = np.concatenate((X1, X2, X3), axis=0)
                    y_train = np.concatenate((y1, y2, y3), axis=0)
                else:
                    if no_S23:
                        X_train = np.concatenate((X_tar_train, X1, X2), axis=0)
                        y_train = np.concatenate((y_tar_train, y1, y2), axis=0)
                    else:
                        X_train = np.concatenate((X_tar_train, X1, X2, X3),
                                                 axis=0)
                        y_train = np.concatenate((y_tar_train, y1, y2, y3),
                                                 axis=0)
                    
            X_test = X_tar_test
            y_test = y_tar_test

            # sc = MinMaxScaler()
            # X_train = sc.fit_transform(X_train)
            # X_test = sc.transform(X_test)

            clf = BaggingClassifier(estimator=SVC(kernel='linear'),
                                    n_estimators=10)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

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
                          'n_folds': n_folds, 'n_comp': n_comp,
                          'algn_type': algn_type,
                          'algn_grouping': algn_grouping,
                          'lab_type': lab_type, 'red_method': red_method,
                          'dim_red': dim_red}
    utils.save_pkl(out_data, filename)


if __name__ == '__main__':
    aligned_decoding()
