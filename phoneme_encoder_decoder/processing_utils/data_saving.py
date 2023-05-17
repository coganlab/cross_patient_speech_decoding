"""
Functions to handle saving of model data.
"""

import os
import pickle


def append_pkl_accs(filename, acc, cmat, acc_key='val_acc', cmat_key='cmat'):
    # load in previous data if it exists
    data = load_pkl_accs(filename)

    # append new data to loaded data list
    if acc_key not in data:
        data[acc_key] = []
        data[cmat_key] = []
    data[acc_key].append(acc)
    data[cmat_key].append(cmat)

    # save new data
    with open(filename, 'wb+') as f:
        pickle.dump(data, f, protocol=-1)


def load_pkl_accs(filename):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}
