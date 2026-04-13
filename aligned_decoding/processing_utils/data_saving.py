"""
Functions to handle saving of model data.
"""

import os
import pickle


def dict_from_lists(param_keys, param_vals):
    """Creates a dictionary by zipping parallel key and value lists.

    Args:
        param_keys (list): Keys for the dictionary.
        param_vals (list): Values corresponding to each key.

    Returns:
        dict: Dictionary mapping each key to its corresponding value.
    """
    return dict(zip(param_keys, param_vals))


def save_pkl_params(filename, param_dict, param_key='params'):
    """Saves a parameter dictionary into a pickle file.

    Loads existing data from the file (if any), adds or overwrites the
    parameter entry, and writes the result back.

    Args:
        filename (str): Path to the pickle file.
        param_dict (dict): Parameter dictionary to store.
        param_key (str, optional): Key under which to store the parameters.
            Defaults to ``'params'``.
    """
    data = load_pkl_accs(filename)
    data[param_key] = param_dict
    with open(filename, 'wb+') as f:
        pickle.dump(data, f, protocol=-1)


def append_pkl_accs(filename, acc, cmat, acc_key='val_acc', cmat_key='cmat'):
    """Appends accuracy and confusion matrix results to a pickle file.

    Creates the key entries if they do not already exist.

    Args:
        filename (str): Path to the pickle file.
        acc: Accuracy value to append.
        cmat: Confusion matrix to append.
        acc_key (str, optional): Key for the accuracy list.
            Defaults to ``'val_acc'``.
        cmat_key (str, optional): Key for the confusion matrix list.
            Defaults to ``'cmat'``.
    """
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
    """Loads data from a pickle file, returning an empty dict if not found.

    Args:
        filename (str): Path to the pickle file.

    Returns:
        dict: Contents of the pickle file, or an empty dictionary if the
            file does not exist.
    """
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}
