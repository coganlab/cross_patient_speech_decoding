"""
Plotting functions for viewing trained seq2seq model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def extend_list_to_length(lst, length):
    """Extends a list to a specified length by repeating the last element.
    Raises error if list is already longer than specified length.

    Args:
        lst (list): List to be extended.
        length (int): Length to extend list to.

    Returns:
        list: Extended list.
    """
    if len(lst) > length:
        raise ValueError("List length is greater than specified length.")
    else:
        lst.extend([lst[-1]] * (length - len(lst)))
    return lst


def extend_history_lists(histories, epochs=100):
    """Extends model history lists to a specified length by repeating the last
    element. Use to ensure all folds have the same number of epochs when number
    of epochs can vary from Early Stopping.

    Args:
        histories (dict): Dictionary of model history lists.
        epochs (int): Length to extend lists to (i.e. max number of epochs).

    Returns:
        dict: Dictionary of extended model history lists in same format as
            input dictionary.
    """
    history_copy = histories.copy()

    # extend histories to specified length
    for key in histories.keys():
        for fold in range(len(histories[key])):
            ext_list = extend_list_to_length(histories[key][fold], epochs)
            history_copy[key][fold] = ext_list
    return history_copy


def remove_duplicate_cols(df):
    return df.loc[:, ~df.columns.duplicated()].copy()


def create_CV_history_df(cv_histories, epochs=100):
    extend_history_lists(cv_histories, epochs=epochs)

    df = pd.DataFrame()
    for key in cv_histories.keys():
        key_arr = np.array(cv_histories[key])
        if df.empty:  # create dataframe from first key
            df = pd.DataFrame(key_arr).melt()
        else:  # add to existing dataframe
            df = pd.concat([df, pd.DataFrame(key_arr).melt()], axis=1)
        # rename dummy column from .melt()
        df.rename(columns={'value': key}, inplace=True)
    # remove duplicate columns
    df = remove_duplicate_cols(df)
    # rename dummy column from .melt()
    df.rename(columns={'variable': 'epoch'}, inplace=True)

    return df


def plot_accuracy_loss(cv_histories, epochs=100, save_fig=False,
                       save_path="../../figures/loss_accuracy.png"):
    train_loss_df = create_CV_history_df(cv_histories, epochs=epochs)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=train_loss_df, x='epoch', y='loss', ax=ax1, color='blue',
                 label='Train')
    sns.lineplot(data=train_loss_df, x='epoch', y='val_loss', ax=ax1,
                 color='orange', label='Validation')
    sns.lineplot(data=train_loss_df, x='epoch', y='seq2seq_val_loss', ax=ax1,
                 color='red', label='Seq2seq Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('RNN Loss')
    ax1.legend()

    sns.lineplot(data=train_loss_df, x='epoch', y='accuracy', ax=ax2,
                 color='blue', label='Train')
    sns.lineplot(data=train_loss_df, x='epoch', y='val_accuracy', ax=ax2,
                 color='orange', label='Validation')
    sns.lineplot(data=train_loss_df, x='epoch', y='seq2seq_val_accuracy',
                 ax=ax2, color='red', label='Seq2seq Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('RNN Accuracy')
    ax2.legend()

    if save_fig:
        plt.savefig(save_path)

    plt.show()
