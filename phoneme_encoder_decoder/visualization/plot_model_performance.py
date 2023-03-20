"""
Plotting functions for viewing trained seq2seq model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def remove_duplicate_cols(df):
    return df.loc[:, ~df.columns.duplicated()].copy()


def create_CV_history_df(cv_histories):
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


def plot_accuracy_loss(cv_histories):
    train_loss_df = create_CV_history_df(cv_histories)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=train_loss_df, x='epoch', y='loss', ax=ax1, color='blue',
                 label='Train')
    sns.lineplot(data=train_loss_df, x='epoch', y='val_loss', ax=ax1,
                 color='orange', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('RNN Loss')
    ax1.legend()

    sns.lineplot(data=train_loss_df, x='epoch', y='accuracy', ax=ax2,
                 color='blue', label='Train')
    sns.lineplot(data=train_loss_df, x='epoch', y='val_accuracy', ax=ax2,
                 color='orange', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('RNN Accuracy')
    ax2.legend()
    plt.show()
