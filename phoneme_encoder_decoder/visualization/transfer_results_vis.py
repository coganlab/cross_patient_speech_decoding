"""
Plotting functions for viewing results of transfer learning experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from .plot_model_performance import create_CV_history_df


def combine_transfer_single(transfer_data, single_data):
    transfer_mean = np.mean(transfer_data, axis=-1)
    single_mean = np.mean(single_data, axis=-1)

    np.fill_diagonal(transfer_mean, single_mean)
    return transfer_mean


def transform_col_by_diag(data):
    diag = np.diag(data)
    return data - diag


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    ADAPTED FROM: https://matplotlib.org/stable/gallery/
                  images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    ADAPTED FROM: https://matplotlib.org/stable/gallery/
                  images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_transfer_loss_acc(t_hist, pre_epochs, conv_epochs, tar_epochs,
                           n_pre, pt_labels=None,
                           save_fig=False,
                           save_path="../../figures/loss_acc.png"):
    if pt_labels is None:
        pt_labels = [f"Pretrain {i+1}" for i in range(n_pre)]
        pt_labels.append("Target")

    total_epochs = n_pre * (pre_epochs + conv_epochs) + tar_epochs

    transfer_df = create_CV_history_df(t_hist, epochs=total_epochs)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=transfer_df, x='epoch', y='loss', ax=ax1, color='blue',
                 label='Train')
    sns.lineplot(data=transfer_df, x='epoch', y='val_loss', ax=ax1,
                 color='orange', label='Validation')
    sns.lineplot(data=transfer_df, x='epoch', y='seq2seq_val_loss', ax=ax1,
                 color='red', label='Seq2seq Validation')

    sns.lineplot(data=transfer_df, x='epoch', y='accuracy', ax=ax2,
                 color='blue', label='Train')
    sns.lineplot(data=transfer_df, x='epoch', y='val_accuracy', ax=ax2,
                 color='orange', label='Validation')
    sns.lineplot(data=transfer_df, x='epoch', y='seq2seq_val_accuracy',
                 ax=ax2, color='red', label='Seq2seq Validation')

    # pretraining annotations
    for i in range(n_pre + 1):
        annotate_transfer_stage((ax1, ax2), i, pt_labels, pre_epochs,
                                conv_epochs, tar_epochs)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    # ax1.set_title('RNN Loss', y=1.0, pad=30)
    ax1.legend()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    # ax2.set_title('RNN Accuracy', y=1.0, pad=30)
    ax2.legend()

    if save_fig:
        plt.savefig(save_path)

    plt.show()


def annotate_transfer_stage(axs, curr_pre_num, pt_labels, pre_epochs,
                            conv_epochs, tar_epochs, stage_color='black'):
    annot_y = 1.01
    for ax in axs:
        conv_x = 0
        if curr_pre_num != 0:  # no conv stage for first patient
            # conv annotation
            conv_x = curr_pre_num*(conv_epochs + pre_epochs)
            ax.axvline(x=conv_x, color=stage_color, linestyle='--')
            # x in data untis, y in axes fraction
            trans = ax.get_xaxis_transform()
            annot_conv_x = conv_x - conv_epochs/2
            ax.annotate(f'{pt_labels[curr_pre_num]}\nConv',
                        xy=(annot_conv_x, annot_y), xycoords=trans,
                        ha='center', fontsize=12)

        # full train annotation
        trans = ax.get_xaxis_transform()
        if curr_pre_num == len(pt_labels) - 1:  # target annotation
            ax.annotate(f'{pt_labels[curr_pre_num]}\nFull',
                        xy=((2*conv_x + tar_epochs)/2, annot_y),
                        xycoords=trans, ha='center', fontsize=12)
        else:
            pre_x = curr_pre_num*conv_epochs + (curr_pre_num+1)*pre_epochs
            ax.axvline(x=pre_x, color=stage_color, linestyle='--')
            ax.annotate(f'{pt_labels[curr_pre_num]}\nFull',
                        xy=((pre_x + conv_x)/2, annot_y), xycoords=trans,
                        ha='center', fontsize=12)
