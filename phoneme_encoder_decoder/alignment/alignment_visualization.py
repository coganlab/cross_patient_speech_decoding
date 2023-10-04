""" Functions to visualize cross-patient alignment results.

Author: Zac Spalding
Cogan & Viventi Labs, Duke University
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_1D_lat_dyn(t, data, labels, label_names, pt_list, pc_ind=0, n_cols=2,
                    title='1D Latent Dynamics', figsize=(12, 10), reorder=None,
                    same_axes=True):
    n_plots = len(data)
    f, axs = plt.subplots(nrows=int(np.ceil(n_plots/n_cols)), ncols=n_cols,
                          figsize=figsize)
    ylims = []
    for i, ax in enumerate(axs.flat):
        if reorder is not None:
            curr_data = data[reorder[i]]
        else:
            curr_data = data[i]
        curr_lab = labels[i]
        for j, _ in enumerate(np.unique(curr_lab)):
            j_locs = np.where(curr_lab == j+1)[0]
            to_plot = np.mean(curr_data[j_locs, :, pc_ind], axis=0)
            ax.plot(t, to_plot, label=label_names[j], linewidth=3)
        ylims.append(ax.get_ylim())
        ax.set_xlabel('Time Relative to Response Onset (s)', weight='bold')
        ax.set_ylabel(f'PC{pc_ind+1}', weight='bold', rotation=0, labelpad=20)
        ax.set_title(f'{pt_list[i]}')
    if same_axes:
        for ax in f.axes:
            ylims = np.array(ylims)
            min_ylim = np.min(ylims[:, 0])
            max_ylim = np.max(ylims[:, 1])
            plt.setp(ax, ylim=(min_ylim, max_ylim))
    # plt.legend(bbox_to_anchor=(1.35, 1), loc="center right")
    handles, labels = ax.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center',
                  ncol=min(10, len(label_names)))
    plt.suptitle(title)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    return f


def plot_2D_lat_dyn(data, labels, label_names, pt_list, n_cols=2,
                    title='2D Latent Dynamics', figsize=(12, 10), reorder=None,
                    same_axes=True):
    n_plots = len(data)
    f, axs = plt.subplots(nrows=int(np.ceil(n_plots/n_cols)), ncols=n_cols,
                          figsize=figsize)
    xlims = []
    ylims = []
    for i, ax in enumerate(axs.flat):
        if reorder is not None:
            curr_data = data[reorder[i]]
        else:
            curr_data = data[i]
        curr_lab = labels[i]
        for j, _ in enumerate(np.unique(curr_lab)):
            j_locs = np.where(curr_lab == j+1)[0]
            to_plot_x = np.mean(curr_data[j_locs, :, 0], axis=0)
            to_plot_y = np.mean(curr_data[j_locs, :, 1], axis=0)
            ax.plot(to_plot_x, to_plot_y, label=label_names[j], linewidth=3)
            ax.scatter(to_plot_x[0], to_plot_y[0], s=50)
        xlims.append(ax.get_xlim())
        ylims.append(ax.get_ylim())
        ax.set_xlabel('PC 1', weight='bold')
        ax.set_ylabel('PC 2', weight='bold')
        ax.set_title(f'{pt_list[i]}')
    if same_axes:
        for ax in f.axes:
            xlims = np.array(xlims)
            min_xlim = np.min(xlims[:, 0])
            max_xlim = np.max(xlims[:, 1])
            ylims = np.array(ylims)
            min_ylim = np.min(ylims[:, 0])
            max_ylim = np.max(ylims[:, 1])
            plt.setp(ax, xlim=(min_xlim, max_xlim), ylim=(min_ylim, max_ylim))
    # plt.legend(bbox_to_anchor=(1.35, 1), loc="center right")
    handles, labels = ax.get_legend_handles_labels()
    # plt.figlegend(handles, labels, loc='lower center',
    #               ncol=min(10, len(label_names)))
    plt.suptitle(title)
    f.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    return f


def plot_3D_lat_dyn(data, labels, label_names, pt_list,
                    title='3D Latent Dynamics', figsize=(12, 12), alpha=0.6,
                    reorder=None, same_axes=True):
    f = plt.figure(figsize=figsize)
    sp = [221, 222, 223, 224]
    xlims = []
    ylims = []
    zlims = []
    for i, sp_type in enumerate(sp):
        ax = f.add_subplot(sp_type, projection='3d')
        if reorder is not None:
            curr_data = data[reorder[i]]
        else:
            curr_data = data[i]
        curr_lab = labels[i]
        for j, _ in enumerate(np.unique(curr_lab)):
            ax.plot(curr_data[j, :, 0], curr_data[j, :, 1], curr_data[j, :, 2],
                    label=label_names[j], linewidth=3, alpha=alpha)
            ax.scatter(curr_data[j, 0, 0], curr_data[j, 0, 1],
                       curr_data[j, 0, 2], s=50)
            # ax.scatter(curr_lat_dyn[j][-1,0], curr_lat_dyn[j][-1,1],
            #            curr_lat_dyn[j][-1,2], s=50, marker='>')
        xlims.append(ax.get_xlim())
        ylims.append(ax.get_ylim())
        zlims.append(ax.get_zlim())
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        ax.set_title(f'{pt_list[i]}')
    if same_axes:
        for ax in f.axes:
            xlims = np.array(xlims)
            ylims = np.array(ylims)
            zlims = np.array(zlims)
            min_xlim, max_xlim = np.min(xlims[:, 0]), np.max(xlims[:, 1])
            min_ylim, max_ylim = np.min(ylims[:, 0]), np.max(ylims[:, 1])
            min_zlim, max_zlim = np.min(zlims[:, 0]), np.max(zlims[:, 1])
            plt.setp(ax, xlim=(min_xlim, max_xlim), ylim=(min_ylim, max_ylim),
                     zlim=(min_zlim, max_zlim))

    plt.legend(bbox_to_anchor=(1.4, 1), loc="center right")
    plt.suptitle(title)
    # plt.show()

    return f


def arrange_subplots(n):

    while is_prime(n) and n > 4:
        n += 1

    pass


def is_prime(n):
    """Checks if a number is prime.

    Args:
        n (int): Number to check.

    Returns:
        bool: True if n is prime, False otherwise.
    """
    if n == 2:
        return True
    if n % 2 == 0 or n <= 1:
        return False
    sqr = int(np.sqrt(n)) + 1
    for divisor in range(3, sqr, 2):
        if n % divisor == 0:
            return False
    return True
