"""Spatial averaging of electrode channels over square contact regions."""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from .grid_subsampling import grid_susbsample_idxs
# from processing_utils.grid_subsampling import grid_susbsample_idxs


def spatial_avg_sig_channels(pt, contactSize, dataPath, useSig=False):
    """Computes spatial averaging indices, optionally filtered by significance.

    Loads a channel map (and optionally significant-channel data) for the
    given subject, then returns the grid indices for each square spatial
    averaging region.

    Args:
        pt (str): Subject identifier string.
        contactSize (int): Edge length of the square averaging region.
        dataPath (str): Root directory containing subject data folders.
        useSig (bool, optional): If True, only return index groups that
            contain at least one significant channel. Defaults to False.

    Returns:
        list: List of ndarrays, each containing 2-D grid indices for one
            spatial averaging region.
    """
    # load in channel map
    chanMap = sio.loadmat(f'{dataPath}/{pt}/{pt}_channelMap.mat')['chanMap']

    # trim full nan edges if necessary
    if chanMap.shape[0] == 24: 
        chanMap = chanMap[1:-1,:]
    elif chanMap.shape[1] == 24:
        chanMap = chanMap[:,1:-1]

    if useSig:
        # load in significant channel data
        sigChan = np.squeeze(
            sio.loadmat(f'{dataPath}/{pt}/{pt}_sigChannel.mat')['sigChannel'])

    # get spatial average indices
    gridSize = chanMap.shape
    avgIdxs = spatial_avg_idxs(gridSize, contactSize)

    if useSig:
        sigIdxList = []
        for idxs in avgIdxs:
            elecPt = chanMap[idxs[:, 0], idxs[:, 1]]
            
            # don't consider sets with over half nans
            if np.sum(np.isnan(elecPt)) >= len(elecPt) / 2:
                continue
            else: # remove nans
                goodIdxs = ~np.isnan(elecPt)
                elecPt = elecPt[goodIdxs].astype(int)
                idxsPt = idxs[goodIdxs]

            # see if any significant channels are in the subsampled set
            commonElec = np.intersect1d(sigChan, elecPt)

            # only save if we sample at least 1 significant channel
            if commonElec.size > 0:
                sigIdxList.append(idxsPt)
        return sigIdxList
    
    # return channels in coordinate form since the averaging may contain non-
    # significant channels and we need to access grid form of data for non-
    # significant channels
    return avgIdxs


def spatial_avg_data(data, avgIdxs):
    """Spatially averages electrode data over grouped channel indices.

    Args:
        data (ndarray): Neural data with shape
            (trials, channels_x, channels_y, timepoints).
        avgIdxs (list): List of ndarrays of 2-D grid indices; each array
            defines a group of channels to average into one channel.

    Returns:
        ndarray: Averaged data with shape (trials, timepoints, avg_channels).
    """
    avgData = np.zeros((data.shape[0], len(avgIdxs), data.shape[-1]))
    for i, idxs in enumerate(avgIdxs):
        avgData[:, i, :] = np.mean(data[:, idxs[:, 0], idxs[:, 1]], axis=1)
    avgData = avgData.transpose(0,2,1) # (trials, time, avg channels)
    return avgData


def spatial_avg_idxs(gridSize, contactSize):
    """Generates non-overlapping square grid indices for spatial averaging.

    Uses ``grid_susbsample_idxs`` with window and step both equal to
    ``contactSize`` so regions tile without overlap, centered in the grid.

    Args:
        gridSize (tuple): Grid dimensions as (rows, cols).
        contactSize (int): Edge length of the square averaging region.

    Returns:
        list: List of ndarrays, each containing 2-D grid indices for one
            averaging region.
    """
    winSize = (contactSize, contactSize)
    step = (contactSize, contactSize)

    # center the windows in the grid
    shiftX = (gridSize[0] % contactSize) // 2
    shiftY = (gridSize[1] % contactSize) // 2
    start = (shiftX, shiftY)

    # gridIdxs is a list where each element is a list of 2d indices for
    # channels that will should be averaged together to a single channel
    avgIdxs = grid_susbsample_idxs(gridSize, winSize, step, start)

    return avgIdxs


if __name__ == '__main__':
    grid_size = (8, 16)
    contact_size = 8
    gridIdxs = spatial_avg_idxs(grid_size, contact_size)

    print(f'Got {len(gridIdxs)} possible grids')

    c_val = 1
    grid = np.zeros(grid_size)
    for idxs in gridIdxs:
        grid[idxs[:, 0], idxs[:, 1]] = c_val
        c_val += 1
    
    plt.imshow(grid.T, cmap='gray', origin='lower', clim=[0, c_val])
    plt.show()
