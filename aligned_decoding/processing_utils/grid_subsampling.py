"""Sliding-window grid subsampling of electrode arrays."""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def grid_subsample_sig_channels(pt, winSize, dataPath, step=(1,1)):
    """Finds significant-channel indices within every sliding sub-grid.

    For each sliding window position on the electrode grid, identifies which
    of the sampled channels are significant and returns their local indices.

    Args:
        pt (str): Subject identifier string.
        winSize (tuple): Window dimensions as (rows, cols).
        dataPath (str): Root directory containing subject data folders.
        step (tuple, optional): Step size in each dimension.
            Defaults to (1, 1).

    Returns:
        list: List of ndarrays, each containing the indices of significant
            channels within a particular sub-grid.
    """
    # load in channel map
    chanMap = sio.loadmat(f'{dataPath}/{pt}/{pt}_channelMap.mat')['chanMap']

    # load in significant channel data
    sigChan = np.squeeze(
        sio.loadmat(f'{dataPath}/{pt}/{pt}_sigChannel.mat')['sigChannel'])

    # trim full nan edges if necessary
    if chanMap.shape[0] == 24: 
        chanMap = chanMap[1:-1,:]
        winSize = (winSize[1], winSize[0]) # transpose window size
    elif chanMap.shape[1] == 24:
        chanMap = chanMap[:,1:-1]

    # get possible subgrids
    gridSize = chanMap.shape
    gridIdxs = grid_susbsample_idxs(gridSize, winSize, step=step)

    sigElecList = []
    for idxs in gridIdxs:
        # convert 2D coordinates to channel numbers
        elecPt = chanMap[idxs[:, 0], idxs[:, 1]]

        # remove any nans from non-rectangular channel maps
        elecPt = elecPt[~np.isnan(elecPt)].astype(int)

        # get indices of significant channels in the subsampled set
        _, sigIdx, _ = np.intersect1d(sigChan, elecPt, return_indices=True)


        # only save if we sample at least 1 significant channel
        if len(sigIdx) > 0:
            sigElecList.append(sigIdx)

    return sigElecList

    
def grid_susbsample_idxs(gridSize, winSize, step=(1,1), start=(0,0)):
    """Generates 2-D index arrays for all sliding window positions on a grid.

    Args:
        gridSize (tuple): Full grid dimensions as (rows, cols).
        winSize (tuple): Window dimensions as (rows, cols).
        step (tuple, optional): Step size in each dimension.
            Defaults to (1, 1).
        start (tuple, optional): Starting offset in each dimension.
            Defaults to (0, 0).

    Returns:
        list: List of ndarrays with shape (winSize[0]*winSize[1], 2), each
            containing the (row, col) indices for one window position.
    """
    # starting indices to placec windows in grid
    startIdxX = np.arange(start[0], gridSize[0] - winSize[0] + 1, step[0])
    startIdxY = np.arange(start[1], gridSize[1] - winSize[1] + 1, step[1])

    # use meshgrid to get all possible combinations of starting indices
    startIdxs = np.array(np.meshgrid(startIdxX, startIdxY))
    startIdxs = startIdxs.reshape(2, -1).T

    gridIdxs = []
    for (x,y) in startIdxs:
        # define x-y span of window from current starting point
        currIdxsX = np.arange(x, x + winSize[0])
        currIdxsY = np.arange(y, y + winSize[1])

        # get full grid indices from x-y span
        currIdxs = np.array(np.meshgrid(currIdxsX, currIdxsY))
        currIdxs = currIdxs.reshape(2, -1).T

        # save to list of possible grids
        gridIdxs.append(currIdxs)
    
    return gridIdxs


if __name__ == '__main__':
    gridSize = (8,16)
    winSize = (6,12)
    step = (1, 1)
    gridIdxs = grid_susbsample_idxs(gridSize, winSize, step=step)

    print(f'Got {len(gridIdxs)} possible grids')

    for idxs in gridIdxs:
        grid = np.zeros(gridSize)
        grid[idxs[:,0], idxs[:,1]] = 1

        plt.imshow(grid.T, cmap='gray', origin='lower', clim=[0,1])
        plt.show()
