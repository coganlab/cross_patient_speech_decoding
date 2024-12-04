import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def grid_subsample_sig_channels(pt, winSize, dataPath, step=(1,1)):
    # load in channel map
    chanMap = sio.loadmat(f'{dataPath}/{pt}/{pt}_channelMap.mat')['chanMap']

    # load in significant channel data
    sigChan = np.squeeze(
        sio.loadmat(f'{dataPath}/{pt}/{pt}_sigChannel.mat')['sigChannel'])

    # trim full nan edges if necessary
    if chanMap.shape[1] == 24:
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

    
def grid_susbsample_idxs(grid_size, win_size, step=(1,1)):
    # starting indices to placec windows in grid
    startIdxX = np.arange(0, grid_size[0] - win_size[0] + 1, step[0])
    startIdxY = np.arange(0, grid_size[1] - win_size[1] + 1, step[1])

    # use meshgrid to get all possible combinations of starting indices
    startIdxs = np.array(np.meshgrid(startIdxX, startIdxY))
    startIdxs = startIdxs.reshape(2, -1).T

    gridIdxs = []
    for (x,y) in startIdxs:
        # define x-y span of window from current starting point
        currIdxsX = np.arange(x, x + win_size[0])
        currIdxsY = np.arange(y, y + win_size[1])

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
