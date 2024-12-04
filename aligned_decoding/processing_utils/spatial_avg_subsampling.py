import numpy as np
import matplotlib.pyplot as plt

from grid_subsampling import grid_susbsample_idxs

def spatial_avg_idxs(gridSize, contactSize):
    # contacts to average are square subsampled grids, so we re-use the grid
    # subsampling function with the window as a square with edge length equal
    # to the contact size and a step equal to the contact size in both
    # dimensions so there is no overlap
    winSize = (contactSize, contactSize)
    step = (contactSize, contactSize)

    # center the windows in the grid
    shiftX = (gridSize[0] % contactSize) // 2
    shiftY = (gridSize[1] % contactSize) // 2
    start = (shiftX, shiftY)

    # gridIdxs is a list where each element is a list of 2d indices for
    # channels that will should be averaged together to a single channel
    gridIdxs = grid_susbsample_idxs(gridSize, winSize, step, start)

    return gridIdxs


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