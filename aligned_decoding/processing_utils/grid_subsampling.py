import numpy as np
import matplotlib.pyplot as plt


def grid_susbsample_idxs(grid_size, window_size, step=(1,1)):
    # starting indices to placec windows in grid
    start_idx_x = np.arange(0, grid_size[0] - window_size[0] + 1, step[0])
    start_idx_y = np.arange(0, grid_size[1] - window_size[1] + 1, step[1])

    # use meshgrid to get all possible combinations of starting indices
    start_idxs = np.array(np.meshgrid(start_idx_x, start_idx_y))
    start_idxs = start_idxs.reshape(2, -1).T

    grid_idxs = []
    for (x,y) in start_idxs:
        # define x-y span of window from current starting point
        curr_idxs_x = np.arange(x, x + window_size[0])
        curr_idxs_y = np.arange(y, y + window_size[1])

        # get full grid indices from x-y span
        curr_idxs = np.array(np.meshgrid(curr_idxs_x, curr_idxs_y))
        curr_idxs = curr_idxs.reshape(2, -1).T

        # save to list of possible grids
        grid_idxs.append(curr_idxs)
    
    return grid_idxs


if __name__ == '__main__':
    grid_size = (8,16)
    window_size = (6,12)
    step = (1, 1)
    grid_idxs = grid_susbsample_idxs(grid_size, window_size, step=step)

    print(f'Got {len(grid_idxs)} possible grids')

    for idxs in grid_idxs:
        grid = np.zeros(grid_size)
        grid[idxs[:,0], idxs[:,1]] = 1

        plt.imshow(grid.T, cmap='gray', origin='lower', clim=[0,1])
        plt.show()
