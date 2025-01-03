import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

def pitch_subsample_sig_channels(pt, pitch, data_path):
    # load in channel map
    chanMap = sio.loadmat(f'{data_path}/{pt}/{pt}_channelMap.mat')['chanMap']

    # load in significant channel data
    sigChan = np.squeeze(
        sio.loadmat(f'{data_path}/{pt}/{pt}_sigChannel.mat')['sigChannel'])

    # trim nan edges if necessary
    if chanMap.shape[1] == 24:
        chanMap = chanMap[:,1:-1]

    # to preserve pitch when sampling across different grid sizes, calculate
    # number of electrodes to sample based on the desired pitch
    if pt in ['S14', 'S22', 'S23', 'S26']:
        mmX = 11.3
        mmY = 22.5
        maxElec = 128
    elif pt in ['S33', 'S39', 'S58', 'S62']:
        mmX = 37.8
        mmY = 20.6
        maxElec = 256
    nElec = round(mmX * mmY / pitch**2)
    
    if nElec >= maxElec:
        # just sample all electrodes if we're sampling more than the max
        elecPt = np.arange(1, maxElec+1)
    else:
        # parameters for poisson disk sampling
        gridX, gridY = chanMap.shape
        domain = (gridX, gridY)
        spacing = np.floor(np.sqrt(gridX * gridY / nElec))

        # do poisson disk sampling and -1 to get 0-indexed
        elecIdx = poisson_disk_sampling(domain, spacing, nElec)
        elecIdx = np.round(elecIdx).astype(int) - 1

        # convert 2D coordinates to channel numbers
        elecPt = chanMap[elecIdx[:, 0], elecIdx[:, 1]].astype(int)

        # check if we need to sample more electrodes
        if elecPt.shape[0] < nElec and spacing == 1:
            nRemaining = nElec - elecPt.shape[0]
            # get unsampled electrodes
            remainingElecs = np.setdiff1d(np.arange(1, gridX * gridY+1), elecPt)
            # uniformly sample remaining electrodes
            extraSampPt = np.random.choice(remainingElecs, nRemaining,
                                            replace=False)
            elecPt = np.concatenate((elecPt, extraSampPt))

    # get indices of significant channels in the subsampled set
    _, sigIdx, _ = np.intersect1d(sigChan, elecPt, return_indices=True)

    # do sampling over if we don't sample at least 1 significant channel
    if len(sigIdx) == 0:
        return pitch_subsample_sig_channels(pt, nElec, data_path)

    return sigIdx

def poisson_disk_sampling(domain, spacing, nPoints, threshold=60,
                          showIter=False, maxIter=1000):
    """
    Poisson Disk Sampling algorithm. Adapted from MATLAB code by Mohak Patel
    (Brown University, 2016). Follows the algorithm in Bridson 2007
    (https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf).
    """
    ##### Initialize the grid #####
    ndim = len(domain)
    cellSize = spacing / np.sqrt(ndim)

    # Construct grid
    sGrid = [np.arange(1, s + 1, cellSize) for s in domain]
    sGrid = np.meshgrid(*sGrid, indexing='ij')
    sizeGrid = sGrid[0].shape

    # Flatten grid points into array of coordinates
    sGrid = np.column_stack([g.ravel() for g in sGrid])

    emptyGrid = np.ones(sGrid.shape[0], dtype=bool)
    nEmptyGrid = np.sum(emptyGrid)
    scoreGrid = np.zeros_like(emptyGrid, dtype=int)

    ##### Dart-throwing #####
    ptsCreated = 0
    pts = []
    iter = 0

    start = time.time()
    while ptsCreated < nPoints and nEmptyGrid > 0:
        if iter > maxIter:
            print(f'Reached max iterations with {ptsCreated} points. Trying sampling again.')
            return poisson_disk_sampling(domain, spacing, nPoints, threshold)

        availGrid = np.where(emptyGrid)[0]
        dataPts = np.minimum(nEmptyGrid, nPoints)
        # sample nPoints from available grid points
        sampPts = np.random.choice(availGrid, dataPts, replace=False)
        # dart throws
        tempPts = sGrid[sampPts] + cellSize * np.random.rand(dataPts, ndim)

        ### Find good dart throws ###
        if len(pts) > 0:
            allPts = np.vstack((pts, tempPts))
        else:
            allPts = tempPts

        # get distance to nearest neighbor
        neighDist = min_neighbor_distance(allPts, tempPts)

        # check which points are valid
        inDomain = np.all(tempPts < domain, axis=1)  # within domain
        goodSpacing = neighDist > spacing  # far enough from other points
        validPts = inDomain & goodSpacing

        scorePts = tempPts[~validPts, :]  # keep scores from bad throws
        tempPts = tempPts[validPts, :]  # save good throws

        ### update tracking grids ###

        # update empty grid
        emptyPts = np.floor((tempPts + cellSize - 1) / cellSize).astype(int)
        # convert to linear index
        emptyPtIdx = np.ravel_multi_index(emptyPts.T-1, sizeGrid)
        # emptyPtIdx = np.ravel_multi_index((emptyPts[:,0], emptyPts[:,1]), sizeGrid)
        emptyGrid[emptyPtIdx] = False

        # update score grid
        scorePts = np.floor((scorePts + cellSize - 1) / cellSize).astype(int)
        scorePtIdx = np.ravel_multi_index(scorePts.T-1, sizeGrid)
        # scorePtIdx = np.ravel_multi_index((scorePts[:,0], scorePts[:,1]), sizeGrid)
        scoreGrid[scorePtIdx] += 1

        # update empty grid if score grid has exceeded threshold
        emptyGrid = emptyGrid & (scoreGrid < threshold)

        # update quantities for next iteration
        nEmptyGrid = np.sum(emptyGrid)
        pts.extend(tempPts)
        ptsCreated += tempPts.shape[0]
        iter += 1

        if showIter:
            elapsed = time.time() - start
            print(f"Iteration: {iter}    Points Created: {ptsCreated}   "
                  f"EmptyGrid: {nEmptyGrid}    Total Time: {elapsed:.3f}")

    # trim points to nPoints if more are created in last iteration
    pts = np.vstack(pts)
    if ptsCreated > nPoints:
        ptIdxs = np.random.choice(pts.shape[0], nPoints, replace=False)
        pts = pts[ptIdxs]
    return pts
                                          

def min_neighbor_distance(pts, newPts):
    # find distances to nearest neighbors
    _, D = knn_search(pts, newPts, 2)
    # since pts and newPts will include the same points, nearest neighbor
    # distance will be the second index, since the first will be the point
    # itself
    return D[:, 1]


def knn_search(pts, newPts, k):
    m = newPts.shape[0]
    D = np.zeros((m, k))
    I = np.zeros((m, k), dtype=int)
    for i in range(m):
        dist = np.sqrt(np.sum((pts - newPts[i])**2, axis=1))
        I[i] = np.argsort(dist)[:k]
        D[i] = np.sort(dist)[:k]
    return I, D


if __name__ == '__main__':
    # testing the grid sampling
    gridX = 8
    gridY = 16

    for nElec in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100]:
    # for pitch in [10, 5, 3, 2, 1.5]:

        # sig, all, extra = subsample_sig_channels('S14', pitch, '../data')
        # grid = np.zeros((gridX, gridY))
        # grid[all[:, 0], all[:, 1]] = 1
        # if extra.shape[0] > 0:
        #     grid[extra[:, 0], extra[:, 1]] = 2
    
        # plt.imshow(grid.T, cmap='gray', origin='lower')
        # plt.show()
    
        print('### Sampling for nElec =', nElec, '###')
        # nElec = 
        domain = (gridX, gridY)
        spacing = np.floor(np.sqrt(gridX * gridY / nElec))
        # print(spacing)

        n_grids = 3
        saved_grids = np.zeros((n_grids, gridX, gridY))
        for i in range(n_grids):
            points = poisson_disk_sampling(domain, spacing, nElec)
            points = np.round(points).astype(int) - 1
            print(f'Sampled {points.shape[0]} points')

            grid = np.zeros((gridX, gridY))
            grid[points[:, 0], points[:, 1]] = 1

            # plt.figure(figsize=(8, 8))
            plt.imshow(grid.T, cmap='gray', origin='lower')
            plt.show()
            saved_grids[i] = grid

        # check that the grids are different
        n_unique_grids = np.unique(saved_grids, axis=0).shape[0]
        print(n_unique_grids, n_unique_grids == n_grids)
