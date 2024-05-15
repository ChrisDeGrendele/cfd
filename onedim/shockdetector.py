from onedim.constants import *
import numpy as np
import numpy as np


# This one looks nice.
# https://www.sciencedirect.com/science/article/pii/S0021999119304796


def a_crappy_1D_shock_detector(grid, std_deviation_above_mean=4):
    """
    1 is shock (true)
    0 is smooth (false)

    std_deviation_above_mean = how many standard deviations above the mean to be a shock?

    fills in grid.shock_mask.
    """

    gradP = np.abs(np.gradient(grid.grid[PCOMP]))
    # gradRho =  np.abs(np.gradient(grid.grid[RHOCOMP]))

    thresholdP = np.mean(gradP) + std_deviation_above_mean * np.std(gradP)
    shock_mask = (gradP > thresholdP).astype(int)

    for i in range(1, len(grid.x) - 1):
        divV = (grid.grid[UCOMP, i + 1] - grid.grid[UCOMP, i - 1]) / (2 * grid.dx)

        if (shock_mask[i] == 1) and (divV <= 0):
            shock_mask[i] = 1
        else:
            shock_mask[i] = 0

    # shock_mask = ((gradP > thresholdP) | (gradRho > threshpoldRho)).astype(int)
    # shock_mask = (gradP > thresholdP).astype(int)

    grid.shock_mask = shock_mask


def gradPFromMood(grid):
    sigmaP = 10 * grid.dx
    sigmaV = grid.dx**2
    shock_mask = np.zeros_like(grid.x)

    for i in range(1, len(grid.x) - 1):
        gradP = np.abs(grid.grid[PCOMP, i + 1] - grid.grid[PCOMP, i - 1]) / (
            min(grid.grid[PCOMP, i + 1], grid.grid[PCOMP, i - 1])
        )

        divV = (grid.grid[UCOMP, i + 1] - grid.grid[UCOMP, i - 1]) / (2 * grid.dx)

        if (gradP > sigmaP) or (divV < -sigmaV):
            shock_mask[i] = 1

    grid.shock_mask = shock_mask
