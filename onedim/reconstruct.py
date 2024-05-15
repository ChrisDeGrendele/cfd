import numpy as np
from onedim.weno5 import *
import onedim.shockdetector as shock
import sys

sys.path.append("/Users/chris/Documents/github/gp-recipe/src_new")
from new_driver import GP_recipe1D, GP_recipe2D
import kernels as kern
import grids as old_grid


def weno5_reconstruction(U, grid):
    return weno5_plus(U, grid)


def SEDAS_apriori(U, grid):
    r_gp = 2

    Uphalf = np.zeros_like(U)

    shock.a_crappy_1D_shock_detector(grid)

    # this is going to be *really* clunky. I'm sorry future me.

    # we also need 1 ghost flux on each side. So NumGhosts should be 1 more than r_gp
    assert grid.Nghost == r_gp + 1
    x_predict = grid.x[grid.Nghost - 1 : grid.Nx + grid.Nghost + 1] + grid.dx / 2
    assert len(x_predict) == 2 + grid.Nx

    # loop through xs and detect shocks, so we don't have to reconstruct this whole grid twice.
    x_predict_smooth = []
    x_predict_shock = []

    innerloop = 0
    for i in range(grid.Nghost - 1, grid.Nx + grid.Nghost + 1):
        if grid.shock_mask[i] == 0:
            x_predict_smooth.append(x_predict[innerloop])
        else:
            x_predict_shock.append(x_predict[innerloop])

        innerloop += 1

    x_predict_smooth = np.array(x_predict_smooth)
    x_predict_shock = np.array(x_predict_shock)

    for icomp in range(NUMQ):
        compGrid = old_grid.Grid1D(grid.xlim, grid.Nx, grid.Nghost)
        compGrid.grid = U[icomp]

        gp_SE = GP_recipe1D(
            compGrid,
            r_gp,
            ell=12 * grid.dx,
            stencil_method="center",
            high_precision=True,
        )
        gp_DAS = GP_recipe1D(compGrid, 1, stencil_method="center", high_precision=True)

        SE_predict = gp_SE.convert_custom(x_predict_smooth, kern.SE, kern.SE)
        DAS_predict = gp_DAS.convert_custom(x_predict_shock, kern.AS, kern.AS)

        smoothIndex = 0
        shockIndex = 0
        for i in range(grid.Nghost - 1, grid.Nx + grid.Nghost + 1):
            Uphalf[icomp, i] = (
                SE_predict[smoothIndex]
                if (grid.shock_mask[i] == 0)
                else DAS_predict[shockIndex]
            )
            if grid.shock_mask[i] == 0:
                smoothIndex += 1
            else:
                shockIndex += 1

    return Uphalf
