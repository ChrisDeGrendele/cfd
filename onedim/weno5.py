# https://apps.dtic.mil/sti/tr/pdf/ADA390653.pdf
from onedim.constants import *
import onedim.euler as euler
import numpy as np


def w_tilda(i, γ, β):
    ε = 1e-6
    return γ[i] / (ε + β[i]) ** 2


def w(i, γ, β):
    sumw = 0
    for k in range(3):
        sumw += w_tilda(k, γ, β)

    return w_tilda(i, γ, β) / sumw


def weno5_plus(F, grid):
    γ = [1 / 10, 3 / 5, 3 / 10]
    β = [0, 0, 0]
    # F = grid.grid

    flux = grid.scratch()

    for icomp in range(NUMQ):
        for i in range(grid.Nghost - 1, grid.Nx + grid.Nghost + 1):
            f1 = (
                (1 / 3) * F[icomp, i - 2]
                - (7 / 6) * F[icomp, i - 1]
                + (11 / 6) * F[icomp, i]
            )
            f2 = (
                (-1 / 6) * F[icomp, i - 1]
                + (5 / 6) * F[icomp, i]
                + (1 / 3) * F[icomp, i + 1]
            )
            f3 = (
                (1 / 3) * F[icomp, i]
                + (5 / 6) * F[icomp, i + 1]
                - (1 / 6) * F[icomp, i + 2]
            )

            β[0] = (13 / 12) * (
                F[icomp, i - 2] - 2 * F[icomp, i - 1] + F[icomp, i]
            ) ** 2 + (1 / 4) * (
                F[icomp, i - 2] - 4 * F[icomp, i - 1] + 3 * F[icomp, i]
            ) ** 2
            β[1] = (13 / 12) * (
                F[icomp, i - 1] - 2 * F[icomp, i] + F[icomp, i + 1]
            ) ** 2 + (1 / 4) * (F[icomp, i - 1] - F[icomp, i + 1]) ** 2
            β[2] = (13 / 12) * (
                F[icomp, i] - 2 * F[icomp, i + 1] + F[icomp, i + 2]
            ) ** 2 + (1 / 4) * (
                3 * F[icomp, i] - 4 * F[icomp, i + 1] + F[icomp, i + 2]
            ) ** 2

            flux[icomp, i] = w(0, γ, β) * f1 + w(1, γ, β) * f2 + w(2, γ, β) * f3

    return flux


def weno5_interp(F, Nx, numGhost):
    γ = [1 / 10, 3 / 5, 3 / 10]
    β = [0, 0, 0]
    assert numGhost >= 2

    flux = np.zeros_like(F)

    for i in range(numGhost - 1, Nx + numGhost + 1):
        f1 = (1 / 3) * F[i - 2] - (7 / 6) * F[i - 1] + (11 / 6) * F[i]
        f2 = (-1 / 6) * F[i - 1] + (5 / 6) * F[i] + (1 / 3) * F[i + 1]
        f3 = (1 / 3) * F[i] + (5 / 6) * F[i + 1] - (1 / 6) * F[i + 2]

        β[0] = (13 / 12) * (F[i - 2] - 2 * F[i - 1] + F[i]) ** 2 + (1 / 4) * (
            F[i - 2] - 4 * F[i - 1] + 3 * F[i]
        ) ** 2
        β[1] = (13 / 12) * (F[i - 1] - 2 * F[i] + F[i + 1]) ** 2 + (1 / 4) * (
            F[i - 1] - F[i + 1]
        ) ** 2
        β[2] = (13 / 12) * (F[i] - 2 * F[i + 1] + F[i + 2]) ** 2 + (1 / 4) * (
            3 * F[i] - 4 * F[i + 1] + F[i + 2]
        ) ** 2

        flux[i] = w(0, γ, β) * f1 + w(1, γ, β) * f2 + w(2, γ, β) * f3

    return flux


def weno5_plus_2DX(F, grid):
    γ = [1 / 10, 3 / 5, 3 / 10]
    β = [0, 0, 0]
    # F = grid.grid

    flux = np.zeros_like(grid.grid)

    for icomp in range(NUMQ):
        for i in range(grid.Nghost - 1, grid.Nx + grid.Nghost + 1):
            for j in range(grid.Nghost - 1, grid.Ny + grid.Nghost + 1):
                f1 = (
                    (1 / 3) * F[icomp, i - 2, j]
                    - (7 / 6) * F[icomp, i - 1, j]
                    + (11 / 6) * F[icomp, i, j]
                )
                f2 = (
                    (-1 / 6) * F[icomp, i - 1, j]
                    + (5 / 6) * F[icomp, i, j]
                    + (1 / 3) * F[icomp, i + 1, j]
                )
                f3 = (
                    (1 / 3) * F[icomp, i, j]
                    + (5 / 6) * F[icomp, i + 1, j]
                    - (1 / 6) * F[icomp, i + 2, j]
                )

                β[0] = (13 / 12) * (
                    F[icomp, i - 2, j] - 2 * F[icomp, i - 1, j] + F[icomp, i, j]
                ) ** 2 + (1 / 4) * (
                    F[icomp, i - 2, j] - 4 * F[icomp, i - 1, j] + 3 * F[icomp, i, j]
                ) ** 2
                β[1] = (13 / 12) * (
                    F[icomp, i - 1, j] - 2 * F[icomp, i, j] + F[icomp, i + 1, j]
                ) ** 2 + (1 / 4) * (F[icomp, i - 1, j] - F[icomp, i + 1, j]) ** 2
                β[2] = (13 / 12) * (
                    F[icomp, i, j] - 2 * F[icomp, i + 1, j] + F[icomp, i + 2, j]
                ) ** 2 + (1 / 4) * (
                    3 * F[icomp, i, j] - 4 * F[icomp, i + 1] + F[icomp, i + 2, j]
                ) ** 2

                flux[icomp, i, j] = w(0, γ, β) * f1 + w(1, γ, β) * f2 + w(2, γ, β) * f3

    return flux


def weno5_plus_2DY(F, grid):
    γ = [1 / 10, 3 / 5, 3 / 10]
    β = [0, 0, 0]
    # F = grid.grid

    flux = np.zeros_like(grid.grid)

    for icomp in range(NUMQ):
        for i in range(grid.Nghost - 1, grid.Nx + grid.Nghost + 1):
            for j in range(grid.Nghost - 1, grid.Ny + grid.Nghost + 1):
                f1 = (
                    (1 / 3) * F[icomp, i, j - 2]
                    - (7 / 6) * F[icomp, i, j - 1]
                    + (11 / 6) * F[icomp, i, j]
                )
                f2 = (
                    (-1 / 6) * F[icomp, i, j - 1]
                    + (5 / 6) * F[icomp, i, j]
                    + (1 / 3) * F[icomp, i, j + 1]
                )
                f3 = (
                    (1 / 3) * F[icomp, i, j]
                    + (5 / 6) * F[icomp, i, j + 1]
                    - (1 / 6) * F[icomp, i, j + 2]
                )

                β[0] = (13 / 12) * (
                    F[icomp, i, j - 2] - 2 * F[icomp, i, j - 1] + F[icomp, i, j]
                ) ** 2 + (1 / 4) * (
                    F[icomp, i, j - 2] - 4 * F[icomp, i, j - 1] + 3 * F[icomp, i, j]
                ) ** 2
                β[1] = (13 / 12) * (
                    F[icomp, i, j - 1] - 2 * F[icomp, i, j] + F[icomp, i, j + 1]
                ) ** 2 + (1 / 4) * (F[icomp, i, j - 1] - F[icomp, i, j + 1]) ** 2
                β[2] = (13 / 12) * (
                    F[icomp, i, j] - 2 * F[icomp, i, j + 1] + F[icomp, i, j + 2]
                ) ** 2 + (1 / 4) * (
                    3 * F[icomp, i, j] - 4 * F[icomp, i, j + 1] + F[icomp, i, j + 2]
                ) ** 2

                flux[icomp, i] = w(0, γ, β) * f1 + w(1, γ, β) * f2 + w(2, γ, β) * f3

    return flux


def weno5_minus(F, grid):
    γ = [1 / 10, 3 / 5, 3 / 10]
    β = [0, 0, 0]

    flux_minus = grid.scratch()

    for icomp in range(NUMQ):
        for i in range(grid.Nghost - 1, grid.Nx + grid.Nghost + 1):
            # Adjust the stencil for f-
            f1_minus = (
                (1 / 3) * F[icomp, i + 2]
                - (7 / 6) * F[icomp, i + 1]
                + (11 / 6) * F[icomp, i]
            )
            f2_minus = (
                (-1 / 6) * F[icomp, i + 1]
                + (5 / 6) * F[icomp, i]
                + (1 / 3) * F[icomp, i - 1]
            )
            f3_minus = (
                (1 / 3) * F[icomp, i]
                + (5 / 6) * F[icomp, i - 1]
                - (1 / 6) * F[icomp, i - 2]
            )

            # Compute β for each stencil
            β[0] = (13 / 12) * (
                F[icomp, i + 2] - 2 * F[icomp, i + 1] + F[icomp, i]
            ) ** 2 + (1 / 4) * (
                F[icomp, i + 2] - 4 * F[icomp, i + 1] + 3 * F[icomp, i]
            ) ** 2
            β[1] = (13 / 12) * (
                F[icomp, i + 1] - 2 * F[icomp, i] + F[icomp, i - 1]
            ) ** 2 + (1 / 4) * (F[icomp, i + 1] - F[icomp, i - 1]) ** 2
            β[2] = (13 / 12) * (
                F[icomp, i] - 2 * F[icomp, i - 1] + F[icomp, i - 2]
            ) ** 2 + (1 / 4) * (
                3 * F[icomp, i] - 4 * F[icomp, i - 1] + F[icomp, i - 2]
            ) ** 2

            # Compute the weighted sum for the flux
            flux_minus[icomp, i] = (
                w(0, γ, β) * f1_minus + w(1, γ, β) * f2_minus + w(2, γ, β) * f3_minus
            )

    return flux_minus
