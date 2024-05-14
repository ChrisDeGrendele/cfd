import numpy as np
from grid import Grid2D
import ics2D
import matplotlib.pyplot as plt
import os
from constants2D import *
import solver
from reconstruct import *
from euler import *
import weno5

# Grid
xlim = (0, 1)
ylim = (0, 1)
Nx = 500
Ny = 500
NumGhost = 3

grid = Grid2D(xlim, ylim, Nx, Ny, NumGhost, 4)

# Time
cfl = 0.5
t_finish = 0.1
t0 = 0.0
Nt = np.inf
timestepNum = 0
t = t0

makeMovie = True


grid.fill_grid(ics2D.rimeann2D)
grid.apply_zero_gradient_bcs()
grid.plot()


# os.makedirs('simulation_frames', exist_ok=True)


while (t < t_finish) and timestepNum < Nt:
    print("Timestep: ", timestepNum, "  Current time: ", t)

    grid.apply_zero_gradient_bcs()

    uPrim = grid.grid
    grid.assert_variable_type("prim")
    a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
    max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
    dt = min(cfl * grid.dx / max_speed, t_finish - t)

    # reconstruct primitive at i+1/2 cell face
    primPX = weno5.weno5_plus_2DX(grid.grid, grid)
    # reconstruct primitive at j+1/2 cell face
    primPY = weno5.weno5_plus_2DY(grid.grid, grid)

    # conservative variables at cell interface.
    mass, mom, energy = prim_to_cons_var2D(
        primPX[RHOCOMP], primPX[UCOMP], primPX[VCOMP], primPX[PCOMP]
    )
    consPX = np.array([mass, mom, energy])

    mass, mom, energy = prim_to_cons_var2D(
        primPY[RHOCOMP], primPY[UCOMP], primPY[VCOMP], primPY[PCOMP]
    )
    consPY = np.array([mass, mom, energy])

    # compute analytical flux at cell interface.
    fR = flux_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])

    U_new = np.ones_like(grid.grid) / 0  # np.nans_like lol
    LFFlux = np.zeros_like(grid.grid)

    # Computes LF Flux
    # Loop through all the cells except for the outermost ghost cells.
    for i in range(1, len(grid.x) - 1):
        for j in range(1, len(grid.y) - 1):
            for icomp in range(NUMQ):
                # Compute the Lax-Friedrichs flux at i+1/2 and i-1/2 interfaces
                LFFlux[icomp, i] = 0.5 * (
                    fRX[icomp, i + 1, j]
                    + fRX[icomp, i, j]
                    + fRY[icomp, i, j + 1]
                    + fRY[icomp, i, j]
                ) - 0.5 * max_speed * (consP[icomp, i + 1] - consP[icomp, i])

    # Update.
    for i in range(grid.Nghost, grid.Nx + grid.Nghost):
        for icomp in range(NUMQ):
            U_new[icomp, i] = consP[icomp, i] - (dt / grid.dx) * (
                LFFlux[icomp, i] - LFFlux[icomp, i - 1]
            )

    grid.set(U_new)
    grid.transform(cons_to_prim, "prim")
    grid.apply_zero_gradient_bcs()

    # DEBUG
    for i in range(len(grid.grid[0])):
        if grid.grid[PCOMP, i] <= 0:
            print("Bad cell: ", i)
            assert False
        for icomp in range(NUMQ):
            if np.isnan(grid.grid[icomp, i]):
                print("Nan cell : ", i)
                assert False
    grid_weno = grid

    # DEBUG
    for i in range(len(grid.grid[0])):
        if grid.grid[PCOMP, i] <= 0:
            print("Bad cell: ", i)
            assert False
        for icomp in range(NUMQ):
            if np.isnan(grid.grid[icomp, i]):
                print("Nan cell : ", i)
                assert False

    t += dt
    timestepNum += 1


grid.plot()
