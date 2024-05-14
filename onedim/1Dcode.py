import numpy as np
from grid import Grid1D
import ics
import matplotlib.pyplot as plt
import os
from constants import *
import solver
from reconstruct import *
from euler import *
import shockdetector as shock

# Grid
xlim = (0, 1)
Nx = 500
NumGhost = 3

grid_weno = Grid1D(xlim, Nx, NumGhost, NUMQ)
grid_sedas = Grid1D(xlim, Nx, NumGhost, NUMQ)
grid_ref = Grid1D(xlim, 5000, NumGhost, NUMQ)


# Time
cfl = 0.5
t_finish = 0.1
t0 = 0.0
Nt = np.inf
timestepNum = 0
t = t0

makeMovie = True


grid_weno.fill_grid(ics.sod_shock_tube)
grid_weno.apply_zero_gradient_bcs()

grid_sedas.fill_grid(ics.sod_shock_tube)
grid_sedas.apply_zero_gradient_bcs()

grid_ref.fill_grid(ics.sod_shock_tube)
grid_ref.apply_zero_gradient_bcs()


os.makedirs("simulation_frames", exist_ok=True)


while (t < t_finish) and timestepNum < Nt:
    print("Timestep: ", timestepNum, "  Current time: ", t)

    grid_weno.apply_zero_gradient_bcs()
    grid_sedas.apply_zero_gradient_bcs()
    grid_ref.apply_zero_gradient_bcs()

    if makeMovie:
        fig, axs = plt.subplots(4, 1, figsize=(10, 15))

        axs[0].plot(grid_ref.x, grid_ref.grid[RHOCOMP, :], label="Reference", c="black")
        axs[0].plot(
            grid_weno.x, grid_weno.grid[RHOCOMP, :], "o", label="FD-Prim WENO", c="red"
        )
        axs[0].plot(
            grid_sedas.x,
            grid_sedas.grid[RHOCOMP, :],
            "+",
            label="FD-Prim SEDAS",
            c="blue",
        )
        axs[0].set_title(f"Time: {t:.2f}")
        axs[0].set_ylabel("Density")
        axs[0].legend()

        axs[1].plot(grid_ref.x, grid_ref.grid[UCOMP, :], label="Reference", c="black")
        axs[1].plot(
            grid_weno.x, grid_weno.grid[UCOMP, :], "o", label="FD-Prim WENO", c="red"
        )
        axs[1].plot(
            grid_sedas.x,
            grid_sedas.grid[UCOMP, :],
            "+",
            label="FD-Prim SEDAS",
            c="blue",
        )
        axs[1].set_ylabel("Velocity")
        axs[1].legend()

        axs[2].plot(grid_ref.x, grid_ref.grid[PCOMP, :], label="Reference", c="black")
        axs[2].plot(
            grid_weno.x, grid_weno.grid[PCOMP, :], "o", label="FD-Prim WENO", c="red"
        )
        axs[2].plot(
            grid_sedas.x,
            grid_sedas.grid[PCOMP, :],
            "+",
            label="FD-Prim SEDAS",
            c="blue",
        )
        axs[2].set_ylabel("Pressure")
        axs[2].legend()

        shock.a_crappy_1D_shock_detector(grid_sedas)
        divV = []
        for i in range(1, len(grid_sedas.x) - 1):
            divV.append(
                (grid_sedas.grid[UCOMP, i + 1] - grid_sedas.grid[UCOMP, i - 1])
                / (2 * grid_sedas.dx)
            )

        divV = np.array(divV)
        divV /= np.max(np.abs(divV))

        axs[3].plot(
            grid_weno.x, grid_weno.shock_mask, "o", label="FD-Prim WENO", c="red"
        )
        axs[3].plot(
            grid_sedas.x, grid_sedas.shock_mask, "+", label="FD-Prim SEDAS", c="blue"
        )
        axs[3].plot(
            grid_sedas.x[1 : len(grid_sedas.x) - 1],
            divV,
            label="Divergence of Velocity",
        )

        axs[3].set_ylabel("Shock Mask")
        axs[3].legend()

        plt.savefig(f"simulation_frames/frame_{timestepNum:04d}.png")
        plt.close(fig)

    ######## FIRST GRID UPDATE ################################################
    grid = grid_weno

    uPrim = grid.grid
    grid.assert_variable_type("prim")
    a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
    max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
    dt = min(cfl * grid.dx / max_speed, t_finish - t)

    # reconstruct primitive at i+1/2 cell face
    primP = weno5_reconstruction(grid.grid, grid)
    # primP = SEDAS_apriori(grid.grid, grid)

    # conservative variables at cell interface.
    mass, mom, energy = prim_to_cons_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])
    consP = np.array([mass, mom, energy])

    # compute analytical flux at cell interface.
    fR = flux_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])

    U_new = np.ones_like(grid.grid) / 0  # np.nans_like lol
    LFFlux = np.zeros_like(grid.grid)

    # Computes LF Flux
    # Loop through all the cells except for the outermost ghost cells.
    for i in range(1, len(grid.x) - 1):
        for icomp in range(NUMQ):
            # Compute the Lax-Friedrichs flux at i+1/2 and i-1/2 interfaces
            LFFlux[icomp, i] = 0.5 * (
                fR[icomp, i + 1] + fR[icomp, i]
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
    ################################################################################################
    ######## SECOND GRID UPDATE ################################################

    grid = grid_sedas

    uPrim = grid.grid
    grid.assert_variable_type("prim")
    a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
    max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
    dt = min(cfl * grid.dx / max_speed, t_finish - t)

    # reconstruct primitive at i+1/2 cell face
    # primP = weno5_reconstruction(grid.grid,grid)
    primP = SEDAS_apriori(grid.grid, grid)

    # conservative variables at cell interface.
    mass, mom, energy = prim_to_cons_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])
    consP = np.array([mass, mom, energy])

    # compute analytical flux at cell interface.
    fR = flux_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])

    U_new = np.ones_like(grid.grid) / 0  # np.nans_like lol
    LFFlux = np.zeros_like(grid.grid)

    # Computes LF Flux
    # Loop through all the cells except for the outermost ghost cells.
    for i in range(1, len(grid.x) - 1):
        for icomp in range(NUMQ):
            # Compute the Lax-Friedrichs flux at i+1/2 and i-1/2 interfaces
            LFFlux[icomp, i] = 0.5 * (
                fR[icomp, i + 1] + fR[icomp, i]
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
    grid_sedas = grid
    ################################################################################################
    ######## Third GRID UPDATE ################################################

    grid = grid_ref

    uPrim = grid.grid
    grid.assert_variable_type("prim")
    a = np.sqrt(gamma * grid.grid[PCOMP] / grid.grid[RHOCOMP])
    max_speed = np.max(np.abs(grid.grid[UCOMP]) + a)
    dt = min(cfl * grid.dx / max_speed, t_finish - t)

    # reconstruct primitive at i+1/2 cell face
    primP = weno5_reconstruction(grid.grid, grid)
    # primP = SEDAS_apriori(grid.grid, grid)

    # conservative variables at cell interface.
    mass, mom, energy = prim_to_cons_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])
    consP = np.array([mass, mom, energy])

    # compute analytical flux at cell interface.
    fR = flux_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])

    U_new = np.ones_like(grid.grid) / 0  # np.nans_like lol
    LFFlux = np.zeros_like(grid.grid)

    # Computes LF Flux
    # Loop through all the cells except for the outermost ghost cells.
    for i in range(1, len(grid.x) - 1):
        for icomp in range(NUMQ):
            # Compute the Lax-Friedrichs flux at i+1/2 and i-1/2 interfaces
            LFFlux[icomp, i] = 0.5 * (
                fR[icomp, i + 1] + fR[icomp, i]
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
    grid_ref = grid
    ################################################################################################

    t += dt
    timestepNum += 1


if makeMovie:
    os.system(
        "ffmpeg -r 30 -f image2 -i simulation_frames/frame_%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p simulation.mp4"
    )


grid.plot()
