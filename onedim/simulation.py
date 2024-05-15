from onedim.grid import Grid1D
from onedim.constants import *
import onedim.ics as ics

from onedim.euler import prim_to_cons_var, flux_var,cons_to_prim

from onedim.reconstruct import weno5_reconstruction

import numpy as np
import matplotlib.pyplot as plt
import os

class Simulation:
    def __init__(self, a_inputs):
        self.inp = a_inputs

        self.grid = Grid1D(
            self.inp.xlim, self.inp.nx, self.inp.numghosts, NUMQ
        )

        self.grid.fill_grid(ics.sod_shock_tube)

        self.grid.apply_zero_gradient_bcs()

        self.t = self.inp.t0
        self.timestepNum = 0


        self.plot()

    def run(self):
        # os.makedirs('simulation_frames', exist_ok=True)


        while (self.t < self.inp.t_finish) and self.timestepNum < self.inp.nt:
            print("Timestep: ", self.timestepNum, "  Current time: ", self.t)

            self.grid.apply_zero_gradient_bcs()


            uPrim = self.grid.grid
            self.grid.assert_variable_type("prim")
            a = np.sqrt(gamma * self.grid.grid[PCOMP] / self.grid.grid[RHOCOMP])
            max_speed = np.max(np.abs(self.grid.grid[UCOMP]) + a)
            dt = min(self.inp.cfl * self.grid.dx / max_speed, self.inp.t_finish - self.t)

            # reconstruct primitive at i+1/2 cell face
            primP = weno5_reconstruction(self.grid.grid, self.grid)
            # primP = SEDAS_apriori(grid.grid, grid)

            # conservative variables at cell interface.
            mass, mom, energy = prim_to_cons_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])
            consP = np.array([mass, mom, energy])

            # compute analytical flux at cell interface.
            fR = flux_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])

            U_new = np.ones_like(self.grid.grid) / 0  # np.nans_like lol
            LFFlux = np.zeros_like(self.grid.grid)

            # Computes LF Flux
            # Loop through all the cells except for the outermost ghost cells.
            for i in range(1, len(self.grid.x) - 1):
                for icomp in range(NUMQ):
                    # Compute the Lax-Friedrichs flux at i+1/2 and i-1/2 interfaces
                    LFFlux[icomp, i] = 0.5 * (
                        fR[icomp, i + 1] + fR[icomp, i]
                    ) - 0.5 * max_speed * (consP[icomp, i + 1] - consP[icomp, i])

            # Update.
            for i in range(self.grid.Nghost, self.grid.Nx + self.grid.Nghost):
                for icomp in range(NUMQ):
                    U_new[icomp, i] = consP[icomp, i] - (dt / self.grid.dx) * (
                        LFFlux[icomp, i] - LFFlux[icomp, i - 1]
                    )

            self.grid.set(U_new)
            self.grid.transform(cons_to_prim, "prim")
            self.grid.apply_zero_gradient_bcs()


            
            self.timestepNum += 1
            self.t += dt

            if self.timestepNum % self.inp.output_freq == 0:
                self.plot()

            # DEBUG
            for i in range(len(self.grid.grid[0])):
                if self.grid.grid[PCOMP, i] <= 0:
                    print("Bad cell: ", i)
                    assert False
                for icomp in range(NUMQ):
                    if np.isnan(self.grid.grid[icomp, i]):
                        print("Nan cell : ", i)
                        assert False

        print("SUCCESS!")

    def plot(self):
        if not os.path.exists(self.inp.output_dir):
            os.makedirs(self.inp.output_dir)

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        axs[0].scatter(self.grid.x, self.grid.grid[RHOCOMP, :], c="black")
        axs[0].set_ylabel("Density")

        axs[1].scatter(self.grid.x, self.grid.grid[UCOMP, :],  c="black")
        axs[1].set_ylabel("Velocity")

        axs[2].scatter(self.grid.x, self.grid.grid[PCOMP, :],  c="black")
        axs[2].set_ylabel("Pressure")

        axs[0].set_title(f"Time: {self.t:.4f}")
        plt.savefig(f"{self.inp.output_dir}/plot_dt{str(self.timestepNum).zfill(6)}")
        plt.close()
