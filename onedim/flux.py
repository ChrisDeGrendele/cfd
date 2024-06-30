
from onedim.euler import prim_to_cons_var, flux_var,cons_to_prim
from onedim.reconstruct import weno5_reconstruction
from onedim.constants import *
from onedim.grid import Grid1D

import numpy as np

class Flux:

    def __init__(self, grid: Grid1D, type: str):

        self.grid = grid
        self.type = type


        if self.type == "weno5_js":
            self.flux_method = self.weno_js
        else:
            raise RuntimeError(f"Flux method not supported: {self.type}")


    def getFlux(self):
        return self.flux_method()

    def weno_js(self):
        self.grid.assert_variable_type("prim")

        a = np.sqrt(gamma * self.grid.grid[PCOMP] / self.grid.grid[RHOCOMP])
        max_speed = np.max(np.abs(self.grid.grid[UCOMP]) + a)

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
        return consP, LFFlux


    def weno_js(self):
        self.grid.assert_variable_type("prim")

        a = np.sqrt(gamma * self.grid.grid[PCOMP] / self.grid.grid[RHOCOMP])
        max_speed = np.max(np.abs(self.grid.grid[UCOMP]) + a)

        # reconstruct primitive at i+1/2 cell face
        primP = weno5_reconstruction(self.grid.grid, self.grid)
        # primP = SEDAS_apriori(grid.grid, grid)

        # conservative variables at cell interface.
        mass, mom, energy = prim_to_cons_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])
        consP = np.array([mass, mom, energy])

        # compute analytical flux at cell interface.
        fR = flux_var(primP[RHOCOMP], primP[UCOMP], primP[PCOMP])

        U_new = np.ones_like(self.grid.grid) #/ 0  # np.nans_like lol
        LFFlux = np.zeros_like(self.grid.grid)

        # Computes LF Flux
        # Loop through all the cells except for the outermost ghost cells.
        for i in range(1, len(self.grid.x) - 1):
            for icomp in range(NUMQ):
                # Compute the Lax-Friedrichs flux at i+1/2 and i-1/2 interfaces
                LFFlux[icomp, i] = 0.5 * (
                    fR[icomp, i + 1] + fR[icomp, i]
                ) - 0.5 * max_speed * (consP[icomp, i + 1] - consP[icomp, i])
        return consP, LFFlux
