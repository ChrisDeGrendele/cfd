import numpy as np
import matplotlib.pyplot as plt
from constants import *

class Grid1D:
    def __init__(self, xlim, Nx, Nghost, num_vars):
        self.xlim = xlim
        self.Nx = Nx
        self.Nghost = Nghost
        self.num_vars = num_vars  # Number of variables (e.g., 3 for Euler equations)
        self.dx = (xlim[1] - xlim[0]) / (Nx - 1)
        self.x = np.linspace(xlim[0] - self.dx*Nghost, xlim[1] + self.dx*Nghost, Nx + 2*Nghost)
        self.ndim = 1
        # Internal x (no ghost cells)
        self.x_int = self.x[self.Nghost:self.Nx + self.Nghost]
        # Initialize the grid for multiple variables
        self.grid = np.zeros((num_vars, Nx + 2*Nghost))
        self.variables = "prim"

    def fill_grid(self, f):
        for var in range(self.num_vars):
            self.grid[var, self.Nghost:self.Nx + self.Nghost] = f(self.x_int,var)

    def plot(self):
        fig, axs = plt.subplots(self.num_vars, 1, figsize=(10, self.num_vars * 3))
        if self.num_vars == 1:
            axs = [axs]  # Make it iterable
        for var in range(self.num_vars):
            axs[var].plot(self.x, self.grid[var, :],'.')
            axs[var].set_title(variable_names[var])
        plt.tight_layout()
        plt.show()

    def apply_periodic_bcs(self):
        for var in range(self.num_vars):
            for i in range(self.Nghost):
                self.grid[var, i] = self.grid[var, self.Nx + self.Nghost - (self.Nghost-i)]
                self.grid[var, self.Nx + self.Nghost + i] = self.grid[var, self.Nghost + i]
            
    def apply_reflective_bcs(self):
        for var in range(self.num_vars):
            # Reflective BCs for left boundary
            if var == UCOMP:  # Assuming UCOMP is the velocity component
                self.grid[var, :self.Nghost] = -self.grid[var, self.Nghost:2*self.Nghost][::-1]
            else:  # For density and pressure, just copy the values from the first interior cells
                self.grid[var, :self.Nghost] = self.grid[var, self.Nghost:2*self.Nghost]

            # Reflective BCs for right boundary
            if var == UCOMP:
                self.grid[var, self.Nx + self.Nghost:] = -self.grid[var, self.Nx:self.Nx + self.Nghost][::-1]
            else:
                self.grid[var, self.Nx + self.Nghost:] = self.grid[var, self.Nx:self.Nx + self.Nghost]

    def apply_zero_gradient_bcs(self):
        # 

        #     self.grid[var,0] = self.grid[var,1]
        #     self.grid[var,self.Nx + self.Nghost] =  self.grid[var,self.Nx]

        #self.grid[UCOMP,0] = 0
        #self.grid[UCOMP,self.Nx + self.Nghost] =  0

        for var in range(self.num_vars):
            for ighost in range(self.Nghost):

                self.grid[var,ighost] = self.grid[var,self.Nghost]
                self.grid[var, self.Nx + self.Nghost + ighost] = self.grid[var, self.Nx + self.Nghost - 1]


    def return_internal_grid(self):
        return self.grid[:, self.Nghost:self.Nx + self.Nghost]
    
    def return_grid(self):
        return self.grid

    def scratch_internal(self):
        return np.zeros_like(self.grid[:, self.Nghost:self.Nx + self.Nghost])
    def scratch(self):
        return np.zeros_like(self.grid)


    def transform(self,f,name):
        self.grid = f(self.grid)
        self.variables = name

    def set_internal(self,U_new):
        self.grid[:, self.Nghost:self.Nx + self.Nghost] = U_new
        
    def set(self,U_new):
        assert(np.shape(U_new) == np.shape(self.grid))
        self.grid = U_new


    def set_grid(self,U_new):
        self.grid = U_new

    def internal_bounds(self):
        return (self.Nghost, self.Nx+self.Nghost)
    
    def assert_variable_type(self,str):
        assert(str == self.variables)