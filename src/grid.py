import numpy as np
import matplotlib.pyplot as plt

class Grid1D:
    def __init__(self, xlim, Nx, Nghost, NumVariables):
        self.xlim = xlim
        self.Nx = Nx
        self.Nghost = Nghost
        self.dx = (xlim[1]-xlim[0])/(Nx-1)
        self.x = np.linspace(xlim[0] - self.dx*Nghost, xlim[1] + self.dx*Nghost, Nx + 2*Nghost)
        self.ndim = 1


        #internal x (no ghost cells)
        self.x_int = self.x[self.Nghost:self.Nx+self.Nghost]

        self.grid = np.zeros((NumVariables,len(self.x)))


    def fill_grid(self, f, ivar):
        self.grid[:,ivar] = f(self.x)

    def plot(self):
        fig = plt.figure()
        plt.plot(self.x, self.grid)
        plt.show()

    def get_scratch_array(self, just_internal=False):

        if just_internal:
            return np.zeros_like(self.grid[self.Nghost:self.Nx+self.Nghost])
        else:
            return np.zeros_like(self.grid)

    def get_var_array(self, a_comp, just_internal=False):
        if just_internal:
            return np.zeros_like(self.grid[a_comp,self.Nghost:self.Nx+self.Nghost])
        else:
            return np.zeros_like(self.grid[a_comp])







class Grid2D:
    def __init__(self, xlim, ylim, Nx, Ny, Nghost):
        self.xlim = xlim
        self.ylim = ylim

        self.Nx = Nx
        self.Ny = Ny
        self.Nghost = Nghost
        self.ndim = 2

        self.dx = (xlim[1]-xlim[0])/(Nx-1)
        self.dy = (ylim[1]-ylim[0])/(Ny-1)

        self.x = np.linspace(xlim[0] - self.dx*Nghost, xlim[1] + self.dx*Nghost, Nx + 2*Nghost)
        self.y = np.linspace(ylim[0] - self.dy*Nghost, ylim[1] + self.dy*Nghost, Ny + 2*Nghost)

        self.x_int = self.x[self.Nghost:self.Nx+self.Nghost]
        self.y_int = self.y[self.Nghost:self.Ny+self.Nghost]

        self.gridX, self.gridY = np.meshgrid(self.x, self.y)



    def fill_grid(self, f):
        self.grid = f(self.gridX, self.gridY)
        self.internal_grid = self.grid[self.Nghost:self.Nx+self.Nghost, self.Nghost:self.Ny+self.Nghost]

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        axp = ax.imshow(self.internal_grid, cmap = 'ocean')
        cb = plt.colorbar(axp,ax=[ax],location='right')
        plt.show()
