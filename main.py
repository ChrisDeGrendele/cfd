from src.grid import Grid1D
import numpy as np


nt = 100
nx = 100
xlim = (0, 1)
nghost = 2
numVariables = 1
ics = np.sin

g = Grid1D(xlim, nx, nghost, numVariables)
g.fill_grid(ics, 0)


# for timestep in range(nt):
