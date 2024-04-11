import numpy as np
from grid import Grid1D
import ics
import matplotlib.pyplot as plt
import os
from constants import *
import solver

#Grid
xlim = (0,1)
Nx = 1000
NumGhost = 3
grid = Grid1D(xlim, Nx, NumGhost, NUMQ)

#Time
cfl = .5
t_finish = 0.1
t0 = 0.0
Nt = np.inf

grid.fill_grid(ics.sod_shock_tube)

grid.apply_zero_gradient_bcs()



grid = solver.weno5solve(grid,t0,t_finish,cfl,Nt)
#grid = solver.LF(grid,t0,t_finish,cfl)

grid.plot()