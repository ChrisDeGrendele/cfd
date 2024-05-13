'''
I'm getting some difference in shock location, I'm trying to understand if that is DAS
or something related to the actual scheme.
'''

import numpy as np
import matplotlib.pyplot as plt
from weno5 import weno5_interp

import sys
sys.path.append('/Users/chris/Documents/github/gp-recipe/src_new')
from new_driver import GP_recipe1D, GP_recipe2D
import kernels as kern
import grids as old_grid

Nx = 100
xlim = (0,2)
NumGhost = 2

x = np.linspace(xlim[0], xlim[1], Nx+2*NumGhost+1)
dx = x[1]-x[0]

def sinShock(x):
    if x <  1:
        return np.sin(np.pi * x)  # Sine wave from 0 to 1
    elif x >= 1 and x < 1.5:
        return 0
    else:
        return 1
vectorized_function = np.vectorize(sinShock)
f = vectorized_function(x)

#gprecipe
compGrid = old_grid.Grid1D(xlim, Nx, NumGhost)
compGrid.fill_grid(vectorized_function)
x_predict = compGrid.x_int + compGrid.dx/2

gp_DAS = GP_recipe1D(compGrid, 1, stencil_method="center", high_precision=True)
gp_DAS_v2 = GP_recipe1D(compGrid, 1, stencil_method="center", high_precision=True)
gp_DAS_v3 = GP_recipe1D(compGrid, 1, stencil_method="center", high_precision=True)

DAS_predict = gp_DAS.convert_custom(x_predict, kern.AS, kern.AS)
DAS_v2_predict = gp_DAS_v2.convert_custom(x_predict, kern.DAS_V8, kern.DAS_V8)
DAS_v3_predict = gp_DAS_v3.convert_custom(x_predict, kern.DAS_V9, kern.DAS_V9)



plt.scatter(x, f,s=1.5, label='Data')
plt.plot(x+dx/2, weno5_interp(f, Nx, NumGhost),'-o', label='Weno5')
plt.plot(x_predict, DAS_predict, 's-', label='DAS')  # Triangle markers
plt.plot(x_predict, DAS_v2_predict, '^-', label='DAS V8')  # Square markers
plt.plot(x_predict, DAS_v3_predict, 'd-', label='DAS V9')  # Diamond markers

plt.legend()
plt.show()
