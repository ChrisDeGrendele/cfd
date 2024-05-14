from constants2D import *
import numpy as np


def rimeann2D(meshX, meshY, var):
    values = np.zeros_like(meshX)

    # ref: https://www.csun.edu/~jb715473/examples/euler2d.htm

    # Define conditions for each quadrant based on position
    # Quadrant 1: top-left
    condition1 = (meshX < 0.5) & (meshY >= 0.5)
    # Quadrant 2: top-right
    condition2 = (meshX >= 0.5) & (meshY >= 0.5)
    # Quadrant 3: bottom-left
    condition3 = (meshX < 0.5) & (meshY < 0.5)
    # Quadrant 4: bottom-right
    condition4 = (meshX >= 0.5) & (meshY < 0.5)

    # Apply conditions based on the variable index
    if var == RHOCOMP:  # Density
        values[condition1] = 0.5323
        values[condition2] = 1.5
        values[condition3] = 0.138
        values[condition4] = 0.5323
    elif var == UCOMP:  # Velocity in x-direction
        values[condition1] = 1.206
        values[condition2] = 0.0
        values[condition3] = 1.206
        values[condition4] = 0.0
    elif var == VCOMP:  # Velocity in y-direction
        values[condition1] = 0.0
        values[condition2] = 0.0
        values[condition3] = 1.206
        values[condition4] = 1.206
    elif var == PCOMP:  # Pressure
        values[condition1] = 0.3
        values[condition2] = 1.5
        values[condition3] = 0.029
        values[condition4] = 0.3

    return values
