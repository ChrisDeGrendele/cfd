from onedim.constants import *
import numpy as np


def sod_shock_tube(a_x, a_var):
    ics = np.zeros_like(a_x)

    for i in range(len(a_x)):
        if a_var == RHOCOMP:
            if a_x[i] < 0.5:
                ics[i] = 1
            else:
                ics[i] = 0.125

        elif a_var == PCOMP:
            if a_x[i] < 0.5:
                ics[i] = 1
            else:
                ics[i] = 0.1

        elif a_var == UCOMP:
            if a_x[i] < 0.5:
                ics[i] = 0
            else:
                ics[i] = 0

        else:
            print("Unea_xpected Variable")
            exit()
    return ics


def shu_osher_shock_tube(a_x, a_var):
    ics = np.zeros_like(a_x)

    # Define the shock position in the scaled domain
    shock_pos = 0.1

    for i in range(len(a_x)):
        if a_var == RHOCOMP:
            if a_x[i] < shock_pos:
                ics[i] = 3.857143
            else:
                ics[i] = 1 + 0.2 * np.sin(5 * (a_x[i] - shock_pos))

        elif a_var == PCOMP:
            if a_x[i] < shock_pos:
                ics[i] = 10.33333
            else:
                ics[i] = 1.0

        elif a_var == UCOMP:
            if a_x[i] < shock_pos:
                ics[i] = 2.629369
            else:
                ics[i] = 0.0

        else:
            print("Unexpected Variable")
            exit()
    return ics


# def rieman2D():
#     # Initialize arrays for density, pressure, and velocities
#     density = np.zeros((ny, nx))
#     pressure = np.zeros((ny, nx))
#     velocity_x = np.zeros((ny, nx))
#     velocity_y = np.zeros((ny, nx))

#     # Define the conditions for each quadrant
#     # Top-left quadrant (Quadrant 1)
#     density[:ny//2, :nx//2] = 1.0
#     pressure[:ny//2, :nx//2] = 1.0
#     velocity_x[:ny//2, :nx//2] = 0.0
#     velocity_y[:ny//2, :nx//2] = 0.0

#     # Top-right quadrant (Quadrant 2)
#     density[:ny//2, nx//2:] = 0.125
#     pressure[:ny//2, nx//2:] = 0.1
#     velocity_x[:ny//2, nx//2:] = 0.0
#     velocity_y[:ny//2, nx//2:] = 0.0

#     # Bottom-left quadrant (Quadrant 3)
#     density[ny//2:, :nx//2] = 1.0
#     pressure[ny//2:, :nx//2] = 0.1
#     velocity_x[ny//2:, :nx//2] = 0.0
#     velocity_y[ny//2:, :nx//2] = 0.0

#     # Bottom-right quadrant (Quadrant 4)
#     density[ny//2:, nx//2:] = 0.125
#     pressure[ny//2:, nx//2:] = 1.0
#     velocity_x[ny//2:, nx//2:] = 0.0
#     velocity_y[ny//2:, nx//2:] = 0.0
