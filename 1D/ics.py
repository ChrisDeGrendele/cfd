from constants import *
import numpy as np

def sod_shock_tube(a_x, a_var):

    ics = np.zeros_like(a_x)

    for i in range(len(a_x)):

        if a_var == RHOCOMP:
            if a_x[i] < 0.5:
                ics[i] = 1
            else:
                ics[i] = .125

        elif a_var == PCOMP:
            if a_x[i] < 0.5:
                ics[i] = 1
            else:
                ics[i] = .1

        elif a_var == UCOMP:
            if a_x[i] < 0.5:
                ics[i] = 0
            else:
                ics[i] = 0

        else:
            print("Unea_xpected Variable")
            ea_xit()
    return ics

