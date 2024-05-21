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
            print("Unexpected Variable")
            exit()
    return ics


def shu_osher_shock_tube(a_x, a_var):

    #http://www.ttctech.com/Samples/shockwave/shockwave.htm
    ics = np.zeros_like(a_x)

    shock_pos = 1/8

    for i in range(len(a_x)):
        if a_var == RHOCOMP:
            if a_x[i] < shock_pos:
                ics[i] = 3.857143
            else:
                ics[i] = 1 + 0.2 * np.sin(8*a_x[i])

        elif a_var == UCOMP:
            if a_x[i] < shock_pos:
                ics[i] = 2.629369
            else:
                ics[i] = 0.0


        elif a_var == PCOMP:
            if a_x[i] < shock_pos:
                ics[i] = 10.33333
            else:
                ics[i] = 1.0

        else:
            print("Unexpected Variable")
            exit()
    return ics


