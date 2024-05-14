from constants import *
import numpy as np


def hllc_flux(q_left, q_right, gamma):
    S_L = min(q_left[UCOMP], q_right[UCOMP]) - max(
        np.sqrt(gamma * q_left[PCOMP] / q_left[RHOCOMP]),
        np.sqrt(gamma * q_right[PCOMP] / q_right[RHOCOMP]),
    )
    S_R = max(q_left[UCOMP], q_right[UCOMP]) + max(
        np.sqrt(gamma * q_left[PCOMP] / q_left[RHOCOMP]),
        np.sqrt(gamma * q_right[PCOMP] / q_right[RHOCOMP]),
    )
    rho_L = q_left[RHOCOMP]
    rho_R = q_right[RHOCOMP]
    u_L = q_left[UCOMP] / rho_L
    u_R = q_right[UCOMP] / rho_R
    p_L = q_left[PCOMP]
    p_R = q_right[PCOMP]

    # Calculate S_M using the pressure and velocity in the left and right states
    S_M = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (
        rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    )

    # Calculate the fluxes for left, right, and star region
    # Here, f(q) would be your flux function for the conserved quantities
    f_L = flux_function(q_left)
    f_R = flux_function(q_right)

    # Compute the intermediate state q_star
    q_star_L = q_left + (S_M - S_L) * (hll_flux(q_left, S_L, S_M) - f_L)
    q_star_R = q_right + (S_M - S_R) * (hll_flux(q_right, S_R, S_M) - f_R)

    # Compute the HLLC flux
    if 0 <= S_L:
        return f_L
    elif S_L <= 0 <= S_M:
        return hll_flux(q_left, S_L, S_M)
    elif S_M <= 0 <= S_R:
        return hll_flux(q_right, S_R, S_M)
    else:  # 0 >= S_R
        return f_R


def hll_flux(q, S_edge, S_M):
    # Compute HLL flux for the edge
    f = flux_function(q)
    return f + S_edge * (q_star(q, S_edge, S_M) - q)


def q_star(q, S_edge, S_M):
    # Compute the intermediate state q_star for HLL
    return q  # Placeholder, actual computation needed


def flux_function(q):
    # Compute the physical flux for the conserved quantities
    return q  # Placeholder, actual computation needed
