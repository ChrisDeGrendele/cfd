import numpy as np
from constants import *


def prim_to_cons(a_prim):
    cons = np.zeros_like(a_prim)

    cons[RHOCOMP] = a_prim[RHOCOMP]
    cons[MCOMP] = a_prim[RHOCOMP] * a_prim[UCOMP]
    E = a_prim[PCOMP] / ((gamma - 1) * a_prim[RHOCOMP]) + 0.5 * a_prim[UCOMP] ** 2
    cons[ECOMP] = E * a_prim[RHOCOMP]

    return cons


def prim_to_cons_var(rho, u, P):
    mass = rho
    mom = rho * u
    E = P / ((gamma - 1) * rho) + 0.5 * u**2
    energy = E * rho

    return mass, mom, energy


def prim_to_cons_var2D(rho, u, v, P):
    mass = rho
    mom_x = rho * u
    mom_y = rho * v
    kinetic_energy = 0.5 * rho * (u**2 + v**2)
    internal_energy = P / (gamma - 1)
    energy = kinetic_energy + internal_energy

    return mass, mom_x, mom_y, energy


def cons_to_prim(a_cons):
    prim = np.zeros_like(a_cons)

    prim[RHOCOMP] = a_cons[RHOCOMP]
    prim[UCOMP] = a_cons[MCOMP] / a_cons[RHOCOMP]
    prim[PCOMP] = (gamma - 1) * (
        a_cons[ECOMP] - 0.5 * a_cons[RHOCOMP] * prim[UCOMP] ** 2
    )

    return prim


def cons_to_prim_var(rho, mom, energy):
    rho = rho
    u = mom / rho
    P = (gamma - 1) * (energy - 0.5 * rho * u**2)

    return rho, u, P


def flux(a_grid_prim):
    """
    Compute the flux for the Euler equations
    """
    a_grid_prim.assert_variable_type("prim")

    a_prim = a_grid_prim.grid

    flux = np.zeros_like(a_prim)

    flux[RHOCOMP, :] = a_prim[RHOCOMP] * a_prim[UCOMP]
    flux[UCOMP, :] = a_prim[RHOCOMP] * a_prim[UCOMP] ** 2 + a_prim[PCOMP]

    e = a_prim[PCOMP] / ((gamma - 1) * a_prim[RHOCOMP])
    E = a_prim[RHOCOMP] * (e + 0.5 * a_prim[RHOCOMP] * a_prim[UCOMP] ** 2)
    flux[PCOMP, :] = (E + a_prim[PCOMP]) * a_prim[UCOMP]

    return flux


def flux_var(rho, u, P):
    """
    Compute the flux for the Euler equations
    """

    rhoFlux = rho * u
    uFlux = rho * u**2 + P
    e = P / ((gamma - 1) * rho)
    E = rho * e + 0.5 * rho * u**2
    pFlux = (E + P) * u

    return np.array([rhoFlux, uFlux, pFlux])


def get_max_speed(a_grid):
    if a_grid.variables == "prim":
        return np.max(a_grid.grid[UCOMP])
    elif a_grid.variables == "cons":
        return np.max(a_grid.grid[MCOMP] / a_grid.grid[RHOCOMP])
    else:
        print("unsupported")
        exit()


def prim_to_char(prim_vars):
    num_points = prim_vars.shape[1]

    # Initialize the array for characteristic variables
    char_vars = np.zeros_like(prim_vars)

    # Initialize a list or 3D array to store the right eigenvectors for each point
    right_eigenvectors_list = []

    for i in range(num_points):
        rho = prim_vars[RHOCOMP, i]
        u = prim_vars[UCOMP, i]
        p = prim_vars[PCOMP, i]

        A = np.array(
            [
                [0, 1, 0],
                [-(u**2) + gamma * p / rho, 2 * u, gamma - 1],
                [
                    -(gamma - 1) * u**3 + gamma * u * p / rho,
                    gamma * u**2 - 1.5 * (gamma - 1) * u**2,
                    gamma * u,
                ],
            ]
        )

        eigenvalues, right_eigenvectors = np.linalg.eig(A)

        # Store the right eigenvectors for each point
        right_eigenvectors_list.append(right_eigenvectors)

        # Inverse of the right eigenvectors matrix
        R_inv = np.linalg.inv(right_eigenvectors)

        U = np.array([rho, u, p])
        char_vars[:, i] = R_inv @ U

    return char_vars, np.array(right_eigenvectors_list)


def char_to_prim(char_vars, right_eigenvectors_list):
    # Assuming char_vars is a 2D array with shape (3, N) where N is the number of spatial points
    num_points = char_vars.shape[1]

    # Initialize the array for primitive variables
    prim_vars = np.zeros_like(char_vars)

    for i in range(num_points):
        # Get the right eigenvectors for the current point
        right_eigenvectors = right_eigenvectors_list[i]

        # Transform characteristic variables back to primitive variables at each point
        prim_vars[:, i] = right_eigenvectors @ char_vars[:, i]

    return prim_vars
