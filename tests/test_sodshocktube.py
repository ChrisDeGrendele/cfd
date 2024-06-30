import pytest
import numpy as np
from onedim.grid import Grid1D
from onedim.inputs import Inputs
from onedim.simulation import Simulation
from onedim.constants import *

@pytest.fixture
def simulation():
    i = Inputs("problems/sodshocktube.ini")
    i.output_freq = -1 #don't output solution files
    i.make_movie = False
    return Simulation(i)


def test_run(simulation):

    final_grid = simulation.run()
    assert final_grid.variables == "prim"
    rho = final_grid.grid[RHOCOMP,:]
    u = final_grid.grid[UCOMP, :]
    p = final_grid.grid[PCOMP,:]


    loaded_matrix = np.loadtxt("tests/solutions/sod_shock_tube.txt")
    rho_sol = loaded_matrix[RHOCOMP, :]
    u_sol = loaded_matrix[UCOMP, :]
    p_sol = loaded_matrix[PCOMP, :]

    print("rho: ", rho)
    print("rho_sol: ", rho_sol)

    np.testing.assert_almost_equal(rho, rho_sol, decimal=6)
    #np.testing.assert_almost_equal(u, u_sol, decimal=6)
    #np.testing.assert_almost_equal(p, p_sol, decimal=6)


