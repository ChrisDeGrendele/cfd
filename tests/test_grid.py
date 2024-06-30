import pytest
import numpy as np
from onedim.grid import Grid1D

@pytest.fixture
def grid():
    xlim = (0, 1)
    Nx = 10
    Nghost = 2
    num_vars = 3
    return Grid1D(xlim, Nx, Nghost, num_vars)

def test_initialization(grid):
    assert grid.xlim == (0, 1)
    assert grid.Nx == 10
    assert grid.Nghost == 2
    assert grid.num_vars == 3
    assert grid.dx == 1 / 9
    assert len(grid.x) == 14  # Nx + 2 * Nghost
    assert grid.ndim == 1
    assert len(grid.x_int) == 10  # Nx
    assert grid.grid.shape == (3, 14)
    assert grid.variables == "prim"
    assert np.array_equal(grid.shock_mask, np.zeros_like(grid.x))

def test_fill_grid(grid):
    def f(x, var):
        return np.sin(x + var)
    
    grid.fill_grid(f)
    for var in range(grid.num_vars):
        np.testing.assert_allclose(grid.grid[var, grid.Nghost:grid.Nx + grid.Nghost], np.sin(grid.x_int + var))



def test_return_internal_grid(grid):
    def f(x, var):
        return np.sin(x + var)
    grid.fill_grid(f)
    internal_grid = grid.return_internal_grid()
    assert internal_grid.shape == (grid.num_vars, grid.Nx)
    expected = np.array([np.sin(grid.x_int + var) for var in range(grid.num_vars)])
    np.testing.assert_allclose(internal_grid, expected)


def test_transform(grid):
    def transform_function(grid_data):
        return grid_data * 2
    
    grid.transform(transform_function, "transformed")
    assert grid.variables == "transformed"
    assert np.array_equal(grid.grid, transform_function(grid.grid))

def test_set_internal(grid):
    new_internal = np.random.random((grid.num_vars, grid.Nx))
    grid.set_internal(new_internal)
    np.testing.assert_allclose(grid.grid[:, grid.Nghost:grid.Nx + grid.Nghost], new_internal)

def test_set(grid):
    new_grid = np.random.random((grid.num_vars, grid.Nx + 2 * grid.Nghost))
    grid.set(new_grid)
    assert np.array_equal(grid.grid, new_grid)

def test_set_grid(grid):
    new_grid = np.random.random((grid.num_vars, grid.Nx + 2 * grid.Nghost))
    grid.set_grid(new_grid)
    assert np.array_equal(grid.grid, new_grid)

def test_internal_bounds(grid):
    assert grid.internal_bounds() == (grid.Nghost, grid.Nx + grid.Nghost)

def test_assert_variable_type(grid):
    grid.assert_variable_type("prim")
    with pytest.raises(AssertionError):
        grid.assert_variable_type("non-prim")
