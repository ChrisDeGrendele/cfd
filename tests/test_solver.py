# import unittest
# from src.grid import NDGrid

# class TestNDGrid(unittest.TestCase):

#     def test_grid_initialization(self):
#         grid = NDGrid(dx=1.0, dy=2.0, dz=3.0, Nx=10, Ny=5, Nz=2, ghost_cells=1)
#         self.assertEqual(grid.dx, 1.0)
#         self.assertEqual(grid.dy, 2.0)
#         self.assertEqual(grid.dz, 3.0)
#         self.assertEqual(grid.Nx, 10)
#         self.assertEqual(grid.Ny, 5)
#         self.assertEqual(grid.Nz, 2)
#         self.assertEqual(grid.ghost_cells, 1)
#         self.assertEqual(grid.get_dimensions(), 3)

#     def test_grid_points_generation(self):
#         grid = NDGrid(dx=1.0, Nx=2, ghost_cells=1)
#         expected_points = [(-1.0,), (0.0,), (1.0,), (2.0,)]
#         self.assertEqual(grid.get_grid_points(), expected_points)

#     def test_ghost_cells_handling(self):
#         grid = NDGrid(dx=1.0, Nx=3, ghost_cells=2)
#         # Expect 3 internal points + 4 ghost points (2 on each side)
#         expected_number_of_points = 3 + 4
#         self.assertEqual(len(grid.get_grid_points()), expected_number_of_points)

# # Additional tests can be added here to cover more scenarios

# if __name__ == '__main__':
#     unittest.main()
