
from onedim.grid import Grid1D
from typing import Callable

class BoundaryConditions:
    
    def __init__(self, grid: Grid1D, type_lo: str, type_hi: str) -> None:

        self.type_lo = type_lo
        self.type_hi = type_hi

        #mutable objects, such as Grid1D are passed by reference in python
        self.grid = grid


    def apply_lo(self, f: Callable[[Grid1D], None]) -> None:    






        
    def apply_hi(self, f):

