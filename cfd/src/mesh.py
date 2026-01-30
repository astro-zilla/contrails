"""
1D mesh with variable area support for quasi-1D flows.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class Mesh1D:
    """
    1D mesh with variable area support.

    Cell-centered finite volume mesh:
    - x_faces: Face locations (n_cells + 1)
    - x_cells: Cell centers (n_cells)
    - A_faces: Area at faces (n_cells + 1)
    - A_cells: Area at cell centers (n_cells)
    - dx: Cell widths (n_cells)
    - vol: Cell volumes = A * dx (n_cells)
    """
    x_faces: np.ndarray
    A_faces: np.ndarray

    def __post_init__(self):
        self.n_cells = len(self.x_faces) - 1
        self.x_cells = 0.5 * (self.x_faces[:-1] + self.x_faces[1:])
        self.A_cells = 0.5 * (self.A_faces[:-1] + self.A_faces[1:])
        self.dx = self.x_faces[1:] - self.x_faces[:-1]
        self.vol = self.A_cells * self.dx

    @classmethod
    def uniform(cls, x_min: float, x_max: float, n_cells: int,
                area_func: Callable[[np.ndarray], np.ndarray]) -> 'Mesh1D':
        """
        Create a uniform mesh with a given area distribution.

        Args:
            x_min, x_max: Domain bounds
            n_cells: Number of cells
            area_func: Function A(x) returning area at position x
        """
        x_faces = np.linspace(x_min, x_max, n_cells + 1)
        A_faces = area_func(x_faces)
        return cls(x_faces=x_faces, A_faces=A_faces)

