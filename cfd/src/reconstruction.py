"""
Spatial reconstruction schemes for 2nd order accuracy.
"""

import numpy as np
from typing import Tuple


def reconstruct_first_order(U: np.ndarray, n_ghost: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    First-order reconstruction (piecewise constant) - most stable.

    Simply uses cell-centered values at faces (no slope reconstruction).
    Very robust, but only 1st order accurate.

    Args:
        U: Conservative variables with ghost cells (n_vars, n_cells + 2*n_ghost)
        n_ghost: Number of ghost cells on each side

    Returns:
        UL: Left states at each face (n_vars, n_faces)
        UR: Right states at each face (n_vars, n_faces)
    """
    n_vars, n_total = U.shape
    n_cells = n_total - 2 * n_ghost
    n_faces = n_cells + 1

    UL = np.zeros((n_vars, n_faces))
    UR = np.zeros((n_vars, n_faces))

    # For first order: left state = left cell value, right state = right cell value
    # No extrapolation, just use cell-centered values directly
    for i in range(n_faces):
        iL = n_ghost + i - 1  # Left cell
        iR = n_ghost + i      # Right cell
        UL[:, i] = U[:, iL]
        UR[:, i] = U[:, iR]

    return UL, UR


def reconstruct_muscl(U: np.ndarray, n_ghost: int = 1, limiter_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    MUSCL reconstruction with minmod limiter for 2nd order accuracy.

    Fully vectorized implementation - NO LOOPS for maximum performance.

    Args:
        U: Conservative variables with ghost cells (n_vars, n_cells + 2*n_ghost)
        n_ghost: Number of ghost cells on each side
        limiter_factor: Factor to reduce reconstruction (0=first order, 1=full MUSCL)

    Returns:
        UL: Left states at each face (n_vars, n_faces)
        UR: Right states at each face (n_vars, n_faces)
    """
    n_vars, n_total = U.shape
    n_cells = n_total - 2 * n_ghost
    n_faces = n_cells + 1

    # Compute all slopes at once (vectorized across all cells)
    # For interior cells (indices 1 to n_total-2)
    dL = U[:, 1:-1] - U[:, :-2]  # Backward differences
    dR = U[:, 2:] - U[:, 1:-1]   # Forward differences

    # Vectorized minmod limiter
    same_sign = (dL * dR) > 0
    abs_dL = np.abs(dL)
    abs_dR = np.abs(dR)
    use_dL = abs_dL < abs_dR

    slopes = np.zeros_like(dL)
    slopes[same_sign & use_dL] = dL[same_sign & use_dL]
    slopes[same_sign & ~use_dL] = dR[same_sign & ~use_dL]

    # Apply limiter factor to reduce reconstruction strength
    slopes *= limiter_factor

    # Reconstruct at ALL faces at once (fully vectorized, no loop!)
    UL = np.zeros((n_vars, n_faces))
    UR = np.zeros((n_vars, n_faces))

    # Left boundary face (face 0): left cell is ghost cell (index 0)
    UL[:, 0] = U[:, n_ghost - 1]  # Ghost cell value

    # Interior faces (1 to n_faces-2): reconstruct from interior cells
    UL[:, 1:n_faces-1] = U[:, n_ghost:n_ghost + n_cells - 1] + 0.5 * slopes[:, n_ghost - 1:n_ghost + n_cells - 2]

    # Right boundary face (face n_faces-1): left cell is last interior cell
    UL[:, n_faces-1] = U[:, n_ghost + n_cells - 1] + 0.5 * slopes[:, n_ghost + n_cells - 2]

    # Right states: extrapolate from right cells
    UR[:, 0] = U[:, n_ghost] - 0.5 * slopes[:, n_ghost - 1]
    UR[:, 1:n_faces-1] = U[:, n_ghost + 1:n_ghost + n_cells] - 0.5 * slopes[:, n_ghost:n_ghost + n_cells - 1]
    UR[:, n_faces-1] = U[:, n_ghost + n_cells]  # Ghost cell value

    return UL, UR
