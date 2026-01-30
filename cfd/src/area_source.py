"""
Area source term for quasi-1D compressible flow.

In quasi-1D flow with variable area, the momentum equation requires
an explicit source term to account for pressure forces on the
changing cross-section.
"""

import numpy as np
from .sources import SourceTerm
from .state import FlowState
from .mesh import Mesh1D


class AreaSourceTerm(SourceTerm):
    """
    Pressure-area source term for quasi-1D flow.

    The momentum equation in quasi-1D flow includes a source term:
        S_momentum = p * dA/dx

    This represents the pressure force acting on the varying cross-section.

    In conservation form:
        d/dt(A*ρu) + d/dx(A*(ρu² + p)) = p * dA/dx

    For cell-centered finite volume, we compute dA/dx at cell centers.
    """

    def compute(self, state: FlowState, mesh: Mesh1D) -> np.ndarray:
        """
        Compute the area source term.

        Args:
            state: Current flow state
            mesh: Computational mesh

        Returns:
            Source term array of shape (n_vars, n_cells)
            Only momentum equation (index 1) has non-zero source
        """
        n_scalars = state.Y.shape[0]
        n_vars = 3 + n_scalars
        S = np.zeros((n_vars, mesh.n_cells))

        # Compute dA/dx at cell centers using centered difference
        # dA/dx ≈ (A_{i+1} - A_{i-1}) / (x_{i+1} - x_{i-1})
        dA_dx = (mesh.A_faces[1:] - mesh.A_faces[:-1]) / mesh.dx

        # Momentum source: S = (p/A) * dA/dx
        # For quasi-1D flow solving per-volume variables (ρ, ρu, ρE),
        # the source term is (p/A) * dA/dx, not p * dA/dx
        # Note: This goes into the momentum equation (index 1)
        S[1, :] = state.p * dA_dx / mesh.A_cells

        return S
