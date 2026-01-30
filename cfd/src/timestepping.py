"""
Time integration schemes and timestep computation.
"""

import numpy as np

from .state import FlowState
from .mesh import Mesh1D
from .gas import GasProperties
from .flux import FluxScheme
from .sources import SourceTerm
from .boundary import BoundaryCondition
from .reconstruction import reconstruct_muscl, reconstruct_first_order


def compute_rhs(U: np.ndarray, mesh: Mesh1D, gas: GasProperties,
                flux_scheme: FluxScheme, source_terms: SourceTerm,
                bc_left: BoundaryCondition, bc_right: BoundaryCondition,
                use_first_order: bool = False) -> np.ndarray:
    """
    Compute the right-hand side of dU/dt = RHS.

    Uses finite volume formulation:
        dU/dt = -1/V * (A_R * F_R - A_L * F_L) + S

    Args:
        U: Conservative variables with ghost cells (n_vars, n_cells + 2)
        mesh: Computational mesh
        gas: Gas properties
        flux_scheme: Numerical flux scheme
        source_terms: Source term calculator
        bc_left, bc_right: Boundary conditions
        use_first_order: Use first-order reconstruction for stability

    Returns:
        RHS: Time derivative (n_vars, n_cells)
    """
    # Apply boundary conditions (in-place)
    bc_left.apply(U, mesh, gas, 'left')
    bc_right.apply(U, mesh, gas, 'right')

    # Reconstruct states at faces
    if use_first_order:
        UL, UR = reconstruct_first_order(U, n_ghost=1)
    else:
        UL, UR = reconstruct_muscl(U, n_ghost=1)

    # Compute fluxes at all faces (vectorized)
    F = flux_scheme.compute_flux_vectorized(UL, UR, gas)

    # Vectorized flux difference computation
    F_scaled = F * mesh.A_faces
    RHS = -(F_scaled[:, 1:] - F_scaled[:, :-1]) / mesh.vol

    # Add source terms (only if there are sources)
    if hasattr(source_terms, 'sources') and source_terms.sources:
        state = FlowState.from_conservative(U[:, 1:-1], gas)
        S = source_terms.compute(state, mesh)
        RHS += S

    return RHS


def forward_euler_step(U: np.ndarray, dt: float, mesh: Mesh1D, gas: GasProperties,
                       flux_scheme: FluxScheme, source_terms: SourceTerm,
                       bc_left: BoundaryCondition, bc_right: BoundaryCondition,
                       use_first_order: bool = False) -> np.ndarray:
    """
    Forward Euler time step - simplest and fastest (1 RHS evaluation).

    Good for steady-state problems where accuracy in time is not needed.
    Requires smaller CFL (~0.3-0.5) for stability.

    Args:
        U: Conservative variables (interior cells only, n_vars x n_cells)
        dt: Time step
        mesh, gas, flux_scheme, source_terms: Solver components
        bc_left, bc_right: Boundary conditions
        use_first_order: Use first-order reconstruction for stability

    Returns:
        U_new: Updated conservative variables
    """
    n_vars, n_cells = U.shape

    # Add ghost cells
    U_ghost = np.zeros((n_vars, n_cells + 2))
    U_ghost[:, 1:-1] = U

    # Compute RHS
    rhs_val = compute_rhs(U_ghost, mesh, gas, flux_scheme, source_terms,
                          bc_left, bc_right, use_first_order)

    U_new = U + dt * rhs_val

    return U_new


def rk2_ssp_step(U: np.ndarray, dt: float, mesh: Mesh1D, gas: GasProperties,
                 flux_scheme: FluxScheme, source_terms: SourceTerm,
                 bc_left: BoundaryCondition, bc_right: BoundaryCondition,
                 use_first_order: bool = False) -> np.ndarray:
    """
    2nd-order Strong Stability Preserving (SSP) Runge-Kutta (RK2).

    Twice as fast as RK4 (2 RHS evaluations vs 4).
    Good balance between speed and accuracy for steady-state convergence.
    Stable up to CFL ~0.8.

    Args:
        U: Conservative variables (interior cells only, n_vars x n_cells)
        dt: Time step
        mesh, gas, flux_scheme, source_terms: Solver components
        bc_left, bc_right: Boundary conditions
        use_first_order: Use first-order reconstruction for stability

    Returns:
        U_new: Updated conservative variables
    """
    n_vars, n_cells = U.shape

    def add_ghosts(U_interior):
        """Add ghost cells to interior solution."""
        U_with_ghosts = np.zeros((n_vars, n_cells + 2))
        U_with_ghosts[:, 1:-1] = U_interior
        return U_with_ghosts

    def rhs(U_interior):
        """Compute RHS with ghost cells."""
        U_ghost = add_ghosts(U_interior)
        return compute_rhs(U_ghost, mesh, gas, flux_scheme, source_terms,
                          bc_left, bc_right, use_first_order)

    # RK2-SSP stages
    U1 = U + dt * rhs(U)
    U_new = 0.5 * U + 0.5 * (U1 + dt * rhs(U1))

    return U_new


def rk4_step(U: np.ndarray, dt: float, mesh: Mesh1D, gas: GasProperties,
             flux_scheme: FluxScheme, source_terms: SourceTerm,
             bc_left: BoundaryCondition, bc_right: BoundaryCondition,
             use_first_order: bool = False) -> np.ndarray:
    """
    Perform one RK4 time step.

    Args:
        U: Conservative variables (interior cells only, n_vars x n_cells)
        dt: Time step
        mesh, gas, flux_scheme, source_terms: Solver components
        bc_left, bc_right: Boundary conditions
        use_first_order: Use first-order reconstruction for stability

    Returns:
        U_new: Updated conservative variables
    """
    n_vars, n_cells = U.shape

    def add_ghosts(U_interior):
        """Add ghost cells to interior solution."""
        U_with_ghosts = np.zeros((n_vars, n_cells + 2))
        U_with_ghosts[:, 1:-1] = U_interior
        return U_with_ghosts

    def rhs(U_interior):
        """Compute RHS with ghost cells."""
        U_ghost = add_ghosts(U_interior)
        return compute_rhs(U_ghost, mesh, gas, flux_scheme, source_terms,
                          bc_left, bc_right, use_first_order)

    # RK4 stages
    k1 = rhs(U)
    k2 = rhs(U + 0.5 * dt * k1)
    k3 = rhs(U + 0.5 * dt * k2)
    k4 = rhs(U + dt * k3)

    U_new = U + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return U_new


def compute_timestep(state: FlowState, mesh: Mesh1D, cfl: float) -> float:
    """
    Compute time step based on CFL condition.

    Args:
        state: Current flow state
        mesh: Computational mesh
        cfl: CFL number (typically 0.5-0.9 for stability)

    Returns:
        dt: Time step
    """
    # Maximum wave speed: |u| + a
    wave_speed = np.abs(state.u) + state.a

    # Minimum dt across all cells
    dt_local = mesh.dx / wave_speed
    dt = cfl * np.min(dt_local)

    return dt
