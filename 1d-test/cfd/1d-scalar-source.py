"""
1D Compressible Flow Solver with Passive Scalar Transport
==========================================================

Solves the quasi-1D Euler equations with variable area and arbitrary passive scalars.
Uses finite volume method with 2nd order spatial reconstruction and RK4 time integration.

Author: Generated for contrails project
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod


# =============================================================================
# Gas Properties
# =============================================================================

@dataclass
class GasProperties:
    """Thermodynamic properties for a calorically perfect gas."""
    gamma: float = 1.4          # Ratio of specific heats
    R: float = 287.0            # Specific gas constant [J/(kg·K)]

    @property
    def cp(self) -> float:
        """Specific heat at constant pressure [J/(kg·K)]."""
        return self.gamma * self.R / (self.gamma - 1)

    @property
    def cv(self) -> float:
        """Specific heat at constant volume [J/(kg·K)]."""
        return self.R / (self.gamma - 1)


# =============================================================================
# Flow State
# =============================================================================

@dataclass
class FlowState:
    """
    Represents the flow state at a point or cell.

    Primitive variables: rho, u, p, T, and scalar mass fractions Y[:]
    Conservative variables: rho, rho*u, rho*E, rho*Y[:]
    """
    rho: np.ndarray     # Density [kg/m³]
    u: np.ndarray       # Velocity [m/s]
    p: np.ndarray       # Pressure [Pa]
    Y: np.ndarray       # Scalar mass fractions [n_scalars, n_cells]
    gas: GasProperties

    @property
    def T(self) -> np.ndarray:
        """Temperature from ideal gas law [K]."""
        return self.p / (self.rho * self.gas.R)

    @property
    def E(self) -> np.ndarray:
        """Total specific energy [J/kg]."""
        return self.p / (self.rho * (self.gas.gamma - 1)) + 0.5 * self.u**2

    @property
    def H(self) -> np.ndarray:
        """Total specific enthalpy [J/kg]."""
        return self.E + self.p / self.rho

    @property
    def a(self) -> np.ndarray:
        """Speed of sound [m/s]."""
        return np.sqrt(self.gas.gamma * self.p / self.rho)

    @property
    def M(self) -> np.ndarray:
        """Mach number."""
        return self.u / self.a

    def to_conservative(self) -> np.ndarray:
        """
        Convert to conservative variables.

        Returns:
            U: Array of shape (3 + n_scalars, n_cells)
               [rho, rho*u, rho*E, rho*Y_0, rho*Y_1, ...]
        """
        n_scalars = self.Y.shape[0]
        n_cells = len(self.rho)
        U = np.zeros((3 + n_scalars, n_cells))

        U[0] = self.rho
        U[1] = self.rho * self.u
        U[2] = self.rho * self.E

        for i in range(n_scalars):
            U[3 + i] = self.rho * self.Y[i]

        return U

    @classmethod
    def from_conservative(cls, U: np.ndarray, gas: GasProperties) -> 'FlowState':
        """
        Create FlowState from conservative variables.

        Args:
            U: Conservative variables [rho, rho*u, rho*E, rho*Y_0, ...]
            gas: Gas properties
        """
        rho = U[0]
        u = U[1] / rho
        rhoE = U[2]

        # p = (gamma - 1) * (rhoE - 0.5*rho*u²)
        p = (gas.gamma - 1) * (rhoE - 0.5 * rho * u**2)

        # Scalars
        n_scalars = U.shape[0] - 3
        Y = np.zeros((n_scalars, len(rho)))
        for i in range(n_scalars):
            Y[i] = U[3 + i] / rho

        return cls(rho=rho, u=u, p=p, Y=Y, gas=gas)


# =============================================================================
# Mesh
# =============================================================================

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


# =============================================================================
# Source Terms (Extensible Architecture)
# =============================================================================

class SourceTerm(ABC):
    """Abstract base class for source terms."""

    @abstractmethod
    def compute(self, state: FlowState, mesh: Mesh1D) -> np.ndarray:
        """
        Compute source term contribution.

        Args:
            state: Current flow state
            mesh: Computational mesh

        Returns:
            Source term array of shape (n_vars, n_cells)
        """
        pass


class AreaSourceTerm(SourceTerm):
    """
    Source term due to variable area (quasi-1D formulation).

    For quasi-1D flow: dA/dx contributes a pressure source to momentum.
    This is handled automatically in the finite volume formulation through
    the flux differences, so this class returns zero.

    Note: In finite volume form with A*F fluxes, the area variation is
    captured through the flux balance. This placeholder is here for
    extensibility if other formulations are needed.
    """

    def compute(self, state: FlowState, mesh: Mesh1D) -> np.ndarray:
        n_vars = 3 + state.Y.shape[0]
        return np.zeros((n_vars, mesh.n_cells))


class ScalarSourceTerm(SourceTerm):
    """
    Source term for passive scalars.

    The user provides a function that computes scalar source rates.
    """

    def __init__(self, source_func: Callable[[FlowState, Mesh1D], np.ndarray]):
        """
        Args:
            source_func: Function(state, mesh) -> sources of shape (n_scalars, n_cells)
                         Returns the source term S in d(rho*Y)/dt = S
        """
        self.source_func = source_func

    def compute(self, state: FlowState, mesh: Mesh1D) -> np.ndarray:
        n_scalars = state.Y.shape[0]
        n_vars = 3 + n_scalars

        S = np.zeros((n_vars, mesh.n_cells))

        # Compute scalar sources
        scalar_sources = self.source_func(state, mesh)

        for i in range(n_scalars):
            S[3 + i] = scalar_sources[i]

        return S


class CompositeSourceTerm(SourceTerm):
    """Combines multiple source terms."""

    def __init__(self, sources: List[SourceTerm] = None):
        self.sources = sources if sources is not None else []

    def add(self, source: SourceTerm):
        """Add a source term to the composite."""
        self.sources.append(source)

    def compute(self, state: FlowState, mesh: Mesh1D) -> np.ndarray:
        if not self.sources:
            n_vars = 3 + state.Y.shape[0]
            return np.zeros((n_vars, mesh.n_cells))

        total = self.sources[0].compute(state, mesh)
        for source in self.sources[1:]:
            total += source.compute(state, mesh)
        return total


# =============================================================================
# Flux Computation (HLLC Scheme)
# =============================================================================

class FluxScheme(ABC):
    """Abstract base class for numerical flux schemes."""

    @abstractmethod
    def compute_flux(self, UL: np.ndarray, UR: np.ndarray,
                     gas: GasProperties) -> np.ndarray:
        """
        Compute numerical flux at a face.

        Args:
            UL: Left state conservative variables (n_vars,)
            UR: Right state conservative variables (n_vars,)
            gas: Gas properties

        Returns:
            Numerical flux (n_vars,)
        """
        pass


class HLLCFlux(FluxScheme):
    """
    HLLC approximate Riemann solver.

    A robust and accurate flux scheme that resolves contact discontinuities.
    """

    def compute_flux(self, UL: np.ndarray, UR: np.ndarray,
                     gas: GasProperties) -> np.ndarray:
        gamma = gas.gamma
        n_vars = len(UL)
        n_scalars = n_vars - 3

        # Extract primitives from left state
        rhoL = UL[0]
        uL = UL[1] / rhoL
        EL = UL[2] / rhoL
        pL = (gamma - 1) * (UL[2] - 0.5 * rhoL * uL**2)
        aL = np.sqrt(gamma * pL / rhoL)
        HL = EL + pL / rhoL

        # Extract primitives from right state
        rhoR = UR[0]
        uR = UR[1] / rhoR
        ER = UR[2] / rhoR
        pR = (gamma - 1) * (UR[2] - 0.5 * rhoR * uR**2)
        aR = np.sqrt(gamma * pR / rhoR)
        HR = ER + pR / rhoR

        # Roe averages for wave speed estimates
        sqrt_rhoL = np.sqrt(rhoL)
        sqrt_rhoR = np.sqrt(rhoR)
        denom = sqrt_rhoL + sqrt_rhoR

        u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) / denom
        H_roe = (sqrt_rhoL * HL + sqrt_rhoR * HR) / denom
        a_roe = np.sqrt((gamma - 1) * (H_roe - 0.5 * u_roe**2))

        # Wave speed estimates
        SL = min(uL - aL, u_roe - a_roe)
        SR = max(uR + aR, u_roe + a_roe)

        # Contact wave speed
        SM = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / \
             (rhoL * (SL - uL) - rhoR * (SR - uR))

        # Physical fluxes
        FL = np.zeros(n_vars)
        FL[0] = rhoL * uL
        FL[1] = rhoL * uL**2 + pL
        FL[2] = rhoL * uL * HL
        for i in range(n_scalars):
            YL_i = UL[3 + i] / rhoL
            FL[3 + i] = rhoL * uL * YL_i

        FR = np.zeros(n_vars)
        FR[0] = rhoR * uR
        FR[1] = rhoR * uR**2 + pR
        FR[2] = rhoR * uR * HR
        for i in range(n_scalars):
            YR_i = UR[3 + i] / rhoR
            FR[3 + i] = rhoR * uR * YR_i

        # HLLC flux
        if SL >= 0:
            return FL
        elif SR <= 0:
            return FR
        elif SM >= 0:
            # Left star state
            pstar = pL + rhoL * (SL - uL) * (SM - uL)
            coeff = rhoL * (SL - uL) / (SL - SM)

            UstarL = np.zeros(n_vars)
            UstarL[0] = coeff
            UstarL[1] = coeff * SM
            UstarL[2] = coeff * (EL + (SM - uL) * (SM + pL / (rhoL * (SL - uL))))
            for i in range(n_scalars):
                YL_i = UL[3 + i] / rhoL
                UstarL[3 + i] = coeff * YL_i

            return FL + SL * (UstarL - UL)
        else:
            # Right star state
            pstar = pR + rhoR * (SR - uR) * (SM - uR)
            coeff = rhoR * (SR - uR) / (SR - SM)

            UstarR = np.zeros(n_vars)
            UstarR[0] = coeff
            UstarR[1] = coeff * SM
            UstarR[2] = coeff * (ER + (SM - uR) * (SM + pR / (rhoR * (SR - uR))))
            for i in range(n_scalars):
                YR_i = UR[3 + i] / rhoR
                UstarR[3 + i] = coeff * YR_i

            return FR + SR * (UstarR - UR)


# =============================================================================
# Reconstruction (2nd Order with Slope Limiting)
# =============================================================================

def minmod(a: float, b: float) -> float:
    """Minmod limiter."""
    if a * b <= 0:
        return 0.0
    elif abs(a) < abs(b):
        return a
    else:
        return b


def reconstruct_muscl(U: np.ndarray, n_ghost: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    MUSCL reconstruction with minmod limiter for 2nd order accuracy.

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

    for var in range(n_vars):
        for i in range(n_faces):
            # Cell indices (accounting for ghost cells)
            iL = n_ghost + i - 1  # Left cell
            iR = n_ghost + i      # Right cell

            # Slopes
            if iL >= 1:
                dL = U[var, iL] - U[var, iL - 1]
                dL_plus = U[var, iL + 1] - U[var, iL]
                slopeL = minmod(dL, dL_plus)
            else:
                slopeL = 0.0

            if iR < n_total - 1:
                dR = U[var, iR] - U[var, iR - 1]
                dR_plus = U[var, iR + 1] - U[var, iR]
                slopeR = minmod(dR, dR_plus)
            else:
                slopeR = 0.0

            # Reconstructed states at face
            UL[var, i] = U[var, iL] + 0.5 * slopeL
            UR[var, i] = U[var, iR] - 0.5 * slopeR

    return UL, UR


# =============================================================================
# Boundary Conditions
# =============================================================================

class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""

    @abstractmethod
    def apply(self, U: np.ndarray, mesh: Mesh1D, gas: GasProperties,
              side: str) -> np.ndarray:
        """
        Apply boundary condition to ghost cells.

        Args:
            U: Conservative variables including ghost cells
            mesh: Computational mesh
            gas: Gas properties
            side: 'left' or 'right'

        Returns:
            Modified U with ghost cells set
        """
        pass


class SubsonicInletBC(BoundaryCondition):
    """
    Subsonic inlet: specify total pressure, total temperature, and scalars.
    Extrapolate velocity (or one wave) from interior.
    """

    def __init__(self, p0: float, T0: float, Y_inlet: np.ndarray):
        """
        Args:
            p0: Total pressure [Pa]
            T0: Total temperature [K]
            Y_inlet: Scalar mass fractions at inlet (n_scalars,)
        """
        self.p0 = p0
        self.T0 = T0
        self.Y_inlet = Y_inlet

    def apply(self, U: np.ndarray, mesh: Mesh1D, gas: GasProperties,
              side: str) -> np.ndarray:
        gamma = gas.gamma

        if side == 'left':
            # Get interior cell velocity
            rho_int = U[0, 1]
            u_int = U[1, 1] / rho_int

            # Use interior Mach number (extrapolated)
            p_int = (gamma - 1) * (U[2, 1] - 0.5 * rho_int * u_int**2)
            a_int = np.sqrt(gamma * p_int / rho_int)
            M = abs(u_int) / a_int

            # Limit Mach number to avoid issues
            M = min(M, 0.99)

            # Isentropic relations
            T = self.T0 / (1 + 0.5 * (gamma - 1) * M**2)
            p = self.p0 / (1 + 0.5 * (gamma - 1) * M**2)**(gamma / (gamma - 1))
            rho = p / (gas.R * T)
            u = M * np.sqrt(gamma * gas.R * T)

            # Set ghost cell
            E = p / (rho * (gamma - 1)) + 0.5 * u**2
            U[0, 0] = rho
            U[1, 0] = rho * u
            U[2, 0] = rho * E

            n_scalars = U.shape[0] - 3
            for i in range(n_scalars):
                U[3 + i, 0] = rho * self.Y_inlet[i]

        return U


class SubsonicOutletBC(BoundaryCondition):
    """
    Subsonic outlet: specify static pressure.
    Extrapolate other variables from interior.
    """

    def __init__(self, p_exit: float):
        """
        Args:
            p_exit: Static pressure at outlet [Pa]
        """
        self.p_exit = p_exit

    def apply(self, U: np.ndarray, mesh: Mesh1D, gas: GasProperties,
              side: str) -> np.ndarray:
        gamma = gas.gamma

        if side == 'right':
            n_total = U.shape[1]

            # Get interior cell values (last real cell)
            rho_int = U[0, -2]
            u_int = U[1, -2] / rho_int

            # Extrapolate density and velocity, impose pressure
            rho = rho_int  # Simple extrapolation
            u = u_int
            p = self.p_exit

            # Compute energy with imposed pressure
            E = p / (rho * (gamma - 1)) + 0.5 * u**2

            # Set ghost cell
            U[0, -1] = rho
            U[1, -1] = rho * u
            U[2, -1] = rho * E

            # Extrapolate scalars
            n_scalars = U.shape[0] - 3
            for i in range(n_scalars):
                Y_int = U[3 + i, -2] / rho_int
                U[3 + i, -1] = rho * Y_int

        return U


class WallBC(BoundaryCondition):
    """
    Inviscid wall (slip): zero normal velocity.
    Reflects the velocity component.
    """

    def apply(self, U: np.ndarray, mesh: Mesh1D, gas: GasProperties,
              side: str) -> np.ndarray:

        if side == 'left':
            # Mirror the first interior cell
            U[0, 0] = U[0, 1]       # rho
            U[1, 0] = -U[1, 1]      # rho*u (reflected)
            U[2, 0] = U[2, 1]       # rho*E

            n_scalars = U.shape[0] - 3
            for i in range(n_scalars):
                U[3 + i, 0] = U[3 + i, 1]

        elif side == 'right':
            U[0, -1] = U[0, -2]
            U[1, -1] = -U[1, -2]
            U[2, -1] = U[2, -2]

            n_scalars = U.shape[0] - 3
            for i in range(n_scalars):
                U[3 + i, -1] = U[3 + i, -2]

        return U


# =============================================================================
# Time Integration (RK4)
# =============================================================================

def compute_rhs(U: np.ndarray, mesh: Mesh1D, gas: GasProperties,
                flux_scheme: FluxScheme, source_terms: SourceTerm,
                bc_left: BoundaryCondition, bc_right: BoundaryCondition) -> np.ndarray:
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

    Returns:
        RHS: Time derivative (n_vars, n_cells)
    """
    n_vars = U.shape[0]
    n_cells = mesh.n_cells

    # Apply boundary conditions
    U = bc_left.apply(U.copy(), mesh, gas, 'left')
    U = bc_right.apply(U, mesh, gas, 'right')

    # Reconstruct states at faces
    UL, UR = reconstruct_muscl(U, n_ghost=1)

    # Compute fluxes at all faces
    n_faces = n_cells + 1
    F = np.zeros((n_vars, n_faces))

    for i in range(n_faces):
        F[:, i] = flux_scheme.compute_flux(UL[:, i], UR[:, i], gas)

    # Compute flux differences (finite volume)
    RHS = np.zeros((n_vars, n_cells))

    for i in range(n_cells):
        A_L = mesh.A_faces[i]
        A_R = mesh.A_faces[i + 1]
        V = mesh.vol[i]

        RHS[:, i] = -(A_R * F[:, i + 1] - A_L * F[:, i]) / V

    # Add source terms
    state = FlowState.from_conservative(U[:, 1:-1], gas)  # Interior cells only
    S = source_terms.compute(state, mesh)
    RHS += S

    return RHS


def rk4_step(U: np.ndarray, dt: float, mesh: Mesh1D, gas: GasProperties,
             flux_scheme: FluxScheme, source_terms: SourceTerm,
             bc_left: BoundaryCondition, bc_right: BoundaryCondition) -> np.ndarray:
    """
    Perform one RK4 time step.

    Args:
        U: Conservative variables (interior cells only, n_vars x n_cells)
        dt: Time step
        mesh, gas, flux_scheme, source_terms: Solver components
        bc_left, bc_right: Boundary conditions

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
                          bc_left, bc_right)

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


# =============================================================================
# Solver Class
# =============================================================================

@dataclass
class SolverConfig:
    """Configuration for the 1D compressible flow solver."""
    cfl: float = 0.5
    max_iter: int = 10000
    convergence_tol: float = 1e-8
    print_interval: int = 100
    output_interval: int = 500


class Solver1D:
    """
    1D Compressible Flow Solver with Passive Scalar Transport.

    Features:
    - Quasi-1D Euler equations with variable area
    - Arbitrary number of passive scalars
    - Extensible source term architecture
    - 2nd order MUSCL reconstruction with minmod limiter
    - HLLC flux scheme
    - RK4 time integration
    """

    def __init__(self, mesh: Mesh1D, gas: GasProperties, n_scalars: int = 0,
                 config: SolverConfig = None):
        """
        Initialize the solver.

        Args:
            mesh: Computational mesh
            gas: Gas properties
            n_scalars: Number of passive scalars
            config: Solver configuration
        """
        self.mesh = mesh
        self.gas = gas
        self.n_scalars = n_scalars
        self.n_vars = 3 + n_scalars
        self.config = config if config is not None else SolverConfig()

        # Numerical components
        self.flux_scheme = HLLCFlux()
        self.source_terms = CompositeSourceTerm()

        # Boundary conditions (must be set before solving)
        self.bc_left = None
        self.bc_right = None

        # Solution storage
        self.U = None
        self.time = 0.0
        self.iteration = 0
        self.residual_history = []

    def set_initial_condition(self, state: FlowState):
        """Set the initial flow state."""
        self.U = state.to_conservative()
        self.time = 0.0
        self.iteration = 0

    def set_boundary_conditions(self, bc_left: BoundaryCondition,
                                 bc_right: BoundaryCondition):
        """Set boundary conditions."""
        self.bc_left = bc_left
        self.bc_right = bc_right

    def add_source_term(self, source: SourceTerm):
        """Add a source term to the solver."""
        self.source_terms.add(source)

    def get_state(self) -> FlowState:
        """Get current flow state."""
        return FlowState.from_conservative(self.U, self.gas)

    def step(self) -> float:
        """
        Perform one time step.

        Returns:
            dt: Time step taken
        """
        # Compute adaptive time step
        state = self.get_state()
        dt = compute_timestep(state, self.mesh, self.config.cfl)

        # Store old solution for residual calculation
        U_old = self.U.copy()

        # RK4 step
        self.U = rk4_step(self.U, dt, self.mesh, self.gas,
                         self.flux_scheme, self.source_terms,
                         self.bc_left, self.bc_right)

        # Update time and iteration
        self.time += dt
        self.iteration += 1

        # Compute residual (L2 norm of change in density)
        residual = np.sqrt(np.mean((self.U[0] - U_old[0])**2)) / dt
        self.residual_history.append(residual)

        return dt

    def solve(self, max_time: float = None) -> Dict:
        """
        Run the solver to convergence or maximum time/iterations.

        Args:
            max_time: Maximum simulation time (optional)

        Returns:
            Dictionary with convergence info
        """
        if self.bc_left is None or self.bc_right is None:
            raise ValueError("Boundary conditions must be set before solving")

        if self.U is None:
            raise ValueError("Initial condition must be set before solving")

        print("Starting 1D Compressible Flow Solver")
        print("=" * 50)
        print(f"Cells: {self.mesh.n_cells}, Scalars: {self.n_scalars}")
        print(f"CFL: {self.config.cfl}, Max iterations: {self.config.max_iter}")
        print("=" * 50)

        converged = False

        for _ in range(self.config.max_iter):
            dt = self.step()

            # Check convergence
            if len(self.residual_history) > 10:
                recent_residual = self.residual_history[-1]
                if recent_residual < self.config.convergence_tol:
                    converged = True
                    print(f"\nConverged at iteration {self.iteration}")
                    break

            # Check max time
            if max_time is not None and self.time >= max_time:
                print(f"\nReached maximum time {max_time:.4e} s")
                break

            # Print progress
            if self.iteration % self.config.print_interval == 0:
                state = self.get_state()
                print(f"Iter {self.iteration:6d}, t = {self.time:.4e}, "
                      f"dt = {dt:.4e}, res = {self.residual_history[-1]:.4e}, "
                      f"M_max = {np.max(state.M):.4f}")

        return {
            'converged': converged,
            'iterations': self.iteration,
            'time': self.time,
            'final_residual': self.residual_history[-1] if self.residual_history else None
        }

    def plot_solution(self, filename: str = None):
        """Plot the current solution."""
        state = self.get_state()
        x = self.mesh.x_cells

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f'1D Compressible Flow Solution (t = {self.time:.4e} s, iter = {self.iteration})')

        # Density
        axes[0, 0].plot(x, state.rho, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('x [m]')
        axes[0, 0].set_ylabel('Density [kg/m³]')
        axes[0, 0].set_title('Density')
        axes[0, 0].grid(True)

        # Velocity
        axes[0, 1].plot(x, state.u, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('x [m]')
        axes[0, 1].set_ylabel('Velocity [m/s]')
        axes[0, 1].set_title('Velocity')
        axes[0, 1].grid(True)

        # Pressure
        axes[0, 2].plot(x, state.p / 1000, 'g-', linewidth=2)
        axes[0, 2].set_xlabel('x [m]')
        axes[0, 2].set_ylabel('Pressure [kPa]')
        axes[0, 2].set_title('Pressure')
        axes[0, 2].grid(True)

        # Temperature
        axes[1, 0].plot(x, state.T, 'm-', linewidth=2)
        axes[1, 0].set_xlabel('x [m]')
        axes[1, 0].set_ylabel('Temperature [K]')
        axes[1, 0].set_title('Temperature')
        axes[1, 0].grid(True)

        # Mach number
        axes[1, 1].plot(x, state.M, 'k-', linewidth=2)
        axes[1, 1].set_xlabel('x [m]')
        axes[1, 1].set_ylabel('Mach number')
        axes[1, 1].set_title('Mach Number')
        axes[1, 1].grid(True)

        # Scalars or Area
        if self.n_scalars > 0:
            for i in range(min(self.n_scalars, 5)):  # Plot up to 5 scalars
                axes[1, 2].plot(x, state.Y[i], linewidth=2, label=f'Y_{i}')
            axes[1, 2].set_xlabel('x [m]')
            axes[1, 2].set_ylabel('Mass fraction')
            axes[1, 2].set_title('Scalars')
            axes[1, 2].legend()
        else:
            axes[1, 2].plot(self.mesh.x_faces, self.mesh.A_faces, 'c-', linewidth=2)
            axes[1, 2].set_xlabel('x [m]')
            axes[1, 2].set_ylabel('Area [m²]')
            axes[1, 2].set_title('Nozzle Area')
        axes[1, 2].grid(True)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {filename}")

        plt.show()

    def plot_convergence(self, filename: str = None):
        """Plot convergence history."""
        plt.figure(figsize=(8, 5))
        plt.semilogy(self.residual_history, 'b-', linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('Convergence History')
        plt.grid(True)

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')

        plt.show()


# =============================================================================
# Test Case: Subsonic Converging-Diverging Nozzle
# =============================================================================

def nozzle_area(x: np.ndarray, x_throat: float = 0.5,
                A_inlet: float = 1.0, A_throat: float = 0.8,
                A_exit: float = 1.0) -> np.ndarray:
    """
    Smooth nozzle area distribution (sinusoidal).

    Args:
        x: Position array (normalized 0 to 1)
        x_throat: Throat location
        A_inlet: Inlet area
        A_throat: Throat area
        A_exit: Exit area
    """
    A = np.zeros_like(x)

    # Converging section (inlet to throat)
    mask_conv = x <= x_throat
    A[mask_conv] = A_inlet + (A_throat - A_inlet) * (1 - np.cos(np.pi * x[mask_conv] / x_throat)) / 2

    # Diverging section (throat to exit)
    mask_div = x > x_throat
    xi = (x[mask_div] - x_throat) / (1 - x_throat)
    A[mask_div] = A_throat + (A_exit - A_throat) * (1 - np.cos(np.pi * xi)) / 2

    return A


def run_subsonic_nozzle_test(n_cells: int = 100, n_scalars: int = 2):
    """
    Run a subsonic nozzle flow test case.

    This test case simulates subsonic flow through a converging-diverging nozzle.
    The flow accelerates through the converging section and decelerates through
    the diverging section, remaining subsonic throughout.

    Args:
        n_cells: Number of cells
        n_scalars: Number of passive scalars to track
    """
    print("\n" + "=" * 60)
    print("SUBSONIC NOZZLE FLOW TEST CASE")
    print("=" * 60 + "\n")

    # Gas properties (air)
    gas = GasProperties(gamma=1.4, R=287.0)

    # Nozzle geometry
    x_min, x_max = 0.0, 1.0
    A_inlet, A_throat, A_exit = 1.0, 0.8, 1.0

    area_func = lambda x: nozzle_area(x, x_throat=0.5,
                                       A_inlet=A_inlet,
                                       A_throat=A_throat,
                                       A_exit=A_exit)

    # Create mesh
    mesh = Mesh1D.uniform(x_min, x_max, n_cells, area_func)

    # Create solver
    config = SolverConfig(
        cfl=0.5,
        max_iter=20000,
        convergence_tol=1e-10,
        print_interval=500
    )
    solver = Solver1D(mesh, gas, n_scalars=n_scalars, config=config)

    # Boundary conditions
    p0 = 101325.0       # Total pressure [Pa]
    T0 = 300.0          # Total temperature [K]
    p_exit = 95000.0    # Exit static pressure [Pa] (subsonic back pressure)

    Y_inlet = np.zeros(n_scalars)
    if n_scalars > 0:
        Y_inlet[0] = 0.01   # 1% mass fraction of first scalar
    if n_scalars > 1:
        Y_inlet[1] = 0.005  # 0.5% mass fraction of second scalar

    bc_left = SubsonicInletBC(p0=p0, T0=T0, Y_inlet=Y_inlet)
    bc_right = SubsonicOutletBC(p_exit=p_exit)
    solver.set_boundary_conditions(bc_left, bc_right)

    # Example scalar source term (demonstrates the interface)
    # This simple source creates/destroys scalar based on local temperature
    def example_scalar_source(state: FlowState, mesh: Mesh1D) -> np.ndarray:
        """
        Example scalar source term.

        For demonstration:
        - Scalar 0: Production proportional to (T - T_ref)
        - Scalar 1: Decay proportional to Y_1
        """
        n_scalars = state.Y.shape[0]
        sources = np.zeros((n_scalars, mesh.n_cells))

        T_ref = 290.0  # Reference temperature

        if n_scalars > 0:
            # Production term for scalar 0
            # S = rho * k * (T - T_ref) where k is a rate constant
            k_prod = 1e-4
            sources[0] = state.rho * k_prod * np.maximum(state.T - T_ref, 0)

        if n_scalars > 1:
            # Decay term for scalar 1
            # S = -rho * k * Y_1
            k_decay = 10.0
            sources[1] = -state.rho * k_decay * state.Y[1]

        return sources

    # Add scalar source term
    if n_scalars > 0:
        solver.add_source_term(ScalarSourceTerm(example_scalar_source))

    # Initial condition (uniform flow)
    # Start with approximate inlet conditions
    M_init = 0.3
    T_init = T0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)
    p_init = p0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)**(gas.gamma / (gas.gamma - 1))
    rho_init = p_init / (gas.R * T_init)
    u_init = M_init * np.sqrt(gas.gamma * gas.R * T_init)

    rho = np.full(n_cells, rho_init)
    u = np.full(n_cells, u_init)
    p = np.full(n_cells, p_init)
    Y = np.zeros((n_scalars, n_cells))
    if n_scalars > 0:
        Y[0, :] = Y_inlet[0]
    if n_scalars > 1:
        Y[1, :] = Y_inlet[1]

    initial_state = FlowState(rho=rho, u=u, p=p, Y=Y, gas=gas)
    solver.set_initial_condition(initial_state)

    # Solve
    result = solver.solve()

    print("\n" + "=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)

    state = solver.get_state()
    print(f"\nInlet:  M = {state.M[0]:.4f}, p = {state.p[0]/1000:.2f} kPa, T = {state.T[0]:.1f} K")
    print(f"Throat: M = {state.M[n_cells//2]:.4f}, p = {state.p[n_cells//2]/1000:.2f} kPa, T = {state.T[n_cells//2]:.1f} K")
    print(f"Exit:   M = {state.M[-1]:.4f}, p = {state.p[-1]/1000:.2f} kPa, T = {state.T[-1]:.1f} K")

    if n_scalars > 0:
        print(f"\nScalar Y_0: inlet = {state.Y[0, 0]:.6f}, exit = {state.Y[0, -1]:.6f}")
    if n_scalars > 1:
        print(f"Scalar Y_1: inlet = {state.Y[1, 0]:.6f}, exit = {state.Y[1, -1]:.6f}")

    # Mass flow rate check
    mdot = state.rho * state.u * mesh.A_cells
    print(f"\nMass flow rate: min = {np.min(mdot):.6f}, max = {np.max(mdot):.6f} kg/s")
    print(f"Mass flow variation: {(np.max(mdot) - np.min(mdot)) / np.mean(mdot) * 100:.4f}%")

    # Plot results
    solver.plot_solution()
    solver.plot_convergence()

    return solver


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run the subsonic nozzle test case
    solver = run_subsonic_nozzle_test(n_cells=100, n_scalars=2)
