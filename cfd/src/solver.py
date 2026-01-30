"""
Main solver class for 1D compressible flow with passive scalar transport.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict

from .gas import GasProperties
from .state import FlowState
from .mesh import Mesh1D
from .flux import HLLCFlux
from .sources import CompositeSourceTerm, SourceTerm
from .area_source import AreaSourceTerm
from .boundary import BoundaryCondition
from .timestepping import rk4_step, rk2_ssp_step, forward_euler_step, compute_timestep


@dataclass
class SolverConfig:
    """Configuration for the 1D compressible flow solver."""
    cfl: float = 0.5
    max_iter: int = 10000
    convergence_tol: float = 1e-8
    print_interval: int = 100
    output_interval: int = 500
    time_scheme: str = 'rk4'  # Options: 'rk4', 'rk2', 'euler'
    use_local_timestepping: bool = False  # Local time stepping for faster steady-state convergence
    use_first_order: bool = False  # Use first-order reconstruction for stability


class Solver1D:
    """
    1D Compressible Flow Solver with Passive Scalar Transport.

    Features:
    - Quasi-1D Euler equations with variable area
    - Arbitrary number of passive scalars
    - Extensible source term architecture
    - 2nd order MUSCL reconstruction with minmod limiter
    - HLLC flux scheme
    - Multiple time integration schemes: RK4 (most accurate), RK2 (2x faster), Euler (4x faster)
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

        # Automatically add area source term for quasi-1D flow
        # This is the standard geometric source term from Anderson's CFD textbook
        area_variation = np.max(mesh.A_faces) - np.min(mesh.A_faces)
        if area_variation > 1e-10:  # Area is variable
            self.source_terms.add(AreaSourceTerm())

        # Select time integration scheme
        if self.config.time_scheme == 'rk4':
            self.time_step_func = rk4_step
        elif self.config.time_scheme == 'rk2':
            self.time_step_func = rk2_ssp_step
        elif self.config.time_scheme == 'euler':
            self.time_step_func = forward_euler_step
        else:
            raise ValueError(f"Unknown time scheme: {self.config.time_scheme}. "
                           "Options: 'rk4', 'rk2', 'euler'")

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
        # Compute adaptive time step directly from U (avoid FlowState creation)
        rho = self.U[0]
        u = self.U[1] / rho
        p = (self.gas.gamma - 1) * (self.U[2] - 0.5 * rho * u**2)
        a = np.sqrt(self.gas.gamma * p / rho)
        wave_speed = np.abs(u) + a
        dt = self.config.cfl * np.min(self.mesh.dx / wave_speed)

        # Time integration step (using selected scheme and reconstruction order)
        self.U = self.time_step_func(self.U, dt, self.mesh, self.gas,
                                     self.flux_scheme, self.source_terms,
                                     self.bc_left, self.bc_right,
                                     self.config.use_first_order)

        # Update time and iteration
        self.time += dt
        self.iteration += 1

        return dt

    def compute_residual(self, U_old: np.ndarray, dt: float) -> float:
        """Compute residual for convergence check (L2 norm of normalized changes)."""
        # Compute L2 norm of relative changes across all variables
        # This is more robust than just using density
        dU = self.U - U_old

        # Normalize by characteristic values to make residual dimensionless
        U_scale = np.maximum(np.abs(self.U), 1e-10)  # Avoid division by zero

        # Compute normalized residual: sqrt(mean((dU/U)^2)) / dt
        residual = np.sqrt(np.mean((dU / U_scale)**2)) / dt

        return residual

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
        print(f"CFL: {self.config.cfl}, Time scheme: {self.config.time_scheme}")
        print(f"Max iterations: {self.config.max_iter}")
        print("=" * 50)

        converged = False
        U_old = self.U.copy()
        check_interval = 100  # Check convergence every N iterations

        for _ in range(self.config.max_iter):
            dt = self.step()

            # Check convergence periodically (not every iteration)
            if self.iteration % check_interval == 0:
                residual = self.compute_residual(U_old, dt * check_interval)
                self.residual_history.append(residual)
                U_old = self.U.copy()

                if residual < self.config.convergence_tol:
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
                res_str = f"{self.residual_history[-1]:.4e}" if self.residual_history else "N/A"
                print(f"Iter {self.iteration:6d}, t = {self.time:.4e}, "
                      f"dt = {dt:.4e}, res = {res_str}, "
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

        # Compute stagnation properties
        T0 = state.T * (1 + 0.5 * (self.gas.gamma - 1) * state.M**2)
        p0 = state.p * (1 + 0.5 * (self.gas.gamma - 1) * state.M**2)**(self.gas.gamma / (self.gas.gamma - 1))

        # Check if we have ice growth scalars (n, vapor, ice)
        has_ice_growth = self.n_scalars >= 3

        if has_ice_growth:
            # Import saturation pressure functions
            from pathlib import Path
            import sys
            scripts_path = Path(__file__).parent.parent / 'scripts'
            if str(scripts_path) not in sys.path:
                sys.path.insert(0, str(scripts_path))
            from ice_growth_source import psat_ice, psat_water, M_w, R

            # Compute vapor pressures
            rho_vapor = state.Y[1] * state.rho  # Vapor density [kg/m³]
            p_vapor = rho_vapor * R / M_w * state.T  # Partial pressure [Pa]
            p_sat_ice_vals = psat_ice(state.T)
            p_sat_water_vals = psat_water(state.T)

            # Create 3x3 grid with vapor pressure plots
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            fig.suptitle(f'1D Contrail Simulation (t = {self.time:.4e} s, iter = {self.iteration})')
        else:
            # Standard 3x2 grid without vapor pressures
            fig, axes = plt.subplots(3, 2, figsize=(10, 12))
            fig.suptitle(f'1D Compressible Flow Solution (t = {self.time:.4e} s, iter = {self.iteration})')

        # Row 1: Velocity, Temperature, Pressure
        ax_idx = 0
        axes.flat[ax_idx].plot(x, state.u, 'r-', linewidth=2)
        axes.flat[ax_idx].set_xlabel('x [m]')
        axes.flat[ax_idx].set_ylabel('Velocity [m/s]')
        axes.flat[ax_idx].set_title('Velocity')
        axes.flat[ax_idx].grid(True)
        ax_idx += 1

        axes.flat[ax_idx].plot(x, state.T, 'm-', linewidth=2, label='Static')
        axes.flat[ax_idx].plot(x, T0, 'm--', linewidth=2, alpha=0.7, label='Stagnation')
        axes.flat[ax_idx].set_xlabel('x [m]')
        axes.flat[ax_idx].set_ylabel('Temperature [K]')
        axes.flat[ax_idx].set_title('Temperature')
        axes.flat[ax_idx].grid(True)
        axes.flat[ax_idx].legend()
        ax_idx += 1

        axes.flat[ax_idx].plot(x, state.p / 1000, 'g-', linewidth=2, label='Static')
        axes.flat[ax_idx].plot(x, p0 / 1000, 'g--', linewidth=2, alpha=0.7, label='Stagnation')
        axes.flat[ax_idx].set_xlabel('x [m]')
        axes.flat[ax_idx].set_ylabel('Pressure [kPa]')
        axes.flat[ax_idx].set_title('Pressure')
        axes.flat[ax_idx].grid(True)
        axes.flat[ax_idx].legend()
        ax_idx += 1

        # Row 2: Mach number, Area, and (if ice growth) vapor partial pressure
        axes.flat[ax_idx].plot(x, state.M, 'k-', linewidth=2)
        axes.flat[ax_idx].set_xlabel('x [m]')
        axes.flat[ax_idx].set_ylabel('Mach number')
        axes.flat[ax_idx].set_title('Mach Number')
        axes.flat[ax_idx].grid(True)
        ax_idx += 1

        axes.flat[ax_idx].plot(self.mesh.x_faces, self.mesh.A_faces, 'c-', linewidth=2)
        axes.flat[ax_idx].set_xlabel('x [m]')
        axes.flat[ax_idx].set_ylabel('Area [m²]')
        axes.flat[ax_idx].set_title('Nozzle Area')
        axes.flat[ax_idx].grid(True)
        ax_idx += 1

        if has_ice_growth:
            # Plot vapor partial pressure
            axes.flat[ax_idx].plot(x, p_vapor, 'b-', linewidth=2, label='Vapor')
            axes.flat[ax_idx].plot(x, p_sat_water_vals, 'r--', linewidth=2, label='Sat (water)')
            axes.flat[ax_idx].plot(x, p_sat_ice_vals, 'c--', linewidth=2, label='Sat (ice)')
            axes.flat[ax_idx].set_xlabel('x [m]')
            axes.flat[ax_idx].set_ylabel('Pressure [Pa]')
            axes.flat[ax_idx].set_title('Water Vapor Pressure')
            axes.flat[ax_idx].grid(True)
            axes.flat[ax_idx].legend()
            ax_idx += 1

        # Row 3: Scalars
        if has_ice_growth:
            # Plot particle number density
            n = state.Y[0] * state.rho
            axes.flat[ax_idx].plot(x, n, 'orange', linewidth=2)
            axes.flat[ax_idx].set_xlabel('x [m]')
            axes.flat[ax_idx].set_ylabel('Number density [#/m³]')
            axes.flat[ax_idx].set_title('Particle Number Density')
            axes.flat[ax_idx].grid(True)
            axes.flat[ax_idx].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax_idx += 1

            # Plot vapor mass fraction
            axes.flat[ax_idx].plot(x, state.Y[1], 'b-', linewidth=2)
            axes.flat[ax_idx].set_xlabel('x [m]')
            axes.flat[ax_idx].set_ylabel('Mass fraction')
            axes.flat[ax_idx].set_title('Water Vapor Mass Fraction')
            axes.flat[ax_idx].grid(True)
            ax_idx += 1

            # Plot ice mass fraction
            axes.flat[ax_idx].plot(x, state.Y[2], 'c-', linewidth=2)
            axes.flat[ax_idx].set_xlabel('x [m]')
            axes.flat[ax_idx].set_ylabel('Mass fraction')
            axes.flat[ax_idx].set_title('Ice Mass Fraction')
            axes.flat[ax_idx].grid(True)
            ax_idx += 1

        elif self.n_scalars > 0:
            # Plot generic scalars
            for i in range(min(self.n_scalars, 3)):
                if ax_idx < len(axes.flat):
                    axes.flat[ax_idx].plot(x, state.Y[i], linewidth=2, label=f'Y_{i}')
                    axes.flat[ax_idx].set_xlabel('x [m]')
                    axes.flat[ax_idx].set_ylabel('Mass fraction')
                    axes.flat[ax_idx].set_title(f'Scalar Y_{i}')
                    axes.flat[ax_idx].grid(True)
                    axes.flat[ax_idx].legend()
                    ax_idx += 1

        # Hide unused axes
        for i in range(ax_idx, len(axes.flat)):
            axes.flat[i].axis('off')

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
