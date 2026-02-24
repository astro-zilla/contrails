"""
Boundary conditions for the 1D compressible flow solver.
"""

import numpy as np
from abc import ABC, abstractmethod

from .gas import GasProperties
from .mesh import Mesh1D


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
    Use characteristic-based boundary condition (extrapolate one Riemann invariant).
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
            # Get interior cell state
            rho_int = U[0, 1]
            u_int = U[1, 1] / rho_int
            p_int = (gamma - 1) * (U[2, 1] - 0.5 * rho_int * u_int**2)
            a_int = np.sqrt(gamma * p_int / rho_int)

            # Characteristic-based BC: extrapolate Riemann invariant from interior
            # Riemann invariant: R = u - 2*a/(gamma-1) (for subsonic inlet)
            R_minus = u_int - 2 * a_int / (gamma - 1)

            # Initial guess for static conditions (from stagnation, assuming M~0.3)
            M_guess = 0.3
            T_guess = self.T0 / (1 + 0.5 * (gamma - 1) * M_guess**2)
            p_guess = self.p0 / (1 + 0.5 * (gamma - 1) * M_guess**2)**(gamma / (gamma - 1))

            # Iteratively solve for static conditions using characteristic BC
            # We need to find T, p such that:
            # 1) Stagnation relations are satisfied
            # 2) Riemann invariant matches interior

            for _ in range(10):  # Newton iteration
                rho_guess = p_guess / (gas.R * T_guess)
                a_guess = np.sqrt(gamma * gas.R * T_guess)

                # From Riemann invariant: u = R + 2*a/(gamma-1)
                u_guess = R_minus + 2 * a_guess / (gamma - 1)

                # Ensure positive flow
                u_guess = max(u_guess, 0.01 * a_guess)

                # Compute Mach number
                M_guess = u_guess / a_guess

                # Update static conditions from stagnation using this Mach number
                T_guess = self.T0 / (1 + 0.5 * (gamma - 1) * M_guess**2)
                p_guess = self.p0 / (1 + 0.5 * (gamma - 1) * M_guess**2)**(gamma / (gamma - 1))

            # Final values
            T = T_guess
            p = p_guess
            rho = p / (gas.R * T)
            a = np.sqrt(gamma * gas.R * T)
            u = R_minus + 2 * a / (gamma - 1)
            u = max(u, 0.01 * a)  # Ensure positive flow

            # Compute total energy
            E = p / (rho * (gamma - 1)) + 0.5 * u**2

            # Set ghost cell
            U[0, 0] = rho
            U[1, 0] = rho * u
            U[2, 0] = rho * E

            # Vectorized scalar assignment
            n_scalars = U.shape[0] - 3
            if n_scalars > 0:
                U[3:, 0] = rho * self.Y_inlet

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
            # Get interior cell values (last real cell)
            rho_int = U[0, -2]
            u_int = U[1, -2] / rho_int

            # Extrapolate density and velocity, impose pressure
            rho = rho_int
            u = u_int
            p = self.p_exit

            # Compute energy with imposed pressure
            E = p / (rho * (gamma - 1)) + 0.5 * u**2

            # Set ghost cell
            U[0, -1] = rho
            U[1, -1] = rho * u
            U[2, -1] = rho * E

            # Vectorized scalar extrapolation
            n_scalars = U.shape[0] - 3
            if n_scalars > 0:
                U[3:, -1] = U[3:, -2] * (rho / rho_int)

        return U


class WallBC(BoundaryCondition):
    """
    Inviscid wall (slip): zero normal velocity.
    Reflects the velocity component.
    """

    def apply(self, U: np.ndarray, mesh: Mesh1D, gas: GasProperties,
              side: str) -> np.ndarray:

        if side == 'left':
            # Mirror the first interior cell (vectorized)
            U[:, 0] = U[:, 1]
            U[1, 0] = -U[1, 1]  # Reflect momentum

        elif side == 'right':
            # Mirror the last interior cell (vectorized)
            U[:, -1] = U[:, -2]
            U[1, -1] = -U[1, -2]  # Reflect momentum

        return U
