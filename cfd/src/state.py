"""
Flow state representation and conversions between primitive and conservative variables.
"""

import numpy as np
from dataclasses import dataclass

from .gas import GasProperties


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

        # Vectorized scalar conversion (no loop)
        if n_scalars > 0:
            U[3:] = self.rho * self.Y

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

        # Vectorized scalar computation (no loop)
        n_scalars = U.shape[0] - 3
        if n_scalars > 0:
            Y = U[3:] / rho
        else:
            Y = np.zeros((0, len(rho)))

        return cls(rho=rho, u=u, p=p, Y=Y, gas=gas)
