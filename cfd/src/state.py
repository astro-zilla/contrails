"""
Flow state representation using conservative variables.

State is defined by:
    rho   - density [kg/m³]
    rhoU  - momentum per volume [kg/(m²·s)]
    rhoE  - total energy per volume [J/m³]
    phi   - scalar concentration per volume [1/m³] (phi = rho * Y)

Source rates use symbol S.
"""

import numpy as np
from dataclasses import dataclass

from .gas import GasProperties


@dataclass
class FlowState:
    """
    Represents the flow state at a point or cell using conservative variables.

    Conservative variables (stored directly):
        rho  : Density [kg/m³]
        rhoU : Momentum per volume [kg/(m²·s)]
        rhoE : Total energy per volume [J/m³]
        phi  : Scalar per volume [1/m³], shape (n_scalars, n_cells)

    Primitive variables (computed as properties):
        u, p, T, Y, a, M, H, e
    """
    rho: np.ndarray     # Density [kg/m³]
    rhoU: np.ndarray    # Momentum per volume [kg/(m²·s)]
    rhoE: np.ndarray    # Total energy per volume [J/m³]
    phi: np.ndarray     # Scalar per volume [1/m³], shape (n_scalars, n_cells)
    gas: GasProperties

    # --- Primitive variables as properties ---

    @property
    def u(self) -> np.ndarray:
        """Velocity [m/s]."""
        return self.rhoU / self.rho

    @property
    def p(self) -> np.ndarray:
        """Pressure from total energy [Pa]."""
        # p = (gamma - 1) * (rhoE - 0.5 * rho * u²)
        return (self.gas.gamma - 1) * (self.rhoE - 0.5 * self.rhoU**2 / self.rho)

    @property
    def T(self) -> np.ndarray:
        """Temperature from ideal gas law [K]."""
        return self.p / (self.rho * self.gas.R)

    @property
    def Y(self) -> np.ndarray:
        """Scalar mass fractions (phi / rho)."""
        if self.phi.shape[0] == 0:
            return self.phi
        return self.phi / self.rho

    @property
    def e(self) -> np.ndarray:
        """Specific internal energy [J/kg]."""
        return self.p / (self.rho * (self.gas.gamma - 1))

    @property
    def E(self) -> np.ndarray:
        """Total specific energy [J/kg]."""
        return self.rhoE / self.rho

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

    # --- Array conversion methods ---

    def to_array(self) -> np.ndarray:
        """
        Convert to conservative variable array.

        Returns:
            U: Array of shape (3 + n_scalars, n_cells)
               [rho, rhoU, rhoE, phi_0, phi_1, ...]
        """
        n_scalars = self.phi.shape[0]
        n_cells = len(self.rho)
        U = np.zeros((3 + n_scalars, n_cells))

        U[0] = self.rho
        U[1] = self.rhoU
        U[2] = self.rhoE

        if n_scalars > 0:
            U[3:] = self.phi

        return U

    # Alias for backward compatibility
    def to_conservative(self) -> np.ndarray:
        """Alias for to_array() for backward compatibility."""
        return self.to_array()

    @classmethod
    def from_array(cls, U: np.ndarray, gas: GasProperties) -> 'FlowState':
        """
        Create FlowState from conservative variable array.

        Args:
            U: Conservative variables [rho, rhoU, rhoE, phi_0, ...]
            gas: Gas properties
        """
        rho = U[0]
        rhoU = U[1]
        rhoE = U[2]

        n_scalars = U.shape[0] - 3
        if n_scalars > 0:
            phi = U[3:]
        else:
            phi = np.zeros((0, len(rho)))

        return cls(rho=rho, rhoU=rhoU, rhoE=rhoE, phi=phi, gas=gas)

    # Alias for backward compatibility
    @classmethod
    def from_conservative(cls, U: np.ndarray, gas: GasProperties) -> 'FlowState':
        """Alias for from_array() for backward compatibility."""
        return cls.from_array(U, gas)

    @classmethod
    def from_primitives(cls, rho: np.ndarray, u: np.ndarray, p: np.ndarray,
                        Y: np.ndarray, gas: GasProperties) -> 'FlowState':
        """
        Create FlowState from primitive variables.

        Args:
            rho: Density [kg/m³]
            u: Velocity [m/s]
            p: Pressure [Pa]
            Y: Scalar mass fractions, shape (n_scalars, n_cells)
            gas: Gas properties
        """
        rhoU = rho * u
        # rhoE = p / (gamma - 1) + 0.5 * rho * u²
        rhoE = p / (gas.gamma - 1) + 0.5 * rho * u**2
        phi = rho * Y if Y.shape[0] > 0 else Y

        return cls(rho=rho, rhoU=rhoU, rhoE=rhoE, phi=phi, gas=gas)
