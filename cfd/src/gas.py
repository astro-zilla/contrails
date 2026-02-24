"""
Gas properties for calorically perfect gas.
"""

from dataclasses import dataclass


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

