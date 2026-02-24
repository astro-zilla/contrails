"""
Source term classes for the 1D compressible flow solver.

Extensible architecture allowing custom source terms for scalars,
momentum, energy, etc.

Notation:
    U   - conservative variable array [rho, rhoU, rhoE, phi_0, ...]
    S   - source rate array (same shape as U)
    phi - scalar concentration per volume (rho * Y)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, List

from .state import FlowState
from .mesh import Mesh1D


class SourceTerm(ABC):
    """Abstract base class for source terms."""

    @abstractmethod
    def compute(self, state: FlowState, mesh: Mesh1D) -> np.ndarray:
        """
        Compute source term contribution.

        Args:
            state: Current flow state (conservative variables)
            mesh: Computational mesh

        Returns:
            S: Source rate array of shape (n_vars, n_cells)
        """
        pass


class ScalarSourceTerm(SourceTerm):
    """
    Source term for passive scalars.

    The user provides a function that computes scalar source rates.
    """

    def __init__(self, source_func: Callable[[FlowState, Mesh1D], np.ndarray]):
        """
        Args:
            source_func: Function(state, mesh) -> S_phi of shape (n_scalars, n_cells)
                         Returns the source term S in d(phi)/dt = S
                         where phi = rho * Y is scalar concentration per volume
        """
        self.source_func = source_func

    def compute(self, state: FlowState, mesh: Mesh1D) -> np.ndarray:
        n_scalars = state.phi.shape[0]
        n_vars = 3 + n_scalars

        S = np.zeros((n_vars, mesh.n_cells))

        # Compute scalar sources
        S_phi = self.source_func(state, mesh)

        # Assign to scalar equations (indices 3 onwards)
        if n_scalars > 0:
            S[3:] = S_phi

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
            n_vars = 3 + state.phi.shape[0]
            return np.zeros((n_vars, mesh.n_cells))

        S_total = self.sources[0].compute(state, mesh)
        for source in self.sources[1:]:
            S_total += source.compute(state, mesh)
        return S_total
