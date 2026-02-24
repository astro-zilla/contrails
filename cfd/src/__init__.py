"""
1D Compressible Flow Solver Package
====================================

A modular solver for quasi-1D compressible flows with passive scalar transport.

Features:
- Full Euler equations with variable area
- Arbitrary number of passive scalars
- Extensible source term architecture
- 2nd order MUSCL reconstruction
- HLLC flux scheme
- RK4 time integration

State representation (conservative variables):
    rho   - density [kg/m³]
    rhoU  - momentum per volume [kg/(m²·s)]
    rhoE  - total energy per volume [J/m³]
    phi   - scalar per volume [1/m³] (phi = rho * Y)

Source rates use symbol S.

Example:
    # Create state from primitive variables
    state = FlowState.from_primitives(rho=rho, u=u, p=p, Y=Y, gas=gas)

    # Access primitive variables as properties
    print(state.u, state.p, state.T, state.Y)

    # Convert to/from array form
    U = state.to_array()  # [rho, rhoU, rhoE, phi_0, phi_1, ...]
    state2 = FlowState.from_array(U, gas)
"""

from .gas import GasProperties
from .state import FlowState
from .mesh import Mesh1D
from .sources import SourceTerm, ScalarSourceTerm, CompositeSourceTerm
from .area_source import AreaSourceTerm
from .flux import FluxScheme, HLLCFlux, RusanovFlux
from .boundary import BoundaryCondition, SubsonicInletBC, SubsonicOutletBC, WallBC
from .solver import Solver1D, SolverConfig

__all__ = [
    # Gas properties
    'GasProperties',

    # Flow state
    'FlowState',

    # Mesh
    'Mesh1D',

    # Source terms
    'SourceTerm',
    'ScalarSourceTerm',
    'CompositeSourceTerm',
    'AreaSourceTerm',

    # Flux schemes
    'FluxScheme',
    'HLLCFlux',
    'RusanovFlux',

    # Boundary conditions
    'BoundaryCondition',
    'SubsonicInletBC',
    'SubsonicOutletBC',
    'WallBC',

    # Solver
    'Solver1D',
    'SolverConfig',
]

__version__ = '1.0.0'
