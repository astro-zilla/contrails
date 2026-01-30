# Copilot Instructions

This file provides context for GitHub Copilot when working in this repository.

## Project Context

This is a **PhD research project** on aircraft contrail formation from the University of Cambridge. The code combines:
- Computational Fluid Dynamics (quasi-1D compressible flow solver)
- Jet engine thermodynamics (turbofan exhaust modeling)
- Ice particle microphysics (contrail crystal growth)
- Mesh generation (Ansys Prime integration)

## Code Patterns

### CFD Solver Pattern
```python
from cfd import GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig
from cfd import SubsonicInletBC, SubsonicOutletBC

# 1. Define gas properties
gas = GasProperties(gamma=1.4, R=287.0)

# 2. Create mesh with area function
mesh = Mesh1D.uniform(0.0, 1.0, n_cells=100, area_func=lambda x: 1.0 - 0.2*np.sin(np.pi*x))

# 3. Configure solver
config = SolverConfig(cfl=0.5, max_iter=20000, convergence_tol=1e-8)
solver = Solver1D(mesh, gas, n_scalars=0, config=config)

# 4. Set boundary conditions
solver.set_boundary_conditions(
    SubsonicInletBC(p0=101325.0, T0=300.0, Y_inlet=np.array([])),
    SubsonicOutletBC(p_exit=95000.0)
)

# 5. Set initial condition and solve
solver.set_initial_condition(initial_state)
result = solver.solve()
```

### Custom Source Term Pattern
```python
from cfd import ScalarSourceTerm

def my_source(state: FlowState, mesh: Mesh1D) -> np.ndarray:
    """Source term for passive scalars."""
    sources = np.zeros((n_scalars, mesh.n_cells))
    sources[0] = state.rho * rate_constant * state.T
    return sources

solver.add_source_term(ScalarSourceTerm(my_source))
```

### Test Pattern
```python
import pytest
from cfd import GasProperties

@pytest.fixture
def gas():
    return GasProperties(gamma=1.4, R=287.0)

class TestFeature:
    def test_something(self, gas):
        # Arrange, Act, Assert
        assert result == expected
```

## Variable Naming

Use standard CFD/thermodynamics notation:
- `rho` - density [kg/m³]
- `u` - velocity [m/s]
- `p` - pressure [Pa]
- `T` - temperature [K]
- `M` - Mach number [-]
- `E` - total energy per unit mass [J/kg]
- `H` - total enthalpy per unit mass [J/kg]
- `gamma` - specific heat ratio [-]
- `R` - specific gas constant [J/(kg·K)]
- `A` - cross-sectional area [m²]
- `mdot` - mass flow rate [kg/s]
- `Y` - mass fraction [-]

## Key Classes

- `GasProperties(gamma, R)` - Ideal gas with specific heat ratio and gas constant
- `FlowState(rho, u, p, Y, gas)` - Primitive flow variables with derived quantities
- `Mesh1D.uniform(x_min, x_max, n_cells, area_func)` - 1D mesh with variable area
- `Solver1D(mesh, gas, n_scalars, config)` - Main solver class
- `SolverConfig(cfl, max_iter, convergence_tol, ...)` - Solver parameters

## Physical Units

This project uses Pint for physical units:
```python
from flightcondition import unit

# Creating quantities
D = 81 * unit('in')
D_m = D.to('m')

# Dimensionless
M = 0.78 * unit('dimensionless')
```

## Files to Understand Before Modifying

1. `cfd/src/solver.py` - Main solver implementation
2. `cfd/src/state.py` - FlowState class (primitive ↔ conservative)
3. `cfd/src/area_source.py` - Geometric source term (critical for nozzle flows)
4. `cfd/docs/BUGFIX_AREA_SOURCE.md` - Documents important bug fix

## Don't Suggest

- Breaking changes to public API without discussion
- Removing type hints
- Non-vectorized NumPy operations (performance critical)
- Changes to `area_source.py` without understanding the physics
