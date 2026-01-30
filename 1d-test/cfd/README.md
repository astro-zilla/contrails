# 1D Compressible Flow Solver

A modular Python package for solving quasi-1D compressible flows with passive scalar transport.

## Features

- **Full Euler equations** with variable area (quasi-1D formulation)
- **Arbitrary number of passive scalars** with custom source terms
- **Extensible architecture** for adding source terms to any equation
- **2nd order spatial accuracy** (MUSCL reconstruction with minmod limiter)
- **HLLC flux scheme** (robust Riemann solver)
- **RK4 time integration** (4th order accurate)
- **Multiple boundary conditions** (subsonic inlet/outlet, wall)

## Package Structure

```
cfd/
├── __init__.py           # Package exports
├── gas.py                # GasProperties class
├── state.py              # FlowState (primitive/conservative conversions)
├── mesh.py               # Mesh1D (variable area support)
├── sources.py            # Source term classes (extensible)
├── flux.py               # Flux schemes (HLLC)
├── reconstruction.py     # MUSCL reconstruction with limiters
├── boundary.py           # Boundary conditions
├── timestepping.py       # RK4 integration and timestep control
├── solver.py             # Main Solver1D class
├── run_nozzle.py         # Example run script
└── test_cases/
    ├── __init__.py
    └── nozzle.py         # Subsonic nozzle test case
```

## Quick Start

### Basic Usage

```python
from cfd import (
    GasProperties, FlowState, Mesh1D,
    Solver1D, SolverConfig,
    SubsonicInletBC, SubsonicOutletBC,
    ScalarSourceTerm
)
import numpy as np

# Define gas properties
gas = GasProperties(gamma=1.4, R=287.0)

# Create mesh with area distribution
area_func = lambda x: 1.0 - 0.2 * np.sin(np.pi * x)  # Example
mesh = Mesh1D.uniform(x_min=0.0, x_max=1.0, n_cells=100, area_func=area_func)

# Configure solver
config = SolverConfig(cfl=0.5, max_iter=10000, convergence_tol=1e-8)
solver = Solver1D(mesh, gas, n_scalars=2, config=config)

# Set boundary conditions
bc_left = SubsonicInletBC(p0=101325.0, T0=300.0, Y_inlet=np.array([0.01, 0.005]))
bc_right = SubsonicOutletBC(p_exit=95000.0)
solver.set_boundary_conditions(bc_left, bc_right)

# Add custom scalar source terms
def my_source(state, mesh):
    sources = np.zeros((2, mesh.n_cells))
    sources[0] = state.rho * 1e-4 * state.T  # Example: temperature-dependent production
    sources[1] = -state.rho * 10.0 * state.Y[1]  # Example: decay
    return sources

solver.add_source_term(ScalarSourceTerm(my_source))

# Set initial condition
rho = np.full(100, 1.2)
u = np.full(100, 100.0)
p = np.full(100, 101325.0)
Y = np.zeros((2, 100))
Y[0, :] = 0.01
Y[1, :] = 0.005

initial_state = FlowState(rho=rho, u=u, p=p, Y=Y, gas=gas)
solver.set_initial_condition(initial_state)

# Solve
result = solver.solve()

# Visualize
solver.plot_solution()
solver.plot_convergence()
```

### Run Test Case

Run from the **project root directory** (not inside the `cfd` folder):

```bash
python run_nozzle_test.py
```

Or from inside the `cfd` directory:

```bash
cd cfd
python -m test_cases.nozzle
```

Or from Python (from project root):

```python
from cfd.test_cases import run_subsonic_nozzle_test

solver = run_subsonic_nozzle_test(n_cells=100, n_scalars=2)
```

## Custom Source Terms

The package uses an extensible architecture for source terms. To add your own:

```python
from cfd.sources import SourceTerm
import numpy as np

class MyCustomSource(SourceTerm):
    def compute(self, state, mesh):
        n_vars = 3 + state.Y.shape[0]
        S = np.zeros((n_vars, mesh.n_cells))
        
        # Add source to momentum equation (index 1)
        S[1] = ... # your momentum source
        
        # Add source to energy equation (index 2)
        S[2] = ... # your energy source
        
        # Add source to scalar equations (index 3+)
        S[3] = ... # scalar 0 source
        S[4] = ... # scalar 1 source
        
        return S

solver.add_source_term(MyCustomSource())
```

For passive scalars specifically, use the convenient `ScalarSourceTerm`:

```python
from cfd import ScalarSourceTerm

def my_scalar_source(state, mesh):
    """
    Args:
        state: FlowState with properties (rho, u, p, T, M, a, Y)
        mesh: Mesh1D with properties (x_cells, dx, A_cells, etc.)
    
    Returns:
        Array of shape (n_scalars, n_cells) with source terms
    """
    n_scalars = state.Y.shape[0]
    sources = np.zeros((n_scalars, mesh.n_cells))
    
    # Your custom source logic here
    # Access: state.rho, state.u, state.p, state.T, state.M, state.Y[i]
    
    return sources

solver.add_source_term(ScalarSourceTerm(my_scalar_source))
```

## FlowState Properties

The `FlowState` class provides convenient access to flow variables:

- `state.rho` - Density [kg/m³]
- `state.u` - Velocity [m/s]
- `state.p` - Pressure [Pa]
- `state.T` - Temperature [K] (computed)
- `state.M` - Mach number (computed)
- `state.a` - Speed of sound [m/s] (computed)
- `state.E` - Total specific energy [J/kg] (computed)
- `state.H` - Total specific enthalpy [J/kg] (computed)
- `state.Y` - Scalar mass fractions [n_scalars, n_cells]

## Mesh Properties

The `Mesh1D` class stores mesh information:

- `mesh.n_cells` - Number of cells
- `mesh.x_cells` - Cell center locations
- `mesh.x_faces` - Face locations
- `mesh.A_cells` - Area at cell centers
- `mesh.A_faces` - Area at faces
- `mesh.dx` - Cell widths
- `mesh.vol` - Cell volumes (A * dx)

## Design Philosophy

The package is designed for:

1. **Readability** - Clear separation of concerns into logical modules
2. **Extensibility** - Easy to add new source terms, flux schemes, BCs
3. **Testability** - Each component can be tested independently
4. **Usability** - Simple API for common use cases

## Requirements

- Python 3.7+
- NumPy
- Matplotlib (for visualization)

## License

See project root for license information.
