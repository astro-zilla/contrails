# 1D Compressible Flow Solver

A modular Python package for solving quasi-1D compressible flows with passive scalar transport and variable area geometries (nozzles, ducts).

## Features

- **Quasi-1D Euler equations** with variable area (includes geometric source term)
- **Passive scalar transport** with arbitrary number of scalars and custom source terms
- **High-order accurate**: 2nd order MUSCL reconstruction, HLLC Riemann solver, RK2/RK4 time integration
- **Robust boundary conditions**: Subsonic inlet/outlet with characteristic-based implementation
- **Automatic area source term**: Correctly handles pressure-area gradient for nozzle flows
- **Vectorized implementation**: Optimized NumPy operations for performance

## Quick Start

### Installation

Ensure you have the required dependencies:
```bash
conda env create -f environment.yml
conda activate contrails
```

Or install manually:
```bash
pip install numpy matplotlib
```

### Basic Example

```python
from cfd import (
    GasProperties, FlowState, Mesh1D,
    Solver1D, SolverConfig,
    SubsonicInletBC, SubsonicOutletBC
)
import numpy as np

# Gas properties (air)
gas = GasProperties(gamma=1.4, R=287.0)

# Create nozzle geometry
def nozzle_area(x):
    # Converging-diverging nozzle
    return 1.0 - 0.2 * np.sin(np.pi * x)

mesh = Mesh1D.uniform(0.0, 1.0, n_cells=100, area_func=nozzle_area)

# Configure solver
config = SolverConfig(
    cfl=0.5,
    max_iter=20000,
    convergence_tol=1e-8,
    time_scheme='rk2',
    use_first_order=False
)

solver = Solver1D(mesh, gas, n_scalars=0, config=config)

# Boundary conditions
bc_left = SubsonicInletBC(p0=101325.0, T0=300.0, Y_inlet=np.array([]))
bc_right = SubsonicOutletBC(p_exit=95000.0)
solver.set_boundary_conditions(bc_left, bc_right)

# Initial condition (uniform flow)
M_init = 0.3
T_init = 300.0 / (1 + 0.5 * 0.4 * M_init**2)
p_init = 101325.0 / (1 + 0.5 * 0.4 * M_init**2)**(1.4/0.4)
rho_init = p_init / (287.0 * T_init)
u_init = M_init * np.sqrt(1.4 * 287.0 * T_init)

initial_state = FlowState(
    rho=np.full(100, rho_init),
    u=np.full(100, u_init),
    p=np.full(100, p_init),
    Y=np.zeros((0, 100)),
    gas=gas
)
solver.set_initial_condition(initial_state)

# Solve
result = solver.solve()
print(f"Converged: {result['converged']}, Iterations: {result['iterations']}")

# Visualize
solver.plot_solution('solution.png')
```

## Directory Structure

```
1d-test/
├── cfd/                      # Main CFD package
│   ├── __init__.py           # Package exports
│   ├── gas.py                # Gas properties
│   ├── state.py              # Flow state (primitive/conservative)
│   ├── mesh.py               # 1D mesh with variable area
│   ├── sources.py            # Source term base classes
│   ├── area_source.py        # Geometric source term for quasi-1D
│   ├── flux.py               # HLLC flux scheme
│   ├── reconstruction.py     # MUSCL with minmod limiter
│   ├── boundary.py           # Boundary conditions
│   ├── timestepping.py       # Time integration (RK2, RK4)
│   ├── solver.py             # Main Solver1D class
│   └── test_cases/
│       ├── nozzle.py         # Nozzle flow test case
│       └── shock_tube.py     # Shock tube test case
│
├── scripts/                  # Test and diagnostic scripts
│   ├── run_nozzle_test.py    # Example: Run nozzle simulation
│   ├── diagnose_nozzle.py    # Diagnostic: Velocity profile analysis
│   ├── validate_area_source.py  # Validation: Compare to analytical solution
│   ├── test_area_source_simple.py  # Simple test case
│   ├── run_shock_tube.py     # Example: Run shock tube simulation
│   └── benchmark_performance.py    # Performance benchmarking
│
└── docs/                     # Documentation
    ├── README.md             # This file
    ├── BUGFIX_AREA_SOURCE.md # Area source term bug fix documentation
    ├── CONVERGENCE_FIX.md    # Convergence improvements
    ├── OPTIMIZATION_SUMMARY.md  # Performance optimization notes
    ├── PERFORMANCE.md        # Performance benchmarks
    └── QUASI_1D_FORMULATION.py  # Quasi-1D equations reference
```

## Running Examples

From the `1d-test` directory:

```bash
# Run basic nozzle test
python scripts/run_nozzle_test.py

# Run diagnostic (checks velocity peaks at throat)
python scripts/diagnose_nozzle.py

# Run validation against analytical solution
python scripts/validate_area_source.py

# Run shock tube test (Sod's problem)
python scripts/run_shock_tube.py

# Benchmark performance
python scripts/benchmark_performance.py
```

## Test Cases

### 1. Subsonic Nozzle Flow
Classic quasi-1D nozzle with converging-diverging geometry. Tests:
- Area source term implementation
- Subsonic flow acceleration/deceleration
- Boundary conditions (subsonic inlet/outlet)
- Passive scalar transport

**Results:** Velocity correctly peaks at throat with <0.3% mass flow variation.

### 2. Sod's Shock Tube
Standard Riemann problem for validation of shock-capturing schemes. Tests:
- Shock wave resolution
- Contact discontinuity capturing
- Rarefaction wave accuracy
- Time-accurate unsteady simulation

**Results:** Errors decrease with grid refinement. At 400 cells:
- Density L1 error: 0.002145 kg/m³
- Velocity L1 error: 1.16 m/s
- Pressure L1 error: 169 Pa

The HLLC flux scheme successfully captures all wave structures.

## Theory: Quasi-1D Equations

The solver implements the quasi-1D Euler equations in conservative form:

**Continuity:**
```
∂(ρ)/∂t + (1/A)·∂(A·ρu)/∂x = 0
```

**Momentum:**
```
∂(ρu)/∂t + (1/A)·∂(A·(ρu² + p))/∂x = (p/A)·dA/dx
```

**Energy:**
```
∂(ρE)/∂t + (1/A)·∂(A·ρuH)/∂x = 0
```

The term `(p/A)·dA/dx` is the **geometric source term** that accounts for pressure forces on the varying cross-section. This is essential for correctly capturing nozzle flow behavior.

## Key Implementation Details

### 1. Geometric Source Term
The solver automatically detects variable-area meshes and adds the momentum source term:
```python
S_momentum = p * dA/dx
```

This is added in the `AreaSourceTerm` class and applied during time integration.

### 2. Finite Volume Discretization
```python
dU/dt = -1/(A*dx) * [A_{i+1/2}*F_{i+1/2} - A_{i-1/2}*F_{i-1/2}] + S
```

where fluxes are computed at faces using HLLC scheme with MUSCL reconstruction.

### 3. Boundary Conditions
- **Subsonic Inlet**: Specify total pressure, total temperature (characteristic-based)
- **Subsonic Outlet**: Specify static pressure (extrapolate other variables)
- Uses iterative Newton method to satisfy both BC and Riemann invariants

## Validation Results

The solver has been validated against analytical isentropic nozzle flow:

| Test Case | Velocity Peak Location | Mass Conservation | Stagnation Pressure |
|-----------|------------------------|-------------------|---------------------|
| Converging-Diverging Nozzle | ✓ At throat (within 0.5% discretization) | 0.22% variation | 2.4% variation |
| Mach Number Range | 0.304 → 0.383 | - | - |
| Velocity Acceleration | 104.4 → 131.0 m/s (25% increase) | - | - |

See `docs/BUGFIX_AREA_SOURCE.md` for detailed before/after comparison.

## Performance

Typical performance on modern hardware:
- **100 cells**: ~1.7 ms/iteration (RK2), ~3.4 ms/iteration (RK4)
- **200 cells**: ~3.5 ms/iteration (RK2)
- **Convergence**: 10,000-20,000 iterations for 1e-8 tolerance

Vectorized NumPy operations provide 10-20x speedup over naive Python loops.

## Adding Custom Source Terms

Example: Temperature-dependent scalar production with decay:

```python
from cfd import ScalarSourceTerm

def my_source_function(state, mesh):
    """
    Custom source term for scalars.
    
    Args:
        state: FlowState with rho, u, p, T, Y
        mesh: Mesh1D with cell geometry
    
    Returns:
        sources: (n_scalars, n_cells) array
    """
    n_scalars = state.Y.shape[0]
    sources = np.zeros((n_scalars, mesh.n_cells))
    
    # Scalar 0: Temperature-dependent production
    sources[0] = state.rho * 1e-4 * state.T
    
    # Scalar 1: First-order decay
    sources[1] = -state.rho * 10.0 * state.Y[1]
    
    return sources

# Add to solver
solver.add_source_term(ScalarSourceTerm(my_source_function))
```

## References

- Anderson, J. D. (1995). *Computational Fluid Dynamics*. McGraw-Hill.
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.
- Hirsch, C. (2007). *Numerical Computation of Internal and External Flows*. Butterworth-Heinemann.

## License

Part of the Cambridge University PhD research project on aircraft contrails.

## Author

PhD Candidate, University of Cambridge
Last updated: January 29, 2026
