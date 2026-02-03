# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) and GitHub Copilot when working with code in this repository.

## Project Overview

This is a PhD research project from the University of Cambridge on **aircraft contrail formation and evolution**. It combines:
- Jet engine thermodynamics modeling
- CFD (1D compressible flow solver)
- Ice particle microphysics
- Mesh generation using Ansys Meshing Prime

## Quick Start

```bash
# Install core dependencies
pip install numpy scipy matplotlib h5py pytest

# Run all tests to verify setup
pytest cfd/tests/ -v

# Validate the CFD solver against analytical solutions
python cfd/scripts/validate_area_source.py
```

## Key Commands

### CFD Solver Tests
```bash
# Run all pytest tests
pytest cfd/tests/ -v

# Run specific test files
pytest cfd/tests/test_nozzle.py -v
pytest cfd/tests/test_shock_tube.py -v
pytest cfd/tests/test_ice_growth.py -v

# Validate area source term against analytical isentropic solution
python cfd/scripts/validate_area_source.py

# Benchmark solver performance
python cfd/scripts/benchmark_performance.py

# Run ice growth simulation (contrail microphysics)
python cfd/scripts/ice_growth_source.py
```

### Using Test Cases Programmatically
```python
from cfd.tests import run_subsonic_nozzle_test, run_shock_tube_test

solver = run_subsonic_nozzle_test(n_cells=100, n_scalars=3)
solver, exact = run_shock_tube_test(n_cells=400, t_final=0.0002)
```

### Mesh Generation (requires Ansys license)
```bash
# Generate mesh from CAD geometry
python mesh.py <path-to-scdoc-file> [--no-display] [-p PROCESSES] [-t THREADS]

# Generate nacelle geometry and boundary conditions
python nacelle.py  # outputs geom/nacelle.json
```

## Architecture

### Root-Level Modules (Jet/Nacelle Design)
- `jet.py` - `JetCondition` class: engine thermodynamics for PW1100G and LEAP1A turbofans. Computes station properties, mass flows, exhaust composition. Uses `thermo` package for flash calculations.
- `nacelle.py` - Traces geometry from SVG (`images/exhaust_traced.svg`), computes jet areas, boundary layer parameters, outputs `geom/nacelle.json`
- `setup.py` - `BoundaryCondition` dataclass (50+ fields), `AdvancedJSONEncoder` for Pint units, `boundary_layer_mesh_stats()` function
- `hygrometry.py` - Magnus formula: `psat_water(p, T)` and `psat_ice(p, T)` saturation pressures
- `geometry.py` - SVG path tools using `svgpathtools`: fillet operations, wake geometry
- `mesh.py` - Ansys Prime mesh generation with size controls, boundary layers, volume meshing

### CFD Package (`cfd/`)

**Core solver in `cfd/src/`:**
- `solver.py` - Main `Solver1D` class with RK2/RK4 time integration
- `gas.py` - `GasProperties(gamma, R)`
- `state.py` - `FlowState`: primitive/conservative variable conversion, provides `.T`, `.M`, `.a`, `.E`, `.H`
- `mesh.py` - `Mesh1D` with variable area support
- `flux.py` - HLLC Riemann solver
- `reconstruction.py` - MUSCL with minmod limiter
- `boundary.py` - `SubsonicInletBC`, `SubsonicOutletBC`
- `area_source.py` - Geometric source term `p·dA/dx` for quasi-1D flows
- `sources.py` - Base `SourceTerm` and `ScalarSourceTerm` classes

**Scripts in `cfd/scripts/`:**
- `validate_area_source.py` - Validation against analytical isentropic nozzle flow
- `benchmark_performance.py` - Performance benchmarking
- `ice_growth_source.py` - `IceGrowthLookupTable` class using `ice_growth_fits.hdf5`, implements Koenig approximation for ice particle growth

### Data Files
- `ice_growth_fits.hdf5` - Lookup table for ice growth coefficients (T, p grids with a, b values)
- `geom/nacelle.json` - Boundary conditions exported from `nacelle.py`
- `images/exhaust_traced.svg` - Engine geometry traced for nacelle generation

## Key Patterns

### CFD Solver Usage
```python
from cfd import GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig
from cfd import SubsonicInletBC, SubsonicOutletBC, ScalarSourceTerm

gas = GasProperties(gamma=1.4, R=287.0)
mesh = Mesh1D.uniform(0.0, 1.0, n_cells=100, area_func=lambda x: 1.0 - 0.2*np.sin(np.pi*x))
config = SolverConfig(cfl=0.5, max_iter=20000, convergence_tol=1e-8, time_scheme='rk2')
solver = Solver1D(mesh, gas, n_scalars=3, config=config)
```

### Custom Scalar Source Terms
```python
def my_source(state: FlowState, mesh: Mesh1D) -> np.ndarray:
    sources = np.zeros((n_scalars, mesh.n_cells))
    sources[0] = ...  # Access state.rho, state.T, state.p, state.Y[i]
    return sources

solver.add_source_term(ScalarSourceTerm(my_source))
```

## Quasi-1D Equations

The CFD solver implements:
```
∂(ρ)/∂t + (1/A)·∂(A·ρu)/∂x = 0
∂(ρu)/∂t + (1/A)·∂(A·(ρu² + p))/∂x = (p/A)·dA/dx  ← geometric source
∂(ρE)/∂t + (1/A)·∂(A·ρuH)/∂x = 0
```

## Dependencies

- NumPy, SciPy, Matplotlib, h5py
- `pint` - unit handling
- `thermo` - thermodynamic flash calculations
- `flightcondition` - flight condition utilities
- `svgpathtools` - SVG geometry
- `ansys.meshing.prime` - mesh generation (commercial)

## Code Style & Conventions

### Python Style
- **Type hints**: Use type hints for function signatures (see `FlowState`, `Mesh1D` classes)
- **Docstrings**: Google-style docstrings with Args/Returns sections
- **NumPy**: Prefer vectorized NumPy operations over Python loops for performance
- **Dataclasses**: Use `@dataclass` for data containers (see `jet.py`, `setup.py`)

### Naming Conventions
- **Physical quantities**: Use standard notation (e.g., `rho` for density, `p` for pressure, `T` for temperature, `M` for Mach number)
- **Greek letters**: Spell out (e.g., `gamma` not `γ`, `rho` not `ρ`)
- **Private methods**: Prefix with underscore (e.g., `_compute_flux`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `M_H2O = 18.02`)

### Units
The project uses Pint for physical units. Import via:
```python
from flightcondition import unit
# e.g., engine.D = (81 * unit('in')).to('m')
```

## Testing Approach

### Test Structure
- Tests live in `cfd/tests/` and follow pytest conventions
- Fixtures defined in test files (not conftest.py currently)
- Test classes group related tests (e.g., `TestMassConservation`, `TestPhysicalBounds`)

### Writing New Tests
```python
import pytest
import numpy as np
from cfd import GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig

@pytest.fixture
def gas():
    """Standard air properties."""
    return GasProperties(gamma=1.4, R=287.0)

class TestNewFeature:
    def test_feature_works(self, gas):
        # Test implementation
        assert result == expected
```

### Test Markers
```bash
# Skip slow tests
pytest -m "not slow"
```

## Common Debugging Tips

### CFD Solver Issues
1. **Solver not converging**: Try reducing CFL number (0.3-0.5), use `use_first_order=True`
2. **Negative density/pressure**: Check boundary conditions, reduce time step
3. **Mass flow variation**: Verify area source term is enabled for variable-area meshes

### Checking Solution Quality
```python
# After solving
state = solver.get_state()

# Check mass conservation
mdot = state.rho * state.u * mesh.A_cells
print(f"Mass flow variation: {(mdot.max() - mdot.min()) / mdot.mean() * 100:.2f}%")

# Check isentropic relations
T0 = state.T * (1 + 0.5 * (gas.gamma - 1) * state.M**2)
print(f"T0 variation: {(T0.max() - T0.min()) / T0.mean() * 100:.2f}%")
```

## Agent Guidelines

### When Making Changes
1. **Run tests first**: `pytest cfd/tests/ -v` to understand baseline
2. **Make minimal changes**: This is research code—preserve existing behavior
3. **Validate physics**: CFD changes should conserve mass, energy, and entropy (for isentropic flows)
4. **Check units**: Pint catches unit errors—don't suppress warnings

### File Organization
- **New CFD features**: Add to `cfd/src/`, export in `cfd/src/__init__.py`
- **New test cases**: Add to `cfd/tests/`, follow existing patterns
- **Documentation**: Update this file and `cfd/README.md` as needed

### Don't Modify Without Understanding
- `cfd/src/area_source.py` - Critical geometric source term (see `cfd/docs/BUGFIX_AREA_SOURCE.md`)
- `cfd/src/flux.py` - HLLC Riemann solver implementation
- `jet.py` - Engine thermodynamics with complex Pint unit handling
