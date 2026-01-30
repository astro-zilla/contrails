# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PhD research project from the University of Cambridge on **aircraft contrail formation and evolution**. It combines:
- Jet engine thermodynamics modeling
- CFD (1D compressible flow solver)
- Ice particle microphysics
- Mesh generation using Ansys Meshing Prime

## Key Commands

### CFD Solver Tests
```bash
# Run nozzle flow test
python cfd/scripts/run_nozzle_test.py

# Run shock tube validation (Sod's problem)
python cfd/scripts/run_shock_tube.py

# Validate area source term against analytical solution
python cfd/scripts/validate_area_source.py

# Benchmark solver performance
python cfd/scripts/benchmark_performance.py

# Run ice growth simulation
python cfd/scripts/ice_growth_source.py
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

### Units
The project uses Pint for physical units. Import via:
```python
from flightcondition import unit
# e.g., engine.D = (81 * unit('in')).to('m')
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
