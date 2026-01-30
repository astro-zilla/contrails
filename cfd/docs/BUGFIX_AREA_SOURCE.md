# Bug Fix: Missing Area Source Term in Quasi-1D Solver

## Problem
The 1D compressible flow solver was not correctly capturing flow acceleration through a converging-diverging nozzle. The velocity profile remained nearly flat instead of peaking at the throat.

## Root Cause
The quasi-1D momentum equation requires an explicit pressure-area gradient source term:

```
∂(ρu)/∂t + ∂(ρu² + p)/∂x = p · dA/dx
```

This term represents the pressure force acting on the varying cross-section. While the code correctly scaled fluxes by area (`A*F`), it was missing this explicit source term.

## Solution
Created `cfd/area_source.py` with `AreaSourceTerm` class that computes:

```python
S_momentum = p * dA/dx
```

The solver now automatically detects variable-area meshes and adds this source term.

## Results

### Before Fix:
- Velocity peak at x = 0.265 m (23.5% away from throat)
- Mach number flat: ~0.305 everywhere
- Velocity change: 104.87 → 104.82 m/s (essentially no acceleration)
- Stagnation pressure variation: 22.3% (non-physical losses)

### After Fix:
- Velocity peak at x = 0.495 m (0.5% from throat) ✓
- Mach number: 0.304 → 0.383 (25% increase) ✓
- Velocity change: 104.4 → 131.0 m/s (proper acceleration) ✓
- Stagnation pressure variation: 2.4% (acceptable)
- Mass flow conservation: 0.22% (excellent)

## Implementation
Modified files:
- `cfd/area_source.py` - New area source term class
- `cfd/__init__.py` - Export AreaSourceTerm
- `cfd/solver.py` - Auto-detect and add area source for variable-area meshes

The fix is transparent to users - existing code automatically benefits from the correction.

## Verification
Run `diagnose_nozzle.py` to verify the fix and visualize the corrected flow field.

