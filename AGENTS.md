# AGENTS.md

This file provides instructions to AI agents (Copilot, Claude, Gemini, etc.) working in this repository.
See also: `CLAUDE.md` for project architecture details.

## Project

PhD research - **aircraft contrail formation and evolution** (University of Cambridge).
Combines quasi-1D CFD, jet engine thermodynamics, ice microphysics, and VTK/Ansys mesh generation.

**Language**: Python 3.x  
**Key packages**: NumPy, SciPy, Matplotlib, h5py, VTK, Pint, `thermo`, `flightcondition`, `ansys.meshing.prime`  
**Tests**: pytest at `cfd/tests/`  
**Shell**: PowerShell on Windows

## Non-Negotiable Rules

- No emojis in commit messages or code comments
- Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- Write tests for all new functionality
- Validate edits with `get_errors` after every file change
- Commit changes to git and push regularly
- Always research API/format specifications before implementation

## Available Custom Agents

Agent profiles are in `.github/agents/`. Invoke via `run_subagent`:

| Agent | Description | When to Use |
|-------|-------------|-------------|
| **Plan** | Orchestrator - breaks tasks into subtasks and delegates | Complex multi-step tasks |
| **researcher** | VTK APIs, web search, CFD docs, code exploration | Before implementing anything involving external APIs |
| **programmer** | Code implementation and editing | Implementing features with known context |
| **tester** | Write and run pytest tests | After implementation, or TDD |
| **reviewer** | PR/code review, numerical correctness | Before merging, quality check |
| **git** | Git branches, commits, PRs, issues | Version control operations |

### Recommended Delegation Patterns

**Feature Implementation:**
```
Plan -> researcher (gather APIs) -> programmer (implement) -> tester (test) -> git (commit)
```

**Bug Fix:**
```
researcher/reviewer (diagnose) -> programmer (fix) -> tester (verify) -> git (commit)
```

**Large/non-urgent upgrades:** Use `github/assign_copilot_to_issue` to offload asynchronously.

## CFD Solver Architecture

```
cfd/src/
  solver.py        - Solver1D (RK2/RK4)
  flux.py          - HLLC Riemann solver
  reconstruction.py- MUSCL + minmod limiter
  boundary.py      - SubsonicInletBC, SubsonicOutletBC
  area_source.py   - p·dA/dx geometric source term
  sources.py       - SourceTerm, ScalarSourceTerm base classes
  state.py         - FlowState (primitive/conservative)
  gas.py           - GasProperties(gamma, R)
  mesh.py          - Mesh1D with variable area
```

**Key equations (quasi-1D):**
```
d(rho)/dt + (1/A) d(A*rho*u)/dx = 0
d(rho*u)/dt + (1/A) d(A*(rho*u^2+p))/dx = (p/A) dA/dx
d(rho*E)/dt + (1/A) d(A*rho*u*H)/dx = 0
```

## Testing

```bash
pytest cfd/tests/ -v                          # all tests
pytest cfd/tests/test_nozzle.py -v            # nozzle
pytest cfd/tests/test_shock_tube.py -v        # shock tube
python cfd/scripts/validate_area_source.py    # analytical validation
```

## File Conventions

- Test files: `cfd/tests/test_<module>.py`
- Scripts: `cfd/scripts/<purpose>.py`
- Mesh/geometry output: `geom/`
- Data files: `*.hdf5` (ice growth coefficients, mesh data)
- Remove temporary test/setup files after use

