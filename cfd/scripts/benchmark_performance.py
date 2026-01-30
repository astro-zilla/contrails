"""
Performance benchmarking script for the 1D CFD solver.

Run from the cfd directory:
    python scripts/benchmark_performance.py
"""

import sys
from pathlib import Path

# Add parent directory (cfd) to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import numpy as np
from cfd import GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig
from cfd.boundary import SubsonicInletBC, SubsonicOutletBC


def run_benchmark(n_cells=100, n_scalars=2, time_scheme='rk4', cfl=0.8, max_iter=1000):
    """Run a benchmark test case."""
    # Gas properties
    gas = GasProperties(gamma=1.4, R=287.0)

    # Simple nozzle
    area_func = lambda x: 1.0 - 0.2 * np.sin(np.pi * x)
    mesh = Mesh1D.uniform(0.0, 1.0, n_cells, area_func)

    # Solver configuration
    config = SolverConfig(
        cfl=cfl,
        max_iter=max_iter,
        convergence_tol=1e-10,
        print_interval=10000,  # Suppress printing
        time_scheme=time_scheme
    )
    solver = Solver1D(mesh, gas, n_scalars=n_scalars, config=config)

    # Boundary conditions
    p0 = 101325.0
    T0 = 300.0
    p_exit = 95000.0
    Y_inlet = np.array([0.01, 0.005])

    bc_left = SubsonicInletBC(p0=p0, T0=T0, Y_inlet=Y_inlet)
    bc_right = SubsonicOutletBC(p_exit=p_exit)
    solver.set_boundary_conditions(bc_left, bc_right)

    # Initial condition
    M_init = 0.3
    T_init = T0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)
    p_init = p0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)**(gas.gamma / (gas.gamma - 1))
    rho_init = p_init / (gas.R * T_init)
    u_init = M_init * np.sqrt(gas.gamma * gas.R * T_init)

    rho = np.full(n_cells, rho_init)
    u = np.full(n_cells, u_init)
    p = np.full(n_cells, p_init)
    Y = np.zeros((n_scalars, n_cells))
    Y[0, :] = Y_inlet[0]
    Y[1, :] = Y_inlet[1]

    initial_state = FlowState(rho=rho, u=u, p=p, Y=Y, gas=gas)
    solver.set_initial_condition(initial_state)

    # Time the solve
    start_time = time.time()
    result = solver.solve()
    end_time = time.time()

    elapsed = end_time - start_time
    return elapsed, solver.iteration


if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"\nTest case: 100 cells, 2 scalars, 1000 iterations")
    print()

    # Test different time integration schemes
    schemes = [
        ('rk4', 0.8, "RK4 (4 RHS evaluations per step)"),
        ('rk2', 0.8, "RK2-SSP (2 RHS evaluations per step)"),
        ('euler', 0.4, "Forward Euler (1 RHS evaluation per step)")
    ]

    results = {}

    for scheme, cfl, description in schemes:
        print(f"\n{'-' * 80}")
        print(f"Testing: {description}")
        print(f"CFL: {cfl}")
        print(f"{'-' * 80}")

        elapsed, iterations = run_benchmark(
            n_cells=100,
            n_scalars=2,
            time_scheme=scheme,
            cfl=cfl,
            max_iter=1000
        )

        results[scheme] = elapsed

        print(f"\n✓ Completed in {elapsed:.3f} seconds")
        print(f"  Time per iteration: {elapsed / iterations * 1000:.2f} ms")
        print(f"  Iterations per second: {iterations / elapsed:.1f}")

    # Performance summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    baseline = results['rk4']

    print(f"\n{'Scheme':<15} {'Time (s)':<12} {'Speedup':<12} {'Notes'}")
    print("-" * 80)

    for scheme, cfl, description in schemes:
        elapsed = results[scheme]
        speedup = baseline / elapsed
        scheme_name = scheme.upper()

        if scheme == 'rk4':
            notes = "Baseline (most accurate)"
        elif scheme == 'rk2':
            notes = "Good balance"
        else:
            notes = "Fastest (less stable)"

        print(f"{scheme_name:<15} {elapsed:<12.3f} {speedup:<12.2f}x {notes}")

    # Estimate time for 15000 iterations
    print("\n" + "=" * 80)
    print("ESTIMATED TIME FOR 15000 ITERATIONS (to convergence)")
    print("=" * 80)

    for scheme in ['rk4', 'rk2', 'euler']:
        time_per_iter = results[scheme] / 1000
        estimated_time = time_per_iter * 15000

        print(f"{scheme.upper():<15} {estimated_time:.1f} seconds ({estimated_time/60:.1f} minutes)")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
For steady-state problems (like nozzle flows):
  
  1. RK2 with CFL=0.8 is RECOMMENDED
     - 2x faster than RK4
     - Still very accurate and stable
     - Best balance of speed and robustness
  
  2. Forward Euler with CFL=0.3-0.4 can be even faster
     - Up to 4x faster than RK4
     - Less stable, may require lower CFL
     - Good for smooth problems
  
  3. RK4 is best for unsteady/transient problems
     - Most accurate in time
     - Unnecessary overhead for steady-state

For your 15000-iteration case with RK2:
  - Expected time: ~{0:.1f} seconds ({1:.1f} minutes)
  - vs RK4: ~{2:.1f} seconds ({3:.1f} minutes)
    """.format(
        results['rk2'] / 1000 * 15000,
        results['rk2'] / 1000 * 15000 / 60,
        results['rk4'] / 1000 * 15000,
        results['rk4'] / 1000 * 15000 / 60
    ))

    print("=" * 80)
