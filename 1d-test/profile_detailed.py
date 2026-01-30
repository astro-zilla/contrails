"""
Detailed performance profiling with cProfile.
"""

import cProfile
import pstats
import io
import numpy as np
from cfd import GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig
from cfd.boundary import SubsonicInletBC, SubsonicOutletBC


def run_test_case():
    """Run a simple test case for profiling."""
    n_cells = 100
    n_scalars = 2

    # Gas properties
    gas = GasProperties(gamma=1.4, R=287.0)

    # Simple nozzle
    area_func = lambda x: 1.0 - 0.2 * np.sin(np.pi * x)
    mesh = Mesh1D.uniform(0.0, 1.0, n_cells, area_func)

    # Solver with 1000 iterations
    config = SolverConfig(
        cfl=0.8,
        max_iter=1000,
        convergence_tol=1e-10,
        print_interval=10000  # Suppress printing during profiling
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

    # Solve
    result = solver.solve()
    return solver


if __name__ == "__main__":
    print("=" * 70)
    print("DETAILED PROFILING WITH cProfile")
    print("=" * 70)
    print("\nRunning 1000 iterations...\n")

    # Create profiler
    profiler = cProfile.Profile()

    # Run with profiling
    profiler.enable()
    solver = run_test_case()
    profiler.disable()

    # Print statistics
    print("\n" + "=" * 70)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 70)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

    print("\n" + "=" * 70)
    print("TOP 30 FUNCTIONS BY INTERNAL TIME")
    print("=" * 70)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('time')
    ps.print_stats(30)
    print(s.getvalue())

    # Save detailed stats to file
    ps.dump_stats('profile_results.prof')
    print("\n" + "=" * 70)
    print("Saved detailed profile to 'profile_results.prof'")
    print("View with: python -m pstats profile_results.prof")
    print("=" * 70)

