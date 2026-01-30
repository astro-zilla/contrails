"""
Performance profiling script for the 1D CFD solver.
"""

import time
import numpy as np
from cfd import GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig
from cfd.boundary import SubsonicInletBC, SubsonicOutletBC

def profile_solver():
    """Profile the solver to find bottlenecks."""

    print("=" * 60)
    print("PERFORMANCE PROFILING")
    print("=" * 60)

    # Small test case
    n_cells = 100
    n_scalars = 2

    # Gas properties
    gas = GasProperties(gamma=1.4, R=287.0)

    # Simple nozzle
    area_func = lambda x: 1.0 - 0.2 * np.sin(np.pi * x)
    mesh = Mesh1D.uniform(0.0, 1.0, n_cells, area_func)

    # Solver with minimal iterations
    config = SolverConfig(
        cfl=0.8,
        max_iter=500,  # Only 500 iterations for profiling
        convergence_tol=1e-6,
        print_interval=100
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
    print(f"\nRunning 500 iterations...")
    start_time = time.time()
    result = solver.solve()
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"\n{'=' * 60}")
    print(f"PROFILING RESULTS")
    print(f"{'=' * 60}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Iterations: {solver.iteration}")
    print(f"Time per iteration: {elapsed / solver.iteration * 1000:.2f} ms")
    print(f"Expected time for 5000 iterations: {elapsed * 10:.1f} seconds")

    if elapsed > 10:
        print(f"\n⚠️  WARNING: Still too slow!")
        print(f"Expected: < 10 seconds for 500 iterations")
        print(f"Actual: {elapsed:.2f} seconds")
        print(f"\nBottleneck is likely still in the flux computation or reconstruction.")
    else:
        print(f"\n✓ Performance is good!")
        print(f"Solver is running at expected speed.")

    return solver, elapsed

if __name__ == "__main__":
    solver, elapsed = profile_solver()

