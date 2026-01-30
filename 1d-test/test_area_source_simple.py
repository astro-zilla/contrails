"""
Simple test to verify area source term with known analytical solution.

Test: Uniform flow through a channel (constant velocity everywhere).
This should be an exact steady-state solution regardless of area variation.
"""

import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cfd import GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig
from cfd.boundary import SubsonicInletBC, SubsonicOutletBC


def test_uniform_flow():
    """
    Test that solver maintains uniform flow through variable area channel.

    For a given mass flow rate, velocity should vary inversely with area:
    u * A = constant
    """
    print("\n" + "=" * 80)
    print("TEST: Uniform Mass Flow Through Variable Area")
    print("=" * 80)

    # Simple linear area variation
    def area_func(x):
        # Linear from 2.0 to 1.0 to 1.5
        if np.isscalar(x):
            x = np.array([x])
        A = np.zeros_like(x)
        mask1 = x <= 0.5
        A[mask1] = 2.0 - 2.0 * x[mask1]  # Converging: 2.0 -> 1.0
        mask2 = x > 0.5
        A[mask2] = 1.0 + (x[mask2] - 0.5)  # Diverging: 1.0 -> 1.5
        return A

    # Setup
    n_cells = 50
    gas = GasProperties(gamma=1.4, R=287.0)
    mesh = Mesh1D.uniform(0.0, 1.0, n_cells, area_func)

    config = SolverConfig(
        cfl=0.5,
        max_iter=30000,
        convergence_tol=1e-8,
        print_interval=5000,
        time_scheme='rk2',
        use_first_order=False
    )

    solver = Solver1D(mesh, gas, n_scalars=0, config=config)

    # Boundary conditions
    p0 = 101325.0
    T0 = 300.0
    p_exit = 98000.0

    bc_left = SubsonicInletBC(p0=p0, T0=T0, Y_inlet=np.array([]))
    bc_right = SubsonicOutletBC(p_exit=p_exit)
    solver.set_boundary_conditions(bc_left, bc_right)

    # Initial condition
    M_init = 0.3
    T_init = T0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)
    p_init = p0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)**(gas.gamma / (gas.gamma - 1))
    rho_init = p_init / (gas.R * T_init)
    u_init = M_init * np.sqrt(gas.gamma * gas.R * T_init)

    initial_state = FlowState(
        rho=np.full(n_cells, rho_init),
        u=np.full(n_cells, u_init),
        p=np.full(n_cells, p_init),
        Y=np.zeros((0, n_cells)),
        gas=gas
    )
    solver.set_initial_condition(initial_state)

    # Solve
    result = solver.solve()
    state = solver.get_state()

    # Check results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final residual: {result['final_residual']:.2e}")

    # Mass flow rate check
    mdot = state.rho * state.u * mesh.A_cells
    mdot_mean = np.mean(mdot)
    mdot_var = (np.max(mdot) - np.min(mdot)) / mdot_mean * 100

    print(f"\nMass flow rate:")
    print(f"  Mean: {mdot_mean:.6f} kg/s")
    print(f"  Min:  {np.min(mdot):.6f} kg/s")
    print(f"  Max:  {np.max(mdot):.6f} kg/s")
    print(f"  Variation: {mdot_var:.4f}%")

    # Check if velocity peaks at throat
    i_throat = np.argmin(mesh.A_cells)
    i_max_vel = np.argmax(state.u)
    i_min_vel = np.argmin(state.u)

    print(f"\nVelocity distribution:")
    print(f"  Inlet (x=0):     {state.u[0]:.2f} m/s, A={mesh.A_cells[0]:.3f}, ρuA={mdot[0]:.4f}")
    print(f"  Throat (x=0.5):  {state.u[i_throat]:.2f} m/s, A={mesh.A_cells[i_throat]:.3f}, ρuA={mdot[i_throat]:.4f}")
    print(f"  Exit (x=1):      {state.u[-1]:.2f} m/s, A={mesh.A_cells[-1]:.3f}, ρuA={mdot[-1]:.4f}")
    print(f"\n  Max velocity at cell {i_max_vel} (x={mesh.x_cells[i_max_vel]:.3f})")
    print(f"  Min velocity at cell {i_min_vel} (x={mesh.x_cells[i_min_vel]:.3f})")
    print(f"  Throat at cell {i_throat} (x={mesh.x_cells[i_throat]:.3f})")

    # Physical check: velocity should be highest where area is smallest
    if i_max_vel == i_throat:
        print(f"\n✓ PASS: Velocity correctly peaks at throat")
    else:
        print(f"\n✗ FAIL: Velocity does NOT peak at throat")
        print(f"         (offset: {abs(i_max_vel - i_throat)} cells)")

    if mdot_var < 1.0:
        print(f"✓ PASS: Mass flow well conserved (<1% variation)")
    else:
        print(f"✗ FAIL: Poor mass flow conservation ({mdot_var:.2f}% variation)")

    return solver, state


if __name__ == "__main__":
    solver, state = test_uniform_flow()

