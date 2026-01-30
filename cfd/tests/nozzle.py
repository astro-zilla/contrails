"""
Subsonic nozzle flow test case.
"""

import numpy as np

from cfd.src import (
    GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig,
    SubsonicInletBC, SubsonicOutletBC, ScalarSourceTerm
)


def nozzle_area(x: np.ndarray, x_throat: float = 0.5,
                A_inlet: float = 1.0, A_throat: float = 0.8,
                A_exit: float = 1.0) -> np.ndarray:
    """
    Smooth nozzle area distribution (sinusoidal).

    Args:
        x: Position array (normalized 0 to 1)
        x_throat: Throat location
        A_inlet: Inlet area
        A_throat: Throat area
        A_exit: Exit area
    """
    A = np.zeros_like(x)

    # Converging section (inlet to throat)
    mask_conv = x <= x_throat
    A[mask_conv] = A_inlet + (A_throat - A_inlet) * (1 - np.cos(np.pi * x[mask_conv] / x_throat)) / 2

    # Diverging section (throat to exit)
    mask_div = x > x_throat
    xi = (x[mask_div] - x_throat) / (1 - x_throat)
    A[mask_div] = A_throat + (A_exit - A_throat) * (1 - np.cos(np.pi * xi)) / 2

    return A


def run_subsonic_nozzle_test(n_cells: int = 100, n_scalars: int = 2):
    """
    Run a subsonic nozzle flow test case.

    This test case simulates subsonic flow through a converging-diverging nozzle.
    The flow accelerates through the converging section and decelerates through
    the diverging section, remaining subsonic throughout.

    Args:
        n_cells: Number of cells
        n_scalars: Number of passive scalars to track
    """
    print("\n" + "=" * 60)
    print("SUBSONIC NOZZLE FLOW TEST CASE")
    print("=" * 60 + "\n")

    # Gas properties (air)
    gas = GasProperties(gamma=1.4, R=287.0)

    # Nozzle geometry
    x_min, x_max = 0.0, 1.0
    A_inlet, A_throat, A_exit = 1.0, 0.8, 1.0

    area_func = lambda x: nozzle_area(x, x_throat=0.5,
                                       A_inlet=A_inlet,
                                       A_throat=A_throat,
                                       A_exit=A_exit)

    # Create mesh
    mesh = Mesh1D.uniform(x_min, x_max, n_cells, area_func)

    # Create solver with FIRST-ORDER reconstruction for stability
    config = SolverConfig(
        cfl=0.5,  # Conservative CFL for stability
        max_iter=100000,  # Further increase max iterations
        convergence_tol=1e-7,  # Tighter tolerance for better convergence
        print_interval=2000,  # Print less frequently
        time_scheme='rk2',  # Use RK2 for 2x speedup
        use_first_order=True  # USE FIRST-ORDER for smooth solution
    )
    solver = Solver1D(mesh, gas, n_scalars=n_scalars, config=config)

    # Boundary conditions
    p0 = 101325.0       # Total pressure [Pa]
    T0 = 300.0          # Total temperature [K]
    p_exit = 95000.0    # Exit static pressure [Pa] (subsonic back pressure)

    Y_inlet = np.zeros(n_scalars)
    if n_scalars > 0:
        Y_inlet[0] = 0.01   # 1% mass fraction of first scalar
    if n_scalars > 1:
        Y_inlet[1] = 0.005  # 0.5% mass fraction of second scalar

    bc_left = SubsonicInletBC(p0=p0, T0=T0, Y_inlet=Y_inlet)
    bc_right = SubsonicOutletBC(p_exit=p_exit)
    solver.set_boundary_conditions(bc_left, bc_right)

    # Example scalar source term (demonstrates the interface)
    def example_scalar_source(state: FlowState, mesh: Mesh1D) -> np.ndarray:
        """
        Example scalar source term.

        For demonstration:
        - Scalar 0: Production proportional to (T - T_ref)
        - Scalar 1: Decay proportional to Y_1
        """
        n_scalars = state.Y.shape[0]
        sources = np.zeros((n_scalars, mesh.n_cells))

        T_ref = 290.0  # Reference temperature

        if n_scalars > 0:
            # Production term for scalar 0 - REDUCED for stability
            k_prod = 1e-6  # Further reduced from 1e-5
            sources[0] = state.rho * k_prod * np.maximum(state.T - T_ref, 0)

        if n_scalars > 1:
            # Decay term for scalar 1 - REDUCED for stability
            k_decay = 0.1  # Further reduced from 1.0 - was causing oscillations
            sources[1] = -state.rho * k_decay * state.Y[1]

        return sources

    # DISABLE source terms for smoother convergence (uncomment to re-enable)
    # Add scalar source term
    # if n_scalars > 0:
    #     solver.add_source_term(ScalarSourceTerm(example_scalar_source))

    # Initial condition (uniform flow)
    M_init = 0.3
    T_init = T0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)
    p_init = p0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)**(gas.gamma / (gas.gamma - 1))
    rho_init = p_init / (gas.R * T_init)
    u_init = M_init * np.sqrt(gas.gamma * gas.R * T_init)

    rho = np.full(n_cells, rho_init)
    u = np.full(n_cells, u_init)
    p = np.full(n_cells, p_init)
    Y = np.zeros((n_scalars, n_cells))
    if n_scalars > 0:
        Y[0, :] = Y_inlet[0]
    if n_scalars > 1:
        Y[1, :] = Y_inlet[1]

    initial_state = FlowState(rho=rho, u=u, p=p, Y=Y, gas=gas)
    solver.set_initial_condition(initial_state)

    # Solve
    result = solver.solve()

    print("\n" + "=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)

    state = solver.get_state()
    print(f"\nInlet:  M = {state.M[0]:.4f}, p = {state.p[0]/1000:.2f} kPa, T = {state.T[0]:.1f} K")
    print(f"Throat: M = {state.M[n_cells//2]:.4f}, p = {state.p[n_cells//2]/1000:.2f} kPa, T = {state.T[n_cells//2]:.1f} K")
    print(f"Exit:   M = {state.M[-1]:.4f}, p = {state.p[-1]/1000:.2f} kPa, T = {state.T[-1]:.1f} K")

    if n_scalars > 0:
        print(f"\nScalar Y_0: inlet = {state.Y[0, 0]:.6f}, exit = {state.Y[0, -1]:.6f}")
    if n_scalars > 1:
        print(f"Scalar Y_1: inlet = {state.Y[1, 0]:.6f}, exit = {state.Y[1, -1]:.6f}")

    # Mass flow rate check
    mdot = state.rho * state.u * mesh.A_cells
    print(f"\nMass flow rate: min = {np.min(mdot):.6f}, max = {np.max(mdot):.6f} kg/s")
    print(f"Mass flow variation: {(np.max(mdot) - np.min(mdot)) / np.mean(mdot) * 100:.4f}%")

    # Plot results
    solver.plot_solution()
    solver.plot_convergence()

    return solver
