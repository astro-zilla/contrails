"""
Validation test case: Quasi-1D isentropic nozzle flow with analytical comparison.

This test compares the numerical solution against the analytical isentropic relations
for quasi-1D nozzle flow to validate the area source term implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cfd import GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig
from cfd.boundary import SubsonicInletBC, SubsonicOutletBC


def nozzle_area_linear(x: np.ndarray, x_throat: float = 0.5,
                       A_inlet: float = 2.0, A_throat: float = 1.0,
                       A_exit: float = 1.5) -> np.ndarray:
    """
    Linear nozzle area distribution for easier analytical comparison.

    Args:
        x: Position array (0 to 1)
        x_throat: Throat location
        A_inlet: Inlet area
        A_throat: Throat area
        A_exit: Exit area
    """
    A = np.zeros_like(x)

    # Converging section (linear)
    mask_conv = x <= x_throat
    A[mask_conv] = A_inlet + (A_throat - A_inlet) * (x[mask_conv] / x_throat)

    # Diverging section (linear)
    mask_div = x > x_throat
    xi = (x[mask_div] - x_throat) / (1 - x_throat)
    A[mask_div] = A_throat + (A_exit - A_throat) * xi

    return A


def analytical_isentropic_nozzle(A_ratio: np.ndarray, M_inlet: float,
                                  gamma: float = 1.4) -> dict:
    """
    Compute analytical isentropic nozzle flow properties.

    For isentropic flow with known inlet Mach number, we can compute
    the area-Mach relation and flow properties.

    Args:
        A_ratio: A/A_throat ratio at each location
        M_inlet: Inlet Mach number
        gamma: Specific heat ratio

    Returns:
        Dictionary with M, p_ratio, T_ratio, rho_ratio, u_ratio
    """
    gm1 = gamma - 1

    # Compute inlet area ratio from inlet Mach number
    A_star_inlet = ((gamma + 1) / 2) ** ((gamma + 1) / (2 * gm1))
    A_star_inlet /= (1 + 0.5 * gm1 * M_inlet**2) ** ((gamma + 1) / (2 * gm1))
    A_inlet_over_A_throat = A_ratio[0]
    A_throat_over_A_star = A_inlet_over_A_throat / A_star_inlet

    # For subsonic flow, solve area-Mach relation iteratively
    M = np.zeros_like(A_ratio)

    for i, A_rat in enumerate(A_ratio):
        # Subsonic branch of area-Mach relation
        A_over_A_star = A_rat / A_throat_over_A_star

        # Newton iteration to solve for M
        M_guess = 0.3 if i == 0 else M[i-1]

        for _ in range(50):
            f = ((gamma + 1) / 2) ** ((gamma + 1) / (2 * gm1))
            f /= (1 + 0.5 * gm1 * M_guess**2) ** ((gamma + 1) / (2 * gm1))
            f /= M_guess
            f -= A_over_A_star

            # Derivative
            numerator = (gamma + 1) / (2 * gm1)
            term = 1 + 0.5 * gm1 * M_guess**2
            df = -f / M_guess
            df += ((gamma + 1) * M_guess) / (2 * term * M_guess * A_over_A_star)

            M_new = M_guess - f / df
            M_new = max(0.01, min(0.99, M_new))  # Keep subsonic

            if abs(M_new - M_guess) < 1e-8:
                break
            M_guess = M_new

        M[i] = M_guess

    # Compute ratios from isentropic relations
    T_ratio = 1 / (1 + 0.5 * gm1 * M**2)
    p_ratio = T_ratio ** (gamma / gm1)
    rho_ratio = T_ratio ** (1 / gm1)

    # Velocity ratio (from continuity)
    u_ratio = M * np.sqrt(T_ratio) * (rho_ratio * A_ratio) / (rho_ratio[0] * A_ratio[0] * M[0] / np.sqrt(T_ratio[0]))

    return {
        'M': M,
        'p_ratio': p_ratio,
        'T_ratio': T_ratio,
        'rho_ratio': rho_ratio,
        'u_ratio': u_ratio
    }


def run_validation_test():
    """
    Run validation test comparing numerical solution to analytical isentropic flow.
    """
    print("\n" + "=" * 80)
    print("VALIDATION TEST: Isentropic Nozzle Flow vs Analytical Solution")
    print("=" * 80 + "\n")

    # Test parameters
    n_cells = 200  # Higher resolution for better comparison

    # Gas properties
    gas = GasProperties(gamma=1.4, R=287.0)

    # Nozzle geometry - linear for easier analytical comparison
    area_func = lambda x: nozzle_area_linear(x, x_throat=0.5,
                                             A_inlet=2.0, A_throat=1.0, A_exit=1.5)
    mesh = Mesh1D.uniform(0.0, 1.0, n_cells, area_func)

    # Solver configuration
    config = SolverConfig(
        cfl=0.5,
        max_iter=50000,
        convergence_tol=1e-8,
        print_interval=5000,
        time_scheme='rk2',
        use_first_order=False  # Use second-order for better accuracy
    )
    solver = Solver1D(mesh, gas, n_scalars=0, config=config)

    # Boundary conditions
    p0 = 101325.0
    T0 = 300.0
    p_exit = 98000.0  # Mild pressure drop

    bc_left = SubsonicInletBC(p0=p0, T0=T0, Y_inlet=np.array([]))
    bc_right = SubsonicOutletBC(p_exit=p_exit)
    solver.set_boundary_conditions(bc_left, bc_right)

    # Initial condition
    M_init = 0.2
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
    print("\n" + "=" * 80)
    print("Running CFD Solver...")
    print("=" * 80)
    result = solver.solve()

    # Get numerical solution
    state = solver.get_state()
    x = mesh.x_cells

    # Compute analytical solution
    print("\n" + "=" * 80)
    print("Computing Analytical Solution...")
    print("=" * 80)
    A_ratio = mesh.A_cells / np.min(mesh.A_cells)
    analytical = analytical_isentropic_nozzle(A_ratio, state.M[0], gas.gamma)

    # Normalize numerical solution for comparison
    p_ratio_num = state.p / state.p[0]
    T_ratio_num = state.T / state.T[0]
    rho_ratio_num = state.rho / state.rho[0]
    u_ratio_num = state.u / state.u[0]

    # Compute errors
    M_error = np.abs(state.M - analytical['M'])
    p_error = np.abs(p_ratio_num - analytical['p_ratio'])
    T_error = np.abs(T_ratio_num - analytical['T_ratio'])

    # Statistics
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"\nSolver converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final residual: {result['final_residual']:.2e}")

    print(f"\n{'Property':<20} {'Max Error':<15} {'RMS Error':<15} {'Mean Error':<15}")
    print("-" * 65)
    print(f"{'Mach number':<20} {np.max(M_error):.6f}      {np.sqrt(np.mean(M_error**2)):.6f}      {np.mean(M_error):.6f}")
    print(f"{'Pressure ratio':<20} {np.max(p_error):.6f}      {np.sqrt(np.mean(p_error**2)):.6f}      {np.mean(p_error):.6f}")
    print(f"{'Temperature ratio':<20} {np.max(T_error):.6f}      {np.sqrt(np.mean(T_error**2)):.6f}      {np.mean(T_error):.6f}")

    # Check conservation
    mdot = state.rho * state.u * mesh.A_cells
    mdot_var = (np.max(mdot) - np.min(mdot)) / np.mean(mdot) * 100

    T0_calc = state.T * (1 + 0.5 * (gas.gamma - 1) * state.M**2)
    p0_calc = state.p * (1 + 0.5 * (gas.gamma - 1) * state.M**2)**(gas.gamma / (gas.gamma - 1))
    T0_var = (np.max(T0_calc) - np.min(T0_calc)) / np.mean(T0_calc) * 100
    p0_var = (np.max(p0_calc) - np.min(p0_calc)) / np.mean(p0_calc) * 100

    print(f"\n{'Conservation Check':<20} {'Variation %':<15}")
    print("-" * 40)
    print(f"{'Mass flow rate':<20} {mdot_var:.4f}%")
    print(f"{'Stagnation temp':<20} {T0_var:.4f}%")
    print(f"{'Stagnation pressure':<20} {p0_var:.4f}%")

    # Throat analysis
    i_throat = np.argmin(mesh.A_cells)
    i_max_vel = np.argmax(state.u)
    i_max_M = np.argmax(state.M)

    print(f"\n{'Throat Analysis':<30} {'Cell Index':<15} {'Position':<15}")
    print("-" * 60)
    print(f"{'Area minimum (throat)':<30} {i_throat:<15} {x[i_throat]:.4f}")
    print(f"{'Maximum velocity':<30} {i_max_vel:<15} {x[i_max_vel]:.4f}")
    print(f"{'Maximum Mach number':<30} {i_max_M:<15} {x[i_max_M]:.4f}")

    throat_aligned = (i_throat == i_max_vel == i_max_M)
    print(f"\n{'Throat alignment check:':<30} {'✓ PASS' if throat_aligned else '✗ FAIL'}")

    # Create comparison plots
    create_validation_plots(x, mesh, state, analytical,
                           A_ratio, M_error, p_error, T_error)

    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    max_M_error = np.max(M_error)
    pass_M = max_M_error < 0.01  # 1% Mach number error
    pass_conservation = mdot_var < 0.5  # 0.5% mass flow variation
    pass_throat = throat_aligned

    print(f"\nMach number accuracy:     {'✓ PASS' if pass_M else '✗ FAIL'} (max error: {max_M_error:.4f})")
    print(f"Mass conservation:        {'✓ PASS' if pass_conservation else '✗ FAIL'} ({mdot_var:.4f}%)")
    print(f"Throat velocity peak:     {'✓ PASS' if pass_throat else '✗ FAIL'}")

    all_pass = pass_M and pass_conservation and pass_throat

    if all_pass:
        print(f"\n{'='*80}")
        print("✓✓✓ ALL TESTS PASSED - Area source term implementation is CORRECT ✓✓✓")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("✗ SOME TESTS FAILED - Further investigation needed")
        print(f"{'='*80}\n")

    return solver, state, analytical


def create_validation_plots(x, mesh, state, analytical, A_ratio,
                            M_error, p_error, T_error):
    """Create comprehensive validation plots."""

    fig = plt.figure(figsize=(16, 10))

    # Main comparison plots (2x2 grid on left)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.4)

    # 1. Mach number comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, state.M, 'b-', linewidth=2, label='CFD')
    ax1.plot(x, analytical['M'], 'r--', linewidth=2, label='Analytical')
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('Mach Number')
    ax1.set_title('Mach Number Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Pressure ratio comparison
    ax2 = fig.add_subplot(gs[0, 1])
    p_ratio_num = state.p / state.p[0]
    ax2.plot(x, p_ratio_num, 'b-', linewidth=2, label='CFD')
    ax2.plot(x, analytical['p_ratio'], 'r--', linewidth=2, label='Analytical')
    ax2.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('p/p₀')
    ax2.set_title('Pressure Ratio Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Temperature ratio comparison
    ax3 = fig.add_subplot(gs[0, 2])
    T_ratio_num = state.T / state.T[0]
    ax3.plot(x, T_ratio_num, 'b-', linewidth=2, label='CFD')
    ax3.plot(x, analytical['T_ratio'], 'r--', linewidth=2, label='Analytical')
    ax3.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('T/T₀')
    ax3.set_title('Temperature Ratio Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Error plots
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(x, M_error, 'b-', linewidth=2)
    ax4.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('x [m]')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Mach Number Error')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.semilogy(x, p_error, 'r-', linewidth=2)
    ax5.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax5.set_xlabel('x [m]')
    ax5.set_ylabel('Absolute Error')
    ax5.set_title('Pressure Ratio Error')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.semilogy(x, T_error, 'g-', linewidth=2)
    ax6.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax6.set_xlabel('x [m]')
    ax6.set_ylabel('Absolute Error')
    ax6.set_title('Temperature Ratio Error')
    ax6.grid(True, alpha=0.3)

    # 5. Area and velocity (with dual axis)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7_twin = ax7.twinx()
    ax7.plot(x, state.u, 'b-', linewidth=2, label='Velocity')
    ax7.set_xlabel('x [m]')
    ax7.set_ylabel('Velocity [m/s]', color='b')
    ax7.tick_params(axis='y', labelcolor='b')
    ax7_twin.plot(mesh.x_faces, mesh.A_faces, 'c--', linewidth=2, label='Area')
    ax7_twin.set_ylabel('Area [m²]', color='c')
    ax7_twin.tick_params(axis='y', labelcolor='c')
    ax7_twin.invert_yaxis()
    ax7.set_title('Velocity vs Area')
    ax7.grid(True, alpha=0.3)

    # 6. Mass flow rate
    ax8 = fig.add_subplot(gs[2, 1])
    mdot = state.rho * state.u * mesh.A_cells
    ax8.plot(x, mdot, 'b-', linewidth=2)
    ax8.axhline(y=np.mean(mdot), color='k', linestyle='--', linewidth=1, label=f'Mean: {np.mean(mdot):.6f}')
    ax8.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax8.set_xlabel('x [m]')
    ax8.set_ylabel('ρuA [kg/s]')
    ax8.set_title('Mass Flow Rate (Conservation Check)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 7. Stagnation properties
    ax9 = fig.add_subplot(gs[2, 2])
    T0_calc = state.T * (1 + 0.5 * (1.4 - 1) * state.M**2)
    p0_calc = state.p * (1 + 0.5 * (1.4 - 1) * state.M**2)**(1.4 / (1.4 - 1))
    ax9.plot(x, T0_calc / T0_calc[0], 'r-', linewidth=2, label='T₀/T₀,inlet')
    ax9.plot(x, p0_calc / p0_calc[0], 'b-', linewidth=2, label='p₀/p₀,inlet')
    ax9.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
    ax9.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax9.set_xlabel('x [m]')
    ax9.set_ylabel('Ratio')
    ax9.set_title('Stagnation Properties (should be constant)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.suptitle('Validation: CFD vs Analytical Isentropic Nozzle Flow',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('validation_isentropic_nozzle.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved validation plots to: validation_isentropic_nozzle.png")


if __name__ == "__main__":
    solver, state, analytical = run_validation_test()
    plt.show()

