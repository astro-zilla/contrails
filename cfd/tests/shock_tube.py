"""
Sod's shock tube test case - classic validation for compressible flow solvers.

The shock tube problem (Sod, 1978) is a Riemann problem with:
- Left state: high pressure gas at rest
- Right state: low pressure gas at rest
- Initial discontinuity at x = 0.5

The exact solution consists of:
1. Left state (undisturbed)
2. Rarefaction fan
3. Contact discontinuity
4. Shock wave
5. Right state (undisturbed)

This provides a rigorous test of:
- Shock capturing ability
- Contact discontinuity resolution
- Rarefaction wave accuracy
"""

import numpy as np
from typing import Tuple

from cfd import (
    GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig,
    BoundaryCondition
)


class ReflectiveWallBC(BoundaryCondition):
    """Reflective wall boundary condition for shock tube."""

    def apply(self, U: np.ndarray, mesh: Mesh1D, gas: GasProperties,
              side: str) -> np.ndarray:
        if side == 'left':
            # Reflect at left boundary
            U[:, 0] = U[:, 1]
            U[1, 0] = -U[1, 1]  # Reverse momentum
        elif side == 'right':
            # Reflect at right boundary
            U[:, -1] = U[:, -2]
            U[1, -1] = -U[1, -2]  # Reverse momentum
        return U


def sod_shock_tube_exact(x: np.ndarray, t: float, gamma: float = 1.4) -> dict:
    """
    Exact solution to Sod's shock tube problem.

    Initial conditions (SI units):
    - Left:  ρ = 1.0 kg/m³, u = 0 m/s, p = 100 kPa
    - Right: ρ = 0.125 kg/m³, u = 0 m/s, p = 10 kPa
    - Diaphragm at x = 0.5 m

    Args:
        x: Position array [m]
        t: Time [s]
        gamma: Specific heat ratio

    Returns:
        Dictionary with exact solution: rho, u, p, e
    """
    # Initial conditions
    rho_L = 1.0
    u_L = 0.0
    p_L = 100000.0

    rho_R = 0.125
    u_R = 0.0
    p_R = 10000.0

    # Speed of sound
    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    # Compute post-shock properties using Rankine-Hugoniot relations
    # This requires iterative solution of the shock relations

    # Use approximate solution (Newton iteration)
    # For exact solution, need to solve: f(p) = 0
    # where f relates pressure jump across shock and rarefaction

    gm1 = gamma - 1
    gp1 = gamma + 1

    # Initial guess for pressure in star region
    p_star_guess = 0.5 * (p_L + p_R)

    # Newton iteration
    for _ in range(20):
        # Rarefaction on left
        A_L = 2 / (gp1 * rho_L)
        B_L = gm1 / gp1 * p_L
        p_rat_L = p_star_guess / p_L

        if p_star_guess > p_L:
            # Shock on left (shouldn't happen for Sod)
            f_L = (p_star_guess - p_L) * np.sqrt(A_L / (p_star_guess + B_L))
            df_L = np.sqrt(A_L / (p_star_guess + B_L)) * (1 - 0.5 * (p_star_guess - p_L) / (p_star_guess + B_L))
        else:
            # Rarefaction on left
            f_L = 2 * a_L / gm1 * (p_rat_L**(gm1/(2*gamma)) - 1)
            df_L = a_L / (gamma * p_L) * p_rat_L**(gm1/(2*gamma))

        # Shock on right
        A_R = 2 / (gp1 * rho_R)
        B_R = gm1 / gp1 * p_R

        if p_star_guess > p_R:
            # Shock on right
            f_R = (p_star_guess - p_R) * np.sqrt(A_R / (p_star_guess + B_R))
            df_R = np.sqrt(A_R / (p_star_guess + B_R)) * (1 - 0.5 * (p_star_guess - p_R) / (p_star_guess + B_R))
        else:
            # Rarefaction on right (shouldn't happen for Sod)
            f_R = 2 * a_R / gm1 * ((p_star_guess/p_R)**(gm1/(2*gamma)) - 1)
            df_R = a_R / (gamma * p_R) * (p_star_guess/p_R)**(gm1/(2*gamma))

        f = f_L + f_R + (u_R - u_L)
        df = df_L + df_R

        p_new = p_star_guess - f / df
        p_new = max(0.001 * p_R, p_new)  # Ensure positive

        if abs(p_new - p_star_guess) / p_star_guess < 1e-8:
            break

        p_star_guess = p_new

    p_star = p_star_guess
    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)

    # Post-shock density (right side)
    p_ratio = p_star / p_R
    rho_star_R = rho_R * (p_ratio + gm1/gp1) / (gm1/gp1 * p_ratio + 1)

    # Post-rarefaction density (left side)
    rho_star_L = rho_L * (p_star / p_L)**(1/gamma)

    # Wave speeds
    # Shock speed (right)
    S = u_R + a_R * np.sqrt(gp1/(2*gamma) * p_ratio + gm1/(2*gamma))

    # Contact discontinuity
    C = u_star

    # Head of rarefaction fan
    H = u_L - a_L

    # Tail of rarefaction fan
    a_star_L = a_L * (p_star / p_L)**(gm1/(2*gamma))
    T = u_star - a_star_L

    # Initialize solution arrays
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)

    # Position of initial discontinuity
    x0 = 0.5

    for i, xi in enumerate(x):
        # Position relative to initial discontinuity
        s = (xi - x0) / t if t > 0 else 0

        if s < H:
            # Left state (undisturbed)
            rho[i] = rho_L
            u[i] = u_L
            p[i] = p_L
        elif s < T:
            # Rarefaction fan
            u[i] = 2/gp1 * (a_L + s)
            a = a_L - 0.5 * gm1 * u[i]
            rho[i] = rho_L * (a / a_L)**(2/gm1)
            p[i] = p_L * (a / a_L)**(2*gamma/gm1)
        elif s < C:
            # Between rarefaction and contact (star region left)
            rho[i] = rho_star_L
            u[i] = u_star
            p[i] = p_star
        elif s < S:
            # Between contact and shock (star region right)
            rho[i] = rho_star_R
            u[i] = u_star
            p[i] = p_star
        else:
            # Right state (undisturbed)
            rho[i] = rho_R
            u[i] = u_R
            p[i] = p_R

    # Compute internal energy
    e = p / ((gamma - 1) * rho)

    return {
        'rho': rho,
        'u': u,
        'p': p,
        'e': e
    }


def run_shock_tube_test(n_cells: int = 200, t_final: float = 0.0002,
                        cfl: float = 0.4, use_first_order: bool = False):
    """
    Run Sod's shock tube test case.

    Args:
        n_cells: Number of computational cells
        t_final: Final simulation time [s]
        cfl: CFL number for time stepping
        use_first_order: Use first-order reconstruction (more dissipative)

    Returns:
        solver: Solver object with final solution
        exact: Exact solution at final time
    """
    print("\n" + "=" * 80)
    print("SOD'S SHOCK TUBE TEST CASE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Cells: {n_cells}")
    print(f"  Final time: {t_final*1000:.2f} ms")
    print(f"  CFL: {cfl}")
    print(f"  Spatial order: {'1st' if use_first_order else '2nd'} order")
    print()

    # Gas properties (air)
    gas = GasProperties(gamma=1.4, R=287.0)

    # Uniform mesh with constant area (1D shock tube)
    area_func = lambda x: np.ones_like(x) if hasattr(x, '__len__') else 1.0
    mesh = Mesh1D.uniform(0.0, 1.0, n_cells, area_func)

    # Solver configuration - unsteady problem!
    config = SolverConfig(
        cfl=cfl,
        max_iter=100000,
        convergence_tol=1e-10,  # Run to final time, not to steady state
        print_interval=1000,
        time_scheme='rk2',
        use_first_order=use_first_order
    )

    solver = Solver1D(mesh, gas, n_scalars=0, config=config)

    # Reflective wall boundary conditions
    bc_wall = ReflectiveWallBC()
    solver.set_boundary_conditions(bc_wall, bc_wall)

    # Initial condition - Sod's shock tube
    # Left state: high pressure
    # Right state: low pressure
    # Discontinuity at x = 0.5

    rho = np.where(mesh.x_cells < 0.5, 1.0, 0.125)
    u = np.zeros(n_cells)
    p = np.where(mesh.x_cells < 0.5, 100000.0, 10000.0)

    initial_state = FlowState(
        rho=rho,
        u=u,
        p=p,
        Y=np.zeros((0, n_cells)),
        gas=gas
    )
    solver.set_initial_condition(initial_state)

    # Solve to final time
    print("Running simulation...")
    result = solver.solve(max_time=t_final)

    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Time reached: {solver.time*1000:.4f} ms")
    print(f"Iterations: {solver.iteration}")
    print(f"Final time step: {result['time']/result['iterations']*1e6:.3f} µs")

    # Get final state
    state = solver.get_state()

    # Compute exact solution at final time
    exact = sod_shock_tube_exact(mesh.x_cells, solver.time, gas.gamma)

    # Compute errors
    rho_error = np.abs(state.rho - exact['rho'])
    u_error = np.abs(state.u - exact['u'])
    p_error = np.abs(state.p - exact['p'])

    print(f"\nError Analysis:")
    print(f"  Density L1 error:  {np.mean(rho_error):.6f}")
    print(f"  Density L∞ error:  {np.max(rho_error):.6f}")
    print(f"  Velocity L1 error: {np.mean(u_error):.6f}")
    print(f"  Velocity L∞ error: {np.max(u_error):.6f}")
    print(f"  Pressure L1 error: {np.mean(p_error):.6f}")
    print(f"  Pressure L∞ error: {np.max(p_error):.6f}")

    return solver, exact


if __name__ == "__main__":
    # Run with different resolutions
    for n_cells in [200, 400]:
        solver, exact = run_shock_tube_test(
            n_cells=n_cells,
            t_final=0.0002,  # 0.2 ms
            cfl=0.4,
            use_first_order=False
        )

