"""
Pytest tests for Sod's shock tube problem.

Tests verify:
1. Shock capturing ability
2. Contact discontinuity resolution
3. Comparison with exact solution
4. Physical bounds maintained
5. Conservation properties
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cfd.src import (
    GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig,
    BoundaryCondition
)


class ReflectiveWallBC(BoundaryCondition):
    """Reflective wall boundary condition for shock tube."""

    def apply(self, U: np.ndarray, mesh: Mesh1D, gas: GasProperties,
              side: str) -> np.ndarray:
        if side == 'left':
            U[:, 0] = U[:, 1]
            U[1, 0] = -U[1, 1]  # Reverse momentum
        elif side == 'right':
            U[:, -1] = U[:, -2]
            U[1, -1] = -U[1, -2]  # Reverse momentum
        return U


def sod_shock_tube_exact(x: np.ndarray, t: float, gamma: float = 1.4) -> dict:
    """
    Exact solution to Sod's shock tube problem.

    Initial conditions (SI units):
    - Left:  rho = 1.0 kg/m³, u = 0 m/s, p = 100 kPa
    - Right: rho = 0.125 kg/m³, u = 0 m/s, p = 10 kPa
    - Diaphragm at x = 0.5 m
    """
    # Initial conditions
    rho_L, u_L, p_L = 1.0, 0.0, 100000.0
    rho_R, u_R, p_R = 0.125, 0.0, 10000.0

    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    gm1 = gamma - 1
    gp1 = gamma + 1

    # Newton iteration for pressure in star region
    p_star_guess = 0.5 * (p_L + p_R)

    for _ in range(20):
        A_L = 2 / (gp1 * rho_L)
        B_L = gm1 / gp1 * p_L
        p_rat_L = p_star_guess / p_L

        if p_star_guess > p_L:
            f_L = (p_star_guess - p_L) * np.sqrt(A_L / (p_star_guess + B_L))
            df_L = np.sqrt(A_L / (p_star_guess + B_L)) * (1 - 0.5 * (p_star_guess - p_L) / (p_star_guess + B_L))
        else:
            f_L = 2 * a_L / gm1 * (p_rat_L**(gm1/(2*gamma)) - 1)
            df_L = a_L / (gamma * p_L) * p_rat_L**(gm1/(2*gamma))

        A_R = 2 / (gp1 * rho_R)
        B_R = gm1 / gp1 * p_R

        if p_star_guess > p_R:
            f_R = (p_star_guess - p_R) * np.sqrt(A_R / (p_star_guess + B_R))
            df_R = np.sqrt(A_R / (p_star_guess + B_R)) * (1 - 0.5 * (p_star_guess - p_R) / (p_star_guess + B_R))
        else:
            f_R = 2 * a_R / gm1 * ((p_star_guess/p_R)**(gm1/(2*gamma)) - 1)
            df_R = a_R / (gamma * p_R) * (p_star_guess/p_R)**(gm1/(2*gamma))

        f = f_L + f_R + (u_R - u_L)
        df = df_L + df_R

        p_new = p_star_guess - f / df
        p_new = max(0.001 * p_R, p_new)

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
    S = u_R + a_R * np.sqrt(gp1/(2*gamma) * p_ratio + gm1/(2*gamma))
    C = u_star
    H = u_L - a_L
    a_star_L = a_L * (p_star / p_L)**(gm1/(2*gamma))
    T = u_star - a_star_L

    # Initialize solution arrays
    rho = np.zeros_like(x)
    u = np.zeros_like(x)
    p = np.zeros_like(x)
    x0 = 0.5

    for i, xi in enumerate(x):
        s = (xi - x0) / t if t > 0 else 0

        if s < H:
            rho[i], u[i], p[i] = rho_L, u_L, p_L
        elif s < T:
            u[i] = 2/gp1 * (a_L + s)
            a = a_L - 0.5 * gm1 * u[i]
            rho[i] = rho_L * (a / a_L)**(2/gm1)
            p[i] = p_L * (a / a_L)**(2*gamma/gm1)
        elif s < C:
            rho[i], u[i], p[i] = rho_star_L, u_star, p_star
        elif s < S:
            rho[i], u[i], p[i] = rho_star_R, u_star, p_star
        else:
            rho[i], u[i], p[i] = rho_R, u_R, p_R

    return {'rho': rho, 'u': u, 'p': p}


@pytest.fixture
def gas():
    """Standard air properties."""
    return GasProperties(gamma=1.4, R=287.0)


@pytest.fixture
def shock_tube_mesh():
    """Standard shock tube mesh."""
    n_cells = 200
    area_func = lambda x: np.ones_like(x) if hasattr(x, '__len__') else 1.0
    return Mesh1D.uniform(0.0, 1.0, n_cells, area_func)


@pytest.fixture
def solver_config():
    """Solver configuration for shock tube tests."""
    return SolverConfig(
        cfl=0.4,
        max_iter=100000,
        convergence_tol=1e-10,  # Run to final time, not steady state
        print_interval=5000,
        time_scheme='rk2',
        use_first_order=False
    )


def create_shock_tube_solver(gas, mesh, config):
    """Create a configured shock tube solver."""
    solver = Solver1D(mesh, gas, n_scalars=0, config=config)

    bc_wall = ReflectiveWallBC()
    solver.set_boundary_conditions(bc_wall, bc_wall)

    # Initial condition - Sod's shock tube
    rho = np.where(mesh.x_cells < 0.5, 1.0, 0.125)
    u = np.zeros(mesh.n_cells)
    p = np.where(mesh.x_cells < 0.5, 100000.0, 10000.0)

    initial_state = FlowState(
        rho=rho, u=u, p=p,
        Y=np.zeros((0, mesh.n_cells)),
        gas=gas
    )
    solver.set_initial_condition(initial_state)

    return solver


class TestShockCapturing:
    """Tests for shock capturing ability."""

    def test_shock_forms(self, gas, shock_tube_mesh, solver_config):
        """A shock wave should form and propagate to the right."""
        solver = create_shock_tube_solver(gas, shock_tube_mesh, solver_config)
        solver.solve(max_time=0.0002)
        state = solver.get_state()

        # Check that pressure has a sharp jump (shock)
        p_gradient = np.abs(np.diff(state.p))
        max_gradient = np.max(p_gradient)

        # There should be a significant pressure gradient (shock)
        # Threshold based on initial pressure jump (100kPa - 10kPa = 90kPa)
        # spread over a few cells
        assert max_gradient > 5000, \
            f"No significant shock detected, max pressure gradient = {max_gradient}"

    def test_shock_moves_right(self, gas, shock_tube_mesh, solver_config):
        """Shock should propagate to the right (x > 0.5)."""
        solver = create_shock_tube_solver(gas, shock_tube_mesh, solver_config)
        solver.solve(max_time=0.0002)
        state = solver.get_state()

        # Find the shock location (maximum pressure gradient)
        p_gradient = np.abs(np.diff(state.p))
        shock_idx = np.argmax(p_gradient)
        shock_x = shock_tube_mesh.x_cells[shock_idx]

        assert shock_x > 0.5, \
            f"Shock should be to the right of initial position, found at x = {shock_x}"


class TestExactSolutionComparison:
    """Tests comparing numerical solution to exact solution."""

    def test_density_accuracy(self, gas, shock_tube_mesh, solver_config):
        """Density should match exact solution within tolerance."""
        solver = create_shock_tube_solver(gas, shock_tube_mesh, solver_config)
        solver.solve(max_time=0.0002)
        state = solver.get_state()

        exact = sod_shock_tube_exact(shock_tube_mesh.x_cells, solver.time, gas.gamma)
        rho_error_l1 = np.mean(np.abs(state.rho - exact['rho']))

        # L1 error should be less than 5% of mean density
        assert rho_error_l1 < 0.05 * np.mean(exact['rho']), \
            f"Density L1 error too large: {rho_error_l1}"

    def test_velocity_accuracy(self, gas, shock_tube_mesh, solver_config):
        """Velocity should match exact solution within tolerance."""
        solver = create_shock_tube_solver(gas, shock_tube_mesh, solver_config)
        solver.solve(max_time=0.0002)
        state = solver.get_state()

        exact = sod_shock_tube_exact(shock_tube_mesh.x_cells, solver.time, gas.gamma)

        # Compute L1 error normalized by max velocity
        u_max = np.max(np.abs(exact['u']))
        u_error_l1 = np.mean(np.abs(state.u - exact['u']))

        assert u_error_l1 < 0.1 * u_max, \
            f"Velocity L1 error too large: {u_error_l1}"

    def test_pressure_accuracy(self, gas, shock_tube_mesh, solver_config):
        """Pressure should match exact solution within tolerance."""
        solver = create_shock_tube_solver(gas, shock_tube_mesh, solver_config)
        solver.solve(max_time=0.0002)
        state = solver.get_state()

        exact = sod_shock_tube_exact(shock_tube_mesh.x_cells, solver.time, gas.gamma)
        p_error_l1 = np.mean(np.abs(state.p - exact['p']))

        # L1 error should be less than 5% of mean pressure
        assert p_error_l1 < 0.05 * np.mean(exact['p']), \
            f"Pressure L1 error too large: {p_error_l1}"


class TestPhysicalBounds:
    """Tests for physical validity of solution."""

    def test_positive_density(self, gas, shock_tube_mesh, solver_config):
        """Density must be positive everywhere."""
        solver = create_shock_tube_solver(gas, shock_tube_mesh, solver_config)
        solver.solve(max_time=0.0002)
        state = solver.get_state()

        assert np.all(state.rho > 0), \
            f"Negative density detected, min = {np.min(state.rho)}"

    def test_positive_pressure(self, gas, shock_tube_mesh, solver_config):
        """Pressure must be positive everywhere."""
        solver = create_shock_tube_solver(gas, shock_tube_mesh, solver_config)
        solver.solve(max_time=0.0002)
        state = solver.get_state()

        assert np.all(state.p > 0), \
            f"Negative pressure detected, min = {np.min(state.p)}"

    def test_positive_temperature(self, gas, shock_tube_mesh, solver_config):
        """Temperature must be positive everywhere."""
        solver = create_shock_tube_solver(gas, shock_tube_mesh, solver_config)
        solver.solve(max_time=0.0002)
        state = solver.get_state()

        assert np.all(state.T > 0), \
            f"Negative temperature detected, min = {np.min(state.T)}"


class TestConservation:
    """Tests for conservation properties."""

    def test_mass_conservation(self, gas, shock_tube_mesh, solver_config):
        """Total mass should be conserved (closed domain)."""
        solver = create_shock_tube_solver(gas, shock_tube_mesh, solver_config)

        # Get initial mass
        state_init = solver.get_state()
        mass_init = np.sum(state_init.rho * shock_tube_mesh.A_cells * shock_tube_mesh.dx)

        # Run simulation
        solver.solve(max_time=0.0002)

        # Get final mass
        state_final = solver.get_state()
        mass_final = np.sum(state_final.rho * shock_tube_mesh.A_cells * shock_tube_mesh.dx)

        # Mass should be conserved to machine precision (reflective BCs)
        mass_error = abs(mass_final - mass_init) / mass_init
        assert mass_error < 1e-10, \
            f"Mass not conserved, error = {mass_error*100:.2e}%"

    def test_energy_conservation(self, gas, shock_tube_mesh, solver_config):
        """Total energy should be conserved (closed domain)."""
        solver = create_shock_tube_solver(gas, shock_tube_mesh, solver_config)

        # Get initial energy
        state_init = solver.get_state()
        E_init = np.sum(state_init.rho * state_init.E *
                       shock_tube_mesh.A_cells * shock_tube_mesh.dx)

        # Run simulation
        solver.solve(max_time=0.0002)

        # Get final energy
        state_final = solver.get_state()
        E_final = np.sum(state_final.rho * state_final.E *
                        shock_tube_mesh.A_cells * shock_tube_mesh.dx)

        # Energy should be conserved to machine precision (reflective BCs)
        energy_error = abs(E_final - E_init) / E_init
        assert energy_error < 1e-10, \
            f"Energy not conserved, error = {energy_error*100:.2e}%"


class TestGridConvergence:
    """Tests for grid convergence."""

    def test_error_decreases_with_resolution(self, gas, solver_config):
        """Error should decrease as grid is refined."""
        errors = []
        resolutions = [100, 200]

        for n_cells in resolutions:
            area_func = lambda x: np.ones_like(x) if hasattr(x, '__len__') else 1.0
            mesh = Mesh1D.uniform(0.0, 1.0, n_cells, area_func)

            solver = create_shock_tube_solver(gas, mesh, solver_config)
            solver.solve(max_time=0.0002)
            state = solver.get_state()

            exact = sod_shock_tube_exact(mesh.x_cells, solver.time, gas.gamma)
            rho_error = np.mean(np.abs(state.rho - exact['rho']))
            errors.append(rho_error)

        # Error should decrease with finer grid
        assert errors[1] < errors[0], \
            f"Error did not decrease with refinement: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
