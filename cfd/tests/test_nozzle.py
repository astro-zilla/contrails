"""
Pytest tests for subsonic nozzle flow.

Tests verify:
1. Solver convergence
2. Mass conservation
3. Maximum Mach number at throat
4. Scalar transport (passive scalars remain constant without sources)
5. Physical bounds (positive density, pressure, temperature)
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
    SubsonicInletBC, SubsonicOutletBC
)


def nozzle_area(x: np.ndarray, x_throat: float = 0.5,
                A_inlet: float = 1.0, A_throat: float = 0.8,
                A_exit: float = 1.0) -> np.ndarray:
    """Smooth nozzle area distribution (sinusoidal)."""
    A = np.zeros_like(x)

    # Converging section (inlet to throat)
    mask_conv = x <= x_throat
    A[mask_conv] = A_inlet + (A_throat - A_inlet) * (1 - np.cos(np.pi * x[mask_conv] / x_throat)) / 2

    # Diverging section (throat to exit)
    mask_div = x > x_throat
    xi = (x[mask_div] - x_throat) / (1 - x_throat)
    A[mask_div] = A_throat + (A_exit - A_throat) * (1 - np.cos(np.pi * xi)) / 2

    return A


@pytest.fixture
def gas():
    """Standard air properties."""
    return GasProperties(gamma=1.4, R=287.0)


@pytest.fixture
def nozzle_mesh():
    """Standard nozzle mesh."""
    n_cells = 100
    area_func = lambda x: nozzle_area(x, x_throat=0.5, A_inlet=1.0, A_throat=0.8, A_exit=1.0)
    return Mesh1D.uniform(0.0, 1.0, n_cells, area_func)


@pytest.fixture
def solver_config():
    """Solver configuration for tests."""
    return SolverConfig(
        cfl=0.5,
        max_iter=50000,
        convergence_tol=1e-6,
        print_interval=10000,
        time_scheme='rk2',
        use_first_order=True
    )


def create_nozzle_solver(gas, mesh, config, n_scalars=0):
    """Create a configured nozzle solver."""
    solver = Solver1D(mesh, gas, n_scalars=n_scalars, config=config)

    # Boundary conditions
    p0 = 101325.0       # Total pressure [Pa]
    T0 = 300.0          # Total temperature [K]
    p_exit = 95000.0    # Exit static pressure [Pa]

    Y_inlet = np.zeros(n_scalars)
    if n_scalars > 0:
        Y_inlet[0] = 0.01  # 1% mass fraction
    if n_scalars > 1:
        Y_inlet[1] = 0.005  # 0.5% mass fraction

    bc_left = SubsonicInletBC(p0=p0, T0=T0, Y_inlet=Y_inlet)
    bc_right = SubsonicOutletBC(p_exit=p_exit)
    solver.set_boundary_conditions(bc_left, bc_right)

    # Initial condition
    M_init = 0.3
    T_init = T0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)
    p_init = p0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)**(gas.gamma / (gas.gamma - 1))
    rho_init = p_init / (gas.R * T_init)
    u_init = M_init * np.sqrt(gas.gamma * gas.R * T_init)

    n_cells = mesh.n_cells
    rho = np.full(n_cells, rho_init)
    u = np.full(n_cells, u_init)
    p = np.full(n_cells, p_init)
    Y = np.zeros((n_scalars, n_cells))
    for i in range(n_scalars):
        Y[i, :] = Y_inlet[i]

    initial_state = FlowState(rho=rho, u=u, p=p, Y=Y, gas=gas)
    solver.set_initial_condition(initial_state)

    return solver


class TestNozzleConvergence:
    """Tests for solver convergence."""

    def test_solver_converges(self, gas, nozzle_mesh, solver_config):
        """Solver should converge to steady state."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        result = solver.solve()

        # Check convergence (either converged or residual dropped significantly)
        assert result['final_residual'] < 1e-3, \
            f"Solver did not converge sufficiently: residual = {result['final_residual']}"

    def test_positive_mach_numbers(self, gas, nozzle_mesh, solver_config):
        """Flow should have positive Mach numbers (flow in correct direction)."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        solver.solve()
        state = solver.get_state()

        assert np.all(state.M > 0), "Mach numbers should be positive (flow moving right)"
        assert np.all(state.u > 0), "Velocity should be positive"


class TestMassConservation:
    """Tests for mass conservation."""

    def test_mass_flow_conservation(self, gas, nozzle_mesh, solver_config):
        """Mass flow rate should be constant through the nozzle."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        solver.solve()
        state = solver.get_state()

        mdot = state.rho * state.u * nozzle_mesh.A_cells
        mdot_variation = (np.max(mdot) - np.min(mdot)) / np.mean(mdot)

        assert mdot_variation < 0.01, \
            f"Mass flow variation too large: {mdot_variation*100:.2f}%"


class TestNozzlePhysics:
    """Tests for correct nozzle physics."""

    def test_max_mach_at_throat(self, gas, nozzle_mesh, solver_config):
        """Maximum Mach number should occur at the throat (minimum area)."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        solver.solve()
        state = solver.get_state()

        # Find throat location (minimum area)
        throat_idx = np.argmin(nozzle_mesh.A_cells)
        max_mach_idx = np.argmax(state.M)

        # Allow for some tolerance (within 5 cells of throat)
        assert abs(max_mach_idx - throat_idx) <= 5, \
            f"Max Mach at cell {max_mach_idx}, throat at cell {throat_idx}"

    def test_min_pressure_at_throat(self, gas, nozzle_mesh, solver_config):
        """Minimum static pressure should occur at the throat."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        solver.solve()
        state = solver.get_state()

        throat_idx = np.argmin(nozzle_mesh.A_cells)
        min_pressure_idx = np.argmin(state.p)

        # Allow for some tolerance (within 5 cells of throat)
        assert abs(min_pressure_idx - throat_idx) <= 5, \
            f"Min pressure at cell {min_pressure_idx}, throat at cell {throat_idx}"

    def test_subsonic_throughout(self, gas, nozzle_mesh, solver_config):
        """Flow should remain subsonic throughout for this configuration."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        solver.solve()
        state = solver.get_state()

        assert np.all(state.M < 1.0), \
            f"Flow should be subsonic, max M = {np.max(state.M)}"


class TestPhysicalBounds:
    """Tests for physical validity of solution."""

    def test_positive_density(self, gas, nozzle_mesh, solver_config):
        """Density must be positive everywhere."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        solver.solve()
        state = solver.get_state()

        assert np.all(state.rho > 0), "Density must be positive"

    def test_positive_pressure(self, gas, nozzle_mesh, solver_config):
        """Pressure must be positive everywhere."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        solver.solve()
        state = solver.get_state()

        assert np.all(state.p > 0), "Pressure must be positive"

    def test_positive_temperature(self, gas, nozzle_mesh, solver_config):
        """Temperature must be positive everywhere."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        solver.solve()
        state = solver.get_state()

        assert np.all(state.T > 0), "Temperature must be positive"


class TestScalarTransport:
    """Tests for passive scalar transport."""

    def test_scalar_conservation(self, gas, nozzle_mesh, solver_config):
        """Passive scalars should be conserved (constant without sources)."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=2)
        solver.solve()
        state = solver.get_state()

        # Check that scalars remain close to inlet values
        Y_inlet = [0.01, 0.005]
        for i, Y_in in enumerate(Y_inlet):
            Y_variation = np.max(np.abs(state.Y[i] - Y_in)) / Y_in
            assert Y_variation < 0.05, \
                f"Scalar {i} variation too large: {Y_variation*100:.1f}%"

    def test_scalar_positive(self, gas, nozzle_mesh, solver_config):
        """Scalar mass fractions should remain non-negative."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=2)
        solver.solve()
        state = solver.get_state()

        for i in range(state.Y.shape[0]):
            assert np.all(state.Y[i] >= -1e-10), \
                f"Scalar {i} has negative values"


class TestStagnationProperties:
    """Tests for isentropic flow properties."""

    def test_stagnation_temperature_constant(self, gas, nozzle_mesh, solver_config):
        """Stagnation temperature should be approximately constant (isentropic)."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        solver.solve()
        state = solver.get_state()

        T0 = state.T * (1 + 0.5 * (gas.gamma - 1) * state.M**2)
        T0_variation = (np.max(T0) - np.min(T0)) / np.mean(T0)

        assert T0_variation < 0.01, \
            f"Stagnation temperature variation too large: {T0_variation*100:.2f}%"

    def test_stagnation_pressure_nearly_constant(self, gas, nozzle_mesh, solver_config):
        """Stagnation pressure should be nearly constant (small numerical dissipation)."""
        solver = create_nozzle_solver(gas, nozzle_mesh, solver_config, n_scalars=0)
        solver.solve()
        state = solver.get_state()

        p0 = state.p * (1 + 0.5 * (gas.gamma - 1) * state.M**2)**(gas.gamma / (gas.gamma - 1))
        p0_variation = (np.max(p0) - np.min(p0)) / np.mean(p0)

        # Allow up to 2% variation due to numerical dissipation
        assert p0_variation < 0.02, \
            f"Stagnation pressure variation too large: {p0_variation*100:.2f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
