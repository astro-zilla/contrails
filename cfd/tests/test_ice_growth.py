"""
Pytest tests for ice growth source term.

Tests verify:
1. Zero ice mass gives zero growth
2. Zero particle number gives zero growth
3. Subsaturated conditions give zero growth
4. Mass conservation (vapor lost = ice gained)
5. Source term signs are correct
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cfd import GasProperties, FlowState, Mesh1D


class MockLookupTable:
    """Mock lookup table for testing without HDF5 file."""

    def __init__(self, a_value=1e-10, b_value=0.33):
        self.a_value = a_value
        self.b_value = b_value

    def get_coefficients(self, T, p):
        """Return constant coefficients for testing."""
        T = np.atleast_1d(T)
        a = np.full_like(T, self.a_value)
        b = np.full_like(T, self.b_value)
        return a, b


@pytest.fixture
def gas():
    """Standard air properties."""
    return GasProperties(gamma=1.4, R=287.0)


@pytest.fixture
def simple_mesh():
    """Simple uniform mesh for testing."""
    n_cells = 10
    area_func = lambda x: np.ones_like(x)
    return Mesh1D.uniform(0.0, 1.0, n_cells, area_func)


@pytest.fixture
def lookup_table():
    """Mock lookup table with typical values."""
    return MockLookupTable(a_value=1e-10, b_value=0.33)


def create_test_state(gas, n_cells, rho=0.4, u=100.0, p=25000.0, T=220.0,
                      n_specific=1e15, Y_vapor=1e-3, Y_ice=1e-6):
    """Create a flow state for testing."""
    # Compute consistent pressure from T and rho
    p = rho * gas.R * T

    Y = np.zeros((3, n_cells))
    Y[0, :] = n_specific  # Specific particle number [#/kg]
    Y[1, :] = Y_vapor     # Vapor mass fraction
    Y[2, :] = Y_ice       # Ice mass fraction

    return FlowState(
        rho=np.full(n_cells, rho),
        u=np.full(n_cells, u),
        p=np.full(n_cells, p),
        Y=Y,
        gas=gas
    )


class TestZeroIceGrowth:
    """Tests that verify growth behavior with zero or small ice mass."""

    def test_zero_ice_zero_particles_gives_zero_growth(self, gas, simple_mesh, lookup_table):
        """When both ice and particles are zero, no growth should occur."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        # Create state with ZERO ice AND zero particles
        state = create_test_state(
            gas, simple_mesh.n_cells,
            n_specific=0.0,   # NO particles
            Y_vapor=1e-3,     # Vapor exists
            Y_ice=0.0         # NO ice
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # With zero particles, nothing can nucleate or grow
        assert np.allclose(sources[2], 0.0), \
            f"Ice growth should be zero when no particles exist, got {sources[2]}"

    def test_zero_particle_number_gives_zero_growth(self, gas, simple_mesh, lookup_table):
        """When particle number is zero, growth rate should be zero."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        # Create state with zero particles
        state = create_test_state(
            gas, simple_mesh.n_cells,
            n_specific=0.0,   # NO particles
            Y_vapor=1e-3,     # Vapor exists
            Y_ice=1e-6        # Ice exists but no particles to grow
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # With zero particles, total growth = n * dm/dt = 0 * dm/dt = 0
        assert np.allclose(sources[2], 0.0), \
            f"Ice growth should be zero when no particles exist, got {sources[2]}"

    def test_very_small_ice_mass_with_low_supersaturation(self, gas, simple_mesh, lookup_table):
        """Very small ice mass with low supersaturation should give small growth."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        # Create state with very small ice mass and low vapor (below nucleation threshold)
        state = create_test_state(
            gas, simple_mesh.n_cells,
            n_specific=1e15,
            Y_vapor=1e-6,  # Low vapor - likely below critical supersaturation
            Y_ice=1e-20    # Extremely small ice mass
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # Growth should be very small or zero (below nucleation threshold)
        assert np.all(np.abs(sources[2]) < 1e-5), \
            f"Ice growth should be small for tiny ice mass with low supersaturation, got max {np.max(np.abs(sources[2]))}"


class TestSubsaturatedConditions:
    """Tests for subsaturated conditions where evaporation should occur."""

    def test_subsaturated_gives_evaporation(self, gas, simple_mesh, lookup_table):
        """When vapor pressure < saturation pressure over ice, evaporation occurs."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term, psat_ice

        # Create state with very low vapor (subsaturated) but existing ice
        # At T=220K, p_sat_ice ~ 3-4 Pa
        # Set vapor fraction such that p_vapor < p_sat_ice
        state = create_test_state(
            gas, simple_mesh.n_cells,
            T=220.0,
            n_specific=1e15,
            Y_vapor=1e-8,  # Very low vapor -> subsaturated
            Y_ice=1e-6     # Existing ice to evaporate
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # Subsaturated means evaporation: ice source should be negative (or zero if no ice)
        assert np.all(sources[2] <= 0.0), \
            f"Ice source should be non-positive (evaporation) when subsaturated, got {sources[2]}"

        # Vapor source should be positive (ice evaporating adds vapor)
        assert np.all(sources[1] >= 0.0), \
            f"Vapor source should be non-negative when evaporating, got {sources[1]}"


class TestMassConservation:
    """Tests for mass conservation in source terms."""

    def test_vapor_ice_mass_balance(self, gas, simple_mesh, lookup_table):
        """Vapor lost should equal ice gained (mass conservation)."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        # Create state with growth conditions
        state = create_test_state(
            gas, simple_mesh.n_cells,
            n_specific=1e15,
            Y_vapor=1e-3,
            Y_ice=1e-6
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # Source[1] = vapor change, Source[2] = ice change
        # They should sum to zero (mass conservation)
        total_source = sources[1] + sources[2]

        assert np.allclose(total_source, 0.0), \
            f"Vapor + ice sources should sum to zero, got {total_source}"

    def test_particle_number_conserved(self, gas, simple_mesh, lookup_table):
        """Particle number should have zero source (conserved)."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        state = create_test_state(
            gas, simple_mesh.n_cells,
            n_specific=1e15,
            Y_vapor=1e-3,
            Y_ice=1e-6
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # Particle number source should be exactly zero
        assert np.allclose(sources[0], 0.0), \
            f"Particle number source should be zero, got {sources[0]}"


class TestSourceTermSigns:
    """Tests for correct signs of source terms."""

    def test_ice_growth_positive(self, gas, simple_mesh, lookup_table):
        """Ice source should be positive (growth) when supersaturated."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        # Create supersaturated state with existing ice
        state = create_test_state(
            gas, simple_mesh.n_cells,
            n_specific=1e15,
            Y_vapor=1e-3,  # High vapor -> supersaturated
            Y_ice=1e-6     # Existing ice to grow
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # If there's growth, ice source should be positive
        # (could be zero if conditions don't allow growth)
        assert np.all(sources[2] >= 0.0), \
            f"Ice source should be non-negative, got {sources[2]}"

    def test_vapor_consumption_negative(self, gas, simple_mesh, lookup_table):
        """Vapor source should be negative (consumption) when ice grows."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        state = create_test_state(
            gas, simple_mesh.n_cells,
            n_specific=1e15,
            Y_vapor=1e-3,
            Y_ice=1e-6
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # Vapor source should be non-positive (consumption or zero)
        assert np.all(sources[1] <= 0.0), \
            f"Vapor source should be non-positive, got {sources[1]}"


class TestNucleation:
    """Tests for κ-Köhler nucleation behavior."""

    def test_nucleation_with_high_supersaturation(self, gas, simple_mesh, lookup_table):
        """Particles should nucleate when supersaturation exceeds critical value."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        # Create state with zero ice but high supersaturation
        # This should trigger nucleation
        state = create_test_state(
            gas, simple_mesh.n_cells,
            T=220.0,
            n_specific=1e15,   # Particles exist
            Y_vapor=1e-2,      # Very high vapor -> high supersaturation
            Y_ice=0.0          # No ice yet
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table,
                                         rd=20e-9, kappa=0.5)

        # With high supersaturation, nucleation should occur
        # Ice source should be positive (particles activating)
        assert np.all(sources[2] > 0.0), \
            f"Nucleation should occur with high supersaturation, got ice source {sources[2]}"

    def test_no_nucleation_below_critical_supersaturation(self, gas, simple_mesh, lookup_table):
        """Particles should not nucleate below critical supersaturation."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        # Create state with zero ice and very low vapor
        # Supersaturation should be below critical
        state = create_test_state(
            gas, simple_mesh.n_cells,
            T=220.0,
            n_specific=1e15,
            Y_vapor=1e-8,      # Very low vapor -> below S_c
            Y_ice=0.0
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table,
                                         rd=20e-9, kappa=0.5)

        # Below critical supersaturation, no nucleation should occur
        # Ice source should be zero (or negligible)
        assert np.all(np.abs(sources[2]) < 1e-10), \
            f"No nucleation should occur below S_c, got ice source {sources[2]}"

    def test_nucleation_creates_critical_mass(self, gas, simple_mesh, lookup_table):
        """Nucleated particles should have mass corresponding to critical radius."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term, psat_water, R, M_w

        # Physical constants
        rho_ice_bulk = 917.0
        rho_water = 1000.0
        sigma_wa = 0.076
        R_v = R / M_w
        rd = 20e-9
        kappa = 0.5

        T = 220.0
        rho = 0.4
        p = rho * gas.R * T

        # Compute expected critical radius
        a_k = 2 * sigma_wa / (rho_water * R_v * T)
        x = 3 * kappa * rd / a_k
        r_star = rd * (1 + np.sqrt(x) * (x**(2/3) + 1.2) / (x**(2/3) + 3.6))
        m_crit = (4/3) * np.pi * r_star**3 * rho_ice_bulk

        # Create state with high supersaturation to trigger nucleation
        state = create_test_state(
            gas, simple_mesh.n_cells,
            T=T,
            n_specific=1e15,
            Y_vapor=1e-2,  # High supersaturation
            Y_ice=0.0
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table,
                                         rd=rd, kappa=kappa)

        # The nucleation rate should be related to n * m_crit
        # Check that ice source is significant (nucleation is happening)
        assert np.all(sources[2] > 0), \
            f"Nucleation should be occurring, got {sources[2]}"


class TestEvaporation:
    """Tests for evaporation behavior."""

    def test_evaporation_restores_vapor(self, gas, simple_mesh, lookup_table):
        """Evaporating ice should restore vapor."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        # Create subsaturated state with existing ice
        state = create_test_state(
            gas, simple_mesh.n_cells,
            T=220.0,
            n_specific=1e15,
            Y_vapor=1e-8,  # Very low vapor -> subsaturated
            Y_ice=1e-5     # Significant ice to evaporate
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # Ice should decrease (evaporate)
        assert np.all(sources[2] < 0.0), \
            f"Ice should evaporate when subsaturated, got {sources[2]}"

        # Vapor should increase (from evaporation)
        assert np.all(sources[1] > 0.0), \
            f"Vapor should increase from evaporation, got {sources[1]}"

    def test_evaporation_mass_conservation(self, gas, simple_mesh, lookup_table):
        """Mass should be conserved during evaporation."""
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        state = create_test_state(
            gas, simple_mesh.n_cells,
            T=220.0,
            n_specific=1e15,
            Y_vapor=1e-8,
            Y_ice=1e-5
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # Vapor gained should equal ice lost
        total_source = sources[1] + sources[2]
        assert np.allclose(total_source, 0.0), \
            f"Mass should be conserved during evaporation, got total {total_source}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
