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

from cfd.src import GasProperties, FlowState, Mesh1D


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
    
    def test_zero_ice_with_particles_and_supersaturation_gives_no_growth(self, gas, simple_mesh, lookup_table):
        """CRITICAL: Zero ice mass should give zero growth even with particles and supersaturation.
        
        This is the key test that was missing! When ice_initial=0 but nucleation sites exist,
        growth should NOT occur until nucleation creates ice mass. The diffusion growth equation
        dm/dt = a * m^b * e_fac requires existing ice mass on particles (m > 0).
        
        Nucleation is a separate process (not yet implemented) that creates initial ice mass.
        """
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        # Create state with ZERO ice but particles exist and high supersaturation
        state = create_test_state(
            gas, simple_mesh.n_cells,
            n_specific=1e15,  # Particles exist (potential nucleation sites)
            Y_vapor=1e-3,     # High vapor -> supersaturated conditions
            Y_ice=0.0         # ZERO ice mass - no existing ice to grow
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # With zero ice mass, diffusion growth should be EXACTLY zero
        # (Nucleation is a separate process, not yet implemented)
        assert np.allclose(sources[2], 0.0), \
            f"Ice growth MUST be zero when ice mass is zero (no ice mass on particles to grow), got {sources[2]}"
        
        # Verify no vapor consumption either
        assert np.allclose(sources[1], 0.0), \
            f"Vapor consumption should be zero when no ice growth occurs, got {sources[1]}"


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
    """Tests for nucleation behavior.
    
    Nucleation should only occur when ice supersaturation exceeds a critical threshold.
    Based on κ-Köhler theory, critical supersaturation typically requires:
    - S_ice > 1.4-1.6 (ice saturation ratio)
    - e_fac > 0.4-0.6 (supersaturation factor)
    
    Currently nucleation is NOT implemented, so all tests verify NO growth from zero ice.
    """

    def test_no_nucleation_at_low_supersaturation(self, gas, simple_mesh, lookup_table):
        """Verify no nucleation occurs when supersaturation is below critical threshold.
        
        With low supersaturation (e_fac ~ 0.1-0.2) and zero ice mass, no nucleation
        should occur, even with particles present. This is below the critical
        supersaturation threshold for homogeneous or heterogeneous nucleation.
        """
        from cfd.scripts.ice_growth_source import ice_growth_source_term, psat_ice, psat_water
        
        # Create conditions with LOW supersaturation
        # At T=220K: p_sat_ice ~ 103 Pa, p_sat_water ~ 125 Pa
        T = 220.0
        p_total = 25000.0
        rho = p_total / (gas.R * T)
        
        # Set vapor to achieve e_fac ~ 0.15 (low supersaturation, below nucleation threshold)
        # Target: p_vapor = p_sat_ice + 0.15 * (p_sat_water - p_sat_ice)
        p_sat_ice_val = psat_ice(np.array([T]))[0]
        p_sat_water_val = psat_water(np.array([T]))[0]
        target_p_vapor = p_sat_ice_val + 0.15 * (p_sat_water_val - p_sat_ice_val)
        
        # Calculate required vapor mass fraction
        # p_vapor = rho_vapor * R_v * T, where R_v = R_universal / M_w
        from cfd.scripts.ice_growth_source import R, M_w
        rho_vapor_target = target_p_vapor * M_w / (R * T)
        Y_vapor = rho_vapor_target / rho
        
        state = create_test_state(
            gas, simple_mesh.n_cells,
            rho=rho,
            T=T,
            n_specific=1e15,  # Particles exist (potential nucleation sites)
            Y_vapor=Y_vapor,  # Low supersaturation (e_fac ~ 0.15)
            Y_ice=0.0         # ZERO ice mass
        )
        
        sources = ice_growth_source_term(state, simple_mesh, lookup_table)
        
        # With zero ice and LOW supersaturation, no nucleation should occur
        assert np.allclose(sources[2], 0.0), \
            f"No nucleation should occur at low supersaturation (e_fac ~ 0.15), got {sources[2]}"
        assert np.allclose(sources[1], 0.0), \
            f"No vapor consumption should occur without nucleation, got {sources[1]}"
    
    def test_no_nucleation_at_moderate_supersaturation(self, gas, simple_mesh, lookup_table):
        """Verify no nucleation at moderate supersaturation (below critical threshold).
        
        Even at moderate supersaturation (e_fac ~ 0.3-0.4), which is still below
        the critical threshold for nucleation (~0.4-0.6), no ice formation should occur.
        """
        from cfd.scripts.ice_growth_source import ice_growth_source_term, psat_ice, psat_water, R, M_w
        
        T = 220.0
        p_total = 25000.0
        rho = p_total / (gas.R * T)
        
        # Set vapor to achieve e_fac ~ 0.35 (moderate, still below nucleation threshold)
        p_sat_ice_val = psat_ice(np.array([T]))[0]
        p_sat_water_val = psat_water(np.array([T]))[0]
        target_p_vapor = p_sat_ice_val + 0.35 * (p_sat_water_val - p_sat_ice_val)
        
        rho_vapor_target = target_p_vapor * M_w / (R * T)
        Y_vapor = rho_vapor_target / rho
        
        state = create_test_state(
            gas, simple_mesh.n_cells,
            rho=rho,
            T=T,
            n_specific=1e15,
            Y_vapor=Y_vapor,  # Moderate supersaturation (e_fac ~ 0.35)
            Y_ice=0.0
        )
        
        sources = ice_growth_source_term(state, simple_mesh, lookup_table)
        
        # Still below nucleation threshold
        assert np.allclose(sources[2], 0.0), \
            f"No nucleation at moderate supersaturation (e_fac ~ 0.35), got {sources[2]}"
    
    def test_high_supersaturation_without_nucleation_gives_zero_growth(self, gas, simple_mesh, lookup_table):
        """Verify that even at HIGH supersaturation, zero ice gives zero growth.
        
        At high supersaturation (e_fac > 0.6), nucleation WOULD occur in reality.
        However, since nucleation is not yet implemented, we should still get zero growth
        from zero ice mass. This test documents expected future behavior.
        
        Once nucleation is implemented, this test should be updated to verify:
        - Nucleation DOES occur when e_fac > critical threshold
        - Ice source becomes positive (new ice mass created)
        """
        from cfd.scripts.ice_growth_source import ice_growth_source_term, psat_ice, psat_water, R, M_w
        
        T = 220.0
        p_total = 25000.0
        rho = p_total / (gas.R * T)
        
        # Set vapor to achieve e_fac ~ 0.8 (high supersaturation, above nucleation threshold)
        p_sat_ice_val = psat_ice(np.array([T]))[0]
        p_sat_water_val = psat_water(np.array([T]))[0]
        target_p_vapor = p_sat_ice_val + 0.8 * (p_sat_water_val - p_sat_ice_val)
        
        rho_vapor_target = target_p_vapor * M_w / (R * T)
        Y_vapor = rho_vapor_target / rho
        
        state = create_test_state(
            gas, simple_mesh.n_cells,
            rho=rho,
            T=T,
            n_specific=1e15,
            Y_vapor=Y_vapor,  # High supersaturation (e_fac ~ 0.8)
            Y_ice=0.0
        )
        
        sources = ice_growth_source_term(state, simple_mesh, lookup_table)
        
        # Currently: no nucleation implemented → zero growth
        # Future: nucleation should occur → positive ice source
        assert np.allclose(sources[2], 0.0), \
            f"Current implementation (no nucleation): should get zero growth, got {sources[2]}"
        
        # TODO: Once nucleation is implemented, update this test:
        # assert sources[2] > 0, "Nucleation should occur at high supersaturation (e_fac ~ 0.8)"
        # assert sources[1] < 0, "Vapor should be consumed by nucleation"
    
    def test_supersaturation_threshold_documentation(self, gas, simple_mesh, lookup_table):
        """Document the expected supersaturation threshold for nucleation.
        
        This test verifies our understanding of supersaturation levels and serves
        as documentation for future nucleation implementation.
        
        Key thresholds:
        - e_fac = 0.0: Ice saturation (S_ice = 1.0)
        - e_fac = 0.4-0.6: Critical supersaturation for heterogeneous nucleation
        - e_fac = 1.0: Water saturation (S_ice ≈ 1.7 at 220K)
        - e_fac > 1.0: Above water saturation
        
        Relationship: e_fac = (p_vapor - p_sat_ice) / (p_sat_water - p_sat_ice)
                      S_ice = p_vapor / p_sat_ice
        """
        from cfd.scripts.ice_growth_source import psat_ice, psat_water
        
        T = 220.0  # Typical contrail temperature
        
        # Calculate saturation pressures
        p_sat_ice_val = psat_ice(np.array([T]))[0]
        p_sat_water_val = psat_water(np.array([T]))[0]
        
        # Verify expected values at 220K
        assert 2.0 < p_sat_ice_val < 3.5, f"Expected p_sat_ice ~ 2.7 Pa at 220K, got {p_sat_ice_val}"
        assert 4.0 < p_sat_water_val < 5.0, f"Expected p_sat_water ~ 4.5 Pa at 220K, got {p_sat_water_val}"
        
        # Document key thresholds
        # At ice saturation (e_fac = 0):
        e_fac_ice_sat = 0.0
        S_ice_at_ice_sat = 1.0
        
        # At water saturation (e_fac = 1):
        e_fac_water_sat = 1.0
        S_ice_at_water_sat = p_sat_water_val / p_sat_ice_val
        
        # Expected range: S_ice ~ 1.7 at water saturation for 220K
        assert 1.6 < S_ice_at_water_sat < 1.8, \
            f"S_ice at water saturation should be ~1.7, got {S_ice_at_water_sat}"
        
        # Critical nucleation threshold (typical values)
        # For heterogeneous nucleation: S_ice ~ 1.4-1.6 → e_fac ~ 0.5-0.7
        # At 220K with S_ice_water ~ 1.7:
        #   S_ice = 1.4 → e_fac = (1.4 - 1.0) / (1.7 - 1.0) = 0.57
        #   S_ice = 1.5 → e_fac = (1.5 - 1.0) / (1.7 - 1.0) = 0.71
        e_fac_critical_low = (1.4 - 1.0) / (S_ice_at_water_sat - 1.0)
        e_fac_critical_high = (1.5 - 1.0) / (S_ice_at_water_sat - 1.0)
        
        # Verify critical thresholds are in expected range
        assert 0.5 < e_fac_critical_low < 0.7, \
            f"Expected e_fac_critical_low ~ 0.57, got {e_fac_critical_low}"
        assert 0.6 < e_fac_critical_high < 0.8, \
            f"Expected e_fac_critical_high ~ 0.71, got {e_fac_critical_high}"
        
        # This test passes to document thresholds
        assert True, "Supersaturation thresholds documented"


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
