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
    
    def test_zero_ice_with_particles_and_low_supersaturation_gives_no_growth(self, gas, simple_mesh, lookup_table):
        """Zero ice mass with LOW supersaturation should give zero growth.
        
        When ice_initial=0 but nucleation sites exist with LOW supersaturation
        (below nucleation threshold), neither diffusion growth nor nucleation should occur.
        The diffusion growth equation dm/dt = a * m^b * e_fac requires existing ice mass (m > 0),
        and nucleation requires e_fac > e_fac_critical (default 0.6).
        """
        from cfd.scripts.ice_growth_source import ice_growth_source_term

        # Create state with ZERO ice, particles exist, but LOW supersaturation
        state = create_test_state(
            gas, simple_mesh.n_cells,
            n_specific=1e15,  # Particles exist (potential nucleation sites)
            Y_vapor=1e-6,     # Low vapor -> below nucleation threshold
            Y_ice=0.0         # ZERO ice mass - no existing ice to grow
        )

        sources = ice_growth_source_term(state, simple_mesh, lookup_table)

        # With zero ice mass and low supersaturation, no growth or nucleation
        assert np.allclose(sources[2], 0.0), \
            f"No growth or nucleation when ice=0 and low supersaturation, got {sources[2]}"
        
        # Verify no vapor consumption either
        assert np.allclose(sources[1], 0.0), \
            f"Vapor consumption should be zero when no growth/nucleation occurs, got {sources[1]}"


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
    """Tests for nucleation behavior based on κ-Köhler theory.
    
    Nucleation occurs when ice supersaturation S_ice exceeds critical threshold.
    Based on κ-Köhler theory (Petters & Kreidenweis):
    - S_ice = p_vapor / p_sat_ice (ice saturation ratio)
    - Critical threshold: S_crit ~ 1.4 (140% RH w.r.t. ice)
    - Depends on aerosol properties (rd, kappa)
    
    IMPORTANT: e_fac is ONLY for growth equation, NOT for nucleation!
    - e_fac scales growth rate: dm/dt = a * m^b * e_fac
    - Nucleation uses S_ice threshold from κ-Köhler theory
    """

    def test_no_nucleation_at_low_supersaturation(self, gas, simple_mesh, lookup_table):
        """Verify no nucleation occurs when S_ice is below critical threshold.
        
        With low ice supersaturation (S_ice ~ 1.1-1.2) and zero ice mass, no nucleation
        should occur, even with particles present. This is below the critical
        supersaturation threshold S_crit ~ 1.4 from κ-Köhler theory.
        """
        from cfd.scripts.ice_growth_source import ice_growth_source_term, psat_ice, psat_water
        
        # Create conditions with LOW ice supersaturation
        # At T=220K: p_sat_ice ~ 2.7 Pa
        T = 220.0
        p_total = 25000.0
        rho = p_total / (gas.R * T)
        
        # Set vapor to achieve S_ice ~ 1.15 (below nucleation threshold of 1.4)
        # S_ice = p_vapor / p_sat_ice
        p_sat_ice_val = psat_ice(np.array([T]))[0]
        target_p_vapor = 1.15 * p_sat_ice_val  # 115% RH w.r.t. ice
        
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
            Y_vapor=Y_vapor,  # S_ice ~ 1.15 (below threshold)
            Y_ice=0.0         # ZERO ice mass
        )
        
        sources = ice_growth_source_term(state, simple_mesh, lookup_table)
        
        # With zero ice and S_ice < S_crit, no nucleation should occur
        assert np.allclose(sources[2], 0.0), \
            f"No nucleation should occur at S_ice ~ 1.15 (< 1.4), got {sources[2]}"
        assert np.allclose(sources[1], 0.0), \
            f"No vapor consumption should occur without nucleation, got {sources[1]}"
    
    def test_no_nucleation_at_moderate_supersaturation(self, gas, simple_mesh, lookup_table):
        """Verify no nucleation at moderate supersaturation (below critical threshold).
        
        Even at moderate ice supersaturation (S_ice ~ 1.3), which is still below
        the critical threshold S_crit ~ 1.4, no ice formation should occur.
        """
        from cfd.scripts.ice_growth_source import ice_growth_source_term, psat_ice, R, M_w
        
        T = 220.0
        p_total = 25000.0
        rho = p_total / (gas.R * T)
        
        # Set vapor to achieve S_ice ~ 1.3 (moderate, still below nucleation threshold)
        p_sat_ice_val = psat_ice(np.array([T]))[0]
        target_p_vapor = 1.3 * p_sat_ice_val  # 130% RH w.r.t. ice
        
        rho_vapor_target = target_p_vapor * M_w / (R * T)
        Y_vapor = rho_vapor_target / rho
        
        state = create_test_state(
            gas, simple_mesh.n_cells,
            rho=rho,
            T=T,
            n_specific=1e15,
            Y_vapor=Y_vapor,  # S_ice ~ 1.3 (below threshold)
            Y_ice=0.0
        )
        
        sources = ice_growth_source_term(state, simple_mesh, lookup_table)
        
        # Still below nucleation threshold - no nucleation should occur
        assert np.allclose(sources[2], 0.0), \
            f"No nucleation at S_ice ~ 1.3 (< 1.4), got {sources[2]}"
    
    def test_nucleation_threshold_behavior(self, gas, simple_mesh, lookup_table):
        """Test nucleation occurs just above threshold but not just below.
        
        Verify that the nucleation threshold is sharp:
        - S_ice = 1.35 (below threshold): no nucleation
        - S_ice = 1.45 (above threshold): nucleation occurs
        """
        from cfd.scripts.ice_growth_source import ice_growth_source_term, psat_ice, R, M_w
        
        T = 220.0
        p_total = 25000.0
        rho = p_total / (gas.R * T)
        
        p_sat_ice_val = psat_ice(np.array([T]))[0]
        
        # Test below threshold (S_ice = 1.35)
        target_p_vapor_below = 1.35 * p_sat_ice_val
        rho_vapor_below = target_p_vapor_below * M_w / (R * T)
        Y_vapor_below = rho_vapor_below / rho
        
        state_below = create_test_state(
            gas, simple_mesh.n_cells,
            rho=rho, T=T,
            n_specific=1e15,
            Y_vapor=Y_vapor_below,
            Y_ice=0.0
        )
        
        sources_below = ice_growth_source_term(state_below, simple_mesh, lookup_table)
        
        # Just below threshold: no nucleation
        assert np.allclose(sources_below[2], 0.0), \
            f"No nucleation below threshold (S_ice=1.35), got {sources_below[2]}"
        
        # Test above threshold (S_ice = 1.45)
        target_p_vapor_above = 1.45 * p_sat_ice_val
        rho_vapor_above = target_p_vapor_above * M_w / (R * T)
        Y_vapor_above = rho_vapor_above / rho
        
        state_above = create_test_state(
            gas, simple_mesh.n_cells,
            rho=rho, T=T,
            n_specific=1e15,
            Y_vapor=Y_vapor_above,
            Y_ice=0.0
        )
        
        sources_above = ice_growth_source_term(state_above, simple_mesh, lookup_table)
        
        # Just above threshold: nucleation occurs
        assert np.all(sources_above[2] > 0), \
            f"Nucleation should occur above threshold (S_ice=1.45), got {sources_above[2]}"
    
    def test_high_supersaturation_with_nucleation_creates_ice(self, gas, simple_mesh, lookup_table):
        """Verify that nucleation occurs at HIGH ice supersaturation.
        
        At high ice supersaturation (S_ice > 1.4), nucleation SHOULD occur when ice mass is zero.
        This test verifies:
        - Nucleation DOES occur when S_ice > S_crit
        - Ice source becomes positive (new ice mass created)
        - Vapor is consumed
        """
        from cfd.scripts.ice_growth_source import ice_growth_source_term, psat_ice, R, M_w
        
        T = 220.0
        p_total = 25000.0
        rho = p_total / (gas.R * T)
        
        # Set vapor to achieve S_ice ~ 1.6 (high, well above nucleation threshold of 1.4)
        p_sat_ice_val = psat_ice(np.array([T]))[0]
        target_p_vapor = 1.6 * p_sat_ice_val  # 160% RH w.r.t. ice
        
        rho_vapor_target = target_p_vapor * M_w / (R * T)
        Y_vapor = rho_vapor_target / rho
        
        state = create_test_state(
            gas, simple_mesh.n_cells,
            rho=rho,
            T=T,
            n_specific=1e15,
            Y_vapor=Y_vapor,  # S_ice ~ 1.6 (above threshold)
            Y_ice=0.0
        )
        
        sources = ice_growth_source_term(state, simple_mesh, lookup_table)
        
        # With nucleation implemented: positive ice source
        assert np.all(sources[2] > 0), \
            f"Nucleation should occur at high S_ice ~ 1.6 (> 1.4), got {sources[2]}"
        
        # Vapor should be consumed
        assert np.all(sources[1] < 0), \
            f"Vapor should be consumed by nucleation, got {sources[1]}"
        
        # Verify mass conservation
        total_source = sources[1] + sources[2]
        assert np.allclose(total_source, 0.0), \
            f"Mass should be conserved during nucleation, got total {total_source}"
    
    def test_supersaturation_threshold_documentation(self, gas, simple_mesh, lookup_table):
        """Document the expected supersaturation threshold for nucleation.
        
        This test verifies our understanding of supersaturation levels and serves
        as documentation for nucleation implementation based on κ-Köhler theory.
        
        Key relationships:
        - S_ice = p_vapor / p_sat_ice (ice saturation ratio)
        - Critical threshold: S_crit ~ 1.4 (140% RH w.r.t. ice)
        - e_fac = (p_vapor - p_sat_ice) / (p_sat_water - p_sat_ice)
        
        IMPORTANT: e_fac is ONLY for growth, NOT for nucleation!
        - Growth: dm/dt = a * m^b * e_fac
        - Nucleation: S_ice > S_crit (from κ-Köhler theory)
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
        # Ice saturation: S_ice = 1.0
        S_ice_saturation = 1.0
        
        # Water saturation: S_ice = p_sat_water / p_sat_ice
        S_ice_at_water_sat = p_sat_water_val / p_sat_ice_val
        
        # Expected range: S_ice ~ 1.7 at water saturation for 220K
        assert 1.6 < S_ice_at_water_sat < 1.8, \
            f"S_ice at water saturation should be ~1.7, got {S_ice_at_water_sat}"
        
        # Critical nucleation threshold from κ-Köhler theory
        # For typical aerosol (rd ~ 0.1 μm, kappa ~ 0.5): S_crit ~ 1.4
        S_crit = 1.4  # 140% RH w.r.t. ice
        
        # Verify critical threshold is reasonable
        assert S_ice_saturation < S_crit < S_ice_at_water_sat, \
            f"S_crit should be between ice and water saturation: {S_ice_saturation} < {S_crit} < {S_ice_at_water_sat}"
        
        # This test passes to document thresholds
        assert True, "Supersaturation thresholds documented for κ-Köhler nucleation"


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
