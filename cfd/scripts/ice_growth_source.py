"""
Ice growth source term for contrail microphysics simulation.

Implements scalar transport for:
- Scalar 0: Particle number density n [#/m³] (passive, conserved)
- Scalar 1: Water vapor mass density ρ_vapor [kg/m³]
- Scalar 2: Ice mass density ρ_ice [kg/m³]

Uses Koenig approximation for ice particle growth:
    dm/mt = a * m^b * e_fac

where:
- a and b are interpolated from pre-computed lookup tables (at saturation)
- e_fac = (e_actual - e_ice) / (e_water - e_ice) is the supersaturation factor
- e = (p_vapor - p_sat_ice) / (p_sat_water - p_sat_ice)

The lookup table stores coefficients computed at water saturation (e=1),
so we multiply by e_fac to get the actual growth rate.
"""

import sys
from pathlib import Path
import numpy as np
import h5py

# Add parent directory to path for src imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cfd import ScalarSourceTerm
from cfd.state import FlowState
from cfd.mesh import Mesh1D


class IceGrowthLookupTable:
    """
    Lookup table for ice growth coefficients a and b.

    Loads and interpolates pre-computed values from HDF5 file.
    """

    def __init__(self, hdf5_path: str = '../../ice_growth_fits.hdf5'):
        """
        Load ice growth lookup table from HDF5 file.

        Args:
            hdf5_path: Path to HDF5 file with ice growth coefficients
        """
        self.hdf5_path = Path(hdf5_path)

        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"Ice growth lookup table not found: {self.hdf5_path}\n"
                f"Expected at: {self.hdf5_path.absolute()}"
            )

        # Load lookup table
        print(f"Loading ice growth lookup table from: {self.hdf5_path}")
        with h5py.File(self.hdf5_path, 'r') as f:
            # Read the table structure
            # Assuming structure like:
            # - 'T' or 'temperature': temperature grid [K]
            # - 'p' or 'pressure': pressure grid [Pa]
            # - 'a': coefficient a values
            # - 'b': coefficient b values

            # Try different possible key names
            temp_keys = ['T', 'temperature', 'temp']
            pres_keys = ['p', 'pressure', 'pres']

            # Find temperature key
            for key in temp_keys:
                if key in f.keys():
                    self.T_grid = np.array(f[key])
                    break
            else:
                raise KeyError(f"Temperature not found in HDF5. Available keys: {list(f.keys())}")

            # Find pressure key
            for key in pres_keys:
                if key in f.keys():
                    self.p_grid = np.array(f[key])
                    break
            else:
                raise KeyError(f"Pressure not found in HDF5. Available keys: {list(f.keys())}")

            # Load coefficients
            self.a_table = np.array(f['a'])
            self.b_table = np.array(f['b'])

            # Store grid info
            print(f"  Temperature range: {self.T_grid.min():.1f} - {self.T_grid.max():.1f} K")
            print(f"  Pressure range: {self.p_grid.min():.1f} - {self.p_grid.max():.1f} Pa")
            print(f"  Grid shape: {self.a_table.shape}")

    def get_coefficients(self, T: np.ndarray, p: np.ndarray) -> tuple:
        """
        Interpolate ice growth coefficients for given T and p.

        Args:
            T: Temperature [K] (can be scalar or array)
            p: Pressure [Pa] (can be scalar or array)

        Returns:
            a, b: Growth coefficients (same shape as inputs)
        """
        # Convert to arrays if needed
        T = np.atleast_1d(T)
        p = np.atleast_1d(p)

        # Initialize output arrays
        a = np.zeros_like(T)
        b = np.zeros_like(T)

        # 2D interpolation (bilinear)
        from scipy.interpolate import RegularGridInterpolator

        # Create interpolators (do this once if called repeatedly)
        if not hasattr(self, '_interp_a'):
            self._interp_a = RegularGridInterpolator(
                (self.T_grid, self.p_grid),
                self.a_table,
                bounds_error=False,
                fill_value=None  # Extrapolate
            )
            self._interp_b = RegularGridInterpolator(
                (self.T_grid, self.p_grid),
                self.b_table,
                bounds_error=False,
                fill_value=None
            )

        # Interpolate
        points = np.column_stack([T.ravel(), p.ravel()])
        a = self._interp_a(points).reshape(T.shape)
        b = self._interp_b(points).reshape(T.shape)

        return a, b


# Physical constants for saturation pressure calculations
M_w = 18.01528e-3  # kg/mol - molar mass of water
M_a = 28.9647e-3   # kg/mol - molar mass of air
R = 8.314462618    # J/(mol·K) - universal gas constant


def psat_ice(T: np.ndarray, p: np.ndarray = None) -> np.ndarray:
    """
    Saturation vapor pressure over ice (Sonntag, 1994).

    Args:
        T: Temperature [K]
        p: Total pressure [Pa] (optional, defaults to None for compatibility)

    Returns:
        Saturation pressure [Pa]
    """
    # Sonntag 1994 expression
    ei = np.exp(-6024.528211 / T + 29.32707 + 1.0613868e-2 * T +
                -1.3198825e-5 * T ** 2 - 0.49382577 * np.log(T))

    # Enhancement factor (pressure correction)
    # If pressure not provided, use standard atmospheric pressure approximation
    if p is None:
        return ei
    else:
        fi = 1.0016 + 3.15e-8 * p - 7.4e-4 / p
        return ei * fi


def psat_water(T: np.ndarray, p: np.ndarray = None) -> np.ndarray:
    """
    Saturation vapor pressure over liquid water (Sonntag, 1994).

    Args:
        T: Temperature [K]
        p: Total pressure [Pa] (optional, defaults to None for compatibility)

    Returns:
        Saturation pressure [Pa]
    """
    # Sonntag 1994 expression
    ew = np.exp(-6096.9385 / T + 21.2409642 - 2.711193e-2 * T +
                1.673952e-5 * T ** 2 + 2.433502 * np.log(T))

    # Enhancement factor (pressure correction)
    # If pressure not provided, use standard atmospheric pressure approximation
    if p is None:
        return ew
    else:
        fw = 1.0016 + 3.15e-8 * p - 7.4e-4 / p
        return ew * fw


def ice_growth_source_term(state: FlowState, mesh: Mesh1D,
                           lookup_table: IceGrowthLookupTable) -> np.ndarray:
    """
    Compute ice growth source terms using Koenig approximation.

    Scalars:
        Y[0] = n/ρ: Specific particle number density [#/kg]
        Y[1] = ρ_vapor/ρ: Vapor mass fraction [-]
        Y[2] = ρ_ice/ρ: Ice mass fraction [-]

    Physics:
        - Particle number is conserved (no source)
        - Vapor condenses onto ice particles: dm/dt = a * m^b * e_fac
        - Ice mass increases from vapor: source = n * dm/dt
        - e_fac accounts for actual supersaturation vs. table values (at saturation)

    Args:
        state: Current flow state
        mesh: Computational mesh
        lookup_table: Pre-loaded ice growth lookup table

    Returns:
        sources: Source terms [kg/(m³·s)] for each scalar
    """
    n_scalars = state.Y.shape[0]
    n_cells = mesh.n_cells
    sources = np.zeros((n_scalars, n_cells))

    if n_scalars < 3:
        raise ValueError(f"Expected 3 scalars (n, vapor, ice), got {n_scalars}")

    # Extract scalar densities
    rho = state.rho                    # Total density [kg/m³]
    n = state.Y[0] * rho               # Particle number density [#/m³]
    rho_vapor = state.Y[1] * rho       # Vapor density [kg/m³]
    rho_ice = state.Y[2] * rho         # Ice density [kg/m³]

    # Compute mean particle mass
    # Avoid division by zero where n → 0
    m_particle = np.where(
        n > 1e-10,  # Threshold for "particles exist"
        rho_ice / n,
        0.0
    )

    # Get growth coefficients from lookup table (computed at water saturation)
    T = state.T  # Temperature [K]
    p = state.p  # Pressure [Pa]

    a, b = lookup_table.get_coefficients(T, p)

    # Compute supersaturation factor e_fac
    # The lookup table gives dm/dt at water saturation (e=1)
    # We need to scale by actual supersaturation
    #
    # The supersaturation factor e_fac is defined as:
    # e_fac = (p_vapor - p_sat_ice) / (p_sat_water - p_sat_ice)
    #
    # This represents the degree of supersaturation:
    # - At ice saturation: e_fac = 0 (no growth)
    # - At water saturation: e_fac = 1 (reference condition for lookup table)
    # - Above water saturation: e_fac > 1 (enhanced growth)
    #
    # The Koenig approximation in the lookup table is calibrated at e_fac = 1,
    # so we multiply by e_fac to account for the actual vapor pressure

    # Partial pressure of water vapor
    p_vapor = rho_vapor * R / M_w * T

    # Saturation pressures
    p_sat_ice = psat_ice(T)
    p_sat_water = psat_water(T)

    # Supersaturation factor w.r.t. ice (dimensionless)
    # e_fac = (p_vapor - p_sat_ice) / (p_sat_water - p_sat_ice)
    # At water saturation: e_fac = 1
    # At ice saturation: e_fac = 0
    # e < 0 means subsaturated (p_vapor < p_sat_ice) - NO GROWTH!

    # CRITICAL: Check if supersaturated w.r.t. ice FIRST
    supersaturated = p_vapor > p_sat_ice

    # Only compute e_fac where supersaturated
    denominator = p_sat_water - p_sat_ice
    e_fac = np.where(
        supersaturated & (np.abs(denominator) > 1e-6),
        (p_vapor - p_sat_ice) / denominator,
        0.0  # Subsaturated or invalid denominator -> no growth
    )

    # Cap extreme supersaturation for numerical stability
    e_fac = np.clip(e_fac, 0.0, 2.0)

    # Compute growth rate: dm/dt = a * m^b * e_fac
    # ONLY where supersaturated (e_fac > 0)
    dm_dt = np.where(
        (n > 1e-10) & (rho_vapor > 1e-20) & (e_fac > 1e-10),  # Require positive e_fac
        a * np.power(np.maximum(m_particle, 1e-20), b) * e_fac,
        0.0
    )

    # Total ice mass growth rate per unit volume
    ice_growth_rate = n * dm_dt

    # Limit growth by available vapor (conservative)
    dt_typical = 1e-3  # seconds
    max_vapor_consumption = rho_vapor / dt_typical
    ice_growth_rate = np.minimum(ice_growth_rate, max_vapor_consumption)

    # FINAL SAFETY: Explicitly zero out growth where subsaturated
    ice_growth_rate = np.where(supersaturated, ice_growth_rate, 0.0)

    # Apply sources
    # Scalar 0: Particle number (conserved)
    sources[0] = 0.0

    # Scalar 1: Water vapor (consumed by ice growth)
    sources[1] = -ice_growth_rate

    # Scalar 2: Ice mass (grows from vapor)
    sources[2] = ice_growth_rate

    return sources


def create_ice_growth_source(hdf5_path: str = '../ice_growth_fits.hdf5') -> ScalarSourceTerm:
    """
    Factory function to create ice growth source term with lookup table.

    Args:
        hdf5_path: Path to ice growth coefficient HDF5 file

    Returns:
        ScalarSourceTerm object ready to add to solver

    Example:
        >>> ice_source = create_ice_growth_source()
        >>> solver.add_source_term(ice_source)
    """
    # Load lookup table once
    lookup_table = IceGrowthLookupTable(hdf5_path)

    # Create closure that captures the lookup table
    def source_function(state: FlowState, mesh: Mesh1D) -> np.ndarray:
        return ice_growth_source_term(state, mesh, lookup_table)

    return ScalarSourceTerm(source_function)


# =============================================================================
# Example usage
# =============================================================================

def example_contrail_simulation():
    """
    Example of setting up a contrail simulation with ice growth.
    """
    print("\n" + "="*80)
    print("EXAMPLE: Contrail Simulation with Ice Growth")
    print("="*80 + "\n")

    from cfd import GasProperties, FlowState, Mesh1D, Solver1D, SolverConfig
    from cfd.boundary import SubsonicInletBC, SubsonicOutletBC

    # Gas properties
    gas = GasProperties(gamma=1.4, R=287.0)

    # Create nozzle mesh (contrail forms in exhaust)
    def nozzle_area(x):
        return 1.0 - 0.4 * np.sin(np.pi * x)

    n_cells = 100
    mesh = Mesh1D.uniform(0.0, 1.0, n_cells, nozzle_area)

    # Solver with 3 scalars
    config = SolverConfig(
        cfl=0.4,  # Lower CFL for stability with source terms
        max_iter=20000,
        convergence_tol=1e-8,
        time_scheme='rk2'
    )

    solver = Solver1D(mesh, gas, n_scalars=3, config=config)

    # Add ice growth source term
    try:
        ice_source = create_ice_growth_source('../ice_growth_fits.hdf5')
        solver.add_source_term(ice_source)
        print("✓ Ice growth source term added successfully\n")
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")
        print("  Continuing without ice growth source term\n")

    # Boundary conditions - CRUISE ALTITUDE CONDITIONS
    # Typical cruise altitude: 10-12 km (FL330-FL390)
    # At 11 km: T ≈ 217 K, p ≈ 23 kPa

    # For a proper nozzle flow simulation, we need realistic area variation
    # A typical aircraft engine nozzle has modest area ratio
    # Use lower Mach number and smaller area variation for stability
    p0 = 23000.0      # Total pressure at cruise [Pa]
    T0 = 217.0        # Total temperature at cruise [K]
    p_exit = 22000.0  # Exit pressure only slightly lower (smaller pressure drop)

    # Initial conditions for scalars (UNCHANGED)
    n_initial = 1e12           # Particle number density [#/m³]
    vapor_initial = 2e-4       # Initial vapor mass fraction
    ice_initial = 1e-11         # Initial ice mass fraction

    # Convert to specific quantities (per unit total mass)
    # Use correct density for cruise altitude
    rho_initial = p0 / (gas.R * T0)  # Correct initial density [kg/m³]
    Y_inlet = np.array([
        n_initial / rho_initial,      # Specific particle number
        vapor_initial,                 # Vapor mass fraction
        ice_initial                    # Ice mass fraction
    ])

    bc_left = SubsonicInletBC(p0=p0, T0=T0, Y_inlet=Y_inlet)
    bc_right = SubsonicOutletBC(p_exit=p_exit)
    solver.set_boundary_conditions(bc_left, bc_right)

    # Initial flow state
    M_init = 0.3
    T_init = T0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)
    p_init = p0 / (1 + 0.5 * (gas.gamma - 1) * M_init**2)**(gas.gamma / (gas.gamma - 1))
    rho_init = p_init / (gas.R * T_init)
    u_init = M_init * np.sqrt(gas.gamma * gas.R * T_init)

    Y = np.zeros((3, n_cells))
    Y[0, :] = Y_inlet[0]
    Y[1, :] = Y_inlet[1]
    Y[2, :] = Y_inlet[2]

    initial_state = FlowState(
        rho=np.full(n_cells, rho_init),
        u=np.full(n_cells, u_init),
        p=np.full(n_cells, p_init),
        Y=Y,
        gas=gas
    )
    solver.set_initial_condition(initial_state)

    print("Configuration complete. Ready to solve.")
    print(f"  Cells: {n_cells}")
    print(f"  Scalars: particle number, vapor, ice")
    print(f"  Initial vapor fraction: {vapor_initial}")
    print(f"  Initial ice fraction: {ice_initial}")
    print(f"  Particle number density: {n_initial:.2e} #/m³")

    # Show vapor pressure diagnostics at initial conditions
    print("\n" + "-"*80)
    print("Vapor Pressure Diagnostics (at initial T={:.1f}K, p={:.1f}Pa):".format(T_init, p_init))
    print("-"*80)

    rho_vapor_init = vapor_initial * rho_init
    p_vapor_init = rho_vapor_init * R / M_w * T_init
    p_sat_ice_init = psat_ice(np.array([T_init]))[0]
    p_sat_water_init = psat_water(np.array([T_init]))[0]

    print(f"  Water vapor partial pressure:     {p_vapor_init:8.2f} Pa")
    print(f"  Saturation pressure over ice:     {p_sat_ice_init:8.2f} Pa")
    print(f"  Saturation pressure over water:   {p_sat_water_init:8.2f} Pa")
    print(f"  Supersaturation factor e_fac:     {(p_vapor_init - p_sat_ice_init)/(p_sat_water_init - p_sat_ice_init):8.4f}")
    print(f"  Relative humidity w.r.t. ice:     {100*p_vapor_init/p_sat_ice_init:8.2f}%")
    print(f"  Relative humidity w.r.t. water:   {100*p_vapor_init/p_sat_water_init:8.2f}%")
    print("-"*80 + "\n")

    # Run simulation
    print("Running simulation...")
    result = solver.solve()

    # Plot results
    print("\nGenerating solution plot...")
    solver.plot_solution('contrail_with_ice_growth.png')
    print("✓ Plot saved as 'contrail_with_ice_growth.png'")


if __name__ == "__main__":
    print(__doc__)
    example_contrail_simulation()

    # Try to load and display lookup table info
    try:
        print("\n" + "="*80)
        print("Ice Growth Lookup Table Information")
        print("="*80)
        lookup = IceGrowthLookupTable('../ice_growth_fits.hdf5')
        print("✓ Lookup table loaded successfully!\n")

        # Test interpolation at a sample point
        T_test = 230.0  # K (typical contrail temperature)
        p_test = 25000.0  # Pa (typical cruise altitude)
        a, b = lookup.get_coefficients(np.array([T_test]), np.array([p_test]))
        print(f"Test interpolation at T={T_test}K, p={p_test}Pa:")
        print(f"  Growth coefficient a: {a[0]:.6e}")
        print(f"  Growth exponent b:    {b[0]:.6f}")

        # Show vapor pressures at this condition
        p_sat_ice_test = psat_ice(np.array([T_test]))[0]
        p_sat_water_test = psat_water(np.array([T_test]))[0]
        print(f"\nSaturation pressures at {T_test}K:")
        print(f"  Over ice:   {p_sat_ice_test:8.2f} Pa")
        print(f"  Over water: {p_sat_water_test:8.2f} Pa")
        print(f"  Difference: {p_sat_water_test - p_sat_ice_test:8.2f} Pa")

    except FileNotFoundError as e:
        print(f"\n⚠ {e}")
        print("\nThe ice growth source term expects an HDF5 file with:")
        print("  - Temperature grid: 'T' or 'temperature'")
        print("  - Pressure grid: 'p' or 'pressure'")
        print("  - Coefficient a: 'a'")
        print("  - Coefficient b: 'b'")

    # Show example usage
    print("\n" + "="*80)
    print("To use in your simulation:")
    print("="*80)
    print("""
from ice_growth_source import create_ice_growth_source

# In your simulation setup:
ice_source = create_ice_growth_source('../ice_growth_fits.hdf5')
solver.add_source_term(ice_source)

# Initial conditions for 3 scalars:
n_particles = 1e12        # Particle number density [#/m³]
vapor_fraction = 1e-3     # Vapor mass fraction
ice_fraction = 1e-4       # Ice mass fraction

Y_inlet = np.array([
    n_particles / rho,    # Specific particle number [#/kg]
    vapor_fraction,       # Vapor mass fraction [-]
    ice_fraction          # Ice mass fraction [-]
])
""")
