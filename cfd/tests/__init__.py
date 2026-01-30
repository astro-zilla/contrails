"""
Test cases for the 1D compressible flow solver.

Run tests with pytest:
    pytest cfd/tests/ -v

Or run individual test files:
    pytest cfd/tests/test_nozzle.py -v
    pytest cfd/tests/test_shock_tube.py -v
"""

# Keep backward compatibility with existing API
from .nozzle import run_subsonic_nozzle_test, nozzle_area
from .shock_tube import run_shock_tube_test, sod_shock_tube_exact, ReflectiveWallBC

__all__ = [
    'run_subsonic_nozzle_test',
    'nozzle_area',
    'run_shock_tube_test',
    'sod_shock_tube_exact',
    'ReflectiveWallBC',
]
