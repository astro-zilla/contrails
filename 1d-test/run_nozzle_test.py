"""
Example script to run the subsonic nozzle test case.

This script should be run from the project root directory.
"""

import sys
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cfd.test_cases import run_subsonic_nozzle_test

if __name__ == "__main__":
    # Run the subsonic nozzle test case
    solver = run_subsonic_nozzle_test(n_cells=100, n_scalars=3)

