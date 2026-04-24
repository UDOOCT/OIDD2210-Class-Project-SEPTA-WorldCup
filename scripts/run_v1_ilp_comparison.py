"""
scripts/run_v1_ilp_comparison.py
---------------------------------
Run the v1 ILP comparison (greedy vs CBC optimal).
Requires: pip install pulp
Identical to: python _run_ilp_comparison.py from the project root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import runpy
runpy.run_path(str(Path(__file__).parents[1] / "_run_ilp_comparison.py"), run_name="__main__")
