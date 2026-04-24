"""
scripts/run_v1_greedy.py
------------------------
Run the v1 greedy integer optimizer + per-line SLSQP.
Identical to: python _run_optimization.py from the project root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import runpy
runpy.run_path(str(Path(__file__).parents[1] / "_run_optimization.py"), run_name="__main__")
