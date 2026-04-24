"""
scripts/run_v2_scenarios.py
----------------------------
Run the v2 multimodal eight-scenario comparison.
Identical to: python run_scenarios.py [--save-csv] from the project root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import runpy
runpy.run_path(str(Path(__file__).parents[1] / "run_scenarios.py"), run_name="__main__")
