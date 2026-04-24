"""
septa_worldcup
--------------
SEPTA World Cup 2026 transit optimization package.

Sub-packages:
  v1  — Regional Rail profit model (18:00–04:00+1, 40 slots × 13 lines)
  v2  — Multimodal policy model  (18:00–04:00+1, 40 slots, RR + BSL)
  common — Shared utilities
"""
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]   # …/septa_worldcup/src/septa_worldcup → ../../..
DATA_DIR  = REPO_ROOT / "data"
