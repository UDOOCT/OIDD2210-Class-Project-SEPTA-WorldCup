"""
src/septa_worldcup/v1/data/parameters.py
-----------------------------------------
Operational and model parameters for the v1 Regional Rail profit model.

ACTIVE TIME WINDOW: 18:00 match day → 03:45+1 (next calendar day)
  Slot 0  = 18:00        Slot 10 = 20:30 (default kickoff)
  Slot 16 = 22:00        Slot 24 = 00:00 (+1)
  Slot 30 = 01:30 (+1)   Slot 39 = 03:45 (+1)
  N_SLOTS = 40  (10 hours × 4 slots/hour, 15-minute resolution)

  v1 fare/time blocks:
    pre_game  : [18:00, 20:30)  — slots  0–9   (10 slots)
    in_game   : [20:30, 22:30)  — slots 10–17  ( 8 slots)
    post_game : [22:30, 01:30)  — slots 18–29  (12 slots)
    late_night: [01:30, 04:00)  — slots 30–39  (10 slots)

HISTORICAL NOTE: the original v1 model used a 6am–9pm window (61 slots,
  decimal-hour TIME_SLOTS).  That window is no longer the active baseline.
  Active v1 now aligns with v2 at 18:00–04:00+1.

DATA SOURCES for costs:
  SEPTA OpenDataPhilly Route Operating Statistics FY2024 Budgeted:
  https://data-septa.opendata.arcgis.com/datasets/26f40aaeb8ae41d291878ba726b57fed_0
  Loaded from: data/costs/cost_summary.json
"""

import json
from pathlib import Path

import numpy as np

from septa_worldcup import DATA_DIR

# ── Time window ───────────────────────────────────────────────────────────────
TIME_START        = "18:00"
TIME_END          = "04:00+1"    # "+1" marks next calendar day
SLOT_MINUTES      = 15
N_SLOTS           = 40           # 10 hours × 4 slots/hour
TIME_SLOTS        = list(range(N_SLOTS))   # [0, 1, …, 39]  — integer slot indices
SLOT_DURATION     = 0.25         # hours (15 minutes)
SLOT_DURATION_MIN = 15           # minutes


def slot_minutes_from_1800(slot_idx: int) -> int:
    """Minutes elapsed since 18:00 for the start of slot_idx."""
    return slot_idx * SLOT_MINUTES


def slot_clock_minutes(slot_idx: int) -> int:
    """Minutes past midnight (0–1439) for the start of slot_idx."""
    return (18 * 60 + slot_idx * SLOT_MINUTES) % (24 * 60)


def slot_label(slot_idx: int) -> str:
    """
    Human-readable HH:MM label. Post-midnight slots include ' (+1)'.
    Examples: slot 0 → '18:00', slot 24 → '00:00 (+1)', slot 39 → '03:45 (+1)'
    """
    total    = 18 * 60 + slot_idx * SLOT_MINUTES
    next_day = total >= 24 * 60
    total    = total % (24 * 60)
    h, m     = total // 60, total % 60
    label    = f"{h:02d}:{m:02d}"
    if next_day:
        label += " (+1)"
    return label


def time_to_slot(hhmm: str) -> int:
    """
    Convert 'HH:MM' clock time to a slot index (0–39) relative to 18:00.
    Times before 18:00 are treated as next-day (post-midnight).
    Examples: '18:00' → 0, '20:30' → 10, '00:00' → 24, '03:45' → 39
    """
    h, m    = int(hhmm[:2]), int(hhmm[3:5])
    total   = h * 60 + m
    start   = 18 * 60
    minutes = (total - start) % (24 * 60)
    return minutes // SLOT_MINUTES


# Named anchor indices
IDX_1800 = 0                        # 18:00 — window start
IDX_2030 = time_to_slot("20:30")    # 10 — default kickoff
IDX_2230 = time_to_slot("22:30")    # 18 — post-game start
IDX_0000 = time_to_slot("00:00")    # 24 — midnight
IDX_0130 = time_to_slot("01:30")    # 30 — late-night start
IDX_0345 = N_SLOTS - 1              # 39 — last slot

# v1 fare/time blocks
TBLOCK_NAMES = ["pre_game", "in_game", "post_game", "late_night"]
TBLOCK_RANGES_V1 = {
    "pre_game":   range(IDX_1800, IDX_2030),          # slots  0–9
    "in_game":    range(IDX_2030, IDX_2230),           # slots 10–17
    "post_game":  range(IDX_2230, IDX_0130),           # slots 18–29
    "late_night": range(IDX_0130, N_SLOTS),            # slots 30–39
}


def is_peak(t_idx: int) -> bool:
    """
    True for high-demand event periods: pre-game and post-game.
    Used to classify slots for display and surge pricing logic.
    """
    return t_idx < IDX_2030 or IDX_2230 <= t_idx < IDX_0130


# ── Train / Fleet ─────────────────────────────────────────────────────────────
SEATS_PER_CAR      = 175
CARS_PER_TRAIN     = 5
TRAIN_CAPACITY     = SEATS_PER_CAR * CARS_PER_TRAIN   # 875 seated
MAX_CAPACITY       = int(TRAIN_CAPACITY * 1.3)         # 1,137 crush load

# ── Costs — loaded from data/costs/cost_summary.json ─────────────────────────
_cs_path = DATA_DIR / "costs" / "cost_summary.json"
if not _cs_path.exists():
    raise FileNotFoundError(f"Required data file missing: {_cs_path}")
_CS = json.loads(_cs_path.read_text())

# Per-trip fixed cost = daily trainset cost / 8 trips per day.
# Raw value $13,535.85 from cost_summary.json is per trainset per day.
FIXED_COST_PER_TRAIN: float = round(13535.85 / 8, 2)   # 1,691.98

# Marginal variable cost per passenger.
VARIABLE_COST_PER_PAX: float = 0.40

DAILY_BUDGET_NORMAL = 250_000
DAILY_BUDGET_EVENT  = 350_000  # World Cup match day

# ── Fares ─────────────────────────────────────────────────────────────────────
ZONE_FARES = {
    "CC": 2.50, "1": 3.75, "2": 4.25, "3": 5.25, "4": 6.00, "NJ": 6.50,
}
FARE_MIN     = 2.50
FARE_MAX     = 9.00
SURGE_FACTOR = 1.5   # max peak/off-peak multiplier

# ── Model ─────────────────────────────────────────────────────────────────────
EQUITY_EPSILON      = 0.80
MIN_TRAINS_PER_SLOT = 0
MAX_TRAINS_PER_SLOT = 8

# Logit calibration (Small & Verhoef 2007)
LOGIT_ALPHA_FARE   = 0.50
LOGIT_ALPHA_WAIT   = 0.30
LOGIT_ALPHA_TRAVEL = 0.15
LOGIT_THETA        = 0.80
LOGIT_NO_TRAVEL_U  = -1.50

PRICE_ELASTICITY   = -0.35

# ── Data provenance ───────────────────────────────────────────────────────────
DATA_SOURCE = {
    "travel_times": "data/gtfs/travel_times.json",
    "ridership":    "data/ridership/ridership_by_line.json",
    "costs":        "data/costs/cost_summary.json",
}
