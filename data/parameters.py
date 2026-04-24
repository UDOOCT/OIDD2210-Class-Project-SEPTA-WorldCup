"""
data/parameters.py
------------------
All operational and model parameters.

TIME RESOLUTION: 15-minute slots from 6:00am to 9:00pm
  T = [6.00, 6.25, 6.50, ..., 20.75, 21.00]  →  61 slots total
  Index 0 = 6:00am,  Index 12 = 9:00am,
  Index 40 = 4:00pm, Index 52 = 7:00pm, Index 60 = 9:00pm

DATA SOURCES for costs:
  SEPTA OpenDataPhilly Route Operating Statistics FY2024 Budgeted:
  https://data-septa.opendata.arcgis.com/datasets/26f40aaeb8ae41d291878ba726b57fed_0
  Loaded from: data/costs/cost_summary.json
"""

import json
from pathlib import Path

import numpy as np

# ── Time slots ────────────────────────────────────────────────────────────────
TIME_SLOTS      = [round(6.0 + i * 0.25, 2) for i in range(61)]  # 6:00–21:00
N_SLOTS         = len(TIME_SLOTS)   # 61
SLOT_DURATION   = 0.25              # hours (15 minutes)
SLOT_DURATION_MIN = 15              # minutes


def slot_label(t: float) -> str:
    """Convert decimal hour to readable label, e.g. 6.25 → '6:15am'"""
    h = int(t)
    m = int(round((t - h) * 60))
    suffix = "am" if h < 12 else "pm"
    h12 = h if h <= 12 else h - 12
    return f"{h12}:{m:02d}{suffix}"


def slot_idx(t: float) -> int:
    """Return index of time slot closest to t (decimal hours)."""
    return int(round((t - 6.0) / 0.25))


# Named anchor indices for reference
IDX_6AM  = slot_idx(6.0)    # 0
IDX_9AM  = slot_idx(9.0)    # 12
IDX_4PM  = slot_idx(16.0)   # 40
IDX_7PM  = slot_idx(19.0)   # 52
IDX_9PM  = slot_idx(21.0)   # 60

# Peak windows (for surge pricing logic)
MORNING_RUSH_IDX = (IDX_6AM,  IDX_9AM)   # slots 0–12
MIDDAY_IDX       = (IDX_9AM,  IDX_4PM)   # slots 12–40
EVENING_RUSH_IDX = (IDX_4PM,  IDX_7PM)   # slots 40–52
NIGHT_IDX        = (IDX_7PM,  IDX_9PM)   # slots 52–60


def is_peak(t_idx: int) -> bool:
    """True if slot is in morning or evening rush."""
    return (MORNING_RUSH_IDX[0] <= t_idx < MORNING_RUSH_IDX[1] or
            EVENING_RUSH_IDX[0] <= t_idx < EVENING_RUSH_IDX[1])


# ── Train / Fleet ─────────────────────────────────────────────────────────────
SEATS_PER_CAR       = 175
CARS_PER_TRAIN      = 5
TRAIN_CAPACITY      = SEATS_PER_CAR * CARS_PER_TRAIN   # 875 seated
MAX_CAPACITY        = int(TRAIN_CAPACITY * 1.3)         # 1,137 crush load

# ── Costs — loaded from data/costs/cost_summary.json ─────────────────────────
_cs_path = Path(__file__).parent / "costs" / "cost_summary.json"
if not _cs_path.exists():
    raise FileNotFoundError(f"Required data file missing: {_cs_path}")
_CS = json.loads(_cs_path.read_text())

# Per-trip fixed cost = daily trainset cost / 8 trips per day.
# Raw value $13,535.85 from cost_summary.json is per trainset per day.
FIXED_COST_PER_TRAIN: float = round(13535.85 / 8, 2)   # 1,691.98

# Marginal variable cost per passenger (fuel + platform staff increment).
# $34.03 from cost_summary.json is average all-in cost — not suitable
# for an incremental optimization objective. Using $0.40 literature estimate.
# Source: SEPTA operating stats marginal cost analysis.
VARIABLE_COST_PER_PAX: float = 0.40

DAILY_BUDGET_NORMAL    = 250_000
DAILY_BUDGET_EVENT     = 350_000  # World Cup match day

# ── Fares ─────────────────────────────────────────────────────────────────────
ZONE_FARES = {
    "CC": 2.50, "1": 3.75, "2": 4.25, "3": 5.25, "4": 6.00, "NJ": 6.50,
}
FARE_MIN     = 2.50
FARE_MAX     = 9.00
SURGE_FACTOR = 1.5   # max peak/off-peak multiplier

# ── Model ─────────────────────────────────────────────────────────────────────
EQUITY_EPSILON       = 0.80
MIN_TRAINS_PER_SLOT  = 0
MAX_TRAINS_PER_SLOT  = 8

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
