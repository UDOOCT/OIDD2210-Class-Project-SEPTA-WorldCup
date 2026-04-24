"""
data/scenario.py
----------------
Master configuration for the multimodal World Cup 2026 transit model.

Time window: 18:00 on match day → 04:00 the next day, 15-minute slots.
  Slot 0  = 18:00   (pre-game, fans starting to arrive)
  Slot 10 = 20:30   (kickoff, default)
  Slot 23 = 23:45   (last slot before midnight)
  Slot 24 = 00:00   (crosses into next calendar day)
  Slot 39 = 03:45   (final slot, late-night tail)

All scenario parameters live here so assumptions are visible in one place.
"""

from __future__ import annotations
from typing import Dict

# ── Match information ──────────────────────────────────────────────────────────
MATCH_DATE           = "2026-06-19"
KICKOFF_TIME_OPTIONS = ["20:30", "21:00"]
DEFAULT_KICKOFF_TIME = "20:30"

# ── Time window ────────────────────────────────────────────────────────────────
TIME_START   = "18:00"
TIME_END     = "04:00+1"   # "+1" marks next calendar day
SLOT_MINUTES = 15
N_SLOTS      = 40          # 10 hours × 4 slots/hour

assert N_SLOTS == (10 * 60) // SLOT_MINUTES, "N_SLOTS must equal 10h / 15min"


def slot_minutes_from_1800(slot_idx: int) -> int:
    """Minutes elapsed since 18:00 for the start of slot_idx."""
    return slot_idx * SLOT_MINUTES


def slot_clock_minutes(slot_idx: int) -> int:
    """Minutes past midnight (0–1439) for the start of slot_idx."""
    return (18 * 60 + slot_idx * SLOT_MINUTES) % (24 * 60)


def slot_label(slot_idx: int) -> str:
    """
    Human-readable HH:MM label for a slot.
    Post-midnight slots include ' (+1)' to indicate next calendar day.
    """
    total = 18 * 60 + slot_idx * SLOT_MINUTES
    next_day = total >= 24 * 60
    total = total % (24 * 60)
    label = f"{total // 60:02d}:{total % 60:02d}"
    return label + " (+1)" if next_day else label


def time_to_slot(hhmm: str) -> int:
    """
    Convert 'HH:MM' to slot index relative to 18:00.

    Times from 00:00 to 04:00 are treated as next-day (post-midnight).
    For example, '02:00' → slot 32  (2:00 AM next day = 8 hours after 18:00).
    """
    h, m = map(int, hhmm.split(":"))
    total_min = h * 60 + m
    ref_min   = 18 * 60      # 18:00 as minutes from midnight
    offset    = total_min - ref_min
    if offset < 0:
        offset += 24 * 60   # handle midnight wrap (e.g. 02:00 → +8h from 18:00)
    if not (0 <= offset < N_SLOTS * SLOT_MINUTES):
        raise ValueError(f"Time '{hhmm}' is outside the model window "
                         f"[18:00, 04:00+1).  offset={offset} min.")
    return offset // SLOT_MINUTES


def generate_slot_labels() -> list:
    """Return list of 40 human-readable time labels (index 0 → '18:00' …)."""
    return [slot_label(t) for t in range(N_SLOTS)]


# ── Stadium & transfer stations ────────────────────────────────────────────────
STADIUM_STATION   = "NRG Station"          # served by BSL/B Line
TRANSFER_STATIONS = [                      # where RR → BSL transfers happen
    "Suburban Station",
    "Jefferson Station",
    "30th Street Station",
    "City Hall / Dilworth Park",
]

# ── Scenario feature toggles ──────────────────────────────────────────────────
# Turn these off to replicate the original RR-only v1 baseline.
INCLUDE_POST_GAME    = True   # model post-game evacuation demand
INCLUDE_BSL          = True   # include Broad Street Line capacity model
FREE_RETURN_FROM_NRG = True   # post-game rides sponsored (fare = $0)
INBOUND_REGULAR_FARE = True   # inbound trips charged at regular zone fare
SPONSOR_SUBSIDY      = True   # sponsor reimburses SEPTA per free-return passenger

# ── Match timing (relative to kickoff) ────────────────────────────────────────
MATCH_DURATION_MINUTES         = 120  # standard 90 min + ~30 min buffer for OT/AET
STOPPAGE_TIME_BUFFER_MINUTES   = 10   # added time beyond official 90+2×15 window
EXIT_DELAY_MEAN_MINUTES        = 25   # minutes after final whistle to clear stadium
POST_GAME_PEAK_OFFSET_MINUTES  = 30   # additional delay for transit queue formation

# ── Fan segmentation ──────────────────────────────────────────────────────────
# Shares must sum to 1.0.  These drive how total demand splits between modes.
FAN_SEGMENTS: Dict[str, float] = {
    "local_city":            0.35,  # reach NRG via BSL from Center City
    "suburban_rr":           0.30,  # Regional Rail → Center City → BSL to NRG
    "visitor_hotel_airport": 0.20,  # half via RR, half direct BSL
    "car_rideshare":         0.15,  # not using SEPTA (excluded from model)
}
assert abs(sum(FAN_SEGMENTS.values()) - 1.0) < 1e-9, "Fan segments must sum to 1.0"

TOTAL_STADIUM_CAPACITY = 69_328
TRANSIT_SHARE          = 0.65
TOTAL_FANS_TRANSIT     = int(TOTAL_STADIUM_CAPACITY * TRANSIT_SHARE)  # ≈ 45,063

# Share of transit fans who also use Regional Rail as a feeder
RR_FEEDER_SHARE = (FAN_SEGMENTS["suburban_rr"] +
                   FAN_SEGMENTS["visitor_hotel_airport"] * 0.5)  # = 0.40

# ── BSL / B Line parameters ───────────────────────────────────────────────────
BSL_CARS_PER_TRAIN   = 8
BSL_SEATS_PER_CAR    = 90
BSL_TRAIN_CAPACITY   = BSL_CARS_PER_TRAIN * BSL_SEATS_PER_CAR  # 720 pax/train

BSL_SAFETY_BUFFER    = 0.85   # effective capacity = nominal × buffer (standing room)

# NRG Station physical capacity constraints
NRG_STATION_THROUGHPUT_CAP  = 4_000  # pax per 15-min slot (platform + fare gates)
NRG_CROWDING_THRESHOLD      = 2_500  # pax/slot above which crowding penalty applies

# Discrete service levels: name → {headway_min, trains_per_slot}
BSL_SERVICE_LEVELS: Dict[str, Dict] = {
    "normal":    {"headway_min": 8, "trains_per_slot": 2},   # standard off-peak
    "enhanced":  {"headway_min": 5, "trains_per_slot": 3},   # event pre-game boost
    "max_event": {"headway_min": 3, "trains_per_slot": 5},   # post-game evacuation
}
BSL_EVENT_HEADWAY_TARGET_MIN = 5   # must achieve ≤ 5 min during event windows

# ── Regional Rail extra service (discrete choices) ────────────────────────────
RR_EXTRA_TRAINS_OPTIONS      = [0, 1, 2, 3]  # extra trains added per slot
RR_BASELINE_TRAINS_EVENING   = 1             # trains/slot during 18:00–22:00
RR_BASELINE_TRAINS_LATENIGHT = 0             # trains/slot after 22:00 (sparse service)

# Slots by period (used to set baseline service)
RR_EVENING_END_SLOT  = time_to_slot("22:00")  # slot 16: last evening slot
RR_LATENIGHT_SLOTS   = range(RR_EVENING_END_SLOT, N_SLOTS)

# ── Cost parameters ───────────────────────────────────────────────────────────
RR_FIXED_COST_PER_TRAIN_TRIP  = 1_692.00  # $/trip (from data/parameters.py)
RR_VAR_COST_PER_PAX           = 0.40      # $/pax
BSL_FIXED_COST_PER_EXTRA_TRIP = 800.00    # $/event trip (shorter route than RR)
BSL_VAR_COST_PER_PAX          = 0.30      # $/pax

DAILY_EVENT_BUDGET = 350_000  # $ total operating budget for match day

# ── Fare policy ───────────────────────────────────────────────────────────────
BASE_INBOUND_FARE            = 2.50   # $ per trip inbound (regular SEPTA zone fare)
FREE_RETURN_FARE             = 0.00   # $ post-game (sponsor-funded)
SPONSOR_REIMBURSEMENT_PER_PAX = 3.00 # $ SEPTA receives per sponsored return trip

# ── Policy objective weights (tune these for sensitivity analysis) ────────────
W_UNMET_DEMAND = 50.0       # $/unserved passenger
W_CROWDING     = 30.0       # $/pax above NRG crowding threshold
W_EQUITY       = 100_000.0  # $ lump-sum if equity threshold violated
W_HEADWAY      = 5_000.0    # $/min headway above target (service reliability)
W_CLEARANCE    = 200.0      # $/min post-game clearance beyond target

POST_GAME_CLEARANCE_TARGET_MIN = 90   # clear NRG within 90 min of match end

# ── Equity thresholds ─────────────────────────────────────────────────────────
# Two distinct concepts — see formulation.md §6 for discussion.
EQUITY_MIN_RAW_COVERAGE       = 0.80  # KPI only: serve ≥ 80% of raw demand
EQUITY_MIN_EFFECTIVE_COVERAGE = 0.90  # Operational constraint: ≥ 90% of effective demand

# ── Regional Rail fleet ───────────────────────────────────────────────────────
# Carried over from data/parameters.py for use in the new 18:00–04:00 model.
TRAIN_CAPACITY_RR = 875   # Bombardier Multilevel, 5-car consist
