"""
data/bsl.py
-----------
Broad Street Line (BSL) / B Line capacity model for the World Cup match day.

BSL serves Lincoln Financial Field directly via NRG Station at the south end
of the line.  This module models:

  - Inbound capacity (Center City → NRG, pre-game direction)
  - Outbound capacity (NRG → Center City, post-game evacuation)
  - NRG Station platform crowding
  - Post-game clearance time
  - Operating cost of event service

The model is intentionally simple: capacity per slot = trains × capacity × buffer.
No microscopic headway simulation or signal-block modeling is performed.
Discrete service levels (normal / enhanced / max_event) replace continuous decisions
to keep the model realistic and interpretable.

Assumption: BSL travel time City Hall ↔ NRG ≈ 15 min (historical ~14–16 min).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from septa_worldcup.v2.config.scenario import (
    N_SLOTS, SLOT_MINUTES,
    BSL_TRAIN_CAPACITY, BSL_SAFETY_BUFFER,
    BSL_SERVICE_LEVELS, BSL_EVENT_HEADWAY_TARGET_MIN,
    NRG_STATION_THROUGHPUT_CAP, NRG_CROWDING_THRESHOLD,
    BSL_FIXED_COST_PER_EXTRA_TRIP, BSL_VAR_COST_PER_PAX,
    POST_GAME_CLEARANCE_TARGET_MIN, SLOT_MINUTES,
    time_to_slot, slot_label,
)


def bsl_capacity_per_slot(service_level: str) -> float:
    """
    Effective BSL passenger capacity for one 15-min slot at a given service level.

      capacity = trains_per_slot × BSL_TRAIN_CAPACITY × BSL_SAFETY_BUFFER

    The safety buffer accounts for operational variance and ADA/standing room policy.
    """
    cfg = BSL_SERVICE_LEVELS[service_level]
    return cfg["trains_per_slot"] * BSL_TRAIN_CAPACITY * BSL_SAFETY_BUFFER


def bsl_slot_cost(service_level: str, baseline_level: str = "normal") -> float:
    """
    Incremental fixed operating cost for one slot of BSL service above baseline.
    Only the *extra* trains above normal headway are charged.
    """
    cfg   = BSL_SERVICE_LEVELS[service_level]
    base  = BSL_SERVICE_LEVELS[baseline_level]
    extra = max(0, cfg["trains_per_slot"] - base["trains_per_slot"])
    return extra * BSL_FIXED_COST_PER_EXTRA_TRIP


def choose_bsl_service_level(demand: float,
                              prefer_max_for_postgame: bool = False) -> str:
    """
    Choose the minimum BSL service level that covers the slot demand.
    During post-game evacuation, always prefer max_event to minimize clearance time.
    """
    if prefer_max_for_postgame:
        return "max_event"
    for level in ["normal", "enhanced", "max_event"]:
        cap = bsl_capacity_per_slot(level)
        if cap >= demand:
            return level
    return "max_event"   # even max_event may not fully cover peak; unmet demand is reported


def allocate_bsl_service(
    bsl_inbound:  np.ndarray,
    bsl_outbound: np.ndarray,
    kickoff_slot: int,
    match_end_slot: int,
    budget_remaining: float = float("inf"),
) -> Dict:
    """
    Greedy slot-by-slot BSL service level assignment.

    Decision rule:
      - Pre-game (slot 0 → kickoff):   use minimum level to cover inbound demand
      - Match window:                   drop to normal (low demand)
      - Post-game (match_end → 04:00):  always max_event (evacuation priority)

    Budget is checked: if over budget, reduce post-game slots to enhanced.

    Returns a dict with per-slot arrays and aggregate metrics.
    """
    levels:    List[str]  = []
    trains:    np.ndarray = np.zeros(N_SLOTS)
    capacity:  np.ndarray = np.zeros(N_SLOTS)
    served_in: np.ndarray = np.zeros(N_SLOTS)
    served_out:np.ndarray = np.zeros(N_SLOTS)
    unmet_in:  np.ndarray = np.zeros(N_SLOTS)
    unmet_out: np.ndarray = np.zeros(N_SLOTS)
    crowding:  np.ndarray = np.zeros(N_SLOTS)
    op_cost    = 0.0

    for t in range(N_SLOTS):
        d_in  = float(bsl_inbound[t])
        d_out = float(bsl_outbound[t])
        d_tot = d_in + d_out

        # Determine service level for this slot
        if t < kickoff_slot:
            level = choose_bsl_service_level(d_tot, prefer_max_for_postgame=False)
        elif kickoff_slot <= t <= match_end_slot:
            level = "normal"                           # low demand during match
        else:
            # Post-game: maximise outbound capacity
            slot_cost = bsl_slot_cost("max_event")
            if budget_remaining >= slot_cost:
                level = "max_event"
            else:
                level = choose_bsl_service_level(d_tot, prefer_max_for_postgame=False)

        cap = bsl_capacity_per_slot(level)
        cfg = BSL_SERVICE_LEVELS[level]
        cost_this_slot = bsl_slot_cost(level)
        op_cost          += cost_this_slot
        budget_remaining -= cost_this_slot

        # Serve inbound first (pre-game arrival more time-sensitive)
        s_in   = min(d_in,  cap)
        rem    = max(0.0, cap - s_in)
        s_out  = min(d_out, rem)

        levels.append(level)
        trains[t]     = cfg["trains_per_slot"]
        capacity[t]   = cap
        served_in[t]  = s_in
        served_out[t] = s_out
        unmet_in[t]   = max(0.0, d_in  - s_in)
        unmet_out[t]  = max(0.0, d_out - s_out)

        # NRG crowding: unmet outbound demand accumulates on the platform
        # Crowding = passengers above the NRG throughput threshold
        pax_on_platform = unmet_out[t]
        crowding[t]     = max(0.0, pax_on_platform - NRG_CROWDING_THRESHOLD)

        # Variable cost on passengers served
        op_cost += BSL_VAR_COST_PER_PAX * (s_in + s_out)

    # ── Post-game clearance time ───────────────────────────────────────────────
    # Clearance = first slot after match_end where cumulative unmet outbound → 0
    clearance_slot = _compute_clearance_slot(
        bsl_outbound, capacity, match_end_slot
    )
    if clearance_slot < N_SLOTS:
        clearance_min = (clearance_slot - match_end_slot) * SLOT_MINUTES
    else:
        clearance_min = (N_SLOTS - match_end_slot) * SLOT_MINUTES  # not cleared by 04:00

    headway_violation = _headway_penalty_minutes(levels, kickoff_slot, match_end_slot)

    return {
        "service_levels":     levels,
        "trains_per_slot":    trains,
        "capacity":           capacity,
        "served_inbound":     served_in,
        "served_outbound":    served_out,
        "unmet_inbound":      unmet_in,
        "unmet_outbound":     unmet_out,
        "nrg_crowding":       crowding,
        "operating_cost":     op_cost,
        "clearance_slot":     clearance_slot,
        "clearance_time_min": clearance_min,
        "headway_violation_min": headway_violation,
        "peak_crowding":      float(crowding.max()),
        "total_served":       float(served_in.sum() + served_out.sum()),
        "total_unmet":        float(unmet_in.sum() + unmet_out.sum()),
        "load_factor":        (float((served_in + served_out).sum()) /
                               max(float(capacity.sum()), 1e-6)),
    }


def _compute_clearance_slot(outbound: np.ndarray,
                             capacity: np.ndarray,
                             match_end_slot: int) -> int:
    """
    Estimate the slot at which the post-game evacuation queue clears.

    We simulate a queue: carry-over unmet demand from slot t spills into t+1.
    Returns the first slot where cumulative backlog reaches zero.
    """
    backlog = 0.0
    for t in range(match_end_slot, N_SLOTS):
        backlog += float(outbound[t])
        served   = min(backlog, float(capacity[t]))
        backlog  = max(0.0, backlog - served)
        if backlog < 1.0:    # effectively cleared
            return t
    return N_SLOTS           # queue not cleared by end of model window


def _headway_penalty_minutes(levels: List[str],
                              kickoff_slot: int,
                              match_end_slot: int) -> float:
    """
    Total minutes of headway exceeding the event target during key windows.
    Penalizes insufficient pre-game and post-game service frequency.
    """
    violation = 0.0
    for t, level in enumerate(levels):
        in_event_window = (t < kickoff_slot) or (t > match_end_slot)
        if not in_event_window:
            continue
        headway = BSL_SERVICE_LEVELS[level]["headway_min"]
        excess  = max(0.0, headway - BSL_EVENT_HEADWAY_TARGET_MIN)
        violation += excess
    return violation


def print_bsl_summary(result: Dict, n_slots: int = N_SLOTS) -> None:
    """Print a terminal summary of BSL service allocation."""
    print(f"  BSL total served:       {result['total_served']:>10,.0f} pax")
    print(f"  BSL total unmet:        {result['total_unmet']:>10,.0f} pax")
    print(f"  BSL load factor:        {result['load_factor']:>10.1%}")
    print(f"  Peak NRG crowding:      {result['peak_crowding']:>10,.0f} pax above threshold")
    print(f"  Post-game clearance:    {result['clearance_time_min']:>10.0f} min "
          f"(target ≤ {POST_GAME_CLEARANCE_TARGET_MIN} min)")
    print(f"  BSL operating cost:     ${result['operating_cost']:>10,.0f}")
    print(f"  Headway violation:      {result['headway_violation_min']:>10.0f} min above target")
