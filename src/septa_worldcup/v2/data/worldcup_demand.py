"""
data/worldcup_demand.py
-----------------------
Extended demand model for the 18:00–04:00 match-day window.

Demand has four components for each 15-minute slot:

  A. Evening baseline commuter demand
     Declining evening profile drawn from FY2024 ridership.
     Assumption: evening (18:00–22:00) accounts for ~10% of daily boardings;
     service is sparse after 22:00 on normal weekdays.

  B. Pre-game fan arrival demand
     Normal distribution centered ~90 min before kickoff.
     Fans arrive between 18:00 and kickoff; this drives inbound RR and BSL load.

  C. In-game / low-activity demand
     Small constant demand during match window (staff, late arrivals, non-fans).

  D. Post-game evacuation demand
     Large spike after match ends, with a long tail through ~04:00.
     This is the primary BSL bottleneck modeled here.

Fan origin segments determine how demand splits between Regional Rail and BSL:
  - local_city fans use BSL directly
  - suburban_rr fans use RR as feeder → Center City → BSL to NRG
  - visitor_hotel_airport fans split 50/50
  - car_rideshare fans are outside the model scope

All assumptions are transparent and documented below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import norm

from septa_worldcup import DATA_DIR
from septa_worldcup.v2.config.scenario import (
    N_SLOTS, SLOT_MINUTES, TOTAL_FANS_TRANSIT, RR_FEEDER_SHARE,
    FAN_SEGMENTS, MATCH_DURATION_MINUTES, STOPPAGE_TIME_BUFFER_MINUTES,
    EXIT_DELAY_MEAN_MINUTES, POST_GAME_PEAK_OFFSET_MINUTES,
    DEFAULT_KICKOFF_TIME, time_to_slot, slot_label,
)

# ── Load FY2024 ridership weights ─────────────────────────────────────────────
_rbl = json.loads((DATA_DIR / "ridership" / "ridership_by_line.json").read_text())
_TOTAL_DAILY_RIDERSHIP = sum(_rbl.values())   # 48,343
_LINE_SHARES = {l: v / _TOTAL_DAILY_RIDERSHIP for l, v in _rbl.items()}

# Assumption: evening window (18:00–22:00) carries ~10% of daily boardings per line.
# Late-night (22:00–04:00) carries ~2%.  Rest of the day (not modeled here) = 88%.
EVENING_SHARE_OF_DAILY   = 0.10   # 18:00–22:00
LATENIGHT_SHARE_OF_DAILY = 0.02   # 22:00–04:00


def _evening_baseline_profile() -> np.ndarray:
    """
    Normalized intraday weight for each of the 40 slots, representing
    background commuter + non-event rider activity.

    Shape: linearly declining from 18:00 (busy) to 22:00 (tapering),
    then very small after midnight.  All values normalized to sum to 1.
    """
    weights = np.zeros(N_SLOTS)
    eve_end = time_to_slot("22:00")   # slot 16

    # 18:00–22:00: linearly declining
    n_eve = eve_end
    for t in range(n_eve):
        weights[t] = 1.0 - (t / n_eve) * 0.7   # declines from 1.0 → 0.3

    # 22:00–04:00: small constant late-night baseline
    for t in range(eve_end, N_SLOTS):
        weights[t] = 0.05

    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


_BASELINE_PROFILE = _evening_baseline_profile()


def compute_baseline_demand(line: str) -> np.ndarray:
    """
    Evening/late-night baseline commuter demand for one line.
    Scaled to match approximately 12% of FY2024 daily weekday boardings
    (10% evening + 2% late-night), distributed over 40 slots.

    Returns: np.array of shape (N_SLOTS,) — passengers per 15-min slot.
    """
    daily_riders     = _rbl.get(line, 0)
    evening_total    = daily_riders * (EVENING_SHARE_OF_DAILY + LATENIGHT_SHARE_OF_DAILY)
    return np.array(evening_total * _BASELINE_PROFILE)


def _normal_wave(peak_slot: float, sigma_slots: float,
                 total_pax: float, start_slot: int = 0,
                 end_slot: int = N_SLOTS) -> np.ndarray:
    """
    Build a Normal-shaped demand wave across slots, truncated to [start_slot, end_slot).

    Args:
        peak_slot   : slot index of the peak (can be fractional)
        sigma_slots : standard deviation in slot units
        total_pax   : total passengers in this wave (integral)
        start_slot  : first slot to include
        end_slot    : one past last slot

    Returns: np.array of shape (N_SLOTS,) — pax per 15-min slot.
    """
    wave = np.zeros(N_SLOTS)
    for t in range(start_slot, end_slot):
        wave[t] = norm.pdf(t, peak_slot, sigma_slots)
    if wave.sum() > 1e-9:
        wave = wave / wave.sum() * total_pax
    return wave


def compute_worldcup_demand(
    kickoff: str = DEFAULT_KICKOFF_TIME,
    total_fans: int = TOTAL_FANS_TRANSIT,
    include_post_game: bool = True,
) -> Dict:
    """
    Compute all World Cup demand components for the 18:00–04:00 window.

    Args:
        kickoff          : kickoff time string 'HH:MM' (default '20:30')
        total_fans       : total fans using SEPTA transit (~45,000)
        include_post_game: whether to include post-game evacuation demand

    Returns a dict with per-slot arrays (shape N_SLOTS each) and metadata:
        raw_worldcup         total WC fan demand per slot (pre + in-game + post)
        pre_game_wave        pre-game arrival component
        ingame_wave          low in-game activity
        post_game_wave       post-game evacuation component
        bsl_inbound          BSL load toward NRG (pre-game direction)
        bsl_outbound         BSL load away from NRG (post-game direction)
        rr_inbound_total     total RR feeder demand across all lines
        rr_outbound_total    total RR demand (fans heading home via RR after game)
        rr_demand            dict: line → np.array(N_SLOTS) for RR inbound+outbound
        post_game_evacuation post-game component only (for KPIs)
        kickoff_slot         integer slot index of kickoff
        match_end_slot       slot index when match approximately ends
        post_game_peak_slot  slot of peak post-game demand
        total_inbound_fans   total fans counted inbound
        total_outbound_fans  total fans counted outbound
    """
    ko_slot = time_to_slot(kickoff)

    # ── Match timing derived from kickoff ──────────────────────────────────────
    # Match end: kickoff + duration + stoppage buffer
    match_end_min_from_18 = (ko_slot * SLOT_MINUTES +
                              MATCH_DURATION_MINUTES +
                              STOPPAGE_TIME_BUFFER_MINUTES)
    match_end_slot = match_end_min_from_18 // SLOT_MINUTES

    # Post-game transit peak: after fans exit stadium and queue at NRG
    post_game_peak_min = (match_end_min_from_18 +
                          EXIT_DELAY_MEAN_MINUTES +
                          POST_GAME_PEAK_OFFSET_MINUTES)
    post_game_peak_slot = post_game_peak_min // SLOT_MINUTES

    # ── A: Pre-game arrival wave ──────────────────────────────────────────────
    # Fans arrive mostly 90 min before kickoff; sigma ~45 min = 3 slots.
    # All inbound: BSL load increases steadily from 18:00 to kickoff.
    pre_peak_slot  = ko_slot - 6    # ~90 min = 6 slots before kickoff
    pre_sigma_slots = 3.0           # spread of ±45 min around pre-peak
    pre_game_wave = _normal_wave(
        peak_slot  = pre_peak_slot,
        sigma_slots = pre_sigma_slots,
        total_pax  = total_fans,
        start_slot = 0,
        end_slot   = ko_slot + 1,   # cut off at kickoff
    )

    # ── B: In-game low demand ─────────────────────────────────────────────────
    # Small constant: late arrivals, stadium workers, nearby residents.
    # Assumption: ~5% of total fans trickle in after kickoff.
    in_game_total = int(total_fans * 0.05)
    ingame_wave   = np.zeros(N_SLOTS)
    if ko_slot < match_end_slot:
        n_ingame = match_end_slot - ko_slot
        ingame_wave[ko_slot:match_end_slot] = in_game_total / max(n_ingame, 1)

    # ── C: Post-game evacuation wave ──────────────────────────────────────────
    if include_post_game:
        # Sharp spike then long tail: sigma ~1 hour = 4 slots
        post_game_wave = _normal_wave(
            peak_slot   = post_game_peak_slot,
            sigma_slots = 4.0,
            total_pax   = total_fans,
            start_slot  = match_end_slot,
            end_slot    = N_SLOTS,
        )
    else:
        post_game_wave = np.zeros(N_SLOTS)

    # ── Aggregate raw WC fan demand ───────────────────────────────────────────
    raw_worldcup = pre_game_wave + ingame_wave + post_game_wave

    # ── BSL demand split ──────────────────────────────────────────────────────
    # Inbound (toward NRG): ALL transit fans eventually use BSL southbound.
    # The pre-game wave drives BSL inbound load.
    bsl_inbound = pre_game_wave + ingame_wave   # fans heading TO NRG

    # Outbound (away from NRG): post-game evacuation, all fans heading home.
    bsl_outbound = post_game_wave                # fans leaving NRG

    # ── Regional Rail feeder demand ───────────────────────────────────────────
    # About RR_FEEDER_SHARE (40%) of all transit fans use RR as a feeder.
    # Inbound: RR trips concentrated ~90 min before kickoff (same wave, scaled).
    rr_total_inbound  = int(total_fans * RR_FEEDER_SHARE)
    rr_inbound_wave   = pre_game_wave * RR_FEEDER_SHARE

    # Outbound: same fans heading back after the game.
    # They get on BSL first (~20 min travel to Center City), then board RR.
    # Shift post-game wave by ~20 min = ~1.3 slots.
    bsl_to_cc_delay_slots = int(20 / SLOT_MINUTES)   # 1 slot ≈ 15 min
    rr_outbound_wave  = np.zeros(N_SLOTS)
    for t in range(N_SLOTS):
        src = t - bsl_to_cc_delay_slots
        if 0 <= src < N_SLOTS:
            rr_outbound_wave[t] = post_game_wave[src] * RR_FEEDER_SHARE

    rr_total_wave = rr_inbound_wave + rr_outbound_wave

    # Distribute RR demand across lines by ridership weight
    rr_demand: Dict[str, np.ndarray] = {
        line: rr_total_wave * share
        for line, share in _LINE_SHARES.items()
    }

    # ── Transfer demand at Center City stations ───────────────────────────────
    # Fans transferring from RR to BSL: concentrated at Suburban/Jefferson/30th St.
    transfer_demand = rr_inbound_wave.copy()

    return {
        "raw_worldcup":         raw_worldcup,
        "pre_game_wave":        pre_game_wave,
        "ingame_wave":          ingame_wave,
        "post_game_wave":       post_game_wave,
        "bsl_inbound":          bsl_inbound,
        "bsl_outbound":         bsl_outbound,
        "rr_inbound_total":     rr_inbound_wave,
        "rr_outbound_total":    rr_outbound_wave,
        "rr_demand":            rr_demand,
        "transfer_demand":      transfer_demand,
        "post_game_evacuation": post_game_wave,
        "kickoff_slot":         ko_slot,
        "match_end_slot":       min(match_end_slot, N_SLOTS - 1),
        "post_game_peak_slot":  min(post_game_peak_slot, N_SLOTS - 1),
        "total_inbound_fans":   float(pre_game_wave.sum() + ingame_wave.sum()),
        "total_outbound_fans":  float(post_game_wave.sum()),
    }


def get_demand(
    kickoff: str = DEFAULT_KICKOFF_TIME,
    total_fans: int = TOTAL_FANS_TRANSIT,
    include_post_game: bool = True,
    include_baseline: bool = True,
) -> Dict:
    """
    Consolidated demand dict for the full 18:00–04:00 window.

    Merges World Cup fan demand with evening baseline commuter demand.
    For each line, rr_demand[line] includes both commuter and WC feeder demand.

    This is the primary interface for reporting.py and run_scenarios.py.
    """
    wc = compute_worldcup_demand(kickoff=kickoff, total_fans=total_fans,
                                 include_post_game=include_post_game)

    if include_baseline:
        from septa_worldcup.v1.data.network import LINES
        for line in LINES:
            baseline = compute_baseline_demand(line)
            if line in wc["rr_demand"]:
                wc["rr_demand"][line] = wc["rr_demand"][line] + baseline

    return wc


def summary_stats(demand: Dict) -> None:
    """Print a quick sanity-check summary of demand components."""
    print(f"  Pre-game inbound total:     {demand['pre_game_wave'].sum():>8,.0f} pax")
    print(f"  In-game low-demand total:   {demand['ingame_wave'].sum():>8,.0f} pax")
    print(f"  Post-game evacuation total: {demand['post_game_evacuation'].sum():>8,.0f} pax")
    print(f"  BSL inbound total:          {demand['bsl_inbound'].sum():>8,.0f} pax")
    print(f"  BSL outbound total:         {demand['bsl_outbound'].sum():>8,.0f} pax")
    print(f"  RR feeder total (in+out):   "
          f"{(demand['rr_inbound_total'] + demand['rr_outbound_total']).sum():>8,.0f} pax")
    ko = demand['kickoff_slot']
    me = demand['match_end_slot']
    pg = demand['post_game_peak_slot']
    print(f"  Kickoff slot:     {ko:3d}  ({slot_label(ko)})")
    print(f"  Match-end slot:   {me:3d}  ({slot_label(me)})")
    print(f"  Post-game peak:   {pg:3d}  ({slot_label(pg)})")
