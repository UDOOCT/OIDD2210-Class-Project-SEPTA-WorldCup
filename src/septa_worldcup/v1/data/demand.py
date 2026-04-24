"""
src/septa_worldcup/v1/data/demand.py
--------------------------------------
Time-series demand model for the v1 Regional Rail profit model.
Active time window: 18:00 match day → 03:45+1 (40 slots, 15-minute resolution).

HISTORICAL NOTE: the original v1 demand used a 6am–9pm bimodal profile
  (morning peak ~8am, evening peak ~5:30pm, 61 slots).  The active baseline
  now uses the 18:00–04:00 event window, consistent with v2.

DEMAND COMPONENTS per (line, slot):
  d[l,t] = d_base[l,t] + d_pregame[l,t] + d_postgame[l,t]

  d_base     : Evening/late-night commuter baseline drawn from FY2024 ridership.
               18:00–22:00: ~10% of daily boardings per line, linearly declining.
               22:00–04:00: ~2% of daily boardings per line, linearly declining.
               Source: data/ridership/ridership_by_line.json

  d_pregame  : Pre-game fan arrivals via Regional Rail.
               Normal distribution centered ~90 min before kickoff.
               Fans split across lines by gateway station proximity.
               Source: WORLDCUP_GATEWAY_WEIGHTS, TOTAL_WC_TRANSIT_USERS

  d_postgame : Post-game fan departures via Regional Rail.
               Normal distribution centered 20 min after match end.
               Match end ≈ kickoff + 90 min game + 15 min stoppage buffer.
               Fans use same gateway weights as pre-game (reverse direction).

All demand arrays have shape (40,) — one entry per 15-minute slot.
All values are nonneg.
Post-midnight slots are indices 24–39 (00:00+1 through 03:45+1).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from septa_worldcup import DATA_DIR
from septa_worldcup.v1.data.network import LINES, WORLDCUP_GATEWAY_WEIGHTS, TOTAL_WC_TRANSIT_USERS
from septa_worldcup.v1.data.parameters import (
    N_SLOTS, SLOT_MINUTES, SLOT_DURATION,
    slot_minutes_from_1800, slot_clock_minutes,
    time_to_slot,
)

# ── Real total ridership from ridership_by_line.json ─────────────────────────
_rbl_path = DATA_DIR / "ridership" / "ridership_by_line.json"
if not _rbl_path.exists():
    raise FileNotFoundError(f"Required data file missing: {_rbl_path}")
_RBL = json.loads(_rbl_path.read_text())
_TOTAL_RIDERSHIP_REAL: int = sum(_RBL.values())   # 48,343 (FY2024 weekday)

# Evening window split (assumption documented here, not in scenario config)
_EVENING_SHARE_OF_DAILY   = 0.10   # 18:00–22:00 → ~10% of daily boardings
_LATENIGHT_SHARE_OF_DAILY = 0.02   # 22:00–04:00 → ~2% of daily boardings
_EVENING_END_SLOT = time_to_slot("22:00")   # slot 16

# Slot mid-times in minutes since 18:00 (used for Normal distribution lookups)
_SLOT_MID_MIN = np.array([slot_minutes_from_1800(t) + SLOT_MINUTES / 2
                           for t in range(N_SLOTS)])


# ── Per-station inbound boarding shares (FY2024 weekday) ─────────────────────
# Not used in line-level demand but available for future station-level models.
def _load_station_shares() -> dict[str, dict[str, float]]:
    csv_path = DATA_DIR / "ridership" / "septa_rr_ridership.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Required data file missing: {csv_path}")

    NAME_OVERRIDES = {"Manyunk/Norristown": "Manayunk/Norristown",
                      "Bala Cynwyd": "Cynwyd"}
    raw: dict[str, dict[str, float]] = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["Year"].strip() != "2024":
                continue
            if row["Service_Ty"].strip() != "Weekday":
                continue
            if row["Direction"].strip() != "Inbound":
                continue
            line    = NAME_OVERRIDES.get(row["Line"].strip(), row["Line"].strip())
            station = row["Station"].strip()
            bs      = row["Boards"].strip()
            if not bs:
                continue
            raw.setdefault(line, {})
            raw[line][station] = raw[line].get(station, 0.0) + float(bs)
    shares: dict[str, dict[str, float]] = {}
    for line, counts in raw.items():
        total = sum(counts.values())
        if total > 0:
            shares[line] = {st: round(v / total, 4) for st, v in counts.items()}
    return shares


STATION_RIDERSHIP_SHARES: dict[str, dict[str, float]] = _load_station_shares()


# ── Base evening commuter demand ──────────────────────────────────────────────

def compute_base_demand(total_ridership: float = _TOTAL_RIDERSHIP_REAL) -> dict:
    """
    Returns demand[line] = np.array of shape (40,) — evening/late-night
    commuter baseline for the 18:00–04:00 window.

    Method:
      Each line gets its share of ridership (from ridership_by_line.json).
      18:00–22:00 (slots 0–15): linearly declining profile, ~10% of daily.
      22:00–04:00 (slots 16–39): linearly declining profile, ~2% of daily.

    Assumption: evening window shares are documented above as class-level
    constants and are intentionally conservative (this is a non-peak night).
    """
    n_eve = _EVENING_END_SLOT              # 16 slots
    n_ln  = N_SLOTS - _EVENING_END_SLOT    # 24 slots

    eve_weights = np.linspace(1.0, 0.3, n_eve)
    eve_weights /= eve_weights.sum()
    ln_weights  = np.linspace(1.0, 0.1, n_ln)
    ln_weights  /= ln_weights.sum()

    demand = {}
    for line, data in LINES.items():
        line_riders = data["weekday_riders"]
        arr = np.zeros(N_SLOTS)
        arr[:n_eve] = line_riders * _EVENING_SHARE_OF_DAILY   * eve_weights
        arr[n_eve:] = line_riders * _LATENIGHT_SHARE_OF_DAILY * ln_weights
        demand[line] = arr
    return demand


# ── World Cup fan demand ──────────────────────────────────────────────────────

_DEFAULT_KICKOFF        = "20:30"
_MATCH_DURATION_MIN     = 90 + 15   # 90 min game + 15 min stoppage buffer
_PRE_PEAK_OFFSET_MIN    = -90       # pre-game peak 90 min before kickoff
_PRE_SIGMA_MIN          = 30        # std dev of fan arrival spread (minutes)
_POST_PEAK_OFFSET_MIN   = 20        # post-game peak 20 min after match end
_POST_SIGMA_MIN         = 25        # std dev of post-game departure spread


def compute_worldcup_demand(
    total_fans:         int   = TOTAL_WC_TRANSIT_USERS,
    kickoff:            str   = _DEFAULT_KICKOFF,
    pre_peak_offset_min: int  = _PRE_PEAK_OFFSET_MIN,
    pre_sigma_min:       float = _PRE_SIGMA_MIN,
    post_peak_offset_min: int = _POST_PEAK_OFFSET_MIN,
    post_sigma_min:      float = _POST_SIGMA_MIN,
    include_post_game:   bool  = True,
) -> dict:
    """
    Returns wc_demand[line] = np.array of shape (40,) — World Cup fan demand
    on Regional Rail during the 18:00–04:00 window.

    Pre-game: Normal(kickoff + pre_peak_offset_min, pre_sigma_min) in minutes-since-1800.
    Post-game: Normal(kickoff + match_duration + post_peak_offset_min, post_sigma_min).
    Fans split across lines by WORLDCUP_GATEWAY_WEIGHTS.
    """
    ko_min = slot_minutes_from_1800(time_to_slot(kickoff))

    # Pre-game distribution (minutes since 18:00)
    pre_mu  = ko_min + pre_peak_offset_min
    pre_pdf = norm.pdf(_SLOT_MID_MIN, pre_mu, pre_sigma_min)
    pre_total = pre_pdf.sum() * SLOT_MINUTES
    if pre_total > 0:
        pre_pdf = pre_pdf / pre_total * total_fans   # scale to fan count
    pre_pdf = np.maximum(pre_pdf * SLOT_MINUTES, 0.0)

    # Post-game distribution
    post_mu  = ko_min + _MATCH_DURATION_MIN + post_peak_offset_min
    post_pdf = norm.pdf(_SLOT_MID_MIN, post_mu, post_sigma_min)
    post_total = post_pdf.sum() * SLOT_MINUTES
    if post_total > 0:
        post_pdf = post_pdf / post_total * total_fans
    post_pdf = np.maximum(post_pdf * SLOT_MINUTES, 0.0)

    combined = pre_pdf + (post_pdf if include_post_game else 0.0)

    wc_demand = {}
    for line, data in LINES.items():
        line_wt = max(
            (WORLDCUP_GATEWAY_WEIGHTS.get(s, 0.01) for s in data["stations"]),
            default=0.01,
        )
        wc_demand[line] = combined * line_wt
    return wc_demand


def get_total_demand(worldcup: bool = True) -> dict:
    """
    Returns demand[line] = np.array of shape (40,) — passengers per 15-min slot,
    18:00–04:00 match-day window.
    """
    base = compute_base_demand()
    if not worldcup:
        return base
    wc = compute_worldcup_demand()
    return {line: base[line] + wc[line] for line in LINES}


def monte_carlo_demand(n_samples: int = 500, seed: int = 42) -> list:
    """
    Returns list of n_samples demand dicts, each with shape (40,) per line.
    Samples uncertain parameters: total_fans, base ridership, arrival timing.
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(n_samples):
        fans         = int(np.clip(rng.normal(45_000, 5_000), 30_000, 65_000))
        base_total   = float(np.clip(rng.normal(_TOTAL_RIDERSHIP_REAL, 4_000), 36_000, 62_000))
        pre_offset   = int(rng.normal(_PRE_PEAK_OFFSET_MIN,  8))
        pre_sig      = float(abs(rng.normal(_PRE_SIGMA_MIN,  5)))
        post_offset  = int(rng.normal(_POST_PEAK_OFFSET_MIN, 5))
        post_sig     = float(abs(rng.normal(_POST_SIGMA_MIN, 5)))

        base = compute_base_demand(total_ridership=base_total)
        wc   = compute_worldcup_demand(
            total_fans=fans,
            pre_peak_offset_min=pre_offset,
            pre_sigma_min=pre_sig,
            post_peak_offset_min=post_offset,
            post_sigma_min=post_sig,
        )
        scenarios.append({line: base[line] + wc[line] for line in LINES})
    return scenarios
