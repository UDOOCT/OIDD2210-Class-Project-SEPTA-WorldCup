"""
data/demand.py
--------------
Time-series demand model at 15-minute resolution (6am–9pm, 61 slots).

CSV FIELDS USED (data/ridership/septa_rr_ridership.csv, FY2024 weekday):
  Used:        Year, Line, Station, Service_Ty, Direction, Boards
  Unavailable: per-slot (time-of-day) boarding counts — CSV only has daily totals.
               Because no intraday time distribution is in the CSV, the bimodal
               normal profile (_PROFILE_PARAMS) is retained for time-shaping.
               Per-station inbound boarding shares are computed and stored in
               STATION_RIDERSHIP_SHARES for future station-level models.

ALGORITHM: Two-component demand curve per (line, slot)
  d[l, t] = d_base[l, t] + d_worldcup[l, t]

  d_base:     intraday profile fitted to SEPTA APC ridership patterns.
              Shape: bimodal — morning peak ~8am, evening peak ~5:30pm.
              Area under curve = FY2024 weekday ridership per line.

  d_worldcup: World Cup fan demand. Pre-game wave (Normal centered ~7:30pm)
              + post-game wave (outside our window: game starts 9pm).
              Only pre-game travel falls in 6am–9pm window.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from data.network import LINES, WORLDCUP_GATEWAY_WEIGHTS, TOTAL_WC_TRANSIT_USERS
from data.parameters import (
    TIME_SLOTS, N_SLOTS, SLOT_DURATION,
    IDX_6AM, IDX_9AM, IDX_4PM, IDX_7PM, IDX_9PM,
)

_BASE = Path(__file__).parent

# ── Real total ridership from ridership_by_line.json ─────────────────────────
_rbl_path = _BASE / "ridership" / "ridership_by_line.json"
if not _rbl_path.exists():
    raise FileNotFoundError(f"Required data file missing: {_rbl_path}")
_RBL = json.loads(_rbl_path.read_text())
_TOTAL_RIDERSHIP_REAL: int = sum(_RBL.values())   # 48,343 (FY2024 weekday)


# ── Per-station inbound boarding shares (FY2024 weekday) ─────────────────────
# Source: septa_rr_ridership.csv — per-station Boards counts, Inbound direction.
# Format: {line_name: {station_name: share_of_line_inbound_total}}
# Available for station-level models; not used in line-level demand functions below.
def _load_station_shares() -> dict[str, dict[str, float]]:
    csv_path = _BASE / "ridership" / "septa_rr_ridership.csv"
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
            line = NAME_OVERRIDES.get(row["Line"].strip(), row["Line"].strip())
            station = row["Station"].strip()
            boards_str = row["Boards"].strip()
            if not boards_str:
                continue
            raw.setdefault(line, {})
            raw[line][station] = raw[line].get(station, 0.0) + float(boards_str)

    shares: dict[str, dict[str, float]] = {}
    for line, counts in raw.items():
        total = sum(counts.values())
        if total > 0:
            shares[line] = {st: round(v / total, 4) for st, v in counts.items()}
    return shares


STATION_RIDERSHIP_SHARES: dict[str, dict[str, float]] = _load_station_shares()


# ── Intraday demand profile ───────────────────────────────────────────────────
# Bimodal normal mixture approximating a typical SEPTA weekday.
# No time-of-day data in CSV; profile shape is retained from original.

_PROFILE_PARAMS = [
    # (mean_hr, sigma_hr, weight)
    (8.0,  0.75, 0.40),   # morning rush peak at 8:00am
    (17.5, 0.80, 0.40),   # evening rush peak at 5:30pm
    (12.5, 1.50, 0.20),   # midday shoulder
]


def _profile_weight(t: float) -> float:
    """Unnormalized intraday weight at decimal hour t."""
    return sum(
        w * norm.pdf(t, mu, sig)
        for mu, sig, w in _PROFILE_PARAMS
    )


_raw_weights = np.array([_profile_weight(t) for t in TIME_SLOTS])
_PROFILE = _raw_weights / _raw_weights.sum()


def compute_base_demand(total_ridership: float = _TOTAL_RIDERSHIP_REAL) -> dict:
    """
    Returns demand[line][slot_index] = passengers in that 15-min slot.

    Method:
      1. Each line gets share = line_ridership / total_ridership
      2. Each slot gets share from _PROFILE (bimodal normal mixture)
      3. demand[l][i] = total × line_share × profile[i]

    Default total_ridership = {real} (FY2024 weekday sum from ridership_by_line.json).
    """.format(real=_TOTAL_RIDERSHIP_REAL)
    demand = {}
    for line, data in LINES.items():
        line_share = data["weekday_riders"] / total_ridership
        demand[line] = np.array([
            total_ridership * line_share * _PROFILE[i]
            for i in range(N_SLOTS)
        ])
    return demand


def compute_worldcup_demand(
    total_fans:       int   = TOTAL_WC_TRANSIT_USERS,
    kickoff_hr:       float = 21.0,
    pre_peak_offset:  float = -1.75,
    pre_sigma:        float = 0.75,
    pre_share:        float = 1.0,
) -> dict:
    """
    Returns wc_demand[line][slot_index] = additional World Cup passengers.

    Distribution: Normal(kickoff + pre_peak_offset, pre_sigma)
                = Normal(19.25h, 0.75) → peaks around 7:15pm
    """
    mu  = kickoff_hr + pre_peak_offset
    sig = pre_sigma

    pdf_vals = np.array([norm.pdf(t, mu, sig) for t in TIME_SLOTS])
    total_pdf = pdf_vals.sum() * SLOT_DURATION
    if total_pdf > 0:
        pdf_vals = pdf_vals / total_pdf * total_fans

    wc_demand = {}
    for line, data in LINES.items():
        line_wt = max(
            (WORLDCUP_GATEWAY_WEIGHTS.get(s, 0.01) for s in data["stations"]),
            default=0.01,
        )
        wc_demand[line] = pdf_vals * line_wt * SLOT_DURATION

    return wc_demand


def get_total_demand(worldcup: bool = True) -> dict:
    """
    Returns demand[line] = np.array of shape (61,) — passengers per 15-min slot.
    """
    base = compute_base_demand()
    if not worldcup:
        return base
    wc = compute_worldcup_demand()
    return {line: base[line] + wc[line] for line in LINES}


def monte_carlo_demand(n_samples: int = 500, seed: int = 42) -> list:
    """
    n_samples demand arrays, each shape (13 lines, 61 slots).
    Samples uncertain parameters: total_fans, base ridership, arrival timing.
    Returns list of demand dicts (same structure as get_total_demand).
    """
    rng = np.random.default_rng(seed)
    scenarios = []
    for _ in range(n_samples):
        fans       = int(np.clip(rng.normal(45_000, 5_000), 30_000, 65_000))
        base_total = np.clip(rng.normal(_TOTAL_RIDERSHIP_REAL, 4_000), 36_000, 62_000)
        pre_offset = rng.normal(-1.75, 0.20)
        pre_sig    = abs(rng.normal(0.75, 0.10))

        base = compute_base_demand(total_ridership=base_total)
        wc   = compute_worldcup_demand(
            total_fans=fans, pre_peak_offset=pre_offset, pre_sigma=pre_sig,
        )
        scenarios.append({line: base[line] + wc[line] for line in LINES})
    return scenarios
