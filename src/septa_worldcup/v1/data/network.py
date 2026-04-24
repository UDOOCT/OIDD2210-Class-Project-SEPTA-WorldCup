"""
data/network.py
---------------
All 13 SEPTA Regional Rail lines with stations and travel times.

DATA SOURCES:
  - Station lists + order: SEPTA GTFS v202603296 stops.txt / trips.txt /
                           stop_times.txt (github.com/septadev/GTFS)
  - Travel times:          Computed from GTFS stop_times.txt →
                           data/gtfs/travel_times.json
  - Ridership:             SEPTA OpenDataPhilly APC FY2024 weekday →
                           data/ridership/ridership_by_line.json
  - Zone fares:            SEPTA official fare table — no fare data in GTFS,
                           values preserved from original estimates.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from septa_worldcup import DATA_DIR

_BASE = DATA_DIR

# ── Load real data files ──────────────────────────────────────────────────────
_tt_path  = _BASE / "gtfs" / "travel_times.json"
_rbl_path = _BASE / "ridership" / "ridership_by_line.json"
for _p in (_tt_path, _rbl_path):
    if not _p.exists():
        raise FileNotFoundError(f"Required data file missing: {_p}")

_TT  = json.loads(_tt_path.read_text())   # {line: {stop_a: {stop_b: minutes}}}
_RBL = json.loads(_rbl_path.read_text())  # {line: weekday_boardings}


# ── Name normalization (mirrors data/gtfs/compute_travel_times.py) ────────────
def _norm(raw: str) -> str:
    """Normalize a raw GTFS stop name to match keys in travel_times.json."""
    name = raw.strip().title()
    # Fix ordinals mangled by str.title(): "30Th" → "30th"
    name = re.sub(r"(\d)(St|Nd|Rd|Th)\b", lambda m: m.group(1) + m.group(2).lower(), name)
    for pat in (r"\s+Transit Center$", r"\s+Station$", r"\s+TC$"):
        name = re.sub(pat, "", name, flags=re.IGNORECASE).strip()
    return name


_DISPLAY_MAP = {
    "Gray 30th St":  "30th Street",
    "Penn Medicine": "University City",
    "Jefferson":     "Jefferson Station",
    "Suburban":      "Suburban Station",
    "NRG":           "NRG Station",
}


def _display(gtfs_name: str) -> str:
    """Convert normalized GTFS name to human-readable display name."""
    if gtfs_name in _DISPLAY_MAP:
        return _DISPLAY_MAP[gtfs_name]
    if gtfs_name.endswith(" TC"):
        return gtfs_name[:-3].strip()
    return gtfs_name


# Names that indicate a Center City terminal — if dir=0 pattern STARTS here,
# the trip is outbound; reverse it to get the inbound (outer→CC) sequence.
_CC_GTFS_STARTS = frozenset({
    "Suburban", "Jefferson", "Temple University",
    "Gray 30th St", "Penn Medicine",
    "North Broad", "Wayne Junction",
})


# ── Load GTFS and build per-line trip patterns ────────────────────────────────
def _load_patterns() -> dict[str, Counter]:
    """Return {line_name: Counter of gtfs-name tuples for direction_id=0 trips}."""
    gtfs = _BASE / "gtfs"

    rail_routes: dict[str, str] = {}
    with open(gtfs / "routes.txt", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["route_type"].strip() == "2":
                name = re.sub(r"\s+Line$", "", r["route_long_name"].strip(),
                              flags=re.IGNORECASE)
                rail_routes[r["route_id"].strip()] = name

    stop_names: dict[str, str] = {}
    with open(gtfs / "stops.txt", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stop_names[r["stop_id"].strip()] = _norm(r["stop_name"])

    trip_meta: dict[str, tuple[str, int]] = {}  # trip_id → (route_id, dir_id)
    with open(gtfs / "trips.txt", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rid = r["route_id"].strip()
            if rid in rail_routes:
                trip_meta[r["trip_id"].strip()] = (rid, int(r["direction_id"].strip()))

    trip_seqs: dict[str, list] = defaultdict(list)
    with open(gtfs / "stop_times.txt", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            tid = r["trip_id"].strip()
            if tid in trip_meta:
                trip_seqs[tid].append((int(r["stop_sequence"]), r["stop_id"].strip()))
    for tid in trip_seqs:
        trip_seqs[tid].sort()

    patterns: dict[str, Counter] = defaultdict(Counter)
    for tid, (rid, did) in trip_meta.items():
        if did == 0:
            line = rail_routes[rid]
            seq = tuple(stop_names.get(s, s) for _, s in trip_seqs[tid])
            patterns[line][seq] += 1

    return dict(patterns)


_PATTERNS = _load_patterns()


# ── Fares — no fare data in GTFS; keep original hardcoded values ──────────────
_FARES: dict[str, float] = {
    "Airport":             5.25,
    "Chestnut Hill East":  4.25,
    "Chestnut Hill West":  4.25,
    "Cynwyd":              4.25,
    "Fox Chase":           5.25,
    "Lansdale/Doylestown": 5.25,
    "Manayunk/Norristown": 4.25,   # Zone 2-3 estimate; no official GTFS fare
    "Media/Wawa":          5.25,
    "Paoli/Thorndale":     5.50,
    "Trenton":             6.00,
    "Warminster":          5.25,
    "West Trenton":        5.75,
    "Wilmington/Newark":   6.00,
}

# Canonical line order (original 12 + Manayunk/Norristown as 7th entry)
_LINE_ORDER = list(_FARES)


def _build_line(line: str) -> dict:
    ctr = _PATTERNS.get(line)
    if not ctr:
        raise ValueError(f"No direction_id=0 GTFS trips found for line: {line!r}")

    # Most common pattern; break ties by preferring the longest sequence
    canon = max(ctr, key=lambda p: (ctr[p], len(p)))

    # Ensure inbound order: outer terminal → Center City.
    # If the first stop is a Center City station the trip is outbound → reverse.
    if canon and canon[0] in _CC_GTFS_STARTS:
        canon = canon[::-1]

    display_stations = [_display(g) for g in canon]

    # Travel times for each consecutive inbound pair (A → B)
    tt_line = _TT.get(line, {})
    travel_times: list[float] = []
    for ga, gb in zip(canon, canon[1:]):
        t = tt_line.get(ga, {}).get(gb)
        if t is None:
            print(f"  [network.py] WARNING: no travel time for "
                  f"{line} / {ga!r} → {gb!r}; using fallback 3.0 min")
            t = 3.0
        travel_times.append(round(float(t), 2))

    # Ridership — case-insensitive lookup
    riders = _RBL.get(line)
    if riders is None:
        low = line.lower()
        riders = next((v for k, v in _RBL.items() if k.lower() == low), 0)

    return {
        "stations":       display_stations,
        "travel_times":   travel_times,
        "weekday_riders": int(riders),
        "avg_fare":       _FARES[line],
    }


LINES: dict[str, dict] = {name: _build_line(name) for name in _LINE_ORDER}


# ── Constants (unchanged from original) ──────────────────────────────────────
CENTER_CITY_STATIONS = {
    "Jefferson Station", "Suburban Station", "30th Street", "University City",
}


def get_transfer_nodes() -> dict:
    """Return {station: [line, ...]} for stations served by 2+ lines."""
    station_lines: dict = defaultdict(list)
    for line, data in LINES.items():
        for s in data["stations"]:
            station_lines[s].append(line)
    return {s: lines for s, lines in station_lines.items() if len(lines) > 1}


WORLDCUP_GATEWAY_WEIGHTS = {
    "Jefferson Station":  0.25,
    "Suburban Station":   0.20,
    "30th Street":        0.15,
    "Temple University":  0.10,
    "North Philadelphia": 0.08,
    "University City":    0.08,
    "Trenton":            0.04,
    "Wilmington":         0.05,
}

TOTAL_WC_TRANSIT_USERS = 45_000


# ── Validation summary (run with: python3 data/network.py) ───────────────────
if __name__ == "__main__":
    all_stations: set[str] = set()
    for _d in LINES.values():
        all_stations.update(_d["stations"])
    _most_st = max(LINES, key=lambda l: len(LINES[l]["stations"]))
    _most_rx = max(LINES, key=lambda l: LINES[l]["weekday_riders"])
    print(f"Lines:             {len(LINES)}")
    print(f"Unique stations:   {len(all_stations)}")
    print(f"Most stations:     {_most_st} ({len(LINES[_most_st]['stations'])} stops)")
    print(f"Highest ridership: {_most_rx} ({LINES[_most_rx]['weekday_riders']:,}/weekday)")
    print()
    for name, data in LINES.items():
        print(f"  {name}: {data['stations']}")
        print(f"    times: {data['travel_times']}")
        print(f"    riders={data['weekday_riders']:,}  fare=${data['avg_fare']}")
