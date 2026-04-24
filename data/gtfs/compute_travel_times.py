"""
Parse SEPTA GTFS files and compute average travel times between consecutive
Regional Rail stops (route_type == 2). Saves data/gtfs/travel_times.json.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent


def parse_time_seconds(t: str) -> int:
    """HH:MM:SS -> total seconds. Handles times > 24:00 for overnight trips."""
    h, m, s = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def normalize_stop_name(raw: str) -> str:
    name = raw.strip().title()
    # Fix ordinal suffixes mangled by str.title(): "30Th" -> "30th"
    name = re.sub(r"(\d)(St|Nd|Rd|Th)\b", lambda m: m.group(1) + m.group(2).lower(), name)
    # Remove trailing suffixes (order matters — longest first)
    suffixes = [
        r"\s+Transit Center$",
        r"\s+Station$",
        r"\s+TC$",
        r"\s+Tc$",
    ]
    for pat in suffixes:
        name = re.sub(pat, "", name, flags=re.IGNORECASE).strip()
    return name


def normalize_line_name(route_long_name: str) -> str:
    """'Paoli/Thorndale Line' -> 'Paoli/Thorndale'"""
    return re.sub(r"\s+Line$", "", route_long_name.strip(), flags=re.IGNORECASE)


# ── 1. Load commuter-rail routes (route_type == 2) ─────────────────────────
rail_routes: dict[str, str] = {}  # route_id -> line_name
with open(DATA_DIR / "routes.txt", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row["route_type"].strip() == "2":
            rail_routes[row["route_id"].strip()] = normalize_line_name(
                row["route_long_name"]
            )

print(f"Commuter rail routes found: {sorted(rail_routes.values())}")

# ── 2. Map trip_id -> line_name ─────────────────────────────────────────────
trip_to_line: dict[str, str] = {}
with open(DATA_DIR / "trips.txt", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        route_id = row["route_id"].strip()
        if route_id in rail_routes:
            trip_to_line[row["trip_id"].strip()] = rail_routes[route_id]

print(f"Rail trips indexed: {len(trip_to_line)}")

# ── 3. Load stop names ───────────────────────────────────────────────────────
stop_names: dict[str, str] = {}
with open(DATA_DIR / "stops.txt", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        stop_names[row["stop_id"].strip()] = normalize_stop_name(row["stop_name"])

# ── 4. Parse stop_times and accumulate leg durations ────────────────────────
# Structure: line -> (stop_a, stop_b) -> [elapsed_minutes, ...]
# We process trip by trip, so we need to group rows by trip first.
# stop_times.txt is sorted by trip_id then stop_sequence in SEPTA's GTFS.

leg_times: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(
    lambda: defaultdict(list)
)

current_trip: str | None = None
current_line: str | None = None
prev_stop: str | None = None
prev_sec: int | None = None

total_rows = 0
with open(DATA_DIR / "stop_times.txt", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_rows += 1
        trip_id = row["trip_id"].strip()
        line = trip_to_line.get(trip_id)
        if line is None:
            continue

        stop_id = row["stop_id"].strip()
        stop_name = stop_names.get(stop_id, stop_id)

        try:
            arr_sec = parse_time_seconds(row["arrival_time"])
        except ValueError:
            continue

        if trip_id != current_trip:
            # New trip — reset state
            current_trip = trip_id
            current_line = line
            prev_stop = stop_name
            prev_sec = arr_sec
            continue

        if current_line == line and prev_stop is not None and prev_sec is not None:
            elapsed = (arr_sec - prev_sec) / 60.0
            if 0 < elapsed < 120:  # sanity: ignore gaps > 2 hours
                leg_times[line][(prev_stop, stop_name)].append(elapsed)

        prev_stop = stop_name
        prev_sec = arr_sec

print(f"stop_times rows read: {total_rows}")

# ── 5. Average each leg and build output dict ───────────────────────────────
travel_times: dict[str, dict[str, dict[str, float]]] = {}

for line, legs in sorted(leg_times.items()):
    travel_times[line] = {}
    for (stop_a, stop_b), times in legs.items():
        avg = round(sum(times) / len(times), 2)
        travel_times[line].setdefault(stop_a, {})[stop_b] = avg

# ── 6. Write JSON ─────────────────────────────────────────────────────────────
out_path = DATA_DIR / "travel_times.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(travel_times, f, indent=2)

print(f"\nWrote {out_path}")

# ── 7. Print first 10 entries for verification ───────────────────────────────
print("\n── First 10 entries of travel_times.json ──")
count = 0
for line, stops in travel_times.items():
    for origin, dests in stops.items():
        for dest, minutes in dests.items():
            print(f'  "{line}" / "{origin}" -> "{dest}": {minutes} min')
            count += 1
            if count >= 10:
                sys.exit(0)
