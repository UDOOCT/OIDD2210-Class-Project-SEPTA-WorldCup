"""
Derive Regional Rail cost summary from SEPTA operating stats CSV.
Uses FY2024 budgeted figures, GTFS trip counts, and ridership data.
Saves data/costs/cost_summary.json.
"""

import csv
import json
from pathlib import Path

BASE = Path(__file__).parent.parent   # project data/ root
COSTS_DIR = BASE / "costs"
GTFS_DIR  = BASE / "gtfs"
RIDERSHIP_DIR = BASE / "ridership"

FISCAL_YEAR = "2024"

# ── 1. Sum Regional Rail operating costs for FY2024 ─────────────────────────
# Column is in $000s. Exclude Depreciation/Contributed Capital (non-cash).
NON_CASH = {"Depreciation/Contributed Capital"}

total_rr_thousands = 0.0
row_detail: list[dict] = []

with open(COSTS_DIR / "septa_route_operating_stats.csv", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row["Fiscal_Year"].strip() != FISCAL_YEAR:
            continue
        division = row["Division"].strip()
        dept = row["Department"].strip()
        label = dept if dept else division
        val_str = row["Regional_Rail__in__000s_"].strip()
        if not val_str:
            continue
        val = float(val_str)
        is_noncash = any(nc.lower() in label.lower() for nc in NON_CASH)
        row_detail.append({"label": label, "value_000s": val, "non_cash": is_noncash})
        if not is_noncash:
            total_rr_thousands += val

total_rr_dollars = total_rr_thousands * 1000
print(f"FY{FISCAL_YEAR} RR operating cost (cash): ${total_rr_dollars:,.0f}")
print(f"  ({total_rr_thousands:,.0f} $000s across {len(row_detail)} line items)")

# ── 2. Count annual weekday train trips from GTFS ────────────────────────────
# strategy: count trips per service_id, then multiply by weekday service days
# from calendar.txt.
import datetime

def gtfs_date(s: str) -> datetime.date:
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:]))

# Count weekday service days per service_id
weekday_days: dict[str, int] = {}
with open(GTFS_DIR / "calendar.txt", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row["monday"].strip() == "1":   # Mon-Fri service
            start = gtfs_date(row["start_date"])
            end   = gtfs_date(row["end_date"])
            days  = (end - start).days + 1
            weekdays = sum(
                1 for d in range(days)
                if (start + datetime.timedelta(d)).weekday() < 5
            )
            weekday_days[row["service_id"].strip()] = weekdays

# Count distinct trips per service_id (rail trips only)
# Reuse the route→type mapping: all routes in google_rail.zip are route_type 2
rail_trip_counts: dict[str, int] = {}
with open(GTFS_DIR / "trips.txt", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        sid = row["service_id"].strip()
        rail_trip_counts[sid] = rail_trip_counts.get(sid, 0) + 1

annual_trips = sum(
    cnt * weekday_days.get(sid, 0)
    for sid, cnt in rail_trip_counts.items()
)
unique_daily_trips = sum(rail_trip_counts.values())
print(f"GTFS daily rail trips (all service IDs): {unique_daily_trips}")
print(f"Estimated annual weekday train-trips: {annual_trips:,}")

# ── 3. Load total weekday ridership ──────────────────────────────────────────
ridership_path = RIDERSHIP_DIR / "ridership_by_line.json"
if ridership_path.exists():
    ridership_by_line = json.loads(ridership_path.read_text())
    total_annual_boardings = sum(ridership_by_line.values())
    # Ridership CSV is a single-day snapshot; scale to annual (260 weekdays)
    annual_boardings = total_annual_boardings * 260
    print(f"Daily weekday boardings (all lines): {total_annual_boardings:,}")
    print(f"Estimated annual weekday boardings: {annual_boardings:,}")
else:
    print("WARNING: ridership_by_line.json not found — skipping per-pax calc")
    annual_boardings = None

# ── 4. Derive cost metrics ────────────────────────────────────────────────────
fixed_cost_per_trip = round(total_rr_dollars / annual_trips, 2) if annual_trips else None

if annual_boardings:
    variable_cost_per_pax = round(total_rr_dollars / annual_boardings, 2)
else:
    variable_cost_per_pax = None

# ── 5. Build and save summary ─────────────────────────────────────────────────
summary = {
    "fiscal_year": int(FISCAL_YEAR),
    "total_rr_operating_cost_usd": round(total_rr_dollars),
    "fixed_cost_per_train_trip": fixed_cost_per_trip,
    "variable_cost_per_pax": variable_cost_per_pax,
    "annual_train_trips_estimated": annual_trips,
    "annual_weekday_boardings_estimated": annual_boardings,
    "notes": (
        "All-in cash operating cost (excludes depreciation). "
        "Per-trip derived from GTFS weekday trips × calendar service days. "
        "Per-pax derived from 2024 ridership snapshot × 260 weekdays."
    ),
    "source": (
        "SEPTA Route Operating Statistics FY2024 Budgeted — "
        "https://opendata.arcgis.com/datasets/26f40aaeb8ae41d291878ba726b57fed_0.csv"
    ),
}

out_path = COSTS_DIR / "cost_summary.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nWrote {out_path}")

# ── 6. Print first 3 entries ──────────────────────────────────────────────────
print("\n── First 3 entries of cost_summary.json ──")
items = list(summary.items())
for k, v in items[:3]:
    print(f'  "{k}": {json.dumps(v)}')
