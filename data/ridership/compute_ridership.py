"""
Compute total weekday boardings per Regional Rail line from the SEPTA
ridership CSV and save data/ridership/ridership_by_line.json.
"""

import csv
import json
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path

DATA_DIR = Path(__file__).parent

TARGET_LINES = [
    "Airport",
    "Chestnut Hill East",
    "Chestnut Hill West",
    "Cynwyd",
    "Fox Chase",
    "Lansdale/Doylestown",
    "Manayunk/Norristown",
    "Media/Wawa",
    "Paoli/Thorndale",
    "Trenton",
    "Warminster",
    "West Trenton",
    "Wilmington/Newark",
]

# Explicit overrides for known bad names in the source data
NAME_OVERRIDES = {
    "Bala Cynwyd":        "Cynwyd",           # historical name
    "Manyunk/Norristown": "Manayunk/Norristown",  # typo in source
}

SKIP_LINES = {"Trunk"}   # shared-trunk rows, not a distinct line

LATEST_YEAR = "2024"

# ── 1. Aggregate weekday boardings by (year, raw_line) ─────────────────────
raw_totals: dict[tuple[str, str], int] = defaultdict(int)
years_seen: set[str] = set()

with open(DATA_DIR / "septa_rr_ridership.csv", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row["Service_Ty"].strip() != "Weekday":
            continue
        year = row["Year"].strip()
        years_seen.add(year)
        if year != LATEST_YEAR:
            continue
        raw_line = row["Line"].strip()
        boards = row["Boards"].strip()
        if boards:
            raw_totals[(year, raw_line)] += int(float(boards))

print(f"Years in dataset: {sorted(years_seen)}  →  using {LATEST_YEAR}")
print(f"Raw lines in {LATEST_YEAR}: {sorted(set(k[1] for k in raw_totals))}")

# ── 2. Map raw line names to canonical target keys ─────────────────────────
line_boardings: dict[str, int] = defaultdict(int)

for (year, raw_line), boards in raw_totals.items():
    if raw_line in SKIP_LINES:
        print(f"  SKIP: '{raw_line}' (not a distinct line)")
        continue

    # Try explicit override first
    if raw_line in NAME_OVERRIDES:
        canonical = NAME_OVERRIDES[raw_line]
        print(f"  NAME OVERRIDE: '{raw_line}' -> '{canonical}'")
        line_boardings[canonical] += boards
        continue

    # Exact match
    if raw_line in TARGET_LINES:
        line_boardings[raw_line] += boards
        continue

    # Fuzzy match
    matches = get_close_matches(raw_line, TARGET_LINES, n=1, cutoff=0.6)
    if matches:
        canonical = matches[0]
        print(f"  FUZZY MATCH: '{raw_line}' -> '{canonical}'")
        line_boardings[canonical] += boards
    else:
        print(f"  WARNING: no match for '{raw_line}' — skipped")

# ── 3. Build output, preserving TARGET_LINES order ──────────────────────────
ridership_by_line = {line: line_boardings.get(line, 0) for line in TARGET_LINES}

# ── 4. Report missing lines ──────────────────────────────────────────────────
for line in TARGET_LINES:
    if ridership_by_line[line] == 0:
        print(f"  MISSING DATA: '{line}' — no weekday boardings found for {LATEST_YEAR}")

# ── 5. Write JSON ────────────────────────────────────────────────────────────
out_path = DATA_DIR / "ridership_by_line.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(ridership_by_line, f, indent=2)

print(f"\nWrote {out_path}")
print(f"Total weekday boardings across all lines: {sum(ridership_by_line.values()):,}")

# ── 6. Print first 3 entries ─────────────────────────────────────────────────
print("\n── First 3 entries of ridership_by_line.json ──")
for i, (line, boards) in enumerate(ridership_by_line.items()):
    if i >= 3:
        break
    print(f'  "{line}": {boards:,}')
