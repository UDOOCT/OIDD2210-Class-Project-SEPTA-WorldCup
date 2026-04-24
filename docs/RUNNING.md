# Running the SEPTA World Cup 2026 Model

## Prerequisites

```bash
pip install -r requirements.txt
```

## Generate all final results (recommended)

```bash
# Single command: v1 greedy + ILP + all 8 v2 scenarios + all figures + CSVs:
python scripts/generate_all_results.py
```

## Individual commands

```bash
# v1 — greedy optimization (~15s including Phase-1 tables):
python _run_optimization.py

# v1 — verify greedy is globally optimal via ILP (~15s):
python _run_ilp_comparison.py

# v2 — eight-scenario comparison (~5 seconds):
python run_scenarios.py
python run_scenarios.py --save-csv      # also writes outputs/tables/
python run_scenarios.py --scenario 2 --verbose  # single scenario + full KPIs

# Smoke test (no results generated):
python scripts/validate_project.py
```

## Generated outputs

```
outputs/
├── tables/
│   ├── v1_greedy_summary.csv
│   ├── v1_ilp_comparison.csv
│   ├── v2_scenario_comparison.csv
│   └── v1_vs_v2_summary.csv
├── figures/                         ← 8 PNG figures
├── raw/
│   ├── v2_s2_bsl_per_slot.csv      ← per-slot BSL data (S2)
│   └── v2_s2_rr_per_line.csv       ← per-line RR equity (S2)
└── validation/
    └── final_validation_summary.txt
```

## Full bilevel pipeline (slow — ~2 hours)

```bash
python main.py --mode upper_only    # global SLSQP, ~4–15 min
python main.py                      # full bilevel, superset of upper_only
python main.py --mode sensitivity   # Optuna sweep, ~hours
```

See `outputs/validation/final_validation_summary.txt` §9 for known runtime limits.

## Time window

Both v1 and v2 use the same window:
- **18:00 match day → 03:45+1 next day**
- 15-minute slots, 40 slots total
- Post-midnight slots labeled with `(+1)`

**Historical note:** the original v1 model used 6am–9pm (61 slots, decimal-hour
slot indexing). That is no longer the active baseline.

## Repository structure

```
src/septa_worldcup/
  v1/             Regional Rail profit model (18:00–04:00, 40 slots)
  v2/             Multimodal policy model  (18:00–04:00, 40 slots)
  common/         Shared utilities (plotting, network graph)
data/             Raw data assets (GTFS, ridership CSVs, cost JSON)
docs/             This documentation
outputs/
  tables/         CSV result tables
  figures/        PNG charts
  raw/            Per-slot / per-line raw data CSVs
  validation/     Audit and validation reports
```
