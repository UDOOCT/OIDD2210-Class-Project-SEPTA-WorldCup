# Running the SEPTA World Cup 2026 Model

## Prerequisites

```bash
pip install -r requirements.txt
```

## Quick start (recommended)

```bash
# v1 — greedy optimization (~4 seconds)
python _run_optimization.py

# v1 — verify greedy is globally optimal via ILP
python _run_ilp_comparison.py

# v2 — eight-scenario comparison (~5 seconds)
python run_scenarios.py
python run_scenarios.py --save-csv   # also writes outputs/tables/scenario_comparison.csv
```

Equivalent calls via `scripts/`:

```bash
python scripts/run_v1_greedy.py
python scripts/run_v1_ilp_comparison.py
python scripts/run_v2_scenarios.py
```

## Validation smoke test

```bash
python scripts/validate_project.py
```

Imports every package module and verifies data files load. No results generated.

## Full bilevel pipeline (slow — ~2 hours)

```bash
python main.py --mode upper_only    # global SLSQP, ~4–15 min
python main.py                      # full bilevel, superset of upper_only
python main.py --mode sensitivity   # Optuna sweep, ~hours
```

See `outputs/validation/validation_summary.txt` §5 and §10 for known runtime limits.

## Time window

Both v1 and v2 now use the same window:
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
  tables/         CSV scenario comparison tables
  validation/     Validation audit reports
  figures/        (empty — figures generated on demand)
```
