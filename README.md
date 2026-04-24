# SEPTA Regional Rail & Broad Street Line — World Cup 2026 Optimization
### OIDD 2210 Final Project

---

## Project Overview

This project models SEPTA's multimodal transit challenge for World Cup 2026 match day (Brazil vs. Haiti, June 19 2026, Lincoln Financial Field). It combines two layers:

**Layer 1 — Regional Rail (v1 baseline):**
Bilevel network optimization over 13 lines × 40 slots (18:00–04:00+1, 15-min). SEPTA maximizes profit by choosing integer train frequencies and fares; passengers respond via Multinomial Logit with World Cup parking penalty. Solved via SLSQP continuous relaxation with iterative best-response.

*Historical note: the original v1 used 6am–9pm (61 slots). The active baseline now aligns with v2 at 18:00–04:00+1 for time-window consistency.*

**Layer 2 — Multimodal extension (new):**
Full 18:00–04:00 match-day window (40 slots × 15 min, crossing midnight). Models Regional Rail as a feeder system and the Broad Street Line (BSL/B Line) as the primary link to NRG Station. Policy objective minimizes operating deficit plus social-cost penalties for unmet demand, crowding, equity failures, and clearance delay. Eight scenarios stress-test the system.

---

## Repo Structure

```
septa_worldcup/
├── src/septa_worldcup/
│   ├── v1/                     ← Regional Rail profit model (18:00–04:00, 40 slots)
│   │   ├── data/network.py     ← 13 RR lines from SEPTA GTFS
│   │   ├── data/parameters.py  ← Cost, fleet, fare, Logit parameters
│   │   ├── data/demand.py      ← Event-window demand (pre-game + post-game + baseline)
│   │   ├── models/upper_level.py ← SLSQP solver (13×40 slots)
│   │   ├── models/lower_level.py ← Multinomial Logit passenger route choice
│   │   ├── models/bilevel.py   ← Iterative best-response bilevel solver
│   │   └── models/sensitivity.py ← Optuna TPE stochastic search
│   ├── v2/                     ← Multimodal policy model (18:00–04:00, 40 slots)
│   │   ├── config/scenario.py  ← Master config: all parameters
│   │   ├── data/worldcup_demand.py ← Demand: pre-game, in-game, post-game
│   │   ├── data/bsl.py         ← BSL/B Line capacity model (NRG Station)
│   │   ├── models/policy_objective.py ← Policy objective + greedy RR allocator
│   │   └── reporting/reporting.py ← KPI computation, display, CSV export
│   └── common/
│       ├── network_builder.py  ← NetworkX multi-line graph
│       └── plotting.py         ← Demand curves, allocation heatmap
├── data/                       ← Raw data assets (GTFS, ridership CSV, cost JSON)
│   ├── gtfs/                   ← SEPTA GTFS v202603296
│   ├── ridership/              ← FY2024 APC weekday boardings
│   └── costs/                  ← Route operating statistics
├── docs/                       ← Documentation
├── scripts/                    ← CLI wrappers
├── outputs/
│   ├── tables/                 ← CSV results
│   └── validation/             ← Audit reports
├── _run_optimization.py        ← v1 greedy integer allocation (entry point)
├── _run_ilp_comparison.py      ← v1 ILP comparison via PuLP/CBC
├── run_scenarios.py            ← v2 eight-scenario comparison runner
├── main.py                     ← v1 bilevel pipeline entry point
└── formulation.md              ← Mathematical formulation index
```

---

## Model Architecture

### Time Window (Multimodal Layer)

The 18:00–04:00 match-day window is divided into **40 slots of 15 minutes** each:

| Slot | Time | Notes |
|------|------|-------|
| 0 | 18:00 | Window start, fans begin traveling |
| 10 | 20:30 | Default kickoff |
| 18 | 22:30 | Approximate final whistle for 20:30 kickoff |
| 20 | 23:00 | Post-game exit / evacuation peak begins |
| 24 | 00:00 (+1) | Crosses into next calendar day |
| 32 | 02:00 (+1) | Late-night evacuation tail |
| 39 | 03:45 (+1) | Last modeled 15-minute slot |

**Cross-midnight handling:** Slot indices are referenced from 18:00. The function `time_to_slot("02:00")` returns slot 32 (8 hours × 4 slots/hour), wrapping times like `00:00–04:00` correctly via modular arithmetic. All slot labels include `(+1)` for post-midnight times.

### Fan Segments

| Segment | Share | Mode to NRG |
|---------|-------|-------------|
| local_city | 35% | BSL direct |
| suburban_rr | 30% | RR → Center City → BSL |
| visitor_hotel_airport | 20% | 50% RR, 50% BSL |
| car_rideshare | 15% | Outside model scope |

**Total transit fans:** ~45,063 (65% of 69,328 stadium capacity).
**RR feeder share:** 40% of transit fans (suburban_rr + half of visitor).

### BSL Service Levels (Discrete)

| Level | Headway | Trains/slot | Effective capacity/slot |
|-------|---------|-------------|------------------------|
| normal | 8 min | 2 | 1,224 pax |
| enhanced | 5 min | 3 | 1,836 pax |
| max_event | 3 min | 5 | 3,060 pax |

Effective capacity = trains × 720 seats × 0.85 safety buffer.

NRG Station platform cap: 4,000 pax/slot throughput; crowding penalty above 2,500 pax/slot.

### Policy Objective (Minimize)

```
Objective = (RR_op_cost + BSL_op_cost)
          − fare_revenue − sponsor_reimbursement
          + 50 × unmet_pax
          + 30 × crowding_pax
          + 100,000 × equity_violations
          + 5,000 × headway_excess_min
          + 200 × clearance_excess_min
```

Lower is better. Weights reflect social cost of stranded passengers vs. financial deficit.

---

## Scenarios

Eight scenarios are pre-configured in `run_scenarios.py`:

| # | Name | Key Changes |
|---|------|-------------|
| 1 | RR-Only Baseline | RR only, no BSL model, no post-game (v1 spirit) |
| 2 | Multimodal Default | BSL + post-game, no free return |
| 3 | Free Return Rides | Sponsor-funded post-game trips |
| 4 | High Attendance | 80% transit share (~55,000 fans), 2 extra RR trains |
| 5 | Delayed Exit Surge | Post-game crowd exits 20 min later |
| 6 | Low Sponsor Subsidy | $1/pax subsidy instead of $3 |
| 7 | Later Kickoff (21:00) | Post-game extends past 02:00 |
| 8 | Overnight Stress | High attendance + late kickoff + delayed exit |

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate all final results (recommended)

```bash
# Single command: runs v1 greedy + v1 ILP + all 8 v2 scenarios,
# writes all CSVs, figures, raw data, and validation summary:
python scripts/generate_all_results.py
```

### Individual commands

```bash
# v1 greedy integer allocation with elastic Logit demand (~15s):
python _run_optimization.py

# v1 greedy vs. ILP optimality comparison (requires pulp, ~15s):
python _run_ilp_comparison.py

# v2 eight-scenario comparison:
python run_scenarios.py
python run_scenarios.py --save-csv           # also writes outputs/tables/
python run_scenarios.py --scenario 4 --verbose  # single scenario, full KPIs

# Import smoke test (no results generated):
python scripts/validate_project.py
```

### Generated outputs

```
outputs/
├── tables/
│   ├── v1_greedy_summary.csv          ← v1 profit, revenue, budget, pax
│   ├── v1_ilp_comparison.csv          ← greedy vs ILP gap (0%)
│   ├── v2_scenario_comparison.csv     ← all 8 scenarios, all KPIs
│   └── v1_vs_v2_summary.csv           ← side-by-side cross-model comparison
├── figures/
│   ├── v2_net_deficit_by_scenario.png
│   ├── v2_unmet_demand_by_scenario.png
│   ├── v2_peak_nrg_crowding_by_scenario.png
│   ├── v2_post_game_clearance_by_scenario.png
│   ├── v2_bsl_load_factor_timeseries.png
│   ├── v2_post_game_evacuation_curve.png
│   ├── v1_vs_v2_key_kpis.png
│   └── v2_equity_coverage_by_line.png
├── raw/
│   ├── v2_s2_bsl_per_slot.csv         ← per-slot BSL demand/served/crowding
│   └── v2_s2_rr_per_line.csv          ← per-line RR equity coverage
└── validation/
    └── final_validation_summary.txt
```

### How to interpret results

- **v1** = Regional Rail-only profit baseline. Maximizes `Σ(fare × pax) − cost`. Logit elastic demand. Greedy integer allocation = ILP optimal (0% gap).
- **v2** = Multimodal policy/capacity model. Minimizes `(cost − revenue − sponsor) + social_cost_penalties`. Models RR feeder + BSL + NRG Station.
- **Both** use 18:00–04:00+1, 40 slots, 15-min resolution.
- **v1 profit** ($288K+) and **v2 net deficit** ($648K+) are different objective types — not directly comparable.
- **BSL/NRG post-game clearance** is the binding bottleneck in v2 (225 min vs 90 min target).
- **Sponsor/free-return policy** changes the tradeoff from pure profit to subsidy–service balance (see S3 vs S6).

### Slow entry points (bilevel / sensitivity)

```bash
python main.py --mode upper_only   # global SLSQP, ~4–15 min
python main.py                     # full bilevel, superset of upper_only
python main.py --mode sensitivity  # Optuna sweep, ~hours
```

---

## Key Parameters

### Regional Rail (v1)

| Parameter | Value | Source |
|-----------|-------|--------|
| Train capacity | 875 seats (5-car Bombardier Multilevel) | SEPTA fleet |
| Fixed cost / trip | $1,692.00 | `$13,535.85/trainset/day ÷ 8 trips` |
| Variable cost / pax | $0.40 | Literature (marginal incremental) |
| Event day budget | $350,000 | Model assumption |
| Equity threshold ε | 0.80 | Must serve ≥ 80% of demand |
| Fare range | $2.50 – $9.00 | SEPTA zone fares + surge cap |
| Logit coefficients (α₁, α₂, α₃) | 0.50, 0.30, 0.15 | Small & Verhoef (2007) |

### BSL / Multimodal (new)

| Parameter | Value | Notes |
|-----------|-------|-------|
| BSL train capacity | 720 pax (8 cars × 90 seats) | SEPTA fleet |
| BSL safety buffer | 0.85 | Effective = nominal × buffer |
| BSL fixed cost / extra trip | $800 | Event premium above baseline |
| BSL variable cost / pax | $0.30 | Marginal incremental |
| NRG throughput cap | 4,000 pax/slot | Platform + fare gate constraint |
| Sponsor reimbursement | $3.00/pax | Default; scenario 6 tests $1.00 |
| Post-game clearance target | 90 min | Must clear NRG within 90 min of whistle |
| Exit delay mean | 25 min | Time from final whistle to transit queue |

---

## Data Sources

| Data | Source | File | Used for |
|------|--------|------|----------|
| Line topology, station order, travel times | SEPTA GTFS v202603296 | `data/gtfs/` | Network structure |
| Weekday boardings by line (FY2024) | SEPTA OpenDataPhilly APC | `data/ridership/ridership_by_line.json` | Demand scaling |
| Per-station boarding shares (FY2024) | SEPTA OpenDataPhilly APC | `data/ridership/septa_rr_ridership.csv` | Station-level shares |
| Route operating costs (FY2024 budgeted) | SEPTA OpenDataPhilly | `data/costs/cost_summary.json` | `c_f = $1,692/trip` |
| Zone-based fares | SEPTA official fare table | `data/network.py` | Fare bounds |
| Match attendance | Lincoln Financial Field cap. 69,328 | — | Transit fan estimate |

---

## Output

`run_scenarios.py` prints a one-line summary per scenario, then a comparison table:

```
Policy objective        |   S1 Baseline |  S2 Multimodal |  ...
Total served (pax)      |        44,000 |         56,000 |  ...
Total unmet (pax)       |         1,200 |             80 |  ...
Net deficit ($)         |       180,000 |        145,000 |  ...
Peak NRG crowding (pax) |         3,200 |           800  |  ...
Clearance (min)         |           120 |            85  |  ...
```

`--verbose` adds a full KPI report per scenario including ridership, load factors, financials, equity, and penalty breakdown.

---

## Limitations & Assumptions

- No microscopic headway simulation; slot-level capacity is deterministic.
- BSL travel time City Hall ↔ NRG assumed constant at 15 min (historical 14–16 min).
- Post-game demand modeled as a Normal distribution; actual exit flow is stochastic.
- RR feeder delay (20 min Center City transfer) is fixed; no variation by line.
- Sponsor subsidy policy is exogenous; no behavioral response to free fares.
- Crowd flow at transfer stations (Suburban, Jefferson, 30th St.) is not explicitly modeled.

---

## Future Work

- Replace Normal post-game wave with agent-based stadium egress simulation.
- Add transfer station capacity constraints (fare gate throughput at Suburban/Jefferson).
- Model variable train dwell time at NRG Station under crowding.
- Extend bilevel to include BSL fare decisions (currently policy-fixed).
- Incorporate actual SEPTA schedule data (departure times, not just frequency).
