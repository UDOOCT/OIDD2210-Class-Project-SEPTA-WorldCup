# SEPTA Regional Rail — World Cup 2026 Network Optimization
### OIDD 2210 Final Project

---

## Project Overview

Bilevel network optimization model to allocate SEPTA Regional Rail trains and set fares on World Cup match day (Brazil vs. Haiti, June 19 2026, 9pm kickoff at Lincoln Financial Field).

**Upper level (SEPTA):** Choose integer train frequencies `f[l,t]` and fares `p[l,t]` across 13 lines × 61 time slots (15-min, 6am–9pm) to maximize profit subject to capacity, budget, equity, and fare-smoothness constraints. Solved via SLSQP continuous relaxation.

**Lower level (Passengers):** Given SEPTA's schedule and fares, passengers choose transit vs. no-travel via Multinomial Logit, producing effective demand `x[l,t]`. The bilevel loop iterates until demand equilibrium.

**Stochastic extension:** Optuna TPE search over a 4-block (morning/midday/evening/night) policy evaluated across 100 Monte Carlo demand scenarios per trial.

---

## Repo Structure

```
septa_worldcup/
├── data/
│   ├── network.py              ← 13 RR lines from GTFS (stations, travel times, ridership)
│   ├── demand.py               ← Bimodal base demand + World Cup overlay + Monte Carlo
│   ├── parameters.py           ← All cost, fleet, fare, and logit parameters
│   ├── gtfs/                   ← SEPTA GTFS v202603296 (stop_times, trips, routes, stops)
│   ├── ridership/              ← FY2024 APC weekday boardings by line and station
│   └── costs/                  ← Route operating statistics (cost_summary.json)
├── models/
│   ├── upper_level.py          ← SLSQP upper-level solver (13×61 slots, 2,196 vars)
│   ├── lower_level.py          ← Multinomial Logit passenger route choice
│   ├── bilevel.py              ← Iterative best-response bilevel solver
│   └── sensitivity.py          ← Optuna TPE stochastic search (4-block policy)
├── utils/
│   ├── network_builder.py      ← NetworkX multi-line graph construction
│   └── visualize.py            ← Demand curves, allocation heatmap, fare profiles
├── outputs/                    ← Generated CSVs and plots (auto-created)
├── notebooks/
│   └── demo.ipynb              ← End-to-end walkthrough
├── _run_optimization.py        ← Greedy integer allocation with elastic Logit demand
├── _run_ilp_comparison.py      ← Multiple-choice knapsack ILP via PuLP/CBC
├── main.py                     ← Entry point: run full pipeline
├── formulation.md              ← Full mathematical formulation (LaTeX)
└── README.md
```

---

## Data Sources

| Data | Source | File | Used for |
|---|---|---|---|
| Line topology, station order, travel times | SEPTA GTFS v202603296 | `data/gtfs/` | Network structure, `τ_l` |
| Weekday boardings by line (FY2024) | SEPTA OpenDataPhilly APC | `data/ridership/ridership_by_line.json` | Base demand scaling |
| Per-station boarding shares (FY2024) | SEPTA OpenDataPhilly APC | `data/ridership/septa_rr_ridership.csv` | Station-level demand shares |
| Route operating costs (FY2024 budgeted) | SEPTA OpenDataPhilly | `data/costs/cost_summary.json` | `c_f = $1,691.98/trip` |
| Zone-based fares | SEPTA official fare table | `data/network.py` | Fare bounds per line |
| Match attendance | Lincoln Financial Field cap. 69,328 | — | `N_fans = 45,000` transit users |

---

## Math Formulation (summary)

See [`formulation.md`](formulation.md) for full derivations and algorithm pseudocode.

**Key dimensions:** L = 13 lines, T = 61 slots (15-min, 6am–9pm), 4 named blocks

**Decision variables:** `f[l,t] ∈ ℤ⁺`, `p[l,t] ∈ ℝ⁺`, `x[l,t] ∈ ℝ⁺`

**Objective:**
```
max  Π = Σ p[l,t]·x[l,t]  −  Σ 1691.98·f[l,t]  −  Σ 0.40·x[l,t]
```

**Constraints (8):**

| # | Name | Expression |
|---|---|---|
| C1 | Capacity | `x[l,t] ≤ 875·f[l,t]` |
| C2 | Demand cap | `x[l,t] ≤ d[l,t]` |
| C3 | Budget | `Σ 1691.98·f[l,t] ≤ $350,000` |
| C4 | Min service | `f[l,t] ≥ 1` (peak slots), `≥ 0` (off-peak) |
| C5 | Equity | `x[l,t] ≥ 0.80·d[l,t]` |
| C6 | Fare bounds | `$2.50 ≤ p[l,t] ≤ $9.00` |
| C7 | Fare smoothness | `|p[l,t] − p[l,t−1]| ≤ $1.00` |
| C8 | Integrality | `f[l,t] ∈ ℤ⁺` |

**Logit lower level:** `P_transit = 1 / (1 + exp(θ·(U_drive + G)))`, `G = 0.50·p + 0.30·(h/2) + 0.15·τ`

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt

# For ILP comparison (optional):
pip install pulp
```

### Main pipeline

```bash
# Bilevel optimization (default): iterative best-response, 40 iterations max
python main.py

# Upper-level SLSQP only (faster, continuous relaxation):
python main.py --mode upper_only

# Stochastic sensitivity search (Optuna TPE, 200 trials × 100 MC scenarios):
python main.py --mode sensitivity

# Run on normal day demand (no World Cup overlay):
python main.py --no-worldcup
```

### Standalone scripts

```bash
# Greedy integer allocation with elastic Logit demand (~3s):
python _run_optimization.py

# Greedy vs. ILP optimality comparison (requires pulp, ~2min for full 13-line):
python _run_ilp_comparison.py
```

### Notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Key Parameters

| Parameter | Value | Source |
|---|---|---|
| Train capacity | 875 seats (5-car Bombardier Multilevel) | SEPTA fleet |
| Fixed cost / trip | $1,691.98 | `$13,535.85/trainset/day ÷ 8 trips` |
| Variable cost / pax | $0.40 | Literature (marginal incremental) |
| Event day budget | $350,000 | Model assumption |
| Equity threshold ε | 0.80 | Must serve ≥ 80% of demand |
| Max trains / slot | 8 | Fleet capacity |
| Fare range | $2.50 – $9.00 | SEPTA zone fares + surge cap |
| World Cup fans (transit) | 45,000 | ~65% of 69,328 capacity |
| Logit coefficients (α₁, α₂, α₃) | 0.50, 0.30, 0.15 | Small & Verhoef (2007) |

---

## Output

`main.py` prints a per-block summary table:

```
Line                      Block      Trains  Avg Fare      Pax   Util  Eq?
----------------------------------------------------------------------
Airport                   morning         …    $x.xx       …    x.x%   ✓
Airport                   midday          …    $x.xx       …    x.x%   ✓
...
```

`--mode sensitivity` also writes `outputs/sensitivity_results.csv` with all 200 trial results.
