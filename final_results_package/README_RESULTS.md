# SEPTA World Cup 2026 Transit Planning Results Package

**OIDD 2210 Final Project**
Match day: Brazil vs. Haiti, June 19 2026, Lincoln Financial Field, Philadelphia

---

## Model Summary

This package contains final reproducible results from two optimization layers:

| Layer | Name | Objective | Scope |
|-------|------|-----------|-------|
| v1 | Regional Rail profit model | Maximize fare revenue − operating cost | RR only (13 lines) |
| v2 | Multimodal event-operations model | Minimize policy deficit + social penalties | RR feeder + BSL + NRG Station |

**Both active models use:**
- Time window: **18:00–04:00+1** (match day → next calendar day)
- Resolution: **15-minute slots, 40 slots total**
- Slot 0 = 18:00 · Slot 10 = 20:30 (kickoff) · Slot 24 = 00:00 (+1) · Slot 39 = 03:45 (+1)

---

## Folder Structure

```
final_results_package/
├── README_RESULTS.md          ← this file
├── demo_commands.txt          ← commands to reproduce results
├── tables/
│   ├── v1_greedy_summary.csv       ← v1 profit, revenue, cost, passengers, timing
│   ├── v1_ilp_comparison.csv       ← greedy vs CBC ILP optimality gap (0%)
│   ├── v2_scenario_comparison.csv  ← all 8 scenarios × 15 KPIs
│   └── v1_vs_v2_summary.csv        ← cross-model side-by-side
├── figures/
│   ├── v2_net_deficit_by_scenario.png
│   ├── v2_unmet_demand_by_scenario.png
│   ├── v2_peak_nrg_crowding_by_scenario.png
│   ├── v2_post_game_clearance_by_scenario.png
│   ├── v2_bsl_load_factor_timeseries.png    ← per-slot BSL load factor (S2)
│   ├── v2_post_game_evacuation_curve.png    ← post-game demand/served/crowding (S2)
│   ├── v1_vs_v2_key_kpis.png               ← ridership and financial comparison
│   └── v2_equity_coverage_by_line.png      ← per-line RR equity coverage (S2)
├── raw/
│   ├── v2_s2_bsl_per_slot.csv   ← 40-row per-slot BSL demand, capacity, served, crowding
│   └── v2_s2_rr_per_line.csv    ← per-line RR equity coverage percentages
└── validation/
    ├── final_validation_summary.txt   ← full audit with commands, checks, KPIs
    └── validation_summary.txt         ← earlier validation audit (pre-restructure)
```

---

## Key Results

### v1 — Regional Rail Greedy (Logit elastic demand, integer trains)

| Metric | Value |
|--------|-------|
| Profit | **$288,306** |
| Revenue | $684,155 |
| Fixed cost | $348,548 |
| Variable cost | $47,301 |
| Budget used | $348,548 / $350,000 |
| Total passengers served | **118,254** |
| Greedy vs ILP gap | **0.0000%** (globally optimal) |
| Equity satisfied | Yes |

### v2 — Multimodal Default (S2, no sponsor subsidy)

| Metric | Value |
|--------|-------|
| Total served (RR + BSL) | **90,489** |
| Total unmet demand | 43,741 |
| Operating cost | $803,088 |
| Fare revenue | $66,328 |
| Sponsor reimbursement | $0 (S2 has no sponsor) |
| Net deficit | **$736,760** |
| Peak NRG crowding above threshold | 0 pax |
| Post-game clearance time | **225 min** (target: 90 min) |
| Equity violations | 8 lines below 90% effective coverage |

### v2 — Eight Scenario Comparison (key column: net deficit)

| Scenario | Served | Unmet | Net Deficit | Clearance |
|----------|--------|-------|-------------|-----------|
| S1 RR-Only Baseline | 23,347 | 47,796 | $651,460 | 330 min |
| S2 Multimodal Default | 90,489 | 43,741 | $736,760 | 225 min |
| S3 Free Return Rides | 90,489 | 43,741 | $648,445 | 225 min |
| S4 High Attendance | 96,779 | 66,989 | $637,198 | 270 min |
| S5 Delayed Exit Surge | 89,033 | 45,197 | $648,056 | 225 min |
| S6 Low Sponsor Subsidy | 90,489 | 43,741 | $712,403 | 225 min |
| S7 Later Kickoff 21:00 | 92,447 | 41,782 | $642,648 | 225 min |
| S8 Overnight Stress | 99,098 | 64,668 | $628,307 | 270 min |

---

## How to Interpret

**v1 and v2 use different objectives — they are not directly comparable:**

- **v1 maximizes** `Σ(fare × pax) − fixed_cost − variable_cost`
  - Treats SEPTA as a profit center
  - Models only Regional Rail (not BSL or NRG Station)
  - Logit elastic demand: high World Cup parking penalty drives P_transit ≈ 57% at peak
  - Greedy integer allocator = CBC ILP optimal (0% gap)

- **v2 minimizes** `(RR_cost + BSL_cost − fare_revenue − sponsor) + social_penalties`
  - Treats SEPTA as a public service operator managing a match-night constraint
  - Models RR as a feeder → Center City → BSL → NRG Station
  - Penalty weights: $50/unmet pax, $30/crowding pax, $100K/equity violation, $200/clearance minute

**Binding bottleneck in v2:** BSL/NRG post-game evacuation. With ~45,000 fans exiting after
the match and BSL max_event capacity of 3,060 pax/slot, the clearance queue takes 225 minutes
to drain (3h45m) vs. the 90-minute target. This is correct queue math — not a model error.

**Sponsor/free-return policy** changes the planning problem from profit maximization (v1)
to a subsidy–service tradeoff (v2 S3 vs S6). Sponsor reimbursement at $3/pax (S3) reduces
the net deficit from $736K to $648K — a $88K improvement — while keeping served pax unchanged.

---

## Limitations

1. **Demand is planning-level, not stochastic simulation.** Pre-game and post-game waves
   use Normal distributions. Actual stadium egress is more heterogeneous.
2. **No microscopic headway simulation.** BSL capacity is deterministic per 15-min slot.
3. **RR feeder delay is fixed** at 20 min Center City transfer. No line-level variation.
4. **BSL travel time** City Hall ↔ NRG assumed constant 15 min.
5. **v1 is RR-only.** It does not model BSL capacity or NRG Station platform constraints.
6. **Equity in v2** is a post-hoc penalty on the 8 under-served lines, not a hard
   allocation constraint. Budget $350K supports only ~5 lines with extra trains.
7. **v1 global SLSQP** (`main.py`) takes ~2 hours and is not the recommended entry point.
   Use `_run_optimization.py` (~15s) for v1 results.

---

## Reproduction

All results in this package are fully reproducible from the repo root:

```bash
pip install -r requirements.txt
python scripts/generate_all_results.py   # ~30 seconds
python scripts/validate_project.py       # 11/11 checks
```

See `demo_commands.txt` for the complete command reference.
