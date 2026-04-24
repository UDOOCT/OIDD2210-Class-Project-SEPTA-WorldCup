"""
reporting.py
------------
KPI computation, terminal display, and CSV output for the multimodal
World Cup 2026 SEPTA optimization model.

Usage:
    from reporting import compute_kpis, print_kpi_report, save_kpis_csv
    kpis = compute_kpis(rr_result, bsl_result, demand, obj_result)
    print_kpi_report(kpis, scenario_name="Baseline")
    save_kpis_csv(kpis, "outputs/baseline_kpis.csv")

For scenario comparison tables use print_scenario_comparison(scenario_results).
"""

from __future__ import annotations

import os
import csv
from typing import Dict, List, Any

import numpy as np

import data.scenario as SC


# ─────────────────────────────────────────────────────────────────────────────
# KPI computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(
    rr_result:  Dict,
    bsl_result: Dict,
    demand:     Dict,
    obj_result: Dict,
    scenario_name: str = "unnamed",
) -> Dict:
    """
    Aggregate all KPIs into a single flat dictionary.

    Inputs:
        rr_result    : from models.policy_objective.evaluate_rr_service()
                       (or models.upper_level.solve() for v1 baseline)
        bsl_result   : from data.bsl.allocate_bsl_service()
        demand       : from data.worldcup_demand.get_demand()
        obj_result   : from models.policy_objective.multimodal_policy_objective()
        scenario_name: label for display / CSV

    Returns: flat dict of KPI name → value.
    """
    midnight_slot = SC.time_to_slot("00:00")

    # ── Ridership ─────────────────────────────────────────────────────────────
    rr_served   = float(rr_result.get("total_served", 0))
    bsl_served  = float(bsl_result.get("total_served", 0))
    total_served = rr_served + bsl_served

    rr_unmet   = float(rr_result.get("total_unmet", 0))
    bsl_unmet  = float(bsl_result.get("total_unmet", 0))
    total_unmet = rr_unmet + bsl_unmet

    bsl_unmet_arr = bsl_result.get("unmet_outbound", np.zeros(SC.N_SLOTS))
    latenight_unmet = float(bsl_unmet_arr[midnight_slot:].sum())

    # ── Load factors ──────────────────────────────────────────────────────────
    rr_capacity = sum(
        float(lr["f"].sum()) * 875   # 875 seats/train
        for lr in rr_result.get("lines", {}).values()
    )
    rr_load_factor = rr_served / max(rr_capacity, 1)
    bsl_load_factor = float(bsl_result.get("load_factor", 0))

    # ── NRG crowding ──────────────────────────────────────────────────────────
    peak_crowding    = float(bsl_result.get("peak_crowding", 0))
    crowding_arr     = bsl_result.get("nrg_crowding", np.zeros(SC.N_SLOTS))
    total_crowding   = float(crowding_arr.sum())

    # ── Post-game clearance ───────────────────────────────────────────────────
    clearance_min    = float(bsl_result.get("clearance_time_min", 0))
    clearance_target = SC.POST_GAME_CLEARANCE_TARGET_MIN
    clearance_delay  = max(0.0, clearance_min - clearance_target)

    # ── Financial ─────────────────────────────────────────────────────────────
    op_cost          = float(obj_result.get("total_operating_cost", 0))
    fare_revenue     = float(obj_result.get("fare_revenue", 0))
    sponsor_reimburse= float(obj_result.get("sponsor_reimburse", 0))
    net_deficit      = float(obj_result.get("net_deficit", 0))
    objective        = float(obj_result.get("objective", 0))

    # ── Equity ────────────────────────────────────────────────────────────────
    raw_rr_demand = sum(
        float(demand["rr_demand"].get(line, np.zeros(1)).sum())
        for line in rr_result.get("lines", {})
    )
    raw_coverage = rr_served / max(raw_rr_demand, 1)
    equity_violations = int(obj_result.get("equity_violations", 0))

    # ── Average wait time (rough estimate from BSL headway) ───────────────────
    # Weighted average headway / 2 across served slots
    bsl_levels   = bsl_result.get("service_levels", ["normal"] * SC.N_SLOTS)
    served_arr   = (bsl_result.get("served_inbound", np.zeros(SC.N_SLOTS)) +
                    bsl_result.get("served_outbound", np.zeros(SC.N_SLOTS)))
    total_svc_pax = served_arr.sum()
    if total_svc_pax > 1e-6:
        weighted_headway = sum(
            served_arr[t] * SC.BSL_SERVICE_LEVELS[bsl_levels[t]]["headway_min"]
            for t in range(SC.N_SLOTS)
        ) / total_svc_pax
        avg_wait_min = weighted_headway / 2.0
    else:
        avg_wait_min = 0.0

    return {
        "scenario":              scenario_name,
        # Ridership
        "total_served":          round(total_served),
        "rr_served":             round(rr_served),
        "bsl_served":            round(bsl_served),
        "total_unmet":           round(total_unmet),
        "rr_unmet":              round(rr_unmet),
        "bsl_unmet":             round(bsl_unmet),
        "latenight_unmet":       round(latenight_unmet),
        # Load factors
        "rr_load_factor":        round(rr_load_factor, 3),
        "bsl_load_factor":       round(bsl_load_factor, 3),
        # NRG
        "peak_nrg_crowding":     round(peak_crowding),
        "total_nrg_crowding":    round(total_crowding),
        # Post-game
        "clearance_time_min":    round(clearance_min),
        "clearance_delay_min":   round(clearance_delay),
        # Wait
        "avg_wait_min":          round(avg_wait_min, 1),
        # Financial
        "operating_cost":        round(op_cost),
        "fare_revenue":          round(fare_revenue),
        "sponsor_reimbursement": round(sponsor_reimburse),
        "net_deficit":           round(net_deficit),
        # Equity
        "raw_coverage":          round(raw_coverage, 3),
        "equity_violations":     equity_violations,
        # Objective
        "policy_objective":      round(objective),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Terminal display
# ─────────────────────────────────────────────────────────────────────────────

def print_kpi_report(kpis: Dict, scenario_name: str = "") -> None:
    """Print a formatted KPI report to the terminal."""
    name = scenario_name or kpis.get("scenario", "")
    print()
    print("=" * 65)
    print(f"  KPI REPORT: {name}")
    print("=" * 65)

    print("\n  [ RIDERSHIP ]")
    print(f"    Total served:        {kpis['total_served']:>10,}  pax")
    print(f"      RR served:         {kpis['rr_served']:>10,}  pax")
    print(f"      BSL served:        {kpis['bsl_served']:>10,}  pax")
    print(f"    Total unmet:         {kpis['total_unmet']:>10,}  pax")
    print(f"      Late-night unmet:  {kpis['latenight_unmet']:>10,}  pax  (post-midnight)")

    print("\n  [ LOAD FACTORS ]")
    print(f"    RR load factor:      {kpis['rr_load_factor']:>10.1%}")
    print(f"    BSL load factor:     {kpis['bsl_load_factor']:>10.1%}")

    print("\n  [ NRG STATION ]")
    print(f"    Peak crowding:       {kpis['peak_nrg_crowding']:>10,}  pax above threshold")
    print(f"    Total crowding:      {kpis['total_nrg_crowding']:>10,}  pax-slots")
    delay = kpis['clearance_delay_min']
    on_time_str = "✓ on time" if delay == 0 else f"✗ {delay} min late"
    print(f"    Clearance time:      {kpis['clearance_time_min']:>10}  min  ({on_time_str})")
    print(f"    Avg BSL wait:        {kpis['avg_wait_min']:>10.1f}  min")

    print("\n  [ FINANCIALS ]")
    print(f"    Operating cost:      ${kpis['operating_cost']:>10,}")
    print(f"    Fare revenue:        ${kpis['fare_revenue']:>10,}")
    print(f"    Sponsor reimburse:   ${kpis['sponsor_reimbursement']:>10,}")
    print(f"    Net deficit:         ${kpis['net_deficit']:>10,}")

    print("\n  [ EQUITY ]")
    print(f"    Raw demand coverage: {kpis['raw_coverage']:>10.1%}  (target ≥ 80%)")
    print(f"    Equity violations:   {kpis['equity_violations']:>10}  lines below 90% effective")

    print("\n  [ POLICY OBJECTIVE ]")
    print(f"    Objective value:     {kpis['policy_objective']:>10,}  (lower = better)")
    print()


def print_scenario_comparison(scenario_results: List[Dict]) -> None:
    """
    Print a compact side-by-side comparison table for multiple scenarios.

    Args:
        scenario_results: list of KPI dicts, one per scenario.
    """
    if not scenario_results:
        return

    metrics = [
        ("Policy objective",   "policy_objective",      "{:>12,}"),
        ("Total served (pax)", "total_served",           "{:>12,}"),
        ("Total unmet (pax)",  "total_unmet",            "{:>12,}"),
        ("Late-night unmet",   "latenight_unmet",        "{:>12,}"),
        ("Operating cost ($)", "operating_cost",         "{:>12,}"),
        ("Fare revenue ($)",   "fare_revenue",           "{:>12,}"),
        ("Sponsor reimb ($)",  "sponsor_reimbursement",  "{:>12,}"),
        ("Net deficit ($)",    "net_deficit",            "{:>12,}"),
        ("Peak NRG crowding",  "peak_nrg_crowding",      "{:>12,}"),
        ("Clearance (min)",    "clearance_time_min",     "{:>12}"),
        ("BSL load factor",    "bsl_load_factor",        "{:>12.1%}"),
        ("Raw coverage",       "raw_coverage",           "{:>12.1%}"),
    ]

    names = [r.get("scenario", f"S{i+1}") for i, r in enumerate(scenario_results)]
    col_w = max(16, max(len(n) for n in names) + 2)

    # Header
    print()
    print("=" * (24 + col_w * len(names)))
    print("  SCENARIO COMPARISON")
    print("=" * (24 + col_w * len(names)))
    header = f"  {'Metric':<22}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("-" * (24 + col_w * len(names)))

    for label, key, fmt in metrics:
        row = f"  {label:<22}"
        for r in scenario_results:
            val = r.get(key, 0)
            row += fmt.format(val).rjust(col_w)
        print(row)

    print("=" * (24 + col_w * len(names)))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_kpis_csv(kpis: Dict, path: str) -> None:
    """Save a single scenario KPI dict to a CSV file (key, value columns)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in kpis.items():
            writer.writerow([k, v])
    print(f"  Saved KPIs: {path}")


def save_comparison_csv(scenario_results: List[Dict], path: str) -> None:
    """Save a scenario comparison table to CSV."""
    if not scenario_results:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = [k for k in scenario_results[0] if k != "scenario"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario"] + keys)
        writer.writeheader()
        for r in scenario_results:
            writer.writerow(r)
    print(f"  Saved comparison: {path}")
