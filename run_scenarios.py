"""
run_scenarios.py
----------------
Eight-scenario comparison runner for the multimodal World Cup 2026 transit model.

Each scenario overrides a subset of data/scenario.py parameters and runs:
  1. Demand generation  (data/worldcup_demand.get_demand)
  2. RR service evaluation  (models/policy_objective.evaluate_rr_service)
  3. BSL service allocation  (data/bsl.allocate_bsl_service)
  4. Policy objective  (models/policy_objective.multimodal_policy_objective)
  5. KPI computation  (reporting.compute_kpis)

Scenarios defined:
  1. Baseline: RR-only, no BSL model, no post-game (v1 model spirit)
  2. Multimodal: BSL + post-game (default config)
  3. Free return rides: multimodal + sponsor-funded return trips
  4. High attendance: 80% transit share instead of 65%
  5. Delayed exit: severe post-game surge (exit delay +20 min)
  6. Lower subsidy: sponsor reimburses $1 instead of $3 per pax
  7. Later kickoff: 21:00 instead of 20:30
  8. Overnight stress: extra post-midnight latent demand

Run with:
    python run_scenarios.py
    python run_scenarios.py --save-csv    # also writes outputs/scenario_comparison.csv
"""

from __future__ import annotations

import argparse
import copy
import types
import numpy as np

import data.scenario as SC
from data.worldcup_demand import get_demand
from data.bsl import allocate_bsl_service
from models.policy_objective import evaluate_rr_service, multimodal_policy_objective
from reporting import compute_kpis, print_kpi_report, print_scenario_comparison, save_comparison_csv


# ─────────────────────────────────────────────────────────────────────────────
# Scenario runner
# ─────────────────────────────────────────────────────────────────────────────

def _make_cfg(**overrides) -> types.SimpleNamespace:
    """Build a scenario config as a SimpleNamespace from data/scenario.py defaults + overrides."""
    cfg = types.SimpleNamespace(**{
        k: v for k, v in vars(SC).items()
        if not k.startswith("_") and not callable(v) and not isinstance(v, type)
    })
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_one_scenario(
    name: str,
    kickoff: str             = SC.DEFAULT_KICKOFF_TIME,
    total_fans: int          = SC.TOTAL_FANS_TRANSIT,
    include_post_game: bool  = True,
    include_bsl: bool        = True,
    extra_rr_trains: int     = 1,
    extra_fans_multiplier: float = 1.0,
    exit_delay_extra_min: int    = 0,
    scenario_cfg_overrides: dict = None,
    verbose: bool = False,
) -> dict:
    """
    Run a single scenario end-to-end and return its KPI dict.

    Args:
        name                   : scenario label for display
        kickoff                : HH:MM kickoff time
        total_fans             : total fans using SEPTA (~45,000 default)
        include_post_game      : whether to generate post-game demand
        include_bsl            : whether to run BSL capacity model
        extra_rr_trains        : extra RR trains per slot above evening baseline
        extra_fans_multiplier  : scale factor applied to total_fans (for high-attendance)
        exit_delay_extra_min   : extra minutes added to EXIT_DELAY_MEAN_MINUTES
        scenario_cfg_overrides : dict of additional data/scenario.py overrides
        verbose                : if True, print detailed KPI report

    Returns: KPI dict (from reporting.compute_kpis).
    """
    # Build scenario config
    cfg_kwargs = dict(scenario_cfg_overrides or {})
    if exit_delay_extra_min:
        cfg_kwargs["EXIT_DELAY_MEAN_MINUTES"] = SC.EXIT_DELAY_MEAN_MINUTES + exit_delay_extra_min
    cfg = _make_cfg(**cfg_kwargs)

    fans = int(total_fans * extra_fans_multiplier)

    # 1. Demand
    demand = get_demand(
        kickoff           = kickoff,
        total_fans        = fans,
        include_post_game = include_post_game,
    )

    # 2. RR service (greedy allocator)
    free_return = getattr(cfg, "FREE_RETURN_FROM_NRG", True)
    rr_result = evaluate_rr_service(
        rr_demand          = demand["rr_demand"],
        extra_trains_per_slot = extra_rr_trains,
        inbound_fare       = getattr(cfg, "BASE_INBOUND_FARE", SC.BASE_INBOUND_FARE),
        free_return        = free_return,
        budget             = getattr(cfg, "DAILY_EVENT_BUDGET", SC.DAILY_EVENT_BUDGET),
    )

    # 3. BSL service (skip if not included)
    if include_bsl:
        bsl_result = allocate_bsl_service(
            bsl_inbound    = demand["bsl_inbound"],
            bsl_outbound   = demand["bsl_outbound"],
            kickoff_slot   = demand["kickoff_slot"],
            match_end_slot = demand["match_end_slot"],
        )
    else:
        # Stub BSL result with zeros for baseline scenario
        bsl_result = {
            "service_levels":     ["normal"] * SC.N_SLOTS,
            "trains_per_slot":    np.zeros(SC.N_SLOTS),
            "capacity":           np.zeros(SC.N_SLOTS),
            "served_inbound":     np.zeros(SC.N_SLOTS),
            "served_outbound":    np.zeros(SC.N_SLOTS),
            "unmet_inbound":      demand["bsl_inbound"].copy(),
            "unmet_outbound":     demand["bsl_outbound"].copy(),
            "nrg_crowding":       demand["bsl_outbound"].copy(),
            "operating_cost":     0.0,
            "clearance_slot":     SC.N_SLOTS,
            "clearance_time_min": (SC.N_SLOTS - demand["match_end_slot"]) * SC.SLOT_MINUTES,
            "headway_violation_min": float(SC.N_SLOTS * 10),
            "peak_crowding":      float(demand["bsl_outbound"].max()),
            "total_served":       0.0,
            "total_unmet":        float(demand["bsl_inbound"].sum() + demand["bsl_outbound"].sum()),
            "load_factor":        0.0,
        }

    # 4. Policy objective
    obj_result = multimodal_policy_objective(
        rr_result   = rr_result,
        bsl_result  = bsl_result,
        demand      = demand,
        scenario_cfg= cfg,
    )

    # 5. KPIs
    kpis = compute_kpis(
        rr_result    = rr_result,
        bsl_result   = bsl_result,
        demand       = demand,
        obj_result   = obj_result,
        scenario_name= name,
    )

    if verbose:
        print_kpi_report(kpis, scenario_name=name)

    return kpis


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name":             "1. RR-Only Baseline",
        "desc":             "Regional Rail only, no BSL model, no post-game (v1 spirit)",
        "include_post_game": False,
        "include_bsl":       False,
        "extra_rr_trains":   1,
        "scenario_cfg_overrides": {
            "FREE_RETURN_FROM_NRG": False,
            "SPONSOR_SUBSIDY":      False,
        },
    },
    {
        "name":             "2. Multimodal Default",
        "desc":             "BSL + post-game evacuation, no free return",
        "include_post_game": True,
        "include_bsl":       True,
        "extra_rr_trains":   1,
        "scenario_cfg_overrides": {
            "FREE_RETURN_FROM_NRG": False,
            "SPONSOR_SUBSIDY":      False,
        },
    },
    {
        "name":             "3. Free Return Rides",
        "desc":             "Multimodal + sponsor-funded free return trips",
        "include_post_game": True,
        "include_bsl":       True,
        "extra_rr_trains":   1,
        "scenario_cfg_overrides": {
            "FREE_RETURN_FROM_NRG": True,
            "SPONSOR_SUBSIDY":      True,
        },
    },
    {
        "name":             "4. High Attendance",
        "desc":             "80% transit share → ~55,000 SEPTA fans",
        "include_post_game": True,
        "include_bsl":       True,
        "extra_rr_trains":   2,
        "extra_fans_multiplier": 1.23,   # 80% / 65% ≈ 1.23
        "scenario_cfg_overrides": {
            "FREE_RETURN_FROM_NRG": True,
            "SPONSOR_SUBSIDY":      True,
        },
    },
    {
        "name":             "5. Delayed Exit Surge",
        "desc":             "Post-game crowd exits 20 min later (severe congestion)",
        "include_post_game": True,
        "include_bsl":       True,
        "extra_rr_trains":   2,
        "exit_delay_extra_min": 20,
        "scenario_cfg_overrides": {
            "FREE_RETURN_FROM_NRG": True,
            "SPONSOR_SUBSIDY":      True,
        },
    },
    {
        "name":             "6. Low Sponsor Subsidy",
        "desc":             "Sponsor pays $1/pax instead of $3 — higher SEPTA deficit",
        "include_post_game": True,
        "include_bsl":       True,
        "extra_rr_trains":   1,
        "scenario_cfg_overrides": {
            "FREE_RETURN_FROM_NRG":         True,
            "SPONSOR_SUBSIDY":              True,
            "SPONSOR_REIMBURSEMENT_PER_PAX": 1.00,
        },
    },
    {
        "name":             "7. Later Kickoff (21:00)",
        "desc":             "Kickoff at 21:00 — post-game evacuation extends past 02:00",
        "kickoff":          "21:00",
        "include_post_game": True,
        "include_bsl":       True,
        "extra_rr_trains":   1,
        "scenario_cfg_overrides": {
            "FREE_RETURN_FROM_NRG": True,
            "SPONSOR_SUBSIDY":      True,
        },
    },
    {
        "name":             "8. Overnight Stress",
        "desc":             "High transit use + later kickoff + delayed exit: worst case",
        "kickoff":          "21:00",
        "include_post_game": True,
        "include_bsl":       True,
        "extra_rr_trains":   3,
        "extra_fans_multiplier": 1.23,
        "exit_delay_extra_min": 20,
        "scenario_cfg_overrides": {
            "FREE_RETURN_FROM_NRG": True,
            "SPONSOR_SUBSIDY":      True,
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SEPTA World Cup 2026 — multimodal scenario comparison")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save comparison table to outputs/scenario_comparison.csv")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full KPI report for each scenario")
    parser.add_argument("--scenario", type=int, default=None,
                        help="Run only scenario N (1-indexed)")
    args = parser.parse_args()

    print()
    print("SEPTA Regional Rail & Broad Street Line — World Cup 2026")
    print("Multimodal Scenario Analysis   (time window: 18:00 → 04:00 +1)")
    print("=" * 65)

    scenarios_to_run = SCENARIOS
    if args.scenario is not None:
        idx = args.scenario - 1
        if 0 <= idx < len(SCENARIOS):
            scenarios_to_run = [SCENARIOS[idx]]
        else:
            print(f"Scenario {args.scenario} not found (valid: 1–{len(SCENARIOS)})")
            return

    all_kpis = []
    for s in scenarios_to_run:
        print(f"\nRunning: {s['name']}")
        print(f"  {s.get('desc', '')}")
        kwargs = {k: v for k, v in s.items() if k not in ("name", "desc")}
        kpis = run_one_scenario(name=s["name"], verbose=args.verbose, **kwargs)
        all_kpis.append(kpis)
        if not args.verbose:
            # Print one-line summary
            print(f"  → served={kpis['total_served']:,}  unmet={kpis['total_unmet']:,}  "
                  f"deficit=${kpis['net_deficit']:,}  "
                  f"clearance={kpis['clearance_time_min']}min  "
                  f"obj={kpis['policy_objective']:,}")

    print_scenario_comparison(all_kpis)

    if args.save_csv:
        save_comparison_csv(all_kpis, "outputs/scenario_comparison.csv")


if __name__ == "__main__":
    main()
