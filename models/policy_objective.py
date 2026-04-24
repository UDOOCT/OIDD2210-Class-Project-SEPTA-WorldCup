"""
models/policy_objective.py
--------------------------
Two objective functions for the SEPTA World Cup transit model:

  1. profit_baseline(rr_result)
     The original v1 Regional Rail profit objective.
     Preserved for backward compatibility and as a comparison baseline.

  2. multimodal_policy_objective(rr_result, bsl_result, demand, scenario_cfg)
     New policy-oriented objective that minimizes operating deficit
     while penalizing unmet demand, crowding, equity failures, and
     poor service reliability.

     Objective =   operating_cost
                 - fare_revenue
                 - sponsor_reimbursement
                 + W_UNMET   × unmet_pax
                 + W_CROWDING × crowding_pax
                 + W_EQUITY  × equity_violations    (lump-sum penalty)
                 + W_HEADWAY × headway_excess_min
                 + W_CLEARANCE × clearance_delay_min

     Lower is better (it is a minimization objective).

Also contains evaluate_rr_service(), a simplified RR allocator for the
18:00–04:00 window that complements the full bilevel SLSQP in models/upper_level.py.

Note on naming: per the task requirements, solvers are named honestly.
  evaluate_rr_service  = greedy_integer_allocator (not a bilevel exact solver)
  profit_baseline      = original v1 continuous_relaxation_solver result wrapper
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any

# Import scenario defaults; callers can override by passing a scenario_cfg dict.
import data.scenario as SC


# ─────────────────────────────────────────────────────────────────────────────
# 1. Original v1 profit baseline (backward-compatible wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def profit_baseline(rr_result: Dict) -> float:
    """
    Extract SEPTA Regional Rail profit from a v1 upper_level.solve() result dict.

    This is the original objective:
      Π = revenue − fixed_cost − variable_cost

    Args:
        rr_result: result dict returned by models/upper_level.solve()

    Returns: float profit in dollars.
    """
    return float(rr_result.get("profit", 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simplified RR evaluator for the new 18:00–04:00 window
#    (greedy_integer_allocator — not the full bilevel SLSQP)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rr_service(
    rr_demand: Dict[str, np.ndarray],
    extra_trains_per_slot: int = 1,
    inbound_fare: float = SC.BASE_INBOUND_FARE,
    free_return:  bool  = True,
    budget:       float = SC.DAILY_EVENT_BUDGET,
) -> Dict:
    """
    greedy_integer_allocator for Regional Rail service in the 18:00–04:00 window.

    Allocates extra trains above the normal evening baseline for each (line, slot)
    based on demand.  Uses fixed fares (not continuous surge pricing) to match
    the realistic World Cup policy (regular inbound fare, sponsored free return).

    This is intentionally simpler than the full bilevel SLSQP in upper_level.py:
    - No Logit lower level (fares are policy-fixed, not optimized)
    - Discrete extra_trains_per_slot (0, 1, 2, or 3 extra trains)
    - Budget is enforced; service is cut if budget is exhausted

    Args:
        rr_demand         : per-line per-slot demand (18:00–04:00 window)
        extra_trains_per_slot: fixed extra trains added during the event window
        inbound_fare      : fare charged for inbound trips
        free_return       : if True, return trips have fare = 0
        budget            : maximum operating budget available

    Returns dict with per-line arrays and aggregate financial metrics.
    """
    from data.scenario import (
        N_SLOTS, RR_BASELINE_TRAINS_EVENING, RR_BASELINE_TRAINS_LATENIGHT,
        RR_FIXED_COST_PER_TRAIN_TRIP, RR_VAR_COST_PER_PAX,
        TRAIN_CAPACITY_RR, RR_EVENING_END_SLOT, SLOT_MINUTES,
    )

    results = {"lines": {}}
    total_rev    = 0.0
    total_fixed  = 0.0
    total_var    = 0.0
    total_served = 0.0
    total_unmet  = 0.0
    budget_used  = 0.0

    for line, demand_arr in rr_demand.items():
        f_arr      = np.zeros(N_SLOTS)
        x_arr      = np.zeros(N_SLOTS)
        rev_arr    = np.zeros(N_SLOTS)
        unmet_arr  = np.zeros(N_SLOTS)

        for t in range(N_SLOTS):
            # Baseline trains for this slot
            baseline = (RR_BASELINE_TRAINS_EVENING
                        if t < RR_EVENING_END_SLOT
                        else RR_BASELINE_TRAINS_LATENIGHT)
            extra  = extra_trains_per_slot
            f_lt   = baseline + extra
            cap_lt = f_lt * TRAIN_CAPACITY_RR
            d_lt   = float(demand_arr[t])

            # Check budget before committing extra trains
            cost_extra = extra * RR_FIXED_COST_PER_TRAIN_TRIP
            if budget_used + cost_extra > budget:
                extra      = 0
                f_lt       = baseline
                cap_lt     = baseline * TRAIN_CAPACITY_RR
                cost_extra = 0  # not deploying extra, so no budget consumed

            x_lt = min(d_lt, cap_lt)
            unmet = max(0.0, d_lt - x_lt)

            # Fare: inbound (pre-game half of demand) vs return (post-game half)
            # Approximate: slots before kickoff = inbound; after = return
            fare = inbound_fare if t < N_SLOTS // 2 else (0.0 if free_return else inbound_fare)

            rev_lt   = fare * x_lt
            fixed_lt = f_lt * RR_FIXED_COST_PER_TRAIN_TRIP
            var_lt   = RR_VAR_COST_PER_PAX * x_lt

            f_arr[t]     = f_lt
            x_arr[t]     = x_lt
            rev_arr[t]   = rev_lt
            unmet_arr[t] = unmet
            budget_used += cost_extra
            total_rev   += rev_lt
            total_fixed += fixed_lt
            total_var   += var_lt
            total_served+= x_lt
            total_unmet += unmet

        results["lines"][line] = {
            "f":     f_arr,
            "x":     x_arr,
            "rev":   rev_arr,
            "unmet": unmet_arr,
        }

    results.update({
        "revenue":      round(total_rev,    2),
        "fixed_cost":   round(total_fixed,  2),
        "var_cost":     round(total_var,    2),
        "total_served": round(total_served),
        "total_unmet":  round(total_unmet),
        "profit":       round(total_rev - total_fixed - total_var, 2),
        "budget_used":  round(budget_used, 2),
    })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multimodal policy objective
# ─────────────────────────────────────────────────────────────────────────────

def multimodal_policy_objective(
    rr_result:   Dict,
    bsl_result:  Dict,
    demand:      Dict,
    scenario_cfg: Any = SC,
) -> Dict:
    """
    Policy-oriented objective for the multimodal World Cup transit model.

    Minimizes:
      net_deficit + social_cost_penalties

    where:
      net_deficit     = (rr_op_cost + bsl_op_cost)
                        − fare_revenue − sponsor_reimbursement
      social penalties = unmet_demand + crowding + equity + headway + clearance

    Lower objective value = better policy outcome.

    Args:
        rr_result:    output of evaluate_rr_service() or upper_level.solve()
        bsl_result:   output of data.bsl.allocate_bsl_service()
        demand:       output of data.worldcup_demand.get_demand()
        scenario_cfg: module or object with scenario parameters

    Returns dict with full objective breakdown.
    """
    cfg = scenario_cfg

    # ── Operating costs ───────────────────────────────────────────────────────
    rr_op_cost  = float(rr_result.get("fixed_cost", 0) + rr_result.get("var_cost", 0))
    bsl_op_cost = float(bsl_result.get("operating_cost", 0))
    total_op_cost = rr_op_cost + bsl_op_cost

    # ── Fare revenue ──────────────────────────────────────────────────────────
    fare_revenue = float(rr_result.get("revenue", 0))

    # ── Sponsor reimbursement ─────────────────────────────────────────────────
    # Sponsor pays SEPTA for each free post-game return trip on BSL+RR.
    if getattr(cfg, "SPONSOR_SUBSIDY", True) and getattr(cfg, "FREE_RETURN_FROM_NRG", True):
        free_return_pax    = float(bsl_result.get("total_served", 0)) * 0.5  # approx post-game share
        sponsor_reimburse  = free_return_pax * cfg.SPONSOR_REIMBURSEMENT_PER_PAX
    else:
        sponsor_reimburse  = 0.0

    # ── Net operating deficit ─────────────────────────────────────────────────
    net_deficit = total_op_cost - fare_revenue - sponsor_reimburse

    # ── Penalty: unmet demand ─────────────────────────────────────────────────
    rr_unmet    = float(rr_result.get("total_unmet", 0))
    bsl_unmet   = float(bsl_result.get("total_unmet", 0))
    total_unmet = rr_unmet + bsl_unmet
    unmet_penalty = cfg.W_UNMET_DEMAND * total_unmet

    # ── Penalty: NRG crowding ─────────────────────────────────────────────────
    total_crowding  = float(bsl_result.get("nrg_crowding", np.zeros(1)).sum())
    crowding_penalty = cfg.W_CROWDING * total_crowding

    # ── Penalty: equity ───────────────────────────────────────────────────────
    # Check that each RR line serves ≥ EQUITY_MIN_EFFECTIVE_COVERAGE of its demand.
    equity_violations = 0
    for line, lr in rr_result.get("lines", {}).items():
        rr_d = demand["rr_demand"].get(line, np.zeros(1))
        served = lr.get("x", np.zeros(1))
        total_d = float(rr_d.sum())
        if total_d > 0:
            coverage = float(served.sum()) / total_d
            if coverage < cfg.EQUITY_MIN_EFFECTIVE_COVERAGE:
                equity_violations += 1
    equity_penalty = cfg.W_EQUITY * equity_violations

    # ── Penalty: headway reliability ──────────────────────────────────────────
    headway_excess  = float(bsl_result.get("headway_violation_min", 0))
    headway_penalty = cfg.W_HEADWAY * headway_excess

    # ── Penalty: post-game clearance delay ────────────────────────────────────
    clearance_min    = float(bsl_result.get("clearance_time_min", 0))
    clearance_excess = max(0.0, clearance_min - cfg.POST_GAME_CLEARANCE_TARGET_MIN)
    clearance_penalty = cfg.W_CLEARANCE * clearance_excess

    # ── Total policy objective (minimize) ─────────────────────────────────────
    total_penalty = (unmet_penalty + crowding_penalty + equity_penalty +
                     headway_penalty + clearance_penalty)
    objective     = net_deficit + total_penalty

    # ── KPI: late-night unmet demand (post-midnight) ──────────────────────────
    from data.scenario import time_to_slot
    midnight_slot  = time_to_slot("00:00")
    bsl_unmet_arr  = bsl_result.get("unmet_outbound", np.zeros(1))
    latenight_unmet = float(bsl_unmet_arr[midnight_slot:].sum())

    return {
        # Main objective
        "objective":           round(objective,        2),
        "net_deficit":         round(net_deficit,       2),

        # Revenue components
        "fare_revenue":        round(fare_revenue,      2),
        "sponsor_reimburse":   round(sponsor_reimburse, 2),

        # Cost components
        "rr_operating_cost":   round(rr_op_cost,       2),
        "bsl_operating_cost":  round(bsl_op_cost,      2),
        "total_operating_cost":round(total_op_cost,    2),

        # Penalty breakdown
        "unmet_penalty":       round(unmet_penalty,    2),
        "crowding_penalty":    round(crowding_penalty,  2),
        "equity_penalty":      round(equity_penalty,    2),
        "headway_penalty":     round(headway_penalty,   2),
        "clearance_penalty":   round(clearance_penalty, 2),
        "total_penalty":       round(total_penalty,     2),

        # Demand KPIs
        "total_unmet":         round(total_unmet),
        "latenight_unmet":     round(latenight_unmet),
        "equity_violations":   equity_violations,

        # BSL KPIs (pass-through)
        "peak_crowding":       bsl_result.get("peak_crowding", 0),
        "clearance_time_min":  clearance_min,
        "bsl_load_factor":     bsl_result.get("load_factor", 0),
    }


TRAIN_CAPACITY_RR = 875   # convenience alias (defined in data/scenario.py)
