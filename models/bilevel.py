"""
models/bilevel.py
-----------------
Bilevel solver over 61 time slots × 12 lines.
"""

import numpy as np
from data.network import LINES
from data.parameters import (
    TIME_SLOTS, N_SLOTS, SLOT_DURATION_MIN,
    TRAIN_CAPACITY, DAILY_BUDGET_EVENT,
)
from models.upper_level import LNAMES, T, solve as upper_solve
from models.lower_level import effective_demand


def run_bilevel(demand: dict, max_iter: int = 40,
                tol: float = 0.01, budget: float = DAILY_BUDGET_EVENT,
                verbose: bool = True) -> dict:
    """
    Iterative best-response bilevel solver.

    demand: dict[line] → np.array shape (61,)  ← per-slot demand
    """
    eff_demand = {l: d.copy() for l, d in demand.items()}
    total_raw  = sum(demand[l].sum() for l in LNAMES)
    history    = []

    for iteration in range(max_iter):
        # ── Step 1: Upper level ──────────────────────────────────────────────
        result = upper_solve(eff_demand, budget=budget, solver="scipy")

        # ── Step 2: Lower level — update per-slot effective demand ───────────
        new_eff    = {}
        total_chg  = 0.0

        for l in LNAMES:
            f_arr  = result["lines"][l]["f"]   # shape (61,)
            p_arr  = result["lines"][l]["p"]
            avg_tt = (sum(LINES[l]["travel_times"]) /
                      max(len(LINES[l]["travel_times"]), 1))
            new_d  = np.zeros(T)

            for t in range(T):
                f_lt  = max(f_arr[t], 1e-6)
                # headway = slot_duration / trains_in_slot (minutes)
                hw    = SLOT_DURATION_MIN / f_lt
                new_d[t] = effective_demand(
                    raw_demand      = demand[l][t],
                    fare            = p_arr[t],
                    headway_min     = hw,
                    travel_time_min = avg_tt,
                )
            total_chg  += float(np.abs(new_d - eff_demand[l]).sum())
            new_eff[l]  = new_d

        history.append({
            "iter":      iteration + 1,
            "delta":     total_chg,
            "profit":    result["profit"],
            "total_pax": result["total_pax"],
        })

        if verbose:
            print(f"  Iter {iteration+1:3d} | Δ={total_chg:8.1f} pax | "
                  f"profit=${result['profit']:>10,.0f} | "
                  f"pax={result['total_pax']:>8,.0f}")

        if total_chg < tol * total_raw:
            if verbose:
                print(f"  ✓ Converged at iteration {iteration+1}")
            break

        eff_demand = new_eff

    result["iterations"] = history
    result["converged"]  = (total_chg < tol * total_raw)
    return result
