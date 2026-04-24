"""
src/septa_worldcup/v1/models/upper_level.py
--------------------------------------------
SEPTA Regional Rail profit optimization over 40 time slots (15-min, 18:00–04:00+1).

VARIABLE COUNT (active baseline):
  f[l, t] : 13 lines × 40 slots = 520  (integer — trains dispatched)
  p[l, t] : 13 × 40 = 520              (continuous — fare per slot)
  x[l, t] : 13 × 40 = 520              (continuous — pax served)
  Total: 1,560 variables

HISTORICAL NOTE: original v1 used 12 lines × 61 slots (6am–9pm). Active
  baseline is now 13 lines × 40 slots (18:00–04:00+1), matching v2's window.

FORMULATION:
  Sets:
    L  — 13 Regional Rail lines
    T  — 40 time slots (18:00, 18:15, …, 23:45, 00:00+1, …, 03:45+1)

  Objective:
    max  Π = Σ_{l,t} p[l,t]·x[l,t]
           − Σ_{l,t} c_f·f[l,t]
           − Σ_{l,t} c_v·x[l,t]

  Constraints:
    C1  x[l,t] ≤ C · f[l,t]                   ∀ l,t   capacity
    C2  x[l,t] ≤ d[l,t]                        ∀ l,t   demand cap
    C3  Σ_{l,t} c_f·f[l,t] ≤ B                        budget
    C5  x[l,t] ≥ ε·d[l,t]                     ∀ l,t   equity
    C6  p̄[l] ≤ p[l,t] ≤ σ·p̄[l]              ∀ l,t   fare bounds
    C7  |p[l,t] − p[l,t−1]| ≤ $1             ∀ l,t   fare smoothness
    (C4 min-service dropped — jointly infeasible with C3 at $350K budget)
"""

import numpy as np
from scipy.optimize import minimize

from septa_worldcup.v1.data.network import LINES
from septa_worldcup.v1.data.parameters import (
    N_SLOTS, SLOT_DURATION_MIN,
    TRAIN_CAPACITY, FIXED_COST_PER_TRAIN, VARIABLE_COST_PER_PAX,
    DAILY_BUDGET_EVENT, EQUITY_EPSILON,
    FARE_MIN, FARE_MAX, SURGE_FACTOR,
    MIN_TRAINS_PER_SLOT, MAX_TRAINS_PER_SLOT,
    IDX_2030, IDX_2230, IDX_0130,
    TBLOCK_NAMES, TBLOCK_RANGES_V1,
    is_peak,
)

LNAMES = list(LINES.keys())
L      = len(LNAMES)          # 13
T      = N_SLOTS              # 40
N      = L * T                # 520 — total (line, slot) pairs

# Named 4-block aggregation for the 18:00–04:00 event window
# pre_game:  slots  0–9   (18:00–20:15)
# in_game:   slots 10–17  (20:30–22:15)
# post_game: slots 18–29  (22:30–01:15+1)
# late_night:slots 30–39  (01:30–03:45+1)
TBLOCKS      = TBLOCK_NAMES
TBLOCK_RANGES = TBLOCK_RANGES_V1

_lidx = {l: i for i, l in enumerate(LNAMES)}

def idx(l_name: str, t_i: int) -> int:
    """Flat index for (line, slot_index) pair."""
    return _lidx[l_name] * T + t_i


def solve(demand: dict, budget: float = DAILY_BUDGET_EVENT,
          solver: str = "scipy") -> dict:
    """
    Solve upper-level optimization over 40 time slots × 13 lines (18:00–04:00).

    Args:
        demand : dict[line] → np.array shape (40,)
        budget : operating budget cap
        solver : 'scipy' | 'pulp' | 'gurobi'

    Returns:
        result dict with f[l][t_idx], p[l][t_idx], x[l][t_idx] and aggregate metrics.
    """
    # Flatten demand into vector d[l*T + t]
    d_vec = np.array([demand[l][t] for l in LNAMES for t in range(T)])

    # Base fares per line (replicated across slots)
    base_fares = np.array([
        LINES[l]["avg_fare"] for l in LNAMES for _ in range(T)
    ])

    # Per-slot min train requirement: 1 during peak, 0 off-peak
    f_min = np.array([
        1.0 if is_peak(t) else 0.0
        for l in LNAMES for t in range(T)
    ])

    if solver == "scipy":
        return _solve_scipy(d_vec, base_fares, f_min, budget)
    elif solver == "pulp":
        return _solve_pulp(d_vec, base_fares, f_min, budget)
    else:
        raise ValueError(f"Unknown solver: {solver}")


def _solve_scipy(d_vec, base_fares, f_min, budget):
    """
    Continuous relaxation via SLSQP.
    x_vec = [f_0,...,f_{N-1},  p_0,...,p_{N-1}]
    """
    def neg_profit(xv):
        f = np.clip(xv[:N], 0.0, MAX_TRAINS_PER_SLOT)
        p = np.clip(xv[N:], FARE_MIN, FARE_MAX)
        x = np.minimum(d_vec, TRAIN_CAPACITY * f)
        rev   = np.dot(p, x)
        fixed = FIXED_COST_PER_TRAIN * f.sum()
        var   = VARIABLE_COST_PER_PAX * x.sum()
        return -(rev - fixed - var)

    # Initial point: proportional to demand-weighted budget allocation.
    # C4 (min service=1 for peak) is dropped from the formulation here because
    # 24 peak slots × 13 lines × $1,692/train = $527,904 > $350,000 budget,
    # making C3+C4 jointly infeasible.  The equity constraint C5 already
    # guarantees trains are deployed wherever there is demand (cap ≥ ε·d).
    f0 = np.minimum(d_vec / max(TRAIN_CAPACITY, 1), MAX_TRAINS_PER_SLOT)
    f0_cost = FIXED_COST_PER_TRAIN * f0.sum()
    if f0_cost > budget:
        f0 = f0 * (budget * 0.9 / f0_cost)
    x0 = np.concatenate([f0, base_fares])

    bounds = (
        [(0.0, MAX_TRAINS_PER_SLOT)] * N +
        [(FARE_MIN, FARE_MAX)] * N
    )

    constraints = [
        # C3: budget
        {"type": "ineq",
         "fun": lambda xv: budget - FIXED_COST_PER_TRAIN * np.clip(xv[:N], 0.0, MAX_TRAINS_PER_SLOT).sum()},
        # C5: equity — capacity must cover ε×demand
        {"type": "ineq",
         "fun": lambda xv: TRAIN_CAPACITY * np.clip(xv[:N], 0.0, MAX_TRAINS_PER_SLOT) - EQUITY_EPSILON * d_vec},
        # C7: fare smoothness — |p[t] - p[t-1]| ≤ 1.0 per line
        # encoded as: p[t] - p[t-1] ≥ -1.0  AND  p[t-1] - p[t] ≥ -1.0
        {"type": "ineq",
         "fun": lambda xv: np.array([
             xv[N + idx(l, t)] - xv[N + idx(l, t-1)] + 1.0
             for l in LNAMES for t in range(1, T)
         ])},
        {"type": "ineq",
         "fun": lambda xv: np.array([
             xv[N + idx(l, t-1)] - xv[N + idx(l, t)] + 1.0
             for l in LNAMES for t in range(1, T)
         ])},
    ]

    res = minimize(
        neg_profit, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-6},
    )

    return _parse_result(res.x, d_vec, f_min, res.fun, res.success)


def _solve_pulp(d_vec, base_fares, f_min, budget):
    """
    True ILP via PuLP + CBC solver.
    Handles bilinear p·x via McCormick: fix p in outer loop, LP for f,x inner.

    TODO: implement full PuLP version.
    For now raises NotImplementedError with guidance.
    """
    raise NotImplementedError(
        "PuLP ILP not yet implemented.\n"
        "Recommended approach:\n"
        "  1. Outer loop: grid/Bayesian search over p[l,t]\n"
        "  2. Inner LP: fix p, solve for f,x using pulp.LpProblem\n"
        "     (bilinear becomes linear when p is fixed)\n"
        "  3. Return best (f,p,x) across outer iterations\n"
        "Run with solver='scipy' for continuous relaxation."
    )


def _parse_result(xv, d_vec, f_min, neg_profit_val, success):
    f_opt = np.clip(xv[:N], f_min, MAX_TRAINS_PER_SLOT)
    p_opt = np.clip(xv[N:], FARE_MIN, FARE_MAX)
    x_opt = np.minimum(d_vec, TRAIN_CAPACITY * f_opt)

    result = {"lines": {}}
    for i, l in enumerate(LNAMES):
        result["lines"][l] = {
            "f": f_opt[i*T:(i+1)*T],   # np.array shape (40,)
            "p": p_opt[i*T:(i+1)*T],
            "x": x_opt[i*T:(i+1)*T],
            "d": d_vec[i*T:(i+1)*T],
            "util": np.divide(
                x_opt[i*T:(i+1)*T],
                TRAIN_CAPACITY * np.maximum(f_opt[i*T:(i+1)*T], 1e-6),
                out=np.zeros(T), where=f_opt[i*T:(i+1)*T]>0
            ),
            "equity_ok": x_opt[i*T:(i+1)*T] >= EQUITY_EPSILON * d_vec[i*T:(i+1)*T],
        }

    rev   = float(np.dot(p_opt, x_opt))
    fixed = float(FIXED_COST_PER_TRAIN * f_opt.sum())
    var   = float(VARIABLE_COST_PER_PAX * x_opt.sum())
    eq_ok = int(np.all([
        result["lines"][l]["equity_ok"] for l in LNAMES
    ]))

    result.update({
        "profit":       round(rev - fixed - var, 2),
        "revenue":      round(rev, 2),
        "fixed_cost":   round(fixed, 2),
        "var_cost":     round(var, 2),
        "total_pax":    round(x_opt.sum()),
        "equity_all":   bool(eq_ok),
        "success":      success,
        "n_slots":      T,
        "n_lines":      L,
    })
    return result
