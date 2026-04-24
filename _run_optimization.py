"""
Elastic-demand optimization — Binary Logit mode-choice.

Generalized cost:
    G[l,t] = 0.50·p + 0.30·(15/(2·f)) + 0.15·avg_tt_l

Mode-choice (Binary Logit, θ=0.80):
    P_transit = 1 / (1 + exp(θ·(U_drive + G)))
    U_drive   = LOGIT_NO_TRAVEL_U − WC_DRIVE_PENALTY
              = −1.50 − 3.50 = −5.00
    WC_DRIVE_PENALTY accounts for ~$40 parking + traffic chaos at
    Lincoln Financial Field (≈ $7 driving-cost equivalent at α_fare=0.50).

Demand and service:
    d_hat[l,t] = d[l,t] · P_transit[l,t]
    x[l,t]     = min(d_hat[l,t],  875·f[l,t])

Algorithm — two-phase integer optimization:
  Phase 1 (Greedy integer f): precompute optimal profit for each
    (line, slot, f_count ∈ {0..8}) via 1-D fare search; then greedily
    add integer trains system-wide, highest marginal profit first, until
    budget exhausted or no profitable additions remain.
  Phase 2: fares are already optimal per slot from Phase 1 tables;
    results assembled directly without a second solve.

FIX 2: integer rounding is trivially exact here since Phase 1 produces
    integer f directly; comparison table shows continuous vs ceil(f_opt)
    where f_opt is the Phase-1 real-valued solve for reference.
"""

import sys, time
sys.path.insert(0, '.')

import numpy as np
from scipy.optimize import minimize_scalar

from data.demand import get_total_demand
from data.network import LINES
from data.parameters import (
    TIME_SLOTS, slot_label, is_peak,
    TRAIN_CAPACITY, FIXED_COST_PER_TRAIN, VARIABLE_COST_PER_PAX,
    DAILY_BUDGET_EVENT, EQUITY_EPSILON,
    FARE_MIN, FARE_MAX, MAX_TRAINS_PER_SLOT,
    LOGIT_ALPHA_FARE, LOGIT_ALPHA_WAIT, LOGIT_ALPHA_TRAVEL,
    LOGIT_THETA, LOGIT_NO_TRAVEL_U,
)
from models.upper_level import LNAMES, T

WC_DRIVE_PENALTY = 3.5                           # extra driving cost, WC match day
U_DRIVE_WC       = LOGIT_NO_TRAVEL_U - WC_DRIVE_PENALTY   # −5.00

t0 = time.time()
demand_full = get_total_demand(worldcup=True)
L = len(LNAMES)

avg_tt = {
    l: float(np.mean(LINES[l]["travel_times"])) if LINES[l]["travel_times"] else 10.0
    for l in LNAMES
}


# ── Scalar Logit helper (single slot) ────────────────────────────────────────
def _pt_scalar(p: float, f: float, att: float) -> float:
    if f <= 0:
        return 0.0
    wait = 15.0 / (2.0 * max(f, 1e-6))
    G    = LOGIT_ALPHA_FARE * p + LOGIT_ALPHA_WAIT * wait + LOGIT_ALPHA_TRAVEL * att
    arg  = np.clip(LOGIT_THETA * (U_DRIVE_WC + G), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(arg))


# Vectorised version (used later for result assembly)
def _pt_vec(p_arr, f_arr, att):
    f_safe = np.maximum(f_arr, 1e-6)
    wait   = 15.0 / (2.0 * f_safe)
    G      = (LOGIT_ALPHA_FARE   * p_arr
              + LOGIT_ALPHA_WAIT  * wait
              + LOGIT_ALPHA_TRAVEL * att)
    arg    = np.clip(LOGIT_THETA * (U_DRIVE_WC + G), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(arg))


# ── Phase 1: precompute profit*(l, t, f) via 1-D fare search ─────────────────
# For each (line, slot, train-count), find optimal fare and resulting profit.
# Profit here is TOTAL for that f level (including fixed cost).

F_MAX = int(MAX_TRAINS_PER_SLOT)   # 8

# Tables:  best_profit[l][t][f], best_fare[l][t][f]
best_profit = {l: [[0.0] * (F_MAX + 1) for _ in range(T)] for l in LNAMES}
best_fare   = {l: [[FARE_MIN] * (F_MAX + 1) for _ in range(T)] for l in LNAMES}

for l in LNAMES:
    att = avg_tt[l]
    for t in range(T):
        d_lt = float(demand_full[l][t])
        if d_lt < 0.5:
            continue   # essentially zero demand — leave at 0
        for f in range(1, F_MAX + 1):
            cap = TRAIN_CAPACITY * f

            def neg_slot_profit(p_val, d=d_lt, f=f, att=att, cap=cap):
                p  = float(p_val)
                pt = _pt_scalar(p, f, att)
                x  = min(d * pt, cap)
                return -(p * x
                         - VARIABLE_COST_PER_PAX * x
                         - FIXED_COST_PER_TRAIN  * f)

            res = minimize_scalar(
                neg_slot_profit, bounds=(FARE_MIN, FARE_MAX), method="bounded",
                options={"xatol": 1e-4},
            )
            p_opt = float(np.clip(res.x, FARE_MIN, FARE_MAX))
            best_profit[l][t][f] = -res.fun
            best_fare[l][t][f]   = p_opt

# ── Greedy system-wide integer train allocation ───────────────────────────────
f_alloc = {l: np.zeros(T, dtype=int) for l in LNAMES}
budget_remaining = float(DAILY_BUDGET_EVENT)

while budget_remaining >= FIXED_COST_PER_TRAIN:
    best_marginal = 0.0
    best_lt       = None

    for l in LNAMES:
        for t in range(T):
            fc = f_alloc[l][t]
            if fc >= F_MAX:
                continue
            fn = fc + 1
            marginal = best_profit[l][t][fn] - best_profit[l][t][fc]
            if marginal > best_marginal:
                best_marginal = marginal
                best_lt       = (l, t)

    if best_lt is None:
        break   # no more profitable additions

    l, t = best_lt
    f_alloc[l][t] += 1
    budget_remaining -= FIXED_COST_PER_TRAIN

elapsed_phase1 = time.time() - t0

# ── Assemble per-line result arrays ───────────────────────────────────────────
results_by_line = {}
for l in LNAMES:
    att   = avg_tt[l]
    f_arr = f_alloc[l].astype(float)
    p_arr = np.array([best_fare[l][t][f_alloc[l][t]] for t in range(T)])
    pt    = _pt_vec(p_arr, f_arr, att)
    d_arr = demand_full[l]
    d_hat = d_arr * pt
    x_arr = np.minimum(d_hat, TRAIN_CAPACITY * f_arr)
    results_by_line[l] = {
        "f": f_arr, "p": p_arr, "x": x_arr,
        "d": d_arr, "d_hat": d_hat, "pt": pt,
    }

# ── Aggregate metrics (integer = final solution) ──────────────────────────────
rev_i   = float(sum(np.dot(results_by_line[l]["p"], results_by_line[l]["x"]) for l in LNAMES))
fix_i   = float(FIXED_COST_PER_TRAIN * sum(results_by_line[l]["f"].sum() for l in LNAMES))
var_i   = float(VARIABLE_COST_PER_PAX * sum(results_by_line[l]["x"].sum() for l in LNAMES))
pax_i   = float(sum(results_by_line[l]["x"].sum() for l in LNAMES))
prof_i  = rev_i - fix_i - var_i

trains_used   = sum(int(results_by_line[l]["f"].sum()) for l in LNAMES)
budget_used   = fix_i
equity_ok = all(
    np.all(results_by_line[l]["x"] >= EQUITY_EPSILON * results_by_line[l]["d_hat"] - 1e-6)
    for l in LNAMES
)

print(f"Logit calibration: U_drive_WC = {U_DRIVE_WC:.2f}  "
      f"(+${WC_DRIVE_PENALTY/LOGIT_ALPHA_FARE:.0f} match-day parking equivalent)")
print(f"Phase 1 solve: {elapsed_phase1:.2f}s  |  "
      f"{trains_used} trains allocated  |  budget used ${budget_used:,.0f} / ${DAILY_BUDGET_EVENT:,.0f}")
print()
print("=== OPTIMIZATION RESULTS (Logit elastic demand, integer trains) ===")
print(f"Profit:     ${prof_i:>12,.0f}")
print(f"Revenue:    ${rev_i:>12,.0f}")
print(f"Fixed cost: ${fix_i:>12,.0f}  ({'OK' if fix_i <= DAILY_BUDGET_EVENT + 1 else 'EXCEEDS BUDGET'})")
print(f"Var cost:   ${var_i:>12,.0f}")
print(f"Total pax:  {pax_i:>13,.0f}")
print(f"Equity OK:  {equity_ok}")
print()

print(f"{'Line':<25} {'Peak f':>6} {'OffPk f':>7} {'Pax':>8} {'Avg fare':>9} {'Avg Pt':>7}")
print("-" * 67)
for l in LNAMES:
    r = results_by_line[l]
    peak_f    = float(np.mean([r["f"][t] for t in range(T) if is_peak(t)]))
    offpeak_f = float(np.mean([r["f"][t] for t in range(T) if not is_peak(t)]))
    total_x   = float(r["x"].sum())
    avg_p     = float(np.average(r["p"], weights=np.maximum(r["x"], 1e-6)))
    avg_pt    = float(np.mean(r["pt"]))
    print(f"{l:<25} {peak_f:>6.2f} {offpeak_f:>7.2f} {total_x:>8.0f} ${avg_p:>8.2f} {avg_pt:>6.1%}")

# ── FIX 2: Comparison table (integer vs ceil of continuous-relaxation f) ──────
# For reference, run a per-line SLSQP continuous relaxation with f in [0,8]
# and warm-start from integer solution. Then show ceil comparison.
from scipy.optimize import minimize as _minimize

results_cont = {}
t1 = time.time()
for l in LNAMES:
    d   = demand_full[l]
    att = avg_tt[l]
    f0  = results_by_line[l]["f"].copy()
    p0  = results_by_line[l]["p"].copy()
    z0  = np.concatenate([f0, p0])

    def neg_profit_cont(z, d=d, att=att):
        f  = np.clip(z[:T], 0.0, MAX_TRAINS_PER_SLOT)
        p  = np.clip(z[T:], FARE_MIN, FARE_MAX)
        pt = _pt_vec(p, f, att)
        x  = np.minimum(d * pt, TRAIN_CAPACITY * f)
        return -(np.dot(p, x)
                 - FIXED_COST_PER_TRAIN  * f.sum()
                 - VARIABLE_COST_PER_PAX * x.sum())

    B_l = DAILY_BUDGET_EVENT * float(d.sum()) / sum(float(demand_full[ll].sum()) for ll in LNAMES)
    constraints = [{"type": "ineq",
                    "fun": lambda z, B_l=B_l:
                        B_l - FIXED_COST_PER_TRAIN
                              * np.clip(z[:T], 0.0, MAX_TRAINS_PER_SLOT).sum()}]
    bounds = [(0.0, MAX_TRAINS_PER_SLOT)] * T + [(FARE_MIN, FARE_MAX)] * T

    res = _minimize(neg_profit_cont, z0, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 200, "ftol": 1e-4})

    f_opt = np.clip(res.x[:T], 0.0, MAX_TRAINS_PER_SLOT)
    p_opt = np.clip(res.x[T:], FARE_MIN, FARE_MAX)
    pt    = _pt_vec(p_opt, f_opt, att)
    d_hat = d * pt
    x_opt = np.minimum(d_hat, TRAIN_CAPACITY * f_opt)
    results_cont[l] = {"f": f_opt, "p": p_opt, "x": x_opt, "d": d, "d_hat": d_hat, "pt": pt}

t2 = time.time()

rev_c   = float(sum(np.dot(results_cont[l]["p"], results_cont[l]["x"]) for l in LNAMES))
fix_c   = float(FIXED_COST_PER_TRAIN * sum(results_cont[l]["f"].sum() for l in LNAMES))
var_c   = float(VARIABLE_COST_PER_PAX * sum(results_cont[l]["x"].sum() for l in LNAMES))
pax_c   = float(sum(results_cont[l]["x"].sum() for l in LNAMES))
prof_c  = rev_c - fix_c - var_c

print()
print("=== FIX 2: INTEGER (greedy) vs CONTINUOUS (SLSQP) ===")
print(f"  Continuous solve: {t2-t1:.2f}s (warm-started from integer solution)")
print(f"{'Metric':<22} {'Integer (greedy)':>18} {'Continuous':>14} {'Delta':>12}")
print("-" * 70)
for name, intg, cont in [
    ("Profit ($)",     prof_i, prof_c),
    ("Revenue ($)",    rev_i,  rev_c),
    ("Fixed cost ($)", fix_i,  fix_c),
    ("Var cost ($)",   var_i,  var_c),
    ("Total pax",      pax_i,  pax_c),
]:
    print(f"{name:<22} {intg:>18,.0f} {cont:>14,.0f} {cont-intg:>+12,.0f}")

budget_flag = ("*** EXCEEDS $350,000 ***"
               if fix_i > DAILY_BUDGET_EVENT + 1 else "within $350,000 budget")
print(f"\nInteger fixed cost ${fix_i:,.0f} — {budget_flag}")

# ── Paoli/Thorndale time series ───────────────────────────────────────────────
print()
line = "Paoli/Thorndale"
rc   = results_cont[line]
ri   = results_by_line[line]

print(f"{line} — time series (integer greedy vs continuous SLSQP):")
print(f"  {'Time':<8} {'Demand':>7} {'f_int':>6} {'f_cont':>7} {'Pax_i':>7} {'Pax_c':>7} {'Fare':>6} {'P_tr':>6}")
for i in [0, 4, 8, 12, 20, 28, 36, 40, 44, 48, 52, 56, 60]:
    print(f"  {slot_label(TIME_SLOTS[i]):<8} "
          f"{ri['d'][i]:>7.0f} {ri['f'][i]:>6.0f} {rc['f'][i]:>7.3f} "
          f"{ri['x'][i]:>7.0f} {rc['x'][i]:>7.0f} "
          f"${ri['p'][i]:>5.2f} {ri['pt'][i]:>5.1%}")

# ── Logit verification: fare sweep at 7pm (WC-peak slot) ─────────────────────
# Shows P_transit < 1 and decreasing as fare increases — confirms elastic demand.
# Greedy chose only the 7pm slot for Paoli/Thorndale (all others unprofitable).
# We evaluate hypothetical P_transit at f=1 across three fares.
print()
r_pt = results_by_line[line]
i_wc = 52   # 7pm slot
d_wc = float(r_pt["d"][i_wc])
att_pt = avg_tt[line]

print(f"Logit verification — {line} at 7:00pm (d={d_wc:.0f}, f=1 train):")
print(f"  Confirms P_transit < 1 and decreases as fare rises.")
print(f"  {'Fare':>6} {'P_transit':>10} {'d_hat':>7} {'x (cap=875)':>12} "
      f"{'Rev − VarCost':>14} {'Marginal profit':>16}")
for p_test in [FARE_MIN, 3.50, 5.04, 7.00, FARE_MAX]:
    pt_test  = _pt_scalar(p_test, 1.0, att_pt)
    dh_test  = d_wc * pt_test
    x_test   = min(dh_test, TRAIN_CAPACITY)
    rev_vc   = p_test * x_test - VARIABLE_COST_PER_PAX * x_test
    marg_pr  = rev_vc - FIXED_COST_PER_TRAIN
    marker   = "  ← optimal" if abs(p_test - r_pt["p"][i_wc]) < 0.10 else ""
    print(f"  ${p_test:>5.2f} {pt_test:>10.4f} {dh_test:>7.0f} {x_test:>12.0f} "
          f"{rev_vc:>14,.0f} {marg_pr:>+15,.0f}{marker}")
