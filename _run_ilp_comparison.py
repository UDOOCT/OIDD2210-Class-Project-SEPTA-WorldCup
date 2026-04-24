"""
ILP comparison: greedy vs PuLP/CBC optimal (Multiple-Choice Knapsack).

Mirrors _run_optimization.py exactly for Logit demand, profit tables,
and greedy algorithm — then replaces the greedy with a provably-optimal
ILP to measure the approximation gap.

Formulation (per line l, slot t, train count f_val ∈ {0..8}):
  Binary    z[l][t][f_val] ∈ {0, 1}
  Table     π[l][t][f_val] = max_p ( p·x − c_v·x − c_f·f )
                             x = min(d[l][t]·P_transit(p,f), 875·f)

  max   Σ_{l,t,f}  π[l][t][f] · z[l][t][f]
  s.t.  Σ_f z[l][t][f] = 1              ∀ l,t   (one choice per slot)
        Σ_{l,t,f} c_f · f · z ≤ B                (global budget)
        z[l][t][f] = 0  if equity violated         (pre-excluded)

Part 1 — Paoli/Thorndale only,  budget = DAILY_BUDGET_EVENT / 13
Part 2 — All 13 lines,          budget = DAILY_BUDGET_EVENT  (global)
Part 3 — Greedy quality assessment
"""

import sys, time
sys.path.insert(0, '.')

import numpy as np
from scipy.optimize import minimize_scalar

try:
    import pulp
except ImportError:
    print("PuLP not installed.  Run:  pip install pulp")
    sys.exit(1)

from data.demand import get_total_demand
from data.network import LINES
from data.parameters import (
    TIME_SLOTS, slot_label,
    TRAIN_CAPACITY, FIXED_COST_PER_TRAIN, VARIABLE_COST_PER_PAX,
    DAILY_BUDGET_EVENT, EQUITY_EPSILON,
    FARE_MIN, FARE_MAX, MAX_TRAINS_PER_SLOT,
    LOGIT_ALPHA_FARE, LOGIT_ALPHA_WAIT, LOGIT_ALPHA_TRAVEL,
    LOGIT_THETA, LOGIT_NO_TRAVEL_U,
)
from models.upper_level import LNAMES, T

# ── Constants (must mirror _run_optimization.py exactly) ──────────────────────
WC_DRIVE_PENALTY = 3.5
U_DRIVE_WC       = LOGIT_NO_TRAVEL_U - WC_DRIVE_PENALTY   # −5.00
F_MAX            = int(MAX_TRAINS_PER_SLOT)                # 8

demand_full = get_total_demand(worldcup=True)
avg_tt = {
    l: float(np.mean(LINES[l]["travel_times"])) if LINES[l]["travel_times"] else 10.0
    for l in LNAMES
}


# ── Logit (identical to _run_optimization.py) ─────────────────────────────────
def _pt(p: float, f: float, att: float) -> float:
    if f <= 0:
        return 0.0
    wait = 15.0 / (2.0 * max(f, 1e-6))
    G    = LOGIT_ALPHA_FARE * p + LOGIT_ALPHA_WAIT * wait + LOGIT_ALPHA_TRAVEL * att
    arg  = float(np.clip(LOGIT_THETA * (U_DRIVE_WC + G), -500.0, 500.0))
    return 1.0 / (1.0 + np.exp(arg))


# ── Profit / fare tables (same 1-D search as greedy) ─────────────────────────
# π[l][t][f] = max_p (p·x − c_v·x − c_f·f)  ←  total slot profit
# p_star[l][t][f] = argmax fare
print("Building Logit profit tables…")
t_pre = time.time()

PI     = {l: [[0.0]   * (F_MAX + 1) for _ in range(T)] for l in LNAMES}
P_STAR = {l: [[FARE_MIN] * (F_MAX + 1) for _ in range(T)] for l in LNAMES}

for l in LNAMES:
    att = avg_tt[l]
    for t in range(T):
        d_lt = float(demand_full[l][t])
        if d_lt < 0.5:
            continue
        for f in range(1, F_MAX + 1):
            cap = TRAIN_CAPACITY * f

            def _neg(pv, d=d_lt, f=f, att=att, cap=cap):
                x = min(d * _pt(float(pv), f, att), cap)
                return -(float(pv) * x
                         - VARIABLE_COST_PER_PAX * x
                         - FIXED_COST_PER_TRAIN  * f)

            res   = minimize_scalar(_neg, bounds=(FARE_MIN, FARE_MAX),
                                    method="bounded", options={"xatol": 1e-4})
            p_opt = float(np.clip(res.x, FARE_MIN, FARE_MAX))
            PI[l][t][f]     = -res.fun
            P_STAR[l][t][f] = p_opt

print(f"  done in {time.time() - t_pre:.2f}s\n")


# ── Greedy (replicated for per-line metrics) ──────────────────────────────────
def run_greedy(budget: float) -> dict:
    f_alloc  = {l: np.zeros(T, dtype=int) for l in LNAMES}
    remaining = float(budget)
    while remaining >= FIXED_COST_PER_TRAIN:
        best_marg, best_lt = 0.0, None
        for l in LNAMES:
            for t in range(T):
                fc = f_alloc[l][t]
                if fc >= F_MAX:
                    continue
                marg = PI[l][t][fc + 1] - PI[l][t][fc]
                if marg > best_marg:
                    best_marg, best_lt = marg, (l, t)
        if best_lt is None:
            break
        l, t = best_lt
        f_alloc[l][t] += 1
        remaining -= FIXED_COST_PER_TRAIN
    return f_alloc


def metrics(f_alloc: dict) -> dict:
    """Compute per-line and total profit/metrics from an integer f-allocation."""
    res = {"lines": {}}
    totals = dict(profit=0.0, revenue=0.0, fixed=0.0, var=0.0, pax=0.0)
    eq_ok  = True
    for l in LNAMES:
        att = avg_tt[l]
        lm  = dict(profit=0.0, revenue=0.0, fixed=0.0, trains=0, pax=0.0)
        for t in range(T):
            fv  = int(f_alloc[l][t])
            pv  = P_STAR[l][t][fv]
            dh  = float(demand_full[l][t]) * _pt(pv, fv, att)
            x   = min(dh, TRAIN_CAPACITY * fv)
            rev = pv * x
            fc  = FIXED_COST_PER_TRAIN * fv
            vc  = VARIABLE_COST_PER_PAX * x
            lm["profit"]  += rev - fc - vc
            lm["revenue"] += rev
            lm["fixed"]   += fc
            lm["trains"]  += fv
            lm["pax"]     += x
            totals["profit"]  += rev - fc - vc
            totals["revenue"] += rev
            totals["fixed"]   += fc
            totals["var"]     += vc
            totals["pax"]     += x
            if x < EQUITY_EPSILON * dh - 1e-6:
                eq_ok = False
        res["lines"][l] = {k: round(v, 2) if isinstance(v, float) else v
                           for k, v in lm.items()}
    res.update(
        profit    = round(totals["profit"],  2),
        revenue   = round(totals["revenue"], 2),
        fixed_cost= round(totals["fixed"],   2),
        var_cost  = round(totals["var"],     2),
        total_pax = round(totals["pax"]),
        trains    = sum(r["trains"] for r in res["lines"].values()),
        equity_ok = eq_ok,
    )
    return res


f_greedy  = run_greedy(DAILY_BUDGET_EVENT)
greedy_m  = metrics(f_greedy)


# ── Solver detection ──────────────────────────────────────────────────────────
def _solver(time_limit=None, msg=0):
    kw = {"msg": msg}
    if time_limit:
        kw["timeLimit"] = time_limit
    try:
        return pulp.PULP_CBC_CMD(**kw), "CBC (bundled)"
    except Exception:
        pass
    try:
        return pulp.getSolver("GLPK_CMD", **kw), "GLPK"
    except Exception:
        return None, None


# ── ILP builder helper ────────────────────────────────────────────────────────
def build_ilp(lines: list, budget: float, name: str):
    """
    Build a PuLP multiple-choice knapsack ILP for the given lines.
    Returns (prob, z_vars) where z_vars[l][t][f] are the binary variables.
    """
    safe = {l: l.replace("/", "_").replace(" ", "_").replace("-", "_")
            for l in lines}

    prob = pulp.LpProblem(name, pulp.LpMaximize)

    z = {l: [[pulp.LpVariable(f"z_{safe[l]}_t{t}_f{f}", cat="Binary")
              for f in range(F_MAX + 1)] for t in range(T)]
         for l in lines}

    # Pre-exclude equity-violating (l,t,f) choices
    obj_terms = []
    for l in lines:
        att = avg_tt[l]
        for t in range(T):
            d_lt = float(demand_full[l][t])
            for f in range(F_MAX + 1):
                if f > 0:
                    dh = d_lt * _pt(P_STAR[l][t][f], f, att)
                    if TRAIN_CAPACITY * f < EQUITY_EPSILON * dh - 1e-6:
                        prob += z[l][t][f] == 0
                        continue
                obj_terms.append(PI[l][t][f] * z[l][t][f])

    prob += pulp.lpSum(obj_terms), "TotalProfit"

    # One choice per (line, slot)
    for l in lines:
        for t in range(T):
            prob += (pulp.lpSum(z[l][t][f] for f in range(F_MAX + 1)) == 1,
                     f"onehot_{safe[l]}_{t}")

    # Budget
    prob += (pulp.lpSum(FIXED_COST_PER_TRAIN * f * z[l][t][f]
                        for l in lines for t in range(T)
                        for f in range(F_MAX + 1)) <= budget,
             "Budget")

    return prob, z


def extract_f(z, lines) -> dict:
    """Extract integer f-allocation from solved binary z-variables."""
    f_alloc = {l: np.zeros(T, dtype=int) for l in lines}
    for l in lines:
        for t in range(T):
            for f in range(F_MAX + 1):
                v = pulp.value(z[l][t][f])
                if v is not None and v > 0.5:
                    f_alloc[l][t] = f
                    break
    return f_alloc


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — Single-line ILP: Paoli/Thorndale
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("PART 1 — Single-line ILP: Paoli/Thorndale")
print("=" * 62)

LINE1   = "Paoli/Thorndale"
BUDGET1 = DAILY_BUDGET_EVENT / 13          # equal per-line share

prob1, z1 = build_ilp([LINE1], BUDGET1, "PT_ILP")
sv1, sname1 = _solver(msg=0)
if sv1 is None:
    print("No solver available."); sys.exit(1)

print(f"Solver : {sname1}")
print(f"Budget : ${BUDGET1:,.0f}  (DAILY_BUDGET / 13)")
print(f"Vars   : {T * (F_MAX + 1)} binary")

t1s = time.time()
prob1.solve(sv1)
t1e = time.time() - t1s

f_ilp1    = extract_f(z1, [LINE1])
ilp1_p    = float(sum(PI[LINE1][t][f_ilp1[LINE1][t]] for t in range(T)))
ilp1_tr   = int(f_ilp1[LINE1].sum())
g1_p      = float(sum(PI[LINE1][t][int(f_greedy[LINE1][t])] for t in range(T)))
g1_tr     = int(f_greedy[LINE1].sum())
gap1      = ilp1_p - g1_p
gap1_pct  = (gap1 / abs(ilp1_p) * 100) if abs(ilp1_p) > 1e-6 else 0.0

print(f"\nILP optimal profit (Paoli only):  ${ilp1_p:>12,.2f}")
print(f"Greedy profit (Paoli only):       ${g1_p:>12,.2f}")
print(f"Gap:                              ${gap1:>+12,.2f}  ({gap1_pct:+.2f}%)")
print(f"ILP solve time:                   {t1e:.3f}s")
print(f"ILP trains deployed:              {ilp1_tr}")
print(f"Greedy trains deployed:           {g1_tr}")
print(f"ILP status:                       {pulp.LpStatus[prob1.status]}")

# Slot-by-slot table — 10 highest-demand slots
top10 = sorted(range(T), key=lambda t: demand_full[LINE1][t], reverse=True)[:10]
print(f"\n{'Slot':<8} {'Demand':>7} {'G_f':>5} {'ILP_f':>6} "
      f"{'G_p':>7} {'ILP_p':>7} {'Δprofit':>9}")
print("-" * 56)
for t in sorted(top10):
    gf = int(f_greedy[LINE1][t]);  if_ = f_ilp1[LINE1][t]
    gp = P_STAR[LINE1][t][gf];    ip  = P_STAR[LINE1][t][if_]
    dp = PI[LINE1][t][if_] - PI[LINE1][t][gf]
    print(f"{slot_label(TIME_SLOTS[t]):<8} {demand_full[LINE1][t]:>7.0f} "
          f"{gf:>5} {if_:>6} ${gp:>6.2f} ${ip:>6.2f} {dp:>+9.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — Full 13-line ILP  (120 s limit, global budget)
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 62)
print("PART 2 — Full 13-line ILP  (120 s limit, global budget)")
print("=" * 62)

prob2, z2 = build_ilp(LNAMES, DAILY_BUDGET_EVENT, "Full13_ILP")
sv2, sname2 = _solver(time_limit=120, msg=1)

n_vars = len(LNAMES) * T * (F_MAX + 1)
print(f"Solver : {sname2}")
print(f"Budget : ${DAILY_BUDGET_EVENT:,.0f}")
print(f"Vars   : {n_vars:,} binary  |  "
      f"{len(LNAMES) * T} one-hot constraints  |  1 budget constraint")
print()
sys.stdout.flush()   # ensure Python output appears before CBC subprocess output

t2s = time.time()
prob2.solve(sv2)
t2e = time.time() - t2s

status2 = pulp.LpStatus[prob2.status]
print(f"\nILP2 status : {status2}  |  Solve time : {t2e:.2f}s")

f_ilp2  = extract_f(z2, LNAMES)
ilp2_m  = metrics(f_ilp2)

print()
print(f"{'Metric':<24} {'Greedy':>14} {'ILP (13-line)':>15} {'Difference':>13}")
print("─" * 68)

def _row(label, gv, iv):
    d   = iv - gv
    pct = f"  ({d / abs(gv) * 100:+.1f}%)" if abs(gv) > 1e-6 else ""
    print(f"{label:<24} {gv:>14,.0f} {iv:>15,.0f} {d:>+12,.0f}{pct}")

_row("Total profit ($)",  greedy_m["profit"],     ilp2_m["profit"])
_row("Total revenue ($)", greedy_m["revenue"],    ilp2_m["revenue"])
_row("Fixed cost ($)",    greedy_m["fixed_cost"], ilp2_m["fixed_cost"])
_row("Total pax served",  greedy_m["total_pax"],  ilp2_m["total_pax"])
_row("Budget used ($)",   greedy_m["fixed_cost"], ilp2_m["fixed_cost"])
_row("Trains deployed",   greedy_m["trains"],     ilp2_m["trains"])
print(f"{'Equity satisfied':<24} {str(greedy_m['equity_ok']):>14} "
      f"{str(ilp2_m['equity_ok']):>15}")
print(f"{'Solve time':<24} {'~0s (greedy)':>14} {t2e:>14.2f}s")


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — Greedy quality assessment
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 62)
print("PART 3 — Greedy quality assessment")
print("=" * 62)

g_p    = greedy_m["profit"]
i_p    = ilp2_m["profit"]
ratio  = (g_p / i_p * 100) if abs(i_p) > 1e-6 else float("nan")
abs_gap = i_p - g_p

print(f"\nGreedy achieves {ratio:.4f}% of ILP optimal")
print(f"Absolute gap:   ${abs_gap:+,.2f}")
if abs(i_p) > 1 and abs(abs_gap / i_p) < 1e-4:
    print("→ Greedy is globally optimal for this demand / cost structure.")

# Per-line breakdown
print(f"\n{'Line':<25} {'Greedy $':>10} {'ILP $':>10} {'Gap $':>9} {'Gap %':>7}")
print("-" * 65)

line_gaps = []
for l in LNAMES:
    gpl = greedy_m["lines"][l]["profit"]
    ipl = ilp2_m["lines"][l]["profit"]
    gap = ipl - gpl
    pct = (gap / abs(ipl) * 100) if abs(ipl) > 1e-6 else 0.0
    line_gaps.append((l, gpl, ipl, gap, pct))

line_gaps.sort(key=lambda r: abs(r[4]), reverse=True)

for l, gpl, ipl, gap, pct in line_gaps:
    print(f"{l:<25} {gpl:>10,.0f} {ipl:>10,.0f} {gap:>+9,.0f} {pct:>+6.2f}%")

# Interpretation of largest-gap line
l_w, g_w, i_w, gap_w, pct_w = line_gaps[0]
diff_t = max(range(T), key=lambda t: abs(int(f_ilp2[l_w][t]) - int(f_greedy[l_w][t])))
gi_f   = int(f_greedy[l_w][diff_t])
ii_f   = int(f_ilp2[l_w][diff_t])

print()
if abs(pct_w) < 0.01:
    print("All lines: greedy matches ILP exactly — optimal allocation confirmed.")
else:
    print(f"Largest gap line: {l_w}  (gap = ${gap_w:+,.0f}, {pct_w:+.2f}%)")
    print(f"  Greedy deploys {gi_f} train(s) vs ILP {ii_f} train(s) "
          f"at slot {slot_label(TIME_SLOTS[diff_t])}.")
    print(f"  The greedy missed a profitable reallocation because it evaluates "
          f"one marginal train at a time and cannot consider cross-slot swaps.")
