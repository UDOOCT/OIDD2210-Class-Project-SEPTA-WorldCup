# Mathematical Formulation
## SEPTA Regional Rail — World Cup 2026 Bilevel Network Optimization
### OIDD 2210 Final Project

---

## 0. System Overview

The model has three algorithmic layers, each implemented as a separate module:

```
data/demand.py          →  Demand model (base ridership + World Cup overlay)
models/upper_level.py   →  Upper-level: SEPTA's profit-maximization (SLSQP)
models/lower_level.py   →  Lower-level: passenger Logit route choice
models/bilevel.py       →  Iterative best-response bilevel solver
_run_optimization.py    →  Greedy integer allocation with elastic demand
_run_ilp_comparison.py  →  Multiple-choice knapsack ILP (PuLP/CBC)
models/sensitivity.py   →  Stochastic search (Optuna TPE + Monte Carlo)
```

**Decision hierarchy:**
1. SEPTA chooses train frequencies $f_{lt}$ and fares $p_{lt}$ to maximize profit
2. Passengers observe $(f^*, p^*)$ and route via Multinomial Logit, producing actual demand $x_{lt}$
3. SEPTA's profit depends on passengers' equilibrium choices → bilevel structure

---

## 1. Sets and Indices

| Symbol | Definition | Size |
|---|---|---|
| $\mathcal{L}$ | Regional Rail lines | $|\mathcal{L}|$ = 13 (12 active per GTFS) |
| $\mathcal{T}$ | 15-minute time slots, $t \in \{0, 1, \ldots, 60\}$ | $|\mathcal{T}|$ = 61 |
| $\mathcal{B}$ | Named time blocks: morning, midday, evening, night | $|\mathcal{B}|$ = 4 |
| $\mathcal{S}$ | All stations (unique across lines) | $|\mathcal{S}| \approx 140$ |

**Time slot mapping** (slot index $t$ → decimal hour $h_t = 6.0 + 0.25t$):

| Block $b$ | Slot range | Hours | Slots |
|---|---|---|---|
| morning | $t \in [0, 11]$ | 6:00am – 9:00am | 12 |
| midday  | $t \in [12, 39]$ | 9:00am – 4:00pm | 28 |
| evening | $t \in [40, 51]$ | 4:00pm – 7:00pm | 12 |
| night   | $t \in [52, 60]$ | 7:00pm – 9:00pm | 9 |

**Lines** (source: SEPTA GTFS v202603296 + `data/ridership/ridership_by_line.json`):

| Line | Avg Fare | FY2024 Weekday Riders |
|---|---|---|
| Airport | $5.25 | from GTFS |
| Chestnut Hill East | $4.25 | from GTFS |
| Chestnut Hill West | $4.25 | from GTFS |
| Cynwyd | $4.25 | from GTFS |
| Fox Chase | $5.25 | from GTFS |
| Lansdale/Doylestown | $5.25 | from GTFS |
| Manayunk/Norristown | $4.25 | from GTFS |
| Media/Wawa | $5.25 | from GTFS |
| Paoli/Thorndale | $5.50 | from GTFS |
| Trenton | $6.00 | from GTFS |
| Warminster | $5.25 | from GTFS |
| West Trenton | $5.75 | from GTFS |
| Wilmington/Newark | $6.00 | from GTFS |

Total FY2024 weekday boardings: **48,343** across all lines.

---

## 2. Parameters

### Cost Parameters
*(Source: SEPTA OpenDataPhilly Route Operating Statistics FY2024 Budgeted)*

| Symbol | Definition | Value |
|---|---|---|
| $c_f$ | Fixed cost per train-trip | $\$1{,}691.98$ ($= \$13{,}535.85 \div 8$ trips/day) |
| $c_v$ | Marginal variable cost per passenger | $\$0.40$ (incremental; all-in avg is $\$34.03$) |
| $B$ | Operating budget cap (event day) | $\$350{,}000$ |
| $B_{\text{normal}}$ | Operating budget cap (normal day) | $\$250{,}000$ |

### Fleet Parameters
*(Bombardier Multilevel consist)*

| Symbol | Definition | Value |
|---|---|---|
| $C$ | Train seated capacity | $875$ ($= 175 \text{ seats} \times 5 \text{ cars}$) |
| $C_{\max}$ | Train crush-load capacity | $1{,}137$ ($= C \times 1.3$) |
| $\bar{f}$ | Max trains per 15-min slot | $8$ |
| $\underline{f}$ | Min trains per slot (peak) | $1$ |
| $\underline{f}$ | Min trains per slot (off-peak) | $0$ |

### Fare Parameters

| Symbol | Definition | Value |
|---|---|---|
| $p_{\min}$ | Minimum fare (system-wide) | $\$2.50$ |
| $p_{\max}$ | Maximum allowed fare | $\$9.00$ |
| $\sigma$ | Maximum surge multiplier over base fare | $1.5$ |
| $\Delta p_{\max}$ | Max fare change between consecutive slots | $\$1.00$ |
| $\bar{p}_l$ | Line-specific base fare | $\$4.25$–$\$6.00$ (zone-based) |

### Equity Parameter

| Symbol | Definition | Value |
|---|---|---|
| $\varepsilon$ | Minimum fraction of demand that must be served | $0.80$ |

### Logit Calibration Parameters
*(Small & Verhoef 2007)*

| Symbol | Definition | Value |
|---|---|---|
| $\alpha_1$ | Fare disutility coefficient | $0.50$ |
| $\alpha_2$ | Wait time disutility coefficient | $0.30$ |
| $\alpha_3$ | In-vehicle travel time coefficient | $0.15$ |
| $\theta$ | Logit scale parameter | $0.80$ |
| $U_0$ | Utility of not traveling (outside option) | $-1.50$ |
| $\delta_{\text{WC}}$ | World Cup driving penalty (match-day parking + traffic) | $3.50$ (≈ $\$7$ parking equivalent at $\alpha_1 = 0.50$) |

---

## 3. Demand Model (`data/demand.py`)

### 3.1 Base Ridership — Bimodal Normal Mixture

Per-slot demand is shaped by a bimodal intraday profile, scaled to FY2024 weekday totals.

$$w(h) = \sum_{k=1}^{3} w_k \cdot \phi\!\left(\frac{h - \mu_k}{\sigma_k}\right)$$

| Component $k$ | $\mu_k$ (hour) | $\sigma_k$ (hr) | $w_k$ | Interpretation |
|---|---|---|---|---|
| 1 | 8.00 | 0.75 | 0.40 | Morning rush peak |
| 2 | 17.50 | 0.80 | 0.40 | Evening rush peak |
| 3 | 12.50 | 1.50 | 0.20 | Midday shoulder |

The normalized profile is $\pi_t = w(h_t) \,/\, \sum_{s} w(h_s)$.

Base demand per (line, slot):

$$d^{\text{base}}_{l,t} = R_l \cdot \pi_t$$

where $R_l$ is FY2024 weekday boardings for line $l$ from `ridership_by_line.json`.

### 3.2 World Cup Demand Overlay

Pre-game fan arrivals modeled as a truncated Normal centered at 7:15pm (1h 45min before 9pm kickoff):

$$\mu_{\text{WC}} = 21.0 + (-1.75) = 19.25 \text{ hr}, \quad \sigma_{\text{WC}} = 0.75 \text{ hr}$$

$$d^{\text{WC}}_{l,t} = N_{\text{fans}} \cdot \omega_l \cdot \frac{\phi\!\left(\frac{h_t - \mu_{\text{WC}}}{\sigma_{\text{WC}}}\right)}{\sum_s \phi\!\left(\frac{h_s - \mu_{\text{WC}}}{\sigma_{\text{WC}}}\right)} \cdot \Delta t$$

where:
- $N_{\text{fans}} = 45{,}000$ total fans using transit (out of 69,328 capacity at Lincoln Financial Field)
- $\omega_l = \max_{s \in \text{stations}(l)} \gamma_s$ is the line's World Cup gateway weight
- $\Delta t = 0.25$ hr (slot duration)
- $\gamma_s$ are gateway weights for key stations (Jefferson: 0.25, Suburban: 0.20, 30th St: 0.15, etc.)

### 3.3 Total Demand

$$d_{l,t} = d^{\text{base}}_{l,t} + d^{\text{WC}}_{l,t} \qquad \forall l \in \mathcal{L},\ t \in \mathcal{T}$$

### 3.4 Monte Carlo Demand Scenarios (`monte_carlo_demand`)

Uncertain parameters drawn per scenario:

| Parameter | Distribution |
|---|---|
| $N_{\text{fans}}$ | $\mathcal{N}(45{,}000,\ 5{,}000^2)$, clipped to $[30{,}000,\ 65{,}000]$ |
| Base total ridership | $\mathcal{N}(48{,}343,\ 4{,}000^2)$, clipped to $[36{,}000,\ 62{,}000]$ |
| Pre-game peak offset | $\mathcal{N}(-1.75,\ 0.20^2)$ hours before kickoff |
| Pre-game spread $\sigma_{\text{WC}}$ | $|\mathcal{N}(0.75,\ 0.10^2)|$ hours |

Returns $N = 500$ demand scenario dicts by default; sensitivity analysis uses $N = 100$.

---

## 4. Decision Variables

$$f_{l,t} \in \mathbb{Z}^+, \quad 0 \leq f_{l,t} \leq 8 \qquad \forall l \in \mathcal{L},\ t \in \mathcal{T} \quad \text{(trains dispatched)}$$

$$p_{l,t} \in \mathbb{R}^+, \quad 2.50 \leq p_{l,t} \leq 9.00 \qquad \forall l \in \mathcal{L},\ t \in \mathcal{T} \quad \text{(fare per trip, \$)}$$

$$x_{l,t} \in \mathbb{R}^+ \qquad \forall l \in \mathcal{L},\ t \in \mathcal{T} \quad \text{(passengers served)}$$

**Variable count (upper level):** $13 \times 61 \times 3 = 2{,}379$ (12 active lines: $12 \times 61 \times 3 = 2{,}196$).

Flat index used in code: $\text{idx}(l, t) = \text{lidx}[l] \cdot T + t$, giving vectors of length $N = L \times T$.

---

## 5. Upper-Level Problem: SEPTA's Profit Maximization

### 5.1 Objective

$$\max_{f,\, p,\, x} \quad \Pi = \underbrace{\sum_{l \in \mathcal{L}} \sum_{t \in \mathcal{T}} p_{l,t} \cdot x_{l,t}}_{\text{Revenue}} - \underbrace{\sum_{l \in \mathcal{L}} \sum_{t \in \mathcal{T}} c_f \cdot f_{l,t}}_{\text{Fixed operating cost}} - \underbrace{\sum_{l \in \mathcal{L}} \sum_{t \in \mathcal{T}} c_v \cdot x_{l,t}}_{\text{Variable cost}}$$

### 5.2 Constraints

$$\text{(C1) Capacity:} \quad x_{l,t} \leq C \cdot f_{l,t} \qquad \forall l, t$$

$$\text{(C2) Demand cap:} \quad x_{l,t} \leq d_{l,t} \qquad \forall l, t$$

$$\text{(C3) Budget:} \quad \sum_{l} \sum_{t} c_f \cdot f_{l,t} \leq B$$

$$\text{(C4) Minimum service:} \quad f_{l,t} \geq \begin{cases} 1 & \text{if } t \in \text{morning} \cup \text{evening (peak)} \\ 0 & \text{otherwise} \end{cases} \qquad \forall l$$

$$\text{(C5) Equity } (\varepsilon\text{-constraint):} \quad x_{l,t} \geq \varepsilon \cdot d_{l,t} \qquad \forall l, t \quad (\varepsilon = 0.80)$$

$$\text{(C6) Fare bounds:} \quad p_{\min} \leq p_{l,t} \leq p_{\max} \qquad \forall l, t$$

$$\text{(C7) Fare smoothness:} \quad |p_{l,t} - p_{l,t-1}| \leq \Delta p_{\max} \qquad \forall l,\ t \geq 1 \quad (\Delta p_{\max} = \$1.00)$$

$$\text{(C8) Integrality:} \quad f_{l,t} \in \mathbb{Z}^+ \qquad \forall l, t$$

**Note on (C1)–(C2):** Both constraints encode $x_{l,t} = \min(d_{l,t},\ C \cdot f_{l,t})$ as a linear upper bound, which is tight at optimality because revenue is monotone in $x$.

**Note on bilinearity:** The term $p_{l,t} \cdot x_{l,t}$ is bilinear (product of two decision variables). The SLSQP solver handles this via gradient-based nonlinear programming on the continuous relaxation.

### 5.3 SLSQP Implementation (`_solve_scipy`)

The problem is solved as a **continuous relaxation** (C8 dropped; $f \in \mathbb{R}^+$) via `scipy.optimize.minimize` with `method='SLSQP'`.

**Decision vector:** $\mathbf{x}_{\text{vec}} = [f_0, \ldots, f_{N-1},\ p_0, \ldots, p_{N-1}] \in \mathbb{R}^{2N}$, $N = L \times T$.

**Objective (negated for minimization):**
```
neg_profit(x_vec):
    f = clip(x_vec[:N], f_min, 8)
    p = clip(x_vec[N:], 2.50, 9.00)
    x = minimum(d_vec, 875 * f)
    return -(dot(p, x) - 1691.98 * sum(f) - 0.40 * sum(x))
```

**Constraint encoding:**
- C3 (budget): single inequality $B - c_f \sum f \geq 0$
- C5 (equity): vector inequality $C \cdot f - \varepsilon \cdot d \geq 0$ (length $N$)
- C7 (fare smoothness): two sets of $L \times (T-1)$ inequalities:
  - $p_{l,t} - p_{l,t-1} + 1 \geq 0$ and $p_{l,t-1} - p_{l,t} + 1 \geq 0$

**Warm start:** $f_0 = 2$ for peak slots, $1$ for off-peak; $p_0 = \bar{p}_l$ (base fares).

**Solver settings:** `maxiter=3000`, `ftol=1e-8`.

---

## 6. Lower-Level Problem: Passenger Route Choice (`models/lower_level.py`)

Given SEPTA's decisions $(f^*, p^*)$, each passenger on line $l$ at slot $t$ faces a binary choice: transit vs. not traveling.

### 6.1 Generalized Travel Cost

$$G_{l,t} = \alpha_1 \cdot p_{l,t} + \alpha_2 \cdot \frac{h_{l,t}}{2} + \alpha_3 \cdot \bar{\tau}_l$$

where:
- $h_{l,t} = \Delta t_{\min} / f_{l,t}$ is the average headway (minutes), $\Delta t_{\min} = 15$
- $\bar{\tau}_l = \text{mean of } \texttt{LINES[l]["travel\_times"]}$ is average in-vehicle travel time (minutes, from GTFS stop_times)
- Wait time = $h_{l,t}/2$ (uniform distribution assumption)

### 6.2 Multinomial Logit Choice

Utility for transit option:

$$U^{\text{transit}}_{l,t} = -\theta \cdot G_{l,t}$$

Utility for outside option (no travel):

$$U^{\text{no-travel}} = \theta \cdot U_0 = 0.80 \times (-1.50) = -1.20$$

For numerical stability, utilities are shifted by their maximum before taking exp. Choice probability:

$$P^{\text{transit}}_{l,t} = \frac{\exp(U^{\text{transit}}_{l,t})}{\exp(U^{\text{transit}}_{l,t}) + \exp(U^{\text{no-travel}})}$$

### 6.3 Effective Demand

$$\hat{d}_{l,t} = d_{l,t} \cdot P^{\text{transit}}_{l,t}(p_{l,t},\ f_{l,t})$$

Passengers actually served (capacity-constrained):

$$x_{l,t} = \min\!\left(\hat{d}_{l,t},\ C \cdot f_{l,t}\right)$$

### 6.4 World Cup Logit Extension (`_run_optimization.py`)

On match day, driving is penalized by $\delta_{\text{WC}} = 3.50$ (parking + congestion ≈ \$7 equivalent):

$$U^{\text{drive}}_{\text{WC}} = U_0 - \delta_{\text{WC}} = -1.50 - 3.50 = -5.00$$

The binary logit becomes:

$$P^{\text{transit}}_{l,t} = \frac{1}{1 + \exp\!\left(\theta \cdot (U^{\text{drive}}_{\text{WC}} + G_{l,t})\right)}$$

This formulation is used in `_run_optimization.py` and `_run_ilp_comparison.py`.

---

## 7. Bilevel Formulation and Iterative Solver (`models/bilevel.py`)

### 7.1 Bilevel Program

$$\max_{f,\, p} \quad \Pi\!\left(f,\, p,\, \hat{x}(f, p)\right)$$

$$\text{subject to (C1)–(C8)}$$

where $\hat{x}_{l,t}(f,p)$ is the lower-level equilibrium:

$$\hat{x}_{l,t}(f, p) = \min\!\left(\hat{d}_{l,t}(p_{l,t}, f_{l,t}),\ C \cdot f_{l,t}\right)$$

This is a Mathematical Program with Equilibrium Constraints (MPEC). Exact reformulation via KKT conditions is non-convex; we solve via iterative best-response.

### 7.2 Iterative Best-Response Algorithm

```
Input:  d[l][t]  (raw exogenous demand, shape 13×61)
        max_iter = 40,  tol = 0.01

Initialize:  eff_demand ← d   (start from raw demand)

For iteration k = 1, 2, ..., max_iter:

  Step 1 (Upper level):
    result ← upper_solve(eff_demand)     # SLSQP on current effective demand
    Extract f*[l][t], p*[l][t] from result

  Step 2 (Lower level — demand update):
    For each l, t:
      h_lt  = 15 / max(f*[l][t], ε)     # headway in minutes
      new_d[l][t] = effective_demand(d[l][t], p*[l][t], h_lt, τ_l)
    Δ = ||new_d − eff_demand||₁

  Step 3 (Convergence check):
    If Δ < tol × ||d||₁:
      STOP — converged
    Else:
      eff_demand ← new_d

Return:  result, convergence flag, iteration history
```

**Convergence criterion:** $\|\hat{d}^{(k)} - \hat{d}^{(k-1)}\|_1 < 0.01 \times \sum_{l,t} d_{l,t}$

The algorithm alternates between SEPTA's best response (upper) and passengers' best response (lower), guaranteed to converge when both best-response correspondences are continuous and the joint feasible set is compact (Facchinei & Pang 2003).

---

## 8. Greedy Integer Allocation with Elastic Demand (`_run_optimization.py`)

This module implements a **two-phase greedy algorithm** that produces integer train counts with elastic (Logit-based) demand.

### 8.1 Profit Table (Phase 1)

For each $(l, t, f) \in \mathcal{L} \times \mathcal{T} \times \{1, \ldots, 8\}$, compute the optimal fare and resulting slot profit via 1-D search:

$$\pi^*(l, t, f) = \max_{p \in [p_{\min},\, p_{\max}]} \left[ p \cdot \min\!\left(d_{l,t} \cdot P^{\text{transit}}(p, f, \bar{\tau}_l),\ C \cdot f\right) - c_v \cdot x_{l,t} - c_f \cdot f \right]$$

$$p^*(l, t, f) = \arg\max_{p} [\cdot]$$

Solved via `scipy.optimize.minimize_scalar` with `method='bounded'`, tolerance $10^{-4}$.

### 8.2 Greedy Allocation

$$f_{l,t} \leftarrow 0 \quad \forall l, t; \qquad \text{budget\_remaining} \leftarrow B$$

**While** budget\_remaining $\geq c_f$:

$$(\ell^*, \tau^*) = \arg\max_{(l,t):\ f_{l,t} < 8} \left[ \pi^*(l, t, f_{l,t}+1) - \pi^*(l, t, f_{l,t}) \right]$$

$$f_{\ell^*, \tau^*} \mathrel{+}= 1; \qquad \text{budget\_remaining} \mathrel{-}= c_f$$

If no $(l,t)$ has positive marginal profit, stop early.

This is equivalent to a **greedy knapsack** on the marginal profit increments, which achieves a $(1 - 1/e)$ approximation guarantee for submodular functions; in practice it matches the ILP optimum (see Section 9.3).

### 8.3 Properties

- Phase 1 runtime: $O(L \cdot T \cdot F_{\max})$ scalar minimizations (≈2.5s for 13×61×8)
- Greedy runtime: $O(B / c_f \cdot L \cdot T)$ comparisons
- Produces **integer** $f$ directly; no rounding needed
- Fares are optimal per-slot given integer $f$

---

## 9. Multiple-Choice Knapsack ILP (`_run_ilp_comparison.py`)

### 9.1 Formulation

Reusing the profit table $\pi^*(l, t, f)$ from Phase 1, define binary variable:

$$z_{l,t,f} \in \{0, 1\} \qquad \forall l \in \mathcal{L},\ t \in \mathcal{T},\ f \in \{0, 1, \ldots, 8\}$$

$$\max_{z} \quad \sum_{l,t,f} \pi^*(l, t, f) \cdot z_{l,t,f}$$

$$\text{s.t.} \quad \sum_{f=0}^{8} z_{l,t,f} = 1 \qquad \forall l, t \quad \text{(one train-count per slot)}$$

$$\sum_{l,t,f} c_f \cdot f \cdot z_{l,t,f} \leq B \quad \text{(global budget)}$$

$$z_{l,t,f} = 0 \quad \text{if } C \cdot f < \varepsilon \cdot \hat{d}_{l,t} \quad \text{(equity pre-exclusion)}$$

$$z_{l,t,f} \in \{0, 1\}$$

**Variable count:** $13 \times 61 \times 9 = 7{,}137$ binary variables.
**Constraints:** $13 \times 61 = 793$ one-hot + 1 budget + equity exclusions.

### 9.2 Solver

Solved via **PuLP + CBC** (bundled COIN-B solver). For the single-line subproblem (Part 1), solves in milliseconds. For the full 13-line system (Part 2), solved with 120-second time limit.

### 9.3 Greedy vs. ILP Comparison (Part 3)

The ratio $\Pi^{\text{greedy}} / \Pi^{\text{ILP}}$ measures greedy approximation quality. In practice, the marginal profit increments $\pi^*(l,t,f+1) - \pi^*(l,t,f)$ are **diminishing in $f$** (concave profit), making the greedy exchange property hold exactly and the greedy solution globally optimal.

---

## 10. Sensitivity Analysis — Stochastic Search (`models/sensitivity.py`)

### 10.1 Policy Parametrization

Instead of optimizing all $L \times T = 732$ fare and frequency values independently, we parametrize with an 8-dimensional **4-block policy**:

$$\boldsymbol{\theta} = \left( p^b_{\text{morning}},\ p^b_{\text{midday}},\ p^b_{\text{evening}},\ p^b_{\text{night}},\ f^b_{\text{morning}},\ f^b_{\text{midday}},\ f^b_{\text{evening}},\ f^b_{\text{night}} \right)$$

Per-slot values are expanded from block values:

$$p_{l,t} = p^b_{b(t)}, \qquad f_{l,t} = f^b_{b(t)}$$

where $b(t) \in \{\text{morning, midday, evening, night}\}$ is the block of slot $t$.

### 10.2 Monotonicity Constraints

The Optuna sampler enforces structural constraints:

$$p^b_{\text{evening}} \geq p^b_{\text{midday}}, \quad p^b_{\text{morning}} \geq p^b_{\text{midday}} \qquad \text{(peak surge)}$$

$$f^b_{\text{evening}} \geq f^b_{\text{midday}}, \quad f^b_{\text{morning}} \geq f^b_{\text{midday}}, \quad f^b_{\text{night}} \leq f^b_{\text{evening}} \qquad \text{(service scaling)}$$

**Search bounds:**

| Parameter | Range |
|---|---|
| $p^b_{\text{midday}}$ | $[\$2.50,\ \$5.00]$ |
| $p^b_{\text{morning}}, p^b_{\text{evening}}$ | $[p^b_{\text{midday}},\ \$9.00]$ |
| $p^b_{\text{night}}$ | $[\$2.50,\ p^b_{\text{evening}}]$ |
| $f^b_{\text{midday}}$ | $\{1, 2, 3, 4\}$ |
| $f^b_{\text{morning}}, f^b_{\text{evening}}$ | $\{f^b_{\text{midday}}, \ldots, 8\}$ |
| $f^b_{\text{night}}$ | $\{1, \ldots, f^b_{\text{evening}}\}$ |

### 10.3 Stochastic Objective (Sample Average Approximation)

For each candidate policy $\boldsymbol{\theta}$, evaluate over $N_{\text{MC}} = 100$ Monte Carlo demand scenarios:

$$\hat{J}(\boldsymbol{\theta}) = \frac{1}{N_{\text{MC}}} \sum_{s=1}^{N_{\text{MC}}} \Pi\!\left(\boldsymbol{\theta},\ \tilde{d}^{(s)}\right) - \lambda_B \cdot \max\!\left(0,\ \hat{B}(\boldsymbol{\theta}) - B\right) - \lambda_E \cdot \max\!\left(0,\ 0.95 - \hat{q}_E(\boldsymbol{\theta})\right)$$

where:
- $\hat{B}(\boldsymbol{\theta}) = c_f \cdot \sum_{t} f^b_{b(t)} \cdot L$ is the implied budget spend
- $\hat{q}_E(\boldsymbol{\theta})$ is the fraction of MC scenarios satisfying $\varepsilon$-equity
- $\lambda_B = 20$ (budget penalty multiplier)
- $\lambda_E = 10^7$ (equity penalty multiplier, hard constraint approximation)

### 10.4 Optuna TPE Algorithm

**Sampler:** Tree-structured Parzen Estimator (TPE), seed=42.
**Trials:** 200 function evaluations.
**Mechanism:** TPE fits separate kernel density estimators on good vs. bad trial regions, samples from the good region, and updates the model after each trial. Balances exploration and exploitation without gradient information.

---

## 11. Equity Analysis — $\varepsilon$-Constraint Pareto Frontier

To trace the profit–equity trade-off, solve parametrically over $\varepsilon \in [0, 1]$:

$$\Pi^*(\varepsilon) = \max_{f, p, x} \Pi \quad \text{subject to (C1)–(C4), (C6)–(C8)}$$

$$\text{and:} \quad x_{l,t} \geq \varepsilon \cdot d_{l,t} \quad \forall l, t$$

Each value of $\varepsilon$ gives one point on the Pareto frontier:
- $x$-axis: minimum service rate $\min_{l,t} \{x_{l,t} / d_{l,t}\}$
- $y$-axis: $\Pi^*(\varepsilon)$

At $\varepsilon = 0$: maximum profit (no equity constraint).
At $\varepsilon = 1$: maximum equity (serve all demand if budget allows).
Model uses $\varepsilon = 0.80$ as the operating constraint.

---

## 12. Data Provenance

| Data | Source | File | Used for |
|---|---|---|---|
| Line topology, station order, travel times | SEPTA GTFS v202603296 | `data/gtfs/` | $\bar{\tau}_l$, station lists |
| Weekday boardings by line (FY2024) | SEPTA OpenDataPhilly APC | `data/ridership/ridership_by_line.json` | $R_l$, base demand scaling |
| Per-station boarding shares (FY2024) | SEPTA OpenDataPhilly APC | `data/ridership/septa_rr_ridership.csv` | `STATION_RIDERSHIP_SHARES` |
| Route operating costs | SEPTA OpenDataPhilly FY2024 Budget | `data/costs/cost_summary.json` | $c_f$ ($\$13{,}535.85$/trainset/day) |
| Average all-in cost per passenger | Same as above | Same | Reference only; $c_v = \$0.40$ used |
| Zone-based fares | SEPTA official fare table | hardcoded in `data/network.py` | $\bar{p}_l$ |
| Match attendance capacity | FIFA / Lincoln Financial Field | — | $N_{\text{fans}} = 45{,}000$ |

---

## 13. Variable and Constraint Count Summary

| Model | Variables | Constraints |
|---|---|---|
| Upper level SLSQP | $2N = 2{,}196$ continuous | $N$ bounds + $1 + N + 2 \times L(T-1)$ |
| Bilevel (per iter) | same as upper | same + lower-level update |
| Sensitivity policy | 8 continuous/integer | monotonicity bounds + penalty terms |
| Greedy ILP | $L \times T$ integer (via table) | budget, greedy selection |
| Knapsack ILP | $L \times T \times 9 = 7{,}137$ binary | $L \times T$ one-hot + 1 budget |

where $N = L \times T = 12 \times 61 = 732$, $L = 12$, $T = 61$.
