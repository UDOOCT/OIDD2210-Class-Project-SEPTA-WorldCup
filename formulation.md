# Mathematical Formulation
## SEPTA Multimodal World Cup 2026 Transit Optimization
### OIDD 2210 Final Project

---

## 0. System Overview

The project has two optimization layers sharing a common data foundation:

**Layer 1 — Regional Rail (v1, `main.py`)**
```
data/demand.py          →  Bimodal base demand + WC overlay + Monte Carlo
models/upper_level.py   →  Upper-level: SEPTA profit-maximization (SLSQP)
models/lower_level.py   →  Lower-level: passenger Multinomial Logit
models/bilevel.py       →  Iterative best-response bilevel solver
_run_optimization.py    →  Greedy integer allocator with elastic demand
_run_ilp_comparison.py  →  Multiple-choice knapsack ILP (PuLP/CBC)
models/sensitivity.py   →  Stochastic sensitivity search (Optuna TPE)
```

**Layer 2 — Multimodal extension (`run_scenarios.py`)**
```
data/scenario.py          →  Master config: 18:00–04:00 window, all parameters
data/worldcup_demand.py   →  Demand model: pre-game, in-game, post-game
data/bsl.py               →  BSL/B Line capacity model (NRG Station)
models/policy_objective.py →  Policy objective + greedy RR evaluator
reporting.py              →  KPI computation, display, and CSV export
run_scenarios.py          →  8-scenario comparison runner
```

---

## Part I — Regional Rail Bilevel Model (v1)

### 1. Sets and Indices

| Symbol | Definition | Size |
|--------|-----------|------|
| $\mathcal{L}$ | Regional Rail lines | 13 (12 active per GTFS) |
| $\mathcal{T}$ | 15-minute slots, $t \in \{0,\ldots,60\}$ | 61 |
| $\mathcal{B}$ | Named blocks: morning, midday, evening, night | 4 |
| $\mathcal{S}$ | Unique stations across all lines | ≈140 |

**Time slot mapping** ($h_t = 6.0 + 0.25t$ decimal hours, 6am–9pm):

| Block | Slot range | Hours |
|-------|-----------|-------|
| morning | $[0,\;11]$ | 6:00–9:00 |
| midday | $[12,\;39]$ | 9:00–16:00 |
| evening | $[40,\;51]$ | 16:00–19:00 |
| night | $[52,\;60]$ | 19:00–21:00 |

**Regional Rail lines and fare zones** (source: SEPTA GTFS v202603296):

| Line | Avg Zone Fare | FY2024 Daily Riders |
|------|--------------|---------------------|
| Airport | $5.25 | — |
| Chestnut Hill East | $4.25 | — |
| Chestnut Hill West | $4.25 | — |
| Cynwyd | $4.25 | — |
| Fox Chase | $5.25 | — |
| Lansdale/Doylestown | $5.25 | — |
| Manayunk/Norristown | $4.25 | — |
| Media/Wawa | $5.25 | — |
| Paoli/Thorndale | $5.50 | — |
| Trenton | $6.00 | — |
| Warminster | $5.25 | — |
| West Trenton | $5.50 | — |
| Wilmington/Newark | $5.75 | — |

---

### 2. Parameters (v1 RR model)

| Symbol | Value | Source / File |
|--------|-------|---------------|
| $c_f$ | $1,692.00/trip | `data/costs/cost_summary.json` |
| $c_v$ | $0.40/pax | Literature |
| $B$ | $350,000 | Assumption |
| $K$ | 875 seats | SEPTA 5-car Bombardier Multilevel |
| $f_{\max}$ | 8 trains/slot | Fleet limit |
| $\varepsilon$ | 0.80 | Equity coverage threshold |
| $p_{\min}$ | $2.50 | SEPTA minimum fare |
| $p_{\max}$ | $9.00 | Surge cap |
| $\Delta p_{\max}$ | $1.00/slot | Fare smoothness |
| $\theta$ | 1.0 | Logit temperature |
| $\alpha_1,\alpha_2,\alpha_3$ | 0.50, 0.30, 0.15 | Small & Verhoef (2007) |

---

### 3. Demand Model (v1)

Base demand is a bimodal normal mixture drawn from FY2024 APC weekday boardings, scaled per line by ridership share. World Cup overlay shifts evening demand upward and adds a return wave.

**Base demand for line $l$, slot $t$:**

$$d_{lt}^{\text{base}} = N_l \left[ 0.40 \cdot \phi\!\left(\frac{t - t_{\text{AM}}}{3}\right) + 0.20 \cdot \phi\!\left(\frac{t - t_{\text{mid}}}{6}\right) + 0.40 \cdot \phi\!\left(\frac{t - t_{\text{PM}}}{4}\right) \right]$$

where $N_l$ = FY2024 daily ridership for line $l$, $\phi(\cdot)$ = standard normal PDF, peaks at $t_{\text{AM}} = 8$, $t_{\text{mid}} = 22$, $t_{\text{PM}} = 40$ (slot indices).

**World Cup overlay** (additive, applied to all lines):

$$d_{lt}^{\text{WC}} = N_{\text{fans}} \cdot s_l \left[ w_1 \cdot \phi\!\left(\frac{t - t_{\text{pre}}}{3}\right) + w_2 \cdot \phi\!\left(\frac{t - t_{\text{post}}}{4}\right) \right]$$

where $N_{\text{fans}} \approx 45{,}000$ transit fans, $s_l$ = line ridership share, pre-game peak at 2 slots before kickoff, post-game peak ~4 slots after match end.

**Monte Carlo:** $N_{\text{fans}} \sim \mathcal{N}(45000, 5000^2)$; demand scale $\sim \mathcal{U}(0.8, 1.2)$. 100 scenarios per Optuna trial.

---

### 4. Decision Variables (v1)

Flat index: $\text{idx}(l,t) = \text{lidx}[l] \cdot T + t$ where $\text{lidx}$ maps line name to integer $\{0,\ldots,12\}$.

| Variable | Domain | Meaning |
|----------|--------|---------|
| $f_{lt}$ | $\mathbb{Z}_{\geq 0}$ | Trains deployed on line $l$, slot $t$ |
| $p_{lt}$ | $\mathbb{R}_{\geq 0}$ | Fare charged on line $l$, slot $t$ |
| $x_{lt}$ | $\mathbb{R}_{\geq 0}$ | Passengers served (Logit equilibrium) |

Total: $13 \times 61 \times 3 = 2,379$ variables (2,196 continuous + integrality rounding).

---

### 5. Upper-Level: SEPTA Profit Maximization (SLSQP)

$$\max_{f,p} \;\; \Pi = \sum_{l,t} p_{lt} x_{lt} \;-\; c_f \sum_{l,t} f_{lt} \;-\; c_v \sum_{l,t} x_{lt}$$

Subject to:

| # | Constraint | Expression |
|---|-----------|-----------|
| C1 | Capacity | $x_{lt} \leq K \cdot f_{lt} \;\forall l,t$ |
| C2 | Demand cap | $x_{lt} \leq d_{lt} \;\forall l,t$ |
| C3 | Budget | $\sum_{l,t} c_f f_{lt} \leq B$ |
| C4 | Min service | $f_{lt} \geq 1$ (peak slots), $\geq 0$ (off-peak) |
| C5 | Equity | $x_{lt} \geq \varepsilon \cdot d_{lt}$ (operative lines) |
| C6 | Fare bounds | $p_{\min} \leq p_{lt} \leq p_{\max}$ |
| C7 | Fare smoothness | $|p_{l,t} - p_{l,t-1}| \leq \Delta p_{\max}$ |
| C8 | Integrality | $f_{lt} \in \mathbb{Z}_{\geq 0}$ (relaxed for SLSQP) |

**Implementation:** `models/upper_level.py` uses `scipy.optimize.minimize(method='SLSQP')` with a flat vector of length 2,196 ($L \times T \times 2$ for $f$ and $p$; $x$ is determined by Logit). Integrality is enforced by rounding $f_{lt}$ to the nearest integer after solving.

---

### 6. Lower-Level: Multinomial Logit Passenger Choice

Given SEPTA's schedule $(f^*, p^*)$, passenger generalized cost for line $l$, slot $t$:

$$G_{lt} = \alpha_1 p_{lt} + \alpha_2 \frac{h_{lt}}{2} + \alpha_3 \tau_l$$

where $h_{lt} = 60/f_{lt}$ is average headway (minutes), $\tau_l$ is travel time (minutes).

**Choice probability:**

$$P_{\text{transit},lt} = \frac{1}{1 + \exp\!\left(\theta (U_{\text{drive}} + G_{lt})\right)}$$

**World Cup drive penalty:** $U_{\text{drive}}$ is increased by a congestion term proportional to total fan travel demand, discouraging car use on match day.

**Effective demand:** $x_{lt} = d_{lt} \cdot P_{\text{transit},lt}$

**Implementation:** `models/lower_level.py` — vectorized NumPy computation; no simulation.

---

### 7. Bilevel Iterative Best-Response

```
Initialize: f⁰, p⁰ from v1 heuristic
For k = 1, 2, ..., K_max:
    x^k ← Logit(f^{k-1}, p^{k-1})      # lower level response
    (f^k, p^k) ← SLSQP(x^k)            # upper level given fixed demand
    If ||x^k - x^{k-1}|| / ||x^{k-1}|| < ε_conv:
        break
Return (f^k, p^k, x^k, Π^k)
```

Convergence threshold $\varepsilon_{\text{conv}} = 0.001$; max 40 iterations. Implemented in `models/bilevel.py`.

---

### 8. Greedy Integer Allocator

For each $(l, t)$ pair independently (Phase 1), solve:

$$\max_{f_{lt} \in \{0,\ldots,f_{\max}\}} \;\; p_{lt} \min(d_{lt},\; K f_{lt}) - c_f f_{lt} - c_v \min(d_{lt},\; K f_{lt})$$

via `scipy.optimize.minimize_scalar` on the continuous relaxation, then round to nearest integer.

Phase 2: if budget is exceeded, greedily drop lowest-marginal-profit $(l,t)$ trains.

---

### 9. Multiple-Choice Knapsack ILP

Binary decision: for each $(l,t)$, choose one service level $k \in \{0,\ldots,f_{\max}\}$.

$$\max_{y} \;\; \sum_{l,t,k} \pi_{ltk} \cdot y_{ltk}$$

subject to:
- $\sum_k y_{ltk} = 1 \;\;\forall l,t$ (exactly one level per slot)
- $\sum_{l,t,k} c_f k \cdot y_{ltk} \leq B$ (budget)
- $y_{ltk} \in \{0,1\}$

where $\pi_{ltk} = p_{lt} \min(d_{lt}, Kk) - c_f k - c_v \min(d_{lt}, Kk)$. Solved via PuLP/CBC. Implemented in `_run_ilp_comparison.py`.

---

### 10. Stochastic Sensitivity Analysis (Optuna TPE)

Search space (8 parameters):

| Parameter | Range | Block |
|-----------|-------|-------|
| fare_morning | [$p_{\min}$, $p_{\max}$] | morning |
| fare_midday | [$p_{\min}$, $p_{\max}$] | midday |
| fare_evening | [$p_{\min}$, $p_{\max}$] | evening |
| fare_night | [$p_{\min}$, $p_{\max}$] | night |
| extra_trains_{morning,midday,evening,night} | {0,1,2,3} | each |

**Objective (SAA):** Average profit across 100 Monte Carlo demand draws per trial.

$$\hat{\Pi}(\theta) = \frac{1}{100} \sum_{s=1}^{100} \Pi\!\left(f(\theta),\; p(\theta),\; d^{(s)}\right)$$

TPE sampler with 200 trials. Results in `outputs/sensitivity_results.csv`.

---

## Part II — Multimodal Extension (18:00–04:00 Window)

### 11. Time Index and Cross-Midnight Handling

The match-day window spans **18:00 to 04:00+1** (next calendar day), divided into $N = 40$ slots of 15 minutes each.

**Slot index** $t \in \{0, 1, \ldots, 39\}$:

$$t = \left\lfloor \frac{(h_{\text{wall}} - 18) \cdot 60 \;\text{mod}\; (24 \cdot 60)}{15} \right\rfloor$$

where $h_{\text{wall}}$ is wall-clock hours. Times from 00:00–04:00 are wrapped by adding 24 hours before the modular subtraction.

**Key slot references** (implemented in `data/scenario.py`):

| Slot | Wall time | Notes |
|------|----------|-------|
| 0 | 18:00 | Window start |
| 10 | 20:30 | Default kickoff |
| 16 | 22:00 | Evening / late-night boundary |
| 24 | 00:00 | Crosses midnight |
| 31 | 23:45+1 | Approx. match end (20:30 + 120+10 min) |
| 39 | 03:45+1 | Window end |

---

### 12. Demand Model (Multimodal)

Four additive components, all in pax/15-min slot:

**A. Evening baseline** (commuter demand, declining):

$$d_t^{\text{base}} = N_l \cdot \left(1 - 0.7 \cdot \frac{t}{t_{\text{eve}}}\right) / Z \quad t < t_{\text{eve}}$$
$$d_t^{\text{base}} = 0.05 \cdot N_l / Z \quad t \geq t_{\text{eve}}$$

where $t_{\text{eve}} = 16$ (slot for 22:00), $Z$ is the normalization constant. Scales from FY2024 ridership via `data/ridership/ridership_by_line.json`.

**B. Pre-game fan arrival** (Normal wave truncated at kickoff):

$$d_t^{\text{pre}} = N_{\text{fans}} \cdot \phi\!\left(\frac{t - (t_{\text{ko}} - 6)}{\sigma_{\text{pre}}}\right) / Z_{\text{pre}} \quad t \leq t_{\text{ko}}$$

Peak slot $= t_{\text{ko}} - 6$ (90 min before kickoff), $\sigma_{\text{pre}} = 3$ slots.

**C. In-game low activity** (uniform during match window):

$$d_t^{\text{game}} = 0.05 \cdot N_{\text{fans}} / (t_{\text{end}} - t_{\text{ko}}) \quad t_{\text{ko}} \leq t < t_{\text{end}}$$

**D. Post-game evacuation** (Normal wave starting after match end):

$$d_t^{\text{post}} = N_{\text{fans}} \cdot \phi\!\left(\frac{t - t_{\text{peak}}^{\text{pg}}}{\sigma_{\text{post}}}\right) / Z_{\text{post}} \quad t \geq t_{\text{end}}$$

where $t_{\text{peak}}^{\text{pg}} = t_{\text{end}} + \lfloor(25 + 30)/15\rfloor \approx t_{\text{end}} + 3$, $\sigma_{\text{post}} = 4$ slots.

**Match-end slot:**
$$t_{\text{end}} = \left\lfloor \frac{t_{\text{ko}} \cdot 15 + 120 + 10}{15} \right\rfloor$$
(kickoff + 120 min match + 10 min stoppage buffer)

---

### 13. Fan Origin Segmentation

| Segment | Share $s_i$ | BSL demand | RR feeder |
|---------|------------|-----------|-----------|
| local_city | 0.35 | yes | no |
| suburban_rr | 0.30 | via transfer | yes |
| visitor_hotel_airport | 0.20 | 50% direct | 50% |
| car_rideshare | 0.15 | excluded | excluded |

**RR feeder share:** $s_{\text{RR}} = 0.30 + 0.20 \times 0.5 = 0.40$

**BSL inbound demand:** all transit fans heading toward NRG (pre-game + in-game waves).
**BSL outbound demand:** all transit fans leaving NRG (post-game wave).
**RR demand per line:** total RR feeder wave $\times$ line ridership share $s_l$.

RR outbound wave is shifted by 1 slot (20 min BSL travel from NRG to Center City) before fans board return trains.

---

### 14. BSL Capacity Model

**Discrete service levels:**

| Level | Headway $h$ (min) | Trains/slot $n$ | Effective capacity/slot |
|-------|-------------------|-----------------|------------------------|
| normal | 8 | 2 | $2 \times 720 \times 0.85 = 1{,}224$ |
| enhanced | 5 | 3 | $3 \times 720 \times 0.85 = 1{,}836$ |
| max\_event | 3 | 5 | $5 \times 720 \times 0.85 = 3{,}060$ |

**Service level assignment** (greedy, slot-by-slot):
- Pre-game: minimum level covering $d_t^{\text{in}} + d_t^{\text{out}}$
- During match: `normal` (low demand)
- Post-game: `max_event` (evacuation priority), fall back to `enhanced` if over budget

**Slot capacity:** $C_t = n_t \times K_{\text{BSL}} \times \beta$ where $K_{\text{BSL}} = 720$, $\beta = 0.85$.

**Served passengers** (inbound priority):

$$s_t^{\text{in}} = \min(d_t^{\text{in}},\; C_t), \quad s_t^{\text{out}} = \min(d_t^{\text{out}},\; C_t - s_t^{\text{in}})$$

**NRG crowding** (pax above platform tolerance threshold):

$$\text{crowd}_t = \max\!\left(0,\; d_t^{\text{out}} - s_t^{\text{out}} - \Gamma\right), \quad \Gamma = 2{,}500 \text{ pax/slot}$$

NRG throughput hard cap: $C_t \leq 4{,}000$ pax/slot.

**Post-game clearance** (queue simulation):

```
backlog ← 0
for t = t_end, ..., N-1:
    backlog += d_t^out
    served = min(backlog, C_t)
    backlog = max(0, backlog - served)
    if backlog < 1: return t   ← clearance slot
```

Clearance time (minutes): $(t_{\text{clear}} - t_{\text{end}}) \times 15$.

**Headway reliability penalty** (sum of excess headway minutes during event windows):

$$H = \sum_{t \notin \text{match}} \max(0,\; h_t - h^*)$$

where $h^* = 5$ min is the event headway target.

---

### 15. Regional Rail Evaluator (Multimodal)

Greedy integer allocator for the 18:00–04:00 window. Simpler than the bilevel v1 solver:
- Fares are policy-fixed (no optimization): inbound at $2.50, return at $0 (sponsored) or $2.50.
- Extra trains $e \in \{0, 1, 2, 3\}$ added above baseline per slot.
- Budget is checked; if exhausted, extra trains are dropped to 0.

**Baseline trains per slot:**

$$f_t^{\text{base}} = \begin{cases} 1 & t < 16 \text{ (18:00–22:00)} \\ 0 & t \geq 16 \text{ (late night)} \end{cases}$$

**Served passengers:** $x_{lt} = \min(d_{lt},\; (f_t^{\text{base}} + e) \cdot K_{\text{RR}})$

where $K_{\text{RR}} = 875$ seats/train.

---

### 16. Policy Objective (Multimodal)

Minimize:

$$\Phi = \underbrace{(C_{\text{RR}} + C_{\text{BSL}}) - R_{\text{fare}} - R_{\text{sponsor}}}_{\text{net operating deficit}} + \underbrace{\lambda_1 U + \lambda_2 V + \lambda_3 E + \lambda_4 H + \lambda_5 D}_{\text{social cost penalties}}$$

**Cost and revenue components:**

$$C_{\text{RR}} = \sum_{l,t} c_f^{\text{RR}} f_{lt} + c_v^{\text{RR}} x_{lt}$$
$$C_{\text{BSL}} = \sum_t \text{cost}(n_t) + c_v^{\text{BSL}} (s_t^{\text{in}} + s_t^{\text{out}})$$
$$R_{\text{fare}} = \sum_{l,t} p_{lt}^{\text{inbound}} x_{lt}$$
$$R_{\text{sponsor}} = r_{\text{sp}} \cdot 0.5 \cdot \sum_t (s_t^{\text{in}} + s_t^{\text{out}}) \quad \text{if sponsor subsidy active}$$

**Penalty terms:**

| Term | Formula | Weight $\lambda$ |
|------|---------|-----------------|
| $U$ = unmet pax | $\sum_{l,t} u_{lt} + \sum_t (u_t^{\text{in}} + u_t^{\text{out}})$ | $\lambda_1 = 50$ $/pax |
| $V$ = NRG crowding | $\sum_t \text{crowd}_t$ | $\lambda_2 = 30$ $/pax |
| $E$ = equity violations | number of RR lines with coverage $< 90\%$ | $\lambda_3 = 100{,}000$ $ (lump-sum) |
| $H$ = headway excess | total headway minutes above $h^* = 5$ min | $\lambda_4 = 5{,}000$ $/min |
| $D$ = clearance delay | $\max(0,\; T_{\text{clear}} - 90)$ minutes | $\lambda_5 = 200$ $/min |

**Equity:** Two definitions (§6 in `data/scenario.py`):
- Raw coverage (KPI only): $\sum_t x_{lt} / \sum_t d_{lt} \geq 0.80$
- Effective coverage (post-hoc penalty): $\sum_t x_{lt} / \sum_t d_{lt} \geq 0.90$ triggers $\lambda_3$ penalty if violated

Note: the greedy RR allocator does **not** enforce the equity threshold during allocation; it is checked post-hoc. Under the default budget ($350,000) and `extra_trains=1`, the budget is exhausted after approximately 5 of 13 lines, leaving remaining lines with zero late-night service. This produces equity violations for those lines, correctly reflected in the penalty. A higher budget or fewer extra trains reduces violations.

Lower $\Phi$ = better policy outcome.

---

### 17. KPI Definitions

| KPI | Formula | Target |
|-----|---------|--------|
| Total served | $\sum_{l,t} x_{lt} + \sum_t (s_t^{\text{in}} + s_t^{\text{out}})$ | — |
| Total unmet | $\sum_{l,t} u_{lt} + \sum_t (u_t^{\text{in}} + u_t^{\text{out}})$ | → 0 |
| Late-night unmet | $\sum_{t \geq t_{\text{midnight}}} u_t^{\text{out}}$ | → 0 |
| RR load factor | $\sum_{l,t} x_{lt} / \sum_{l,t} K_{\text{RR}} f_{lt}$ | < 1.0 |
| BSL load factor | $\sum_t (s_t^{\text{in}} + s_t^{\text{out}}) / \sum_t C_t$ | < 1.0 |
| Peak NRG crowding | $\max_t \text{crowd}_t$ | → 0 |
| Clearance time | $(t_{\text{clear}} - t_{\text{end}}) \times 15$ min | ≤ 90 min |
| Net deficit | $(C_{\text{RR}} + C_{\text{BSL}}) - R_{\text{fare}} - R_{\text{sponsor}}$ | minimize |
| Policy objective $\Phi$ | full formula above | minimize |
| Raw coverage | $\sum_t x_{lt} / \sum_t d_{lt}$ per line | ≥ 80% |

---

### 18. Variable and Constraint Summary

**V1 Regional Rail model:**
- Decision variables: $13 \times 61 \times 2 = 1{,}586$ ($f_{lt}$, $p_{lt}$) + $13 \times 61 = 793$ auxiliary ($x_{lt}$) = **2,379 total**
- Constraints: C1–C8 as above; SLSQP enforces via gradient projection

**Multimodal model:**
- Discrete service level per slot: 40 choices from {normal, enhanced, max\_event}
- Extra trains per slot: 40 choices from {0, 1, 2, 3}
- All other quantities (demand, capacity, served, cost, KPIs) are derived

---

### 19. Data Provenance

| Quantity | Value | Source |
|----------|-------|--------|
| 12 active RR lines | from SEPTA GTFS | `data/gtfs/` |
| FY2024 daily boardings per line | — | `data/ridership/ridership_by_line.json` |
| $c_f = 1{,}692$/trip | $13{,}535.85 \div 8$ | `data/costs/cost_summary.json` |
| $K_{\text{RR}} = 875$ seats | 5-car Bombardier Multilevel | SEPTA fleet data |
| $K_{\text{BSL}} = 720$ seats | 8 cars × 90 seats | SEPTA fleet data |
| $N_{\text{fans}} \approx 45{,}063$ | $69{,}328 \times 0.65$ | LFF capacity × transit share |
| $r_{\text{sp}} = \$3.00$/pax | sponsor reimbursement | Model assumption |
| $s_{\text{local}} = 0.35,\; s_{\text{suburban}} = 0.30,\ldots$ | fan segments | Model assumption |
| $\Gamma = 2{,}500$ pax/slot | NRG crowding threshold | Model assumption |
| $\beta = 0.85$ | BSL safety buffer | Model assumption |
