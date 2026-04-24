# Model Assumptions

## Time window (both v1 and v2)

Both active models use the same time window:
- **Start:** 18:00 match day
- **End:** 03:45+1 (next calendar day)
- **Resolution:** 15-minute slots, 40 slots total
- **Post-midnight:** slots 24–39 (00:00+1 through 03:45+1)

Historical note: the original v1 model used 6am–9pm (61 slots). That is no
longer the active baseline. Both v1 and v2 are now time-window consistent.

## v1 fare/time blocks (Regional Rail)

| Block | Slots | Clock time | Description |
|---|---|---|---|
| pre_game | 0–9 | 18:00–20:15 | Pre-game fan arrivals |
| in_game | 10–17 | 20:30–22:15 | Match in progress |
| post_game | 18–29 | 22:30–01:15+1 | Post-game evacuation |
| late_night | 30–39 | 01:30–03:45+1 | Late tail demand |

## Demand

### v1 (Regional Rail only)
- Evening baseline: linearly declining profile, ~10% of FY2024 daily ridership in 18:00–22:00, ~2% in 22:00–04:00
- Pre-game RR fan demand: Normal distribution, peak at kickoff − 90 min (σ = 30 min)
- Post-game RR fan demand: Normal distribution, peak at match end + 20 min (σ = 25 min)
- Fan-to-line routing: by gateway station proximity (WORLDCUP_GATEWAY_WEIGHTS)

### v2 (Multimodal)
- Total World Cup transit users: 65% of 69,500 stadium capacity ≈ 45,175 fans
- Fan segments: 35% local city (BSL direct), 30% suburban RR feeder, 20% visitor hotel/airport (50/50 split), 15% car/rideshare (outside model)
- Pre-game fan arrival: Normal, centered 90 min before kickoff
- Post-game evacuation: Normal, centered 20 min after match end + stoppage buffer
- Evening baseline: same shares as v1

## Regional Rail (v1)

- Train capacity: 875 seated (5 cars × 175 seats)
- Budget cap: $350,000
- Fixed cost: $1,691.98/train-trip
- Variable cost: $0.40/pax
- Fare range: $2.50–$9.00
- Max trains per slot: 8
- Logit mode-choice: θ=0.80, U_drive_WC = −5.00 (includes $40 parking equivalent)

## Regional Rail (v2 window)

- Baseline trains: 2/slot evening (18:00–22:00), 1/slot late-night (22:00–04:00)
- Extra trains: 1 additional per slot, budget-capped at $350K
- Fares: fixed inbound; post-game return free (Scenario 3+) or paid
- Cost: $1,692/train trip fixed + $1.30/pax variable

## BSL (v2 only)

- Service levels: normal (8 min headway, 2 trains/slot), enhanced (5 min, 3), max_event (3 min, 5)
- Train capacity: 800 pax × 85% safety buffer = 680 effective
- NRG station throughput cap: 4,000 pax/slot; crowding threshold: 3,000 pax

## Equity

- v1: equity constraint ε = 0.80 — capacity must cover 80% of Logit-adjusted demand
- v2: raw coverage KPI ≥ 80% (reported only); effective coverage ≥ 90% triggers $100K/line penalty (post-hoc, not enforced during allocation)

## Financial (v2 policy objective)

- Deficit = (RR_cost + BSL_cost) − fare_revenue − sponsor_reimbursement
- Sponsor reimburses $3/pax for free post-game return trips (when `SPONSOR_SUBSIDY=True`)
- Penalty weights: unmet=$50/pax, crowding=$30/pax, equity=$100K/line, headway=$5K/min, clearance=$200/min
