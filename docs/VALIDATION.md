# Validation Notes

Full audit report: `outputs/validation/validation_summary.txt`

## Time-window alignment (2026-04-24 update)

Active v1 and active v2 now both use **18:00–04:00+1** (40 slots, 15-minute resolution).
- v1 was updated from 6am–9pm (61 slots) to 18:00–04:00+1 (40 slots).
- v2 was unchanged.
- All 40-slot assertions in `scripts/validate_project.py` pass for both.

## Bugs fixed during original validation audit

**BUG 1** — `budget_used` phantom accounting (`v2/models/policy_objective.py`)
- Severity: Low (reporting only)
- Fix: `cost_extra = 0` when extra trains are not deployed

**BUG 2** — SLSQP infeasible initial point (`v1/models/upper_level.py`)
- Severity: High (caused `main.py` to run indefinitely)
- C3 budget ($350K) + C4 min-service ($527K min cost) jointly infeasible
- Fix: dropped C4, demand-proportional initial point, `maxiter=500`

## v1 result numbers (18:00–04:00 window, after time-window update)

| Metric | Value |
|---|---|
| v1 time slots | 40 |
| v1 first slot | 18:00 |
| v1 last slot | 03:45 (+1) |
| Profit (greedy) | $288,306 |
| Revenue | $684,155 |
| Budget used | $348,548 (within $350K) |
| Total pax | 118,254 |
| Greedy vs ILP gap | $0 (0.00%) — globally optimal |

**Why v1 profit changed from old audit ($59,448 → $288,306):**
The old audit profit applied the 6am–9pm model to a full-day 48K ridership
profile. The active 18:00–04:00 window has concentrated event demand (pre-game
fan wave + post-game wave) that is more elastic and profitable per seat.
The Logit mode-choice model assigns high P_transit during the event window
(U_drive_WC = −5.00 vs U_drive = −1.50 normal days), driving higher revenue
at optimal event fares.

## Known limitations (unchanged)

| # | Description |
|---|---|
| L1 | `main.py` global SLSQP: ~14s/iter × 500 iters ≈ 2 hours. Use `_run_optimization.py` (~4s) instead. |
| L2 | 8 equity violations expected under $350K budget in v2 (correct behavior). |
| L3 | BSL clearance 225 min is correct queue math for 45K post-game fans. |
| L4 | `main.py --mode sensitivity` also slow (200 Optuna trials × SLSQP). |
| L5 | Equity in greedy RR allocator (v2) is post-hoc penalty, not enforced during allocation. |
| L6 | BSL demand uses Normal distribution for post-game wave. |
