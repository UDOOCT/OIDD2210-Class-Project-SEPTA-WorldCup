"""
models/sensitivity.py
---------------------
OptQuest equivalent — Optuna TPE search over 61 time slots × 12 lines.

Policy parametrization (keeps search space tractable):
  Instead of searching 61×12=732 fare values independently,
  we parametrize with 4 numbers that define a piecewise fare curve:
    fare_morning : fare during morning rush  (6–9am, slots 0–11)
    fare_midday  : fare during midday        (9am–4pm, slots 12–39)
    fare_evening : fare during evening rush  (4–7pm, slots 40–51)
    fare_night   : fare during night         (7–9pm, slots 52–60)
  Similarly for frequency: freq_morning, freq_midday, freq_evening, freq_night.

  This is the 4-block structure from your proposal, used as the
  policy parametrization for the stochastic search.
  The full 61-slot model then inherits these values per block.
"""

import numpy as np
import pandas as pd

from data.demand import get_total_demand, monte_carlo_demand
from data.network import LINES
from data.parameters import (
    TIME_SLOTS, N_SLOTS, SLOT_DURATION_MIN,
    TRAIN_CAPACITY, FIXED_COST_PER_TRAIN, VARIABLE_COST_PER_PAX,
    DAILY_BUDGET_EVENT, EQUITY_EPSILON,
    FARE_MIN, FARE_MAX,
    IDX_9AM, IDX_4PM, IDX_7PM,
)
from models.lower_level import effective_demand

LNAMES = list(LINES.keys())
T      = N_SLOTS  # 61

# Slot-to-block mapping
def _block_of(t_idx: int) -> str:
    if t_idx < IDX_9AM:  return "morning"
    if t_idx < IDX_4PM:  return "midday"
    if t_idx < IDX_7PM:  return "evening"
    return "night"


def policy_to_arrays(policy: dict) -> tuple:
    """
    Expand 4-block policy dict into per-slot arrays (length 61 each).
    Returns (fare_arr, freq_arr) each shape (61,).
    """
    fare_arr = np.array([policy[f"fare_{_block_of(t)}"] for t in range(T)])
    freq_arr = np.array([policy[f"freq_{_block_of(t)}"] for t in range(T)], dtype=float)
    return fare_arr, freq_arr


def evaluate_policy(policy: dict, demand_scenario: dict) -> dict:
    fare_arr, freq_arr = policy_to_arrays(policy)
    total_rev = total_fixed = total_var = total_pax = 0.0
    equity_checks = []

    for l in LNAMES:
        avg_tt = (sum(LINES[l]["travel_times"]) /
                  max(len(LINES[l]["travel_times"]), 1))
        for t in range(T):
            fare  = fare_arr[t]
            freq  = max(freq_arr[t], 0.01)
            hw    = SLOT_DURATION_MIN / freq
            d_raw = demand_scenario[l][t]

            eff_d  = effective_demand(d_raw, fare, hw, avg_tt)
            served = min(eff_d, TRAIN_CAPACITY * freq)

            total_rev   += fare * served
            total_fixed += FIXED_COST_PER_TRAIN * freq
            total_var   += VARIABLE_COST_PER_PAX * served
            total_pax   += served
            if d_raw > 0:
                equity_checks.append(served >= EQUITY_EPSILON * d_raw)

    return {
        "profit":      total_rev - total_fixed - total_var,
        "total_pax":   total_pax,
        "equity_rate": float(np.mean(equity_checks)) if equity_checks else 1.0,
    }


def run_sensitivity(n_trials: int = 200, n_mc_samples: int = 100,
                    seed: int = 42) -> dict:
    """
    Optuna TPE search over 4-block fare+frequency policy.
    Each trial evaluated over n_mc_samples Monte Carlo demand scenarios.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("pip install optuna")

    mc_scenarios = monte_carlo_demand(n_samples=n_mc_samples, seed=seed)

    def objective(trial):
        # 4-block fare policy (monotone: evening ≥ midday, morning ≥ midday)
        fare_mid  = trial.suggest_float("fare_midday",  FARE_MIN, 5.00)
        fare_morn = trial.suggest_float("fare_morning", fare_mid, FARE_MAX)
        fare_eve  = trial.suggest_float("fare_evening", fare_mid, FARE_MAX)
        fare_ngt  = trial.suggest_float("fare_night",   FARE_MIN, fare_eve)

        # 4-block frequency policy
        freq_mid  = trial.suggest_int("freq_midday",  1, 4)
        freq_morn = trial.suggest_int("freq_morning", freq_mid, 8)
        freq_eve  = trial.suggest_int("freq_evening", freq_mid, 8)
        freq_ngt  = trial.suggest_int("freq_night",   1, freq_eve)

        policy = {
            "fare_morning": fare_morn, "fare_midday": fare_mid,
            "fare_evening": fare_eve,  "fare_night":  fare_ngt,
            "freq_morning": freq_morn, "freq_midday": freq_mid,
            "freq_evening": freq_eve,  "freq_night":  freq_ngt,
        }

        profits, equity_rates = [], []
        for sc in mc_scenarios:
            ev = evaluate_policy(policy, sc)
            profits.append(ev["profit"])
            equity_rates.append(ev["equity_rate"])

        mean_profit   = float(np.mean(profits))
        equity_ok_pct = float(np.mean([e >= EQUITY_EPSILON for e in equity_rates]))

        # Budget constraint check
        freq_arr = policy_to_arrays(policy)[1]
        budget_use = FIXED_COST_PER_TRAIN * freq_arr.sum() * len(LNAMES)
        budget_pen = max(0, budget_use - DAILY_BUDGET_EVENT) * 20
        equity_pen = max(0, 0.95 - equity_ok_pct) * 1e7

        return mean_profit - budget_pen - equity_pen

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_policy": study.best_params,
        "best_value":  study.best_value,
        "trials_df":   study.trials_dataframe(),
        "study":       study,
    }
