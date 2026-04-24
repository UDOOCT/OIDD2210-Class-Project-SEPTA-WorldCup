"""
src/septa_worldcup/v1/models/sensitivity.py
--------------------------------------------
Optuna TPE stochastic sensitivity search over 40 time slots × 13 lines (18:00–04:00).

Policy parametrization — 4 event-window blocks:
  fare_pre_game  : 18:00–20:15  (slots  0–9)
  fare_in_game   : 20:30–22:15  (slots 10–17)
  fare_post_game : 22:30–01:15  (slots 18–29)
  fare_late_night: 01:30–03:45  (slots 30–39)
  Similarly for frequency: freq_pre_game, freq_in_game, etc.

HISTORICAL NOTE: original v1 used morning/midday/evening/night blocks over
  61 slots (6am–9pm). Active baseline now uses event-window blocks.
"""

import numpy as np
import pandas as pd

from septa_worldcup.v1.data.demand import get_total_demand, monte_carlo_demand
from septa_worldcup.v1.data.network import LINES
from septa_worldcup.v1.data.parameters import (
    N_SLOTS, SLOT_DURATION_MIN,
    TRAIN_CAPACITY, FIXED_COST_PER_TRAIN, VARIABLE_COST_PER_PAX,
    DAILY_BUDGET_EVENT, EQUITY_EPSILON,
    FARE_MIN, FARE_MAX,
    IDX_2030, IDX_2230, IDX_0130,
    TBLOCK_RANGES_V1,
)
from septa_worldcup.v1.models.lower_level import effective_demand

LNAMES = list(LINES.keys())
T      = N_SLOTS  # 40

# Slot-to-block mapping for the 18:00–04:00 event window
def _block_of(t_idx: int) -> str:
    if t_idx < IDX_2030:   return "pre_game"
    if t_idx < IDX_2230:   return "in_game"
    if t_idx < IDX_0130:   return "post_game"
    return "late_night"


def policy_to_arrays(policy: dict) -> tuple:
    """
    Expand 4-block policy dict into per-slot arrays (length 40 each).
    Returns (fare_arr, freq_arr) each shape (40,).
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
        # 4-block fare policy for event window (post_game ≥ in_game)
        fare_in   = trial.suggest_float("fare_in_game",    FARE_MIN, 5.00)
        fare_pre  = trial.suggest_float("fare_pre_game",   fare_in,  FARE_MAX)
        fare_post = trial.suggest_float("fare_post_game",  fare_in,  FARE_MAX)
        fare_ln   = trial.suggest_float("fare_late_night", FARE_MIN, fare_post)

        # 4-block frequency policy
        freq_in   = trial.suggest_int("freq_in_game",    1, 4)
        freq_pre  = trial.suggest_int("freq_pre_game",   freq_in, 8)
        freq_post = trial.suggest_int("freq_post_game",  freq_in, 8)
        freq_ln   = trial.suggest_int("freq_late_night", 1, freq_post)

        policy = {
            "fare_pre_game":   fare_pre,  "fare_in_game":    fare_in,
            "fare_post_game":  fare_post, "fare_late_night": fare_ln,
            "freq_pre_game":   freq_pre,  "freq_in_game":    freq_in,
            "freq_post_game":  freq_post, "freq_late_night": freq_ln,
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
