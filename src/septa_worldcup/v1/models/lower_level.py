"""
models/lower_level.py
---------------------
Passenger route choice model (lower level of bilevel).

ALGORITHM: Multinomial Logit (MNL)
  Passengers minimize generalized travel cost across available routes.

  Utility:  U_r = −(α1·fare + α2·wait_time + α3·travel_time) + ε_r
  ε_r ~ Gumbel(0, 1/θ)  →  gives closed-form logit probabilities

  P(route r | OD k) = exp(θ·U_r) / Σ_{s∈R_k} exp(θ·U_s)

  "No travel" option always included with fixed utility U_0 = LOGIT_NO_TRAVEL_U

VALIDITY:
  MNL assumes IIA (independence of irrelevant alternatives).
  For Regional Rail this is acceptable at the line level since lines
  serve distinct geographic corridors — adding the Airport Line doesn't
  affect the Paoli/Thorndale vs Wilmington substitution.
  IIA would be problematic for nested alternatives (e.g. two lines sharing
  a corridor) — in that case use Nested Logit (future extension).
"""

from __future__ import annotations
from typing import List
import numpy as np
from septa_worldcup.v1.data.parameters import (
    LOGIT_ALPHA_FARE, LOGIT_ALPHA_WAIT, LOGIT_ALPHA_TRAVEL,
    LOGIT_THETA, LOGIT_NO_TRAVEL_U, TRAIN_CAPACITY,
)


def generalized_cost(fare: float, headway_min: float, travel_time_min: float) -> float:
    """G = α1·fare + α2·(headway/2) + α3·travel_time"""
    wait = headway_min / 2
    return (LOGIT_ALPHA_FARE   * fare +
            LOGIT_ALPHA_WAIT   * wait +
            LOGIT_ALPHA_TRAVEL * travel_time_min)


def logit_probs(options: List[dict], include_no_travel: bool = True) -> np.ndarray:
    """
    Args:
        options: list of dicts with keys 'fare', 'headway_min', 'travel_time_min'
        include_no_travel: if True, add outside option with fixed utility

    Returns: probability array (length = len(options) [+ 1 if no_travel])
    """
    utils = np.array([
        -LOGIT_THETA * generalized_cost(o["fare"], o["headway_min"], o["travel_time_min"])
        for o in options
    ])
    if include_no_travel:
        utils = np.append(utils, LOGIT_THETA * LOGIT_NO_TRAVEL_U)

    utils -= utils.max()   # numerical stability
    exp_u = np.exp(utils)
    return exp_u / exp_u.sum()


def effective_demand(raw_demand: float, fare: float,
                     headway_min: float, travel_time_min: float) -> float:
    """
    Fraction of raw demand that actually boards, given service level.
    Equivalent to: raw_demand × P(travel) from binary logit.
    """
    probs = logit_probs(
        [{"fare": fare, "headway_min": headway_min, "travel_time_min": travel_time_min}],
        include_no_travel=True,
    )
    return raw_demand * probs[0]   # probs[0] = P(use transit), probs[1] = P(no travel)


def headway_from_freq(trains_per_block: int, block_duration_hr: float) -> float:
    """Average headway in minutes given train frequency and block duration."""
    block_min = block_duration_hr * 60
    if trains_per_block <= 0:
        return block_min
    return block_min / trains_per_block
