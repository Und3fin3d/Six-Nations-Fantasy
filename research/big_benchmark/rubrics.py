"""Deterministic API-observable fantasy rubrics for actual and expected points.

Every rubric is restricted to components that the cached API statistics
observe for club and international matches alike, and the same restriction is
applied to forecasts, so labels and predictions are on one scale.

Components of the official games that are NOT observable and therefore absent:

* Six Nations: 50-22 kicks (7), lineout steals (7), scrums won (1),
  kicks retained (2), player of the match (15). Breakdown steals (5) use the
  API ``tackle_turnover`` count as a proxy.
* NCR: lineout steals (5), lineouts won (1; the API count is diagnostic-only
  and not forecast), interceptions (5), front-row scrums won (2), player of
  the match (15).
* Super Rugby Pacific style: lineout steals (5). Line breaks use API
  ``clean_breaks``; turnovers won use ``tackle_turnover``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from model.unified.contracts import RawPrediction


@dataclass(frozen=True)
class Rubric:
    name: str
    try_forward: float
    try_back: float
    weights: Mapping[str, float]
    metres_per_point: float | None = None  # floor(metres / k) points when set
    missing: tuple[str, ...] = field(default_factory=tuple)

    @property
    def events(self) -> tuple[str, ...]:
        extra = ('metres',) if self.metres_per_point else ()
        return ('tries', *self.weights, *extra)


SIX_NATIONS = Rubric(
    'six_nations', 15.0, 10.0,
    {'try_assists': 4, 'conversion_goals': 2, 'penalty_goals': 3, 'drop_goals_converted': 4,
     'defenders_beaten': 2, 'offload': 2, 'tackles': 1, 'tackle_turnover': 5,
     'penalties_conceded': -1, 'yellow_cards': -5, 'red_cards': -8},
    metres_per_point=10.0,
    missing=('fifty_22 (7)', 'lineout_steals (7)', 'scrums_won (1)', 'kicks_retained (2)', 'potm (15)'),
)
NCR = Rubric(
    'ncr', 12.0, 12.0,
    {'try_assists': 5, 'conversion_goals': 2, 'missed_conversion_goals': -1, 'penalty_goals': 3,
     'missed_penalty_goals': -1, 'drop_goals_converted': 5, 'defenders_beaten': 2, 'offload': 2,
     'clean_breaks': 3, 'tackles': 1, 'missed_tackles': -1, 'tackle_turnover': 4,
     'turnovers_conceded': -1, 'penalties_conceded': -1, 'yellow_cards': -5, 'red_cards': -10},
    missing=('lineout_steals (5)', 'lineouts_won (1)', 'interceptions (5)', 'front-row scrums_won (2)',
             'potm (15)'),
)
SUPER_RUGBY = Rubric(
    'srp', 15.0, 15.0,
    {'try_assists': 9, 'conversion_goals': 2, 'missed_conversion_goals': -1, 'penalty_goals': 3,
     'missed_penalty_goals': -1, 'drop_goals_converted': 3, 'clean_breaks': 7, 'tackles': 1,
     'missed_tackles': -1, 'tackle_turnover': 4, 'defenders_beaten': 2, 'offload': 2,
     'penalties_conceded': -1, 'yellow_cards': -5, 'red_cards': -10},
    metres_per_point=10.0,
    missing=('lineout_steals (5)',),
)
RUBRICS = {r.name: r for r in (SIX_NATIONS, NCR, SUPER_RUGBY)}


def actual_points(frame: pd.DataFrame, rubric: Rubric) -> np.ndarray:
    """Realised rubric points per row; NaN where any rubric event is unobserved."""
    forward = frame['is_forward'].fillna(False).astype(bool).to_numpy()
    valid = np.ones(len(frame), dtype=bool)
    values = {}
    for event in rubric.events:
        value = pd.to_numeric(frame[event], errors='coerce').to_numpy(float)
        available = frame.get(f'available__{event}')
        if available is not None:
            valid &= available.fillna(False).astype(bool).to_numpy()
        valid &= np.isfinite(value)
        values[event] = np.nan_to_num(value)
    total = np.where(forward, rubric.try_forward, rubric.try_back)*values['tries']
    for event, weight in rubric.weights.items():
        total = total + weight*values[event]
    if rubric.metres_per_point:
        total = total + np.floor(values['metres']/rubric.metres_per_point)
    return np.where(valid, total, np.nan)


def expected_floor(mean: np.ndarray, variance: np.ndarray, step: float) -> np.ndarray:
    """E[floor(X/step)] for lognormal X with the given mean and variance.

    ``E[floor(X/k)] = sum_j P(X >= j k)``; the sum is truncated where the
    survival probability is below 1e-10 (as in ``rolling_eval.expected_points``).
    """
    mean = np.asarray(mean, float)
    variance = np.asarray(variance, float)
    out = np.zeros(len(mean))
    positive = mean > 0
    if not positive.any():
        return out
    m, v = mean[positive], np.maximum(variance[positive], 1e-12)
    sigma2 = np.log1p(v/m**2)
    sigma, scale = np.sqrt(sigma2), np.exp(np.log(m) - sigma2/2)
    end = int(np.ceil(np.max(lognorm.isf(1e-10, sigma, scale=scale))/step))
    if end > 100_000:
        raise ValueError('metres tail exceeds the bounded expectation calculation')
    result = np.zeros(len(m))
    for start in range(1, end + 1, 512):
        grid = step*np.arange(start, min(start + 512, end + 1))
        result += lognorm.sf(grid[None, :], sigma[:, None], scale=scale[:, None]).sum(axis=1)
    out[positive] = result
    return out


def event_means(predictions: Sequence[RawPrediction], events: Sequence[str]) -> dict[str, np.ndarray]:
    return {e: np.array([p.events[e].mean if e in p.events else 0.0 for p in predictions], float) for e in events}


def expected_points(predictions: Sequence[RawPrediction], rubric: Rubric) -> np.ndarray:
    """Exact expectation of the observable rubric under each forecast distribution."""
    forward = np.array([bool(p.is_forward) for p in predictions])
    means = event_means(predictions, ('tries', *rubric.weights))
    total = np.where(forward, rubric.try_forward, rubric.try_back)*means['tries']
    for event, weight in rubric.weights.items():
        total = total + weight*means[event]
    if rubric.metres_per_point:
        metres = [p.events.get('metres') for p in predictions]
        mean = np.array([0.0 if d is None else d.mean for d in metres])
        variance = np.array([1.0 if d is None else d.dispersion for d in metres])
        total = total + expected_floor(mean, variance, rubric.metres_per_point)
    return total


# Rubric configs for the empirical fantasy baseline (``model.empirical_unified``),
# restricted to the same observable components: official-only terms (50-22,
# kicks retained, lineout steals) are removed and the front-row scrum bonus is
# zero. Its hard-coded player-of-the-match term is removed afterwards by
# ``strip_potm``.
BASELINE_CONFIGS = {
    'six_nations': dict(
        try_weight=lambda fwd: 15.0 if bool(fwd) else 10.0,
        att={'try_assists': 4, 'conversion_goals': 2, 'penalty_goals': 3, 'drop_goals_converted': 4,
             'defenders_beaten': 2, 'offload': 2, 'metres': 0.1},
        dfn={'tackles': 1, 'tackle_turnover': 5},
        dsc={'penalties_conceded': -1, 'yellow_cards': -5, 'red_cards': -8},
        rp_att={'try_assists': 4, 'defenders_beaten': 2, 'offloads': 2, 'metres': 0.1},
        rp_dfn={'tackles': 1, 'turnovers_won': 5},
        front_row_bonus=0.0,
    ),
    'ncr': dict(
        try_weight=lambda fwd: 12.0,
        att={'try_assists': 5, 'conversion_goals': 2, 'missed_conversion_goals': -1, 'penalty_goals': 3,
             'missed_penalty_goals': -1, 'drop_goals_converted': 5, 'defenders_beaten': 2, 'offload': 2,
             'clean_breaks': 3},
        dfn={'tackles': 1, 'missed_tackles': -1, 'tackle_turnover': 4},
        dsc={'turnovers_conceded': -1, 'penalties_conceded': -1, 'yellow_cards': -5, 'red_cards': -10},
        rp_att={'try_assists': 5, 'defenders_beaten': 2, 'offloads': 2, 'clean_breaks': 3},
        rp_dfn={'tackles': 1, 'missed_tackles': -1, 'turnovers_won': 4},
        front_row_bonus=0.0,
    ),
    'srp': dict(
        try_weight=lambda fwd: 15.0,
        att={'try_assists': 9, 'conversion_goals': 2, 'missed_conversion_goals': -1, 'penalty_goals': 3,
             'missed_penalty_goals': -1, 'drop_goals_converted': 3, 'defenders_beaten': 2, 'offload': 2,
             'clean_breaks': 7, 'metres': 0.1},
        dfn={'tackles': 1, 'missed_tackles': -1, 'tackle_turnover': 4},
        dsc={'penalties_conceded': -1, 'yellow_cards': -5, 'red_cards': -10},
        rp_att={'try_assists': 9, 'defenders_beaten': 2, 'offloads': 2, 'clean_breaks': 7, 'metres': 0.1},
        rp_dfn={'tackles': 1, 'missed_tackles': -1, 'turnovers_won': 4},
        front_row_bonus=0.0,
    ),
}
POTM_SCALE, POTM_POINTS = 145.0, 15.0


def strip_potm(predicted: np.ndarray, started: np.ndarray) -> np.ndarray:
    """Invert ``project_candidates``' player-of-the-match term exactly.

    It adds ``15 * clip(base / 145, 0, cap)`` with cap 0.13 for starters and
    0.05 for replacements; that map is strictly increasing, so ``base`` is
    recovered piecewise.
    """
    predicted = np.asarray(predicted, float)
    cap = np.where(np.asarray(started, bool), 0.13, 0.05)
    ratio = 1.0 + POTM_POINTS/POTM_SCALE
    knee = cap*POTM_SCALE*ratio
    return np.where(predicted <= 0, predicted,
                    np.where(predicted < knee, predicted/ratio, predicted - POTM_POINTS*cap))
