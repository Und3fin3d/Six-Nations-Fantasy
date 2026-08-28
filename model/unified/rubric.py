"""Canonical scoring-rubric vector used to condition the pooled stage-2 model.

The whole point of one unified model is that competition differences live in the
scoring rules, not in separate learned heads.  To let a single model span both
competitions we hand it the rules themselves: a fixed-order numeric vector of
per-event point weights, read straight from the deterministic scorers so it can
never drift from what the converter actually applies.

Position-dependent Six Nations tries are encoded as two components
(``try_forward`` / ``try_back``); metres-per-10 is encoded as its per-metre
weight.  NCR's flat try value populates both try components equally.
"""

from __future__ import annotations

import numpy as np

from .scoring import NationsChampionshipScorer, SixNationsScorer, scorer_for

# Fixed feature order. Adding a rule means appending here (never reordering),
# so existing artifacts keep their column meaning.
RUBRIC_EVENTS: tuple[str, ...] = (
    "try_forward", "try_back", "try_assists", "conversion_goals",
    "missed_conversion_goals", "penalty_goals", "missed_penalty_goals",
    "drop_goals_converted", "defenders_beaten", "offload", "clean_breaks",
    "tackles", "missed_tackles", "tackle_turnover", "turnovers_conceded",
    "penalties_conceded", "yellow_cards", "red_cards", "lineouts_won",
    "lineout_steals", "scrums_won", "interceptions", "fifty_22",
    "kicks_retained", "metres_per_metre", "potm",
)


def _weight(weights: dict, key: str) -> float:
    return float(weights.get(key, 0.0))


def rubric_vector(competition: str) -> np.ndarray:
    """Return the ordered point-weight vector for ``competition``."""
    scorer = scorer_for(competition)
    w = dict(scorer.weights)
    vec = {event: 0.0 for event in RUBRIC_EVENTS}
    for event in RUBRIC_EVENTS:
        if event in w:
            vec[event] = _weight(w, event)
    if isinstance(scorer, SixNationsScorer):
        vec["try_forward"] = 15.0
        vec["try_back"] = 10.0
        vec["metres_per_metre"] = 0.1  # floor(metres/10)
        vec["scrums_won"] = _weight(w, "scrums_won")
    elif isinstance(scorer, NationsChampionshipScorer):
        vec["try_forward"] = float(w["tries"])
        vec["try_back"] = float(w["tries"])
        vec["metres_per_metre"] = 0.0
        vec["scrums_won"] = 2.0  # front-row only; added outside the weights dict
    return np.array([vec[event] for event in RUBRIC_EVENTS], dtype=float)


def rubric_columns() -> list[str]:
    return [f"rubric__{event}" for event in RUBRIC_EVENTS]
