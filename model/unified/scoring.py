"""Deterministic competition scoring adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .contracts import RawPrediction


@dataclass(frozen=True)
class ScoreSummary:
    mean: float
    p10: float
    median: float
    p90: float
    ceiling_probability: float


class CompetitionScorer:
    """Interface implemented by each competition's rules adapter."""

    name = "base"

    def score_samples(self, events: Mapping[str, np.ndarray], *, is_forward: bool) -> np.ndarray:
        raise NotImplementedError

    def score_prediction(
        self, prediction: RawPrediction, n: int = 4000, seed: int = 0,
        ceiling: float = 40.0,
    ) -> ScoreSummary:
        samples = self.score_samples(prediction.sample(n=n, seed=seed), is_forward=prediction.is_forward)
        return ScoreSummary(
            mean=float(samples.mean()), p10=float(np.quantile(samples, .10)),
            median=float(np.quantile(samples, .50)), p90=float(np.quantile(samples, .90)),
            ceiling_probability=float((samples >= ceiling).mean()),
        )

    @staticmethod
    def _event(events: Mapping[str, np.ndarray], name: str) -> np.ndarray:
        if name in events:
            return np.asarray(events[name], dtype=float)
        size = len(next(iter(events.values()))) if events else 1
        return np.zeros(size)


class NationsChampionshipScorer(CompetitionScorer):
    name = "ncr"
    weights = {
        "tries": 12, "try_assists": 5, "conversion_goals": 2,
        "missed_conversion_goals": -1, "penalty_goals": 3,
        "missed_penalty_goals": -1, "drop_goals_converted": 5,
        "defenders_beaten": 2, "offload": 2, "clean_breaks": 3,
        "tackles": 1, "missed_tackles": -1, "tackle_turnover": 4,
        "turnovers_conceded": -1, "penalties_conceded": -1,
        "yellow_cards": -5, "red_cards": -10, "lineout_steals": 5,
        "lineouts_won": 1, "interceptions": 5, "potm": 15,
    }

    def score_samples(self, events: Mapping[str, np.ndarray], *, is_forward: bool) -> np.ndarray:
        result = sum(self.weights[k] * self._event(events, k) for k in self.weights)
        # Scrum won applies only to front-row players; callers encode that role as
        # an allocated player event. Interception is not currently observable.
        result += 2 * self._event(events, "scrums_won")
        return np.asarray(result, dtype=float)


class SixNationsScorer(CompetitionScorer):
    name = "six_nations"
    weights = {
        "try_assists": 4, "conversion_goals": 2, "penalty_goals": 3,
        "drop_goals_converted": 4, "defenders_beaten": 2, "offload": 2,
        "tackles": 1, "tackle_turnover": 5, "penalties_conceded": -1,
        "yellow_cards": -5, "red_cards": -8, "fifty_22": 7,
        "lineout_steals": 7, "scrums_won": 1, "kicks_retained": 2, "potm": 15,
    }

    def score_samples(self, events: Mapping[str, np.ndarray], *, is_forward: bool) -> np.ndarray:
        result = sum(self.weights[k] * self._event(events, k) for k in self.weights)
        result += (15 if is_forward else 10) * self._event(events, "tries")
        result += np.floor(self._event(events, "metres") / 10)
        return np.asarray(result, dtype=float)


def scorer_for(competition: str) -> CompetitionScorer:
    key = competition.lower().replace("-", "_").replace(" ", "_")
    if key in {"ncr", "nations_championship", "nations_championship_rugby"}:
        return NationsChampionshipScorer()
    if key in {"6n", "six_nations", "sixnations"}:
        return SixNationsScorer()
    raise ValueError(f"unknown scoring rules {competition!r}")
