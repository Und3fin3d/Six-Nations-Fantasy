"""Deterministic competition scoring adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .contracts import RawPrediction

NCR_SCORING_VERSION = "ncr_front_row_v2"
NCR_LEGACY_SCORING_VERSION = "ncr_unallocated_scrums_v1"
SIX_NATIONS_LEGACY_SCORING_VERSION = "six_nations_legacy_v1"
SIX_NATIONS_2026_SCORING_VERSION = "six_nations_2026_v2"


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

    def score_samples(
        self, events: Mapping[str, np.ndarray], *, is_forward: bool,
        position: str | None = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def score_prediction(
        self, prediction: RawPrediction, n: int = 4000, seed: int = 0,
        ceiling: float = 40.0,
    ) -> ScoreSummary:
        samples = self.score_samples(
            prediction.sample(n=n, seed=seed), is_forward=prediction.is_forward,
            position=prediction.position,
        )
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

    def __init__(self, version: str = NCR_SCORING_VERSION):
        if version not in {NCR_SCORING_VERSION, NCR_LEGACY_SCORING_VERSION}:
            raise ValueError(f"unknown NCR scoring version {version!r}")
        self.version = version

    def scrum_weight(self, position: str | None) -> float:
        if self.version == NCR_LEGACY_SCORING_VERSION:
            return 2.0
        if not isinstance(position, str) or not position:
            raise ValueError("NCR scrum scoring requires the player's position")
        return 2.0 if position in {"Prop", "Hooker"} else 0.0

    def score_samples(
        self, events: Mapping[str, np.ndarray], *, is_forward: bool,
        position: str | None = None,
    ) -> np.ndarray:
        result = sum(self.weights[k] * self._event(events, k) for k in self.weights)
        scrums = self._event(events, "scrums_won")
        if np.any(scrums):
            result += self.scrum_weight(position) * scrums
        return np.asarray(result, dtype=float)

    def expected_points(self, prediction: RawPrediction) -> float:
        events = {name: np.array([dist.mean]) for name, dist in prediction.events.items()}
        return float(self.score_samples(
            events, is_forward=prediction.is_forward, position=prediction.position,
        )[0])


class SixNationsScorer(CompetitionScorer):
    name = "six_nations"
    weights = {
        "try_assists": 4, "conversion_goals": 2, "penalty_goals": 3,
        "drop_goals_converted": 4, "defenders_beaten": 2, "offload": 2,
        "tackles": 1, "tackle_turnover": 5, "penalties_conceded": -1,
        "yellow_cards": -5, "red_cards": -8, "fifty_22": 7,
        "lineout_steals": 7, "scrums_won": 1, "kicks_retained": 2, "potm": 15,
    }

    def __init__(self, *, season: int | None = None, version: str | None = None):
        self.version = version or (
            SIX_NATIONS_2026_SCORING_VERSION if season == 2026
            else SIX_NATIONS_LEGACY_SCORING_VERSION
        )
        if self.version not in {SIX_NATIONS_LEGACY_SCORING_VERSION, SIX_NATIONS_2026_SCORING_VERSION}:
            raise ValueError(f"unknown Six Nations scoring version {self.version!r}")
        self.weights = dict(type(self).weights)
        if self.version == SIX_NATIONS_2026_SCORING_VERSION:
            self.weights["drop_goals_converted"] = 5

    def score_samples(
        self, events: Mapping[str, np.ndarray], *, is_forward: bool,
        position: str | None = None,
    ) -> np.ndarray:
        result = sum(self.weights[k] * self._event(events, k) for k in self.weights)
        result += (15 if is_forward else 10) * self._event(events, "tries")
        result += np.floor(self._event(events, "metres") / 10)
        return np.asarray(result, dtype=float)


def scorer_for(
    competition: str, *, version: str | None = None, season: int | None = None,
) -> CompetitionScorer:
    key = competition.lower().replace("-", "_").replace(" ", "_")
    if key in {"ncr", "nations_championship", "nations_championship_rugby"}:
        return NationsChampionshipScorer(version=version or NCR_SCORING_VERSION)
    if key in {"6n", "six_nations", "sixnations"}:
        return SixNationsScorer(season=season, version=version)
    raise ValueError(f"unknown scoring rules {competition!r}")
