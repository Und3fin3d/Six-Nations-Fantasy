"""Team-level goal-kicking concentration for raw-event forecasts.

A team's place kicks are normally taken by one player, but independent
player forecasts spread the team's expected attempts across every plausible
kicker. This keeps each team's expected attempts and each player's success
split, and reallocates attempts in proportion to ``attempts ** gamma``.
``gamma = 1`` is the identity; larger values concentrate on the most likely
kicker. Only the four goal-kicking events are changed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np

from .contracts import EventDistribution, RawPrediction

KICKING_EVENTS = ("conversion_goals", "missed_conversion_goals", "penalty_goals", "missed_penalty_goals")


def _attempts(prediction: RawPrediction) -> float:
    return float(sum(prediction.events[e].mean for e in KICKING_EVENTS if e in prediction.events))


def concentrate_kicking(predictions: list[RawPrediction], gamma: float) -> list[RawPrediction]:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    teams: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, prediction in enumerate(predictions):
        teams[(prediction.fixture_id, prediction.team)].append(i)
    output = list(predictions)
    for members in teams.values():
        attempts = np.array([_attempts(predictions[i]) for i in members])
        total = attempts.sum()
        if total <= 0:
            continue
        weights = np.power(np.maximum(attempts, 0.0), gamma)
        new = weights / weights.sum() * total
        for i, before, after in zip(members, attempts, new):
            if before <= 0:
                continue
            factor = after / before
            events = dict(predictions[i].events)
            for event in KICKING_EVENTS:
                if event in events:
                    dist = events[event]
                    events[event] = EventDistribution(dist.family, dist.mean * factor, dist.dispersion)
            output[i] = replace(predictions[i], events=events)
    return output


@dataclass
class KickingConcentration:
    """Wrap a fitted raw-event model with team-level kicking concentration."""

    model: object
    gamma: float = 1.5

    def predict_frame(self, frame) -> list[RawPrediction]:
        return concentrate_kicking(self.model.predict_frame(frame), self.gamma)
