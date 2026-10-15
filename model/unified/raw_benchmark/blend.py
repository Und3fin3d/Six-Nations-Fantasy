from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

from ..contracts import EventDistribution, RawPrediction
from ..schema import distribution_family


def _variance(distribution: EventDistribution) -> float:
    mean = distribution.mean
    if distribution.family == "bernoulli":
        return mean * (1.0 - mean)
    if distribution.family == "negative_binomial":
        return mean + mean * mean / max(distribution.dispersion, 1e-9)
    return distribution.dispersion


def _blend_distribution(left: EventDistribution, right: EventDistribution, weight_left: float, *, target: str) -> EventDistribution:
    weight_right = 1.0 - weight_left
    mean = weight_left * left.mean + weight_right * right.mean
    variance = weight_left * (_variance(left) + (left.mean - mean) ** 2) + weight_right * (_variance(right) + (right.mean - mean) ** 2)
    family = "lognormal" if target == "minutes" else distribution_family(target)
    if family == "bernoulli":
        return EventDistribution(family, min(mean, 1.0), 1.0)
    if family == "negative_binomial":
        dispersion = max(mean * mean / max(variance - mean, 1e-6), 0.05)
    else:
        dispersion = max(variance, 1e-6)
    return EventDistribution(family, max(mean, 0.0), dispersion)


def _blend(empirical: list[RawPrediction], v4: list[RawPrediction], weight: float, metadata: dict) -> list[RawPrediction]:
    return [
        RawPrediction(
            fixture_id=v.fixture_id,
            player_id=v.player_id,
            player_name=v.player_name,
            team=v.team,
            opponent=v.opponent,
            position=v.position,
            is_forward=v.is_forward,
            minutes=_blend_distribution(v.minutes, m.minutes, weight, target="minutes"),
            metadata=dict(metadata),
            events={
                e: _blend_distribution(v.events[e], m.events[e], weight, target=e) if e in v.events and e in m.events else v.events.get(e, m.events.get(e))
                for e in sorted(set(m.events) | set(v.events))
            },
        )
        for m, v in zip(empirical, v4)
    ]


@dataclass
class EventBlend50:
    empirical: object
    v4: object
    weight_v4: float = 0.5

    def predict_frame(self, frame) -> list[RawPrediction]:
        return _blend(
            self.empirical.predict_frame(frame), self.v4.predict_frame(frame), self.weight_v4,
            {"model": "p3_event_50", "weight_v4": self.weight_v4},
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str) -> "EventBlend50":
        with open(path, "rb") as handle:
            return pickle.load(handle)
