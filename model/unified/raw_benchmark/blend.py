"""Fixed global event-distribution blend for empirical and v4."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..contracts import EventDistribution, RawPrediction
from ..schema import EVENTS, distribution_family


def _variance(distribution: EventDistribution) -> float:
    mean = distribution.mean
    if distribution.family == "bernoulli":
        return mean * (1.0 - mean)
    if distribution.family == "negative_binomial":
        return mean + mean * mean / max(distribution.dispersion, 1e-9)
    return distribution.dispersion


def _blend_distribution(
    left: EventDistribution, right: EventDistribution, weight_left: float,
    *, target: str,
) -> EventDistribution:
    weight_right = 1.0 - weight_left
    mean = weight_left * left.mean + weight_right * right.mean
    variance = (
        weight_left * (_variance(left) + (left.mean - mean) ** 2)
        + weight_right * (_variance(right) + (right.mean - mean) ** 2)
    )
    family = "lognormal" if target == "minutes" else distribution_family(target)
    if family == "bernoulli":
        return EventDistribution(family, min(mean, 1.0), 1.0)
    if family == "negative_binomial":
        dispersion = max(mean * mean / max(variance - mean, 1e-6), 0.05)
    else:
        dispersion = max(variance, 1e-6)
    return EventDistribution(family, max(mean, 0.0), dispersion)


@dataclass
class EventBlend50:
    empirical: object
    v4: object
    weight_v4: float = 0.5

    def predict_frame(self, frame) -> list[RawPrediction]:
        empirical_predictions = self.empirical.predict_frame(frame)
        v4_predictions = self.v4.predict_frame(frame)
        if len(empirical_predictions) != len(v4_predictions):
            raise ValueError("event blend component prediction lengths differ")
        output = []
        for empirical, v4 in zip(empirical_predictions, v4_predictions):
            if (empirical.fixture_id, empirical.player_id) != (v4.fixture_id, v4.player_id):
                raise ValueError("event blend component rows are misaligned")
            events = {}
            for event in sorted(set(empirical.events) | set(v4.events)):
                left, right = v4.events.get(event), empirical.events.get(event)
                if left is None:
                    events[event] = right
                elif right is None:
                    events[event] = left
                else:
                    events[event] = _blend_distribution(
                        left, right, self.weight_v4, target=event,
                    )
            minutes = _blend_distribution(
                v4.minutes, empirical.minutes, self.weight_v4, target="minutes",
            )
            output.append(RawPrediction(
                fixture_id=v4.fixture_id, player_id=v4.player_id,
                player_name=v4.player_name, team=v4.team, opponent=v4.opponent,
                position=v4.position, is_forward=v4.is_forward, events=events,
                minutes=minutes,
                metadata={"model": "p3_event_50", "weight_v4": self.weight_v4},
            ))
        return output

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "EventBlend50":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} is not EventBlend50")
        return model


def _validate_weights(default_weight_v4: float, event_weights_v4: Mapping[str, float]) -> None:
    weights = {"default": default_weight_v4, **dict(event_weights_v4)}
    invalid = {name: value for name, value in weights.items() if not 0.0 <= value <= 1.0}
    if invalid:
        raise ValueError(f"blend weights must be in [0, 1]: {invalid}")
    unknown = set(event_weights_v4) - {"minutes", *EVENTS}
    if unknown:
        raise ValueError(f"unknown event blend weights: {sorted(unknown)}")


def _predict_weighted(
    empirical_model, v4_model, frame, *, default_weight_v4: float,
    event_weights_v4: Mapping[str, float], model_name: str,
) -> list[RawPrediction]:
    _validate_weights(default_weight_v4, event_weights_v4)
    empirical_predictions = empirical_model.predict_frame(frame)
    v4_predictions = v4_model.predict_frame(frame)
    if len(empirical_predictions) != len(v4_predictions):
        raise ValueError("event blend component prediction lengths differ")
    output = []
    for empirical, v4 in zip(empirical_predictions, v4_predictions):
        empirical_key = (empirical.fixture_id, empirical.player_id, empirical.team)
        v4_key = (v4.fixture_id, v4.player_id, v4.team)
        if empirical_key != v4_key:
            raise ValueError("event blend component rows are misaligned")
        events = {}
        for event in sorted(set(empirical.events) | set(v4.events)):
            left, right = v4.events.get(event), empirical.events.get(event)
            if left is None:
                events[event] = right
            elif right is None:
                events[event] = left
            else:
                events[event] = _blend_distribution(
                    left, right, event_weights_v4.get(event, default_weight_v4),
                    target=event,
                )
        minutes = _blend_distribution(
            v4.minutes, empirical.minutes,
            event_weights_v4.get("minutes", default_weight_v4), target="minutes",
        )
        metadata = {"model": model_name, "weight_v4": default_weight_v4}
        if event_weights_v4:
            metadata["event_weights_v4"] = dict(sorted(event_weights_v4.items()))
        output.append(RawPrediction(
            fixture_id=v4.fixture_id, player_id=v4.player_id,
            player_name=v4.player_name, team=v4.team, opponent=v4.opponent,
            position=v4.position, is_forward=v4.is_forward, events=events,
            minutes=minutes,
            metadata=metadata,
        ))
    return output


@dataclass
class EventWeightedBlend:
    """Competition-independent P3 blend with optional per-event weights."""

    empirical: object
    v4: object
    weight_v4: float = 0.5
    event_weights_v4: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_weights_v4 = dict(self.event_weights_v4)
        _validate_weights(self.weight_v4, self.event_weights_v4)

    def predict_frame(self, frame) -> list[RawPrediction]:
        return _predict_weighted(
            self.empirical, self.v4, frame,
            default_weight_v4=self.weight_v4,
            event_weights_v4=self.event_weights_v4,
            model_name="p3_event_weighted",
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "EventWeightedBlend":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} is not EventWeightedBlend")
        return model
