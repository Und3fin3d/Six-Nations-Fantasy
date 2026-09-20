"""Equal-weight predictive mixtures; no fitted weights or tournament routing."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import EventDistribution, RawPrediction


def _variance(value: EventDistribution) -> float:
    if value.family == "bernoulli":
        return value.mean * (1.0 - value.mean)
    if value.family == "poisson":
        return value.mean
    if value.family == "negative_binomial":
        return value.mean + value.mean ** 2 / value.dispersion
    return value.dispersion


def mean_distribution(values: Sequence[EventDistribution]) -> EventDistribution:
    """Moment-match a uniform mixture, not an average of independent outcomes."""
    if not values:
        raise ValueError("distribution ensemble must not be empty")
    if len({value.family for value in values}) != 1:
        raise ValueError("component distribution families differ")
    if len(values) == 1:
        return values[0]
    mean = float(np.mean([value.mean for value in values]))
    variance = float(np.mean([
        _variance(value) + (value.mean - mean) ** 2 for value in values
    ]))
    family = values[0].family
    if family == "bernoulli":
        return EventDistribution(family, min(mean, 1.0), 1.0)
    if family == "poisson" and np.isclose(variance, mean, rtol=1e-12, atol=1e-12):
        return EventDistribution(family, mean)
    if family in {"poisson", "negative_binomial"}:
        dispersion = max(mean ** 2 / max(variance - mean, 1e-12), 1e-12)
        return EventDistribution("negative_binomial", mean, dispersion)
    return EventDistribution(family, mean, max(variance, 1e-12))


def mean_predictions(components: Sequence[Sequence[RawPrediction]]) -> list[RawPrediction]:
    """Reject incomplete or differently ordered cohorts rather than averaging them."""
    if not components:
        raise ValueError("prediction ensemble must not be empty")
    keys = [(p.fixture_id, p.player_id, p.team) for p in components[0]]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate prediction keys")
    for component in components[1:]:
        if [(p.fixture_id, p.player_id, p.team) for p in component] != keys:
            raise ValueError("component fixture/player/team keys differ")
    if len(components) == 1:
        return list(components[0])
    result = []
    for rows in zip(*components):
        first = rows[0]
        for row in rows[1:]:
            if (row.player_name, row.opponent, row.position, row.is_forward) != (
                first.player_name, first.opponent, first.position, first.is_forward
            ):
                raise ValueError("component player context differs")
            if set(row.events) != set(first.events):
                raise ValueError("component event support differs")
        result.append(replace(
            first,
            events={name: mean_distribution([row.events[name] for row in rows])
                    for name in sorted(first.events)},
            minutes=mean_distribution([row.minutes for row in rows]),
            metadata={"model": "uniform_raw_ensemble", "members": len(rows)},
        ))
    return result


@dataclass
class UniformRawEnsemble:
    """Serializable collection of already-fitted raw-event models."""

    members: tuple[object, ...]

    def __post_init__(self) -> None:
        self.members = tuple(self.members)
        if not self.members or any(not callable(getattr(m, "predict_frame", None))
                                   for m in self.members):
            raise ValueError("members must be nonempty raw-event prediction models")

    def predict_frame(self, frame) -> list[RawPrediction]:
        return mean_predictions([member.predict_frame(frame) for member in self.members])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "UniformRawEnsemble":
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"{path} is not a UniformRawEnsemble")
        return model
