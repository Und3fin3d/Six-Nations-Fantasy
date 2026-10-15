from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class EventDistribution:
    family: str
    mean: float
    dispersion: float = 1.0

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        mean = float(self.mean)
        if self.family == "poisson":
            return rng.poisson(mean, size=size).astype(float)
        if self.family == "negative_binomial":
            if mean == 0:
                return np.zeros(size)
            shape = self.dispersion
            rate = rng.gamma(shape=shape, scale=mean / shape, size=size)
            return rng.poisson(rate).astype(float)
        if self.family == "bernoulli":
            return rng.binomial(1, min(mean, 1.0), size=size).astype(float)
        if mean == 0:
            return np.zeros(size)
        sigma2 = np.log1p(self.dispersion / max(mean * mean, 1e-9))
        mu = np.log(mean) - sigma2 / 2
        return rng.lognormal(mu, np.sqrt(sigma2), size=size)


@dataclass(frozen=True)
class RawPrediction:
    fixture_id: str
    player_id: str
    player_name: str
    team: str
    opponent: str
    position: str
    is_forward: bool
    events: Mapping[str, EventDistribution]
    minutes: EventDistribution
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def sample(self, n: int = 2000, seed: int = 0) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        out = {name: dist.sample(rng, n) for name, dist in self.events.items()}
        out["minutes"] = np.clip(self.minutes.sample(rng, n), 0, 80)
        return out

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events"] = {k: asdict(v) for k, v in self.events.items()}
        payload["minutes"] = asdict(self.minutes)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RawPrediction":
        d = dict(payload)
        d["events"] = {k: EventDistribution(**v) for k, v in d["events"].items()}
        d["minutes"] = EventDistribution(**d["minutes"])
        return cls(**d)
