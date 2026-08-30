"""Blend rules and offline scoring for the all-rugby P3 hill-climb.

Every rule maps (empirical, v4) component predictions to a blended
``RawPrediction``. The frozen incumbent is ``GlobalWeight(0.5)``, which
reproduces ``p3_event_50`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import EventDistribution, RawPrediction
from ..schema import distribution_family
from .blend import _variance


def _family(target: str) -> str:
    return "lognormal" if target == "minutes" else distribution_family(target)


def _combine(
    left: EventDistribution, right: EventDistribution, weight_left: float,
    *, target: str, log_space: bool = False,
) -> EventDistribution:
    """Blend v4 (``left``) with empirical (``right``) at ``weight_left`` on v4.

    ``log_space`` blends the means geometrically on the log1p scale, which is
    the natural link for the Poisson-family and lognormal losses; the frozen
    rule always blends means arithmetically.
    """
    weight_right = 1.0 - weight_left
    family = _family(target)
    if log_space and family != "bernoulli":
        mean = float(np.expm1(
            weight_left * np.log1p(max(left.mean, 0.0))
            + weight_right * np.log1p(max(right.mean, 0.0))
        ))
    else:
        mean = weight_left * left.mean + weight_right * right.mean
    variance = (
        weight_left * (_variance(left) + (left.mean - mean) ** 2)
        + weight_right * (_variance(right) + (right.mean - mean) ** 2)
    )
    if family == "bernoulli":
        return EventDistribution(family, min(mean, 1.0), 1.0)
    if family == "negative_binomial":
        dispersion = max(mean * mean / max(variance - mean, 1e-6), 0.05)
    else:
        dispersion = max(variance, 1e-6)
    return EventDistribution(family, max(mean, 0.0), dispersion)


@dataclass
class BlendRule:
    """Per-target v4 weights plus per-target mean-space/log-space choice."""

    name: str
    weights: dict[str, float]
    default_weight: float = 0.5
    log_space: frozenset[str] = frozenset()

    def weight_for(self, target: str) -> float:
        return float(self.weights.get(target, self.default_weight))

    def apply(
        self, empirical: list[RawPrediction], v4: list[RawPrediction],
    ) -> list[RawPrediction]:
        if len(empirical) != len(v4):
            raise ValueError("component prediction lengths differ")
        output = []
        for left_row, right_row in zip(v4, empirical):
            if (left_row.fixture_id, left_row.player_id) != (right_row.fixture_id, right_row.player_id):
                raise ValueError("component rows are misaligned")
            events = {}
            for target in sorted(set(left_row.events) | set(right_row.events)):
                left, right = left_row.events.get(target), right_row.events.get(target)
                if left is None:
                    events[target] = right
                elif right is None:
                    events[target] = left
                else:
                    events[target] = _combine(
                        left, right, self.weight_for(target), target=target,
                        log_space=target in self.log_space,
                    )
            minutes = _combine(
                left_row.minutes, right_row.minutes, self.weight_for("minutes"),
                target="minutes", log_space="minutes" in self.log_space,
            )
            output.append(RawPrediction(
                fixture_id=left_row.fixture_id, player_id=left_row.player_id,
                player_name=left_row.player_name, team=left_row.team,
                opponent=left_row.opponent, position=left_row.position,
                is_forward=left_row.is_forward, events=events, minutes=minutes,
                metadata={"model": self.name},
            ))
        return output


def global_rule(name: str, weight: float, *, log_space: bool = False) -> BlendRule:
    return BlendRule(
        name=name, weights={}, default_weight=weight,
        log_space=frozenset(ALL_TARGETS) if log_space else frozenset(),
    )


from .config import EXTENDED_EVENTS, STABLE_EVENTS  # noqa: E402

ALL_TARGETS = ("minutes", *STABLE_EVENTS, *EXTENDED_EVENTS)
SCORED_TARGETS = ("minutes", *STABLE_EVENTS)
