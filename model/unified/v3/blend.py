"""One optional global event-level blend of the two unified v3 models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..contracts import EventDistribution, RawPrediction
from .gbdt import ExposureRateGBDT
from .neural import ExposureRateNeural


def blend_predictions(
    gbdt: list[RawPrediction], neural: list[RawPrediction], neural_weight: float,
) -> list[RawPrediction]:
    if len(gbdt) != len(neural):
        raise ValueError("blend inputs have different row counts")
    weight = float(neural_weight)
    if not 0 <= weight <= 1:
        raise ValueError("neural blend weight must be in [0, 1]")
    output = []
    for left, right in zip(gbdt, neural):
        identity = (left.fixture_id, left.player_id, left.team)
        if identity != (right.fixture_id, right.player_id, right.team):
            raise ValueError(f"blend prediction identity mismatch: {identity}")
        events = {}
        for event in sorted(set(left.events) & set(right.events)):
            a, b = left.events[event], right.events[event]
            if a.family != b.family:
                raise ValueError(f"{event}: distribution-family mismatch")
            events[event] = EventDistribution(
                a.family,
                (1 - weight) * a.mean + weight * b.mean,
                max((1 - weight) * a.dispersion + weight * b.dispersion, 1e-6),
            )
        output.append(RawPrediction(
            fixture_id=left.fixture_id, player_id=left.player_id,
            player_name=left.player_name, team=left.team, opponent=left.opponent,
            position=left.position, is_forward=left.is_forward, events=events,
            minutes=EventDistribution(
                "lognormal",
                (1 - weight) * left.minutes.mean + weight * right.minutes.mean,
                max(
                    (1 - weight) * left.minutes.dispersion
                    + weight * right.minutes.dispersion,
                    1e-6,
                ),
            ),
            metadata={
                "model": "event_blend_v3", "neural_weight": weight,
                "gbdt": left.metadata, "neural": right.metadata,
            },
        ))
    return output


@dataclass
class EventBlendModel:
    gbdt: ExposureRateGBDT
    neural: ExposureRateNeural
    neural_weight: float

    def predict_frame(self, frame):
        return blend_predictions(
            self.gbdt.predict_frame(frame), self.neural.predict_frame(frame),
            self.neural_weight,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        gbdt_path = path.with_suffix(path.suffix + ".gbdt.pkl")
        neural_path = path.with_suffix(path.suffix + ".neural.pt")
        self.gbdt.save(gbdt_path)
        self.neural.save(neural_path)
        path.write_text(json.dumps({
            "schema_version": 1, "kind": "event_blend_v3",
            "neural_weight": self.neural_weight,
            "gbdt_artifact": gbdt_path.name, "neural_artifact": neural_path.name,
        }, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EventBlendModel":
        payload = json.loads(path.read_text())
        return cls(
            gbdt=ExposureRateGBDT.load(path.with_name(payload["gbdt_artifact"])),
            neural=ExposureRateNeural.load(path.with_name(payload["neural_artifact"])),
            neural_weight=float(payload["neural_weight"]),
        )
