"""Frozen configuration contracts for unified model v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BaselineConfig:
    weighting: str = "level_balanced"
    time_half_life_days: float | None = None
    n_estimators: int = 180
    num_leaves: int = 23
    seed: int = 17

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineConfig":
        return cls(**payload)


@dataclass
class GBDTV3Config:
    """Configuration shared by all exposure/rate LightGBM heads."""

    learning_rate: float = 0.045
    n_estimators: int = 220
    num_leaves: int = 31
    min_child_samples: int = 40
    reg_lambda: float = 4.0
    subsample: float = 0.9
    colsample_bytree: float = 0.85
    weighting: str = "level_balanced"
    time_half_life_days: float | None = 1095.0
    rate_cap_quantile: float = 0.995
    use_appearance: bool = True
    context_blocks: tuple[str, ...] = ("wr",)
    seed: int = 17

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["context_blocks"] = list(self.context_blocks)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GBDTV3Config":
        data = dict(payload)
        data["context_blocks"] = tuple(data.get("context_blocks", ()))
        return cls(**data)


@dataclass
class NeuralV3Config:
    """One candidate in the fixed 20-configuration neural search."""

    hidden: int = 128
    depth: int = 2
    dropout: float = 0.15
    embedding_scale: float = 1.0
    epochs: int = 40
    batch_size: int = 1024
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    patience: int = 5
    weighting: str = "level_balanced"
    time_half_life_days: float | None = 1095.0
    rate_cap_quantile: float = 0.995
    use_appearance: bool = True
    context_blocks: tuple[str, ...] = ("wr",)
    seed: int = 17

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["context_blocks"] = list(self.context_blocks)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NeuralV3Config":
        data = dict(payload)
        data["context_blocks"] = tuple(data.get("context_blocks", ()))
        return cls(**data)


@dataclass
class SearchConfig:
    """Bounded search and model-selection policy."""

    rolling_years: tuple[int, ...] = (2022, 2023, 2024)
    neural_candidates: int = 20
    neural_rungs: tuple[int, ...] = (6, 15, 40)
    neural_survivors: tuple[int, ...] = (8, 3)
    observable_correlation_gate: float = 0.85
    blend_deviance_margin: float = 0.03
    blend_residual_correlation_max: float = 0.95
    blend_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    seed: int = 17
    gbdt_candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
