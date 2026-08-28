"""Raw-event, observable-points, uncertainty, and promotion metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..contracts import RawPrediction
from ..schema import EVENTS, distribution_family
from ..scoring import NationsChampionshipScorer, SixNationsScorer


def scoring_importance() -> dict[str, float]:
    scorers = (NationsChampionshipScorer(), SixNationsScorer())
    weights = {event: 1.0 for event in EVENTS}
    for scorer in scorers:
        for event, weight in scorer.weights.items():
            if event in weights:
                weights[event] = max(weights[event], abs(float(weight)))
    weights["tries"] = max(weights["tries"], 15.0)
    weights["metres"] = max(weights["metres"], 0.1)
    weights["scrums_won"] = max(weights["scrums_won"], 2.0)
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def observable_events(frame: pd.DataFrame, threshold: float = 0.80) -> tuple[str, ...]:
    scored = set(NationsChampionshipScorer.weights) | set(SixNationsScorer.weights) | {
        "scrums_won", "metres",
    }
    return tuple(
        event for event in EVENTS
        if event in scored
        and f"available__{event}" in frame
        and float(frame[f"available__{event}"].fillna(False).astype(bool).mean()) >= threshold
    )


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-7, None)
    term = np.where(y > 0, y * np.log(np.clip(y, 1e-7, None) / mu), 0.0)
    return float(np.mean(2 * (term - (y - mu))))


def raw_event_deviance(frame: pd.DataFrame, predictions: list[RawPrediction]) -> dict[str, float]:
    if len(frame) != len(predictions):
        raise ValueError("frame/prediction length mismatch")
    importance = scoring_importance()
    rows: dict[str, float] = {}
    total = 0.0
    used_weight = 0.0
    for event in EVENTS:
        available = frame.get(f"available__{event}", pd.Series(False, index=frame.index))
        valid = available.fillna(False).astype(bool).to_numpy()
        pred = np.array([
            p.events[event].mean if event in p.events else np.nan for p in predictions
        ])
        valid &= np.isfinite(pred)
        if not valid.any():
            continue
        y = pd.to_numeric(frame.loc[valid, event], errors="coerce").to_numpy(float)
        mu = np.clip(pred[valid], 0, None)
        family = distribution_family(event)
        if family == "bernoulli":
            p = np.clip(mu, 1e-7, 1 - 1e-7)
            loss = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
        elif family == "lognormal":
            loss = float(np.mean((np.log1p(y) - np.log1p(mu)) ** 2))
        else:
            loss = _poisson_deviance(y, mu) / max(1.0, float(np.mean(y) + 1))
        rows[f"deviance__{event}"] = loss
        weight = importance[event]
        total += weight * loss
        used_weight += weight
    minutes = np.array([p.minutes.mean for p in predictions])
    valid_min = frame.get("available__minutes", pd.Series(True, index=frame.index)).astype(bool).to_numpy()
    minute_loss = float(np.mean(
        np.abs(minutes[valid_min] - frame.loc[valid_min, "minutes"].to_numpy(float))
    ) / 80.0)
    rows["minutes_scaled_mae"] = minute_loss
    rows["raw_deviance"] = (total / max(used_weight, 1e-12) + minute_loss) / 2
    return rows


def raw_residual_vector(
    frame: pd.DataFrame, predictions: list[RawPrediction],
    events: Iterable[str] = EVENTS,
) -> np.ndarray:
    """Flatten comparable standardized raw-event residuals for diversity checks."""
    residuals = []
    for event in events:
        available = frame.get(f"available__{event}", pd.Series(False, index=frame.index))
        valid = available.fillna(False).astype(bool).to_numpy()
        predicted = np.array([
            prediction.events[event].mean if event in prediction.events else np.nan
            for prediction in predictions
        ])
        valid &= np.isfinite(predicted)
        if not valid.any():
            continue
        actual = pd.to_numeric(frame.loc[valid, event], errors="coerce").to_numpy(float)
        scale = max(float(np.std(actual)), 1.0)
        residuals.append((actual - predicted[valid]) / scale)
    if not residuals:
        return np.empty(0, dtype=float)
    return np.concatenate(residuals)


def observable_points_actual(
    frame: pd.DataFrame, competition: str, allowed_events: Iterable[str],
) -> np.ndarray:
    scorer = SixNationsScorer() if competition == "six_nations" else NationsChampionshipScorer()
    allowed = set(allowed_events)
    rows = []
    for row in frame.itertuples(index=False):
        events = {
            event: np.array([float(getattr(row, event))])
            for event in allowed
            if hasattr(row, event) and pd.notna(getattr(row, event))
        }
        rows.append(float(scorer.score_samples(events, is_forward=bool(row.is_forward))[0]))
    return np.asarray(rows)


def observable_points_predicted(
    predictions: list[RawPrediction], competition: str, allowed_events: Iterable[str],
) -> np.ndarray:
    scorer = SixNationsScorer() if competition == "six_nations" else NationsChampionshipScorer()
    allowed = set(allowed_events)
    rows = []
    for prediction in predictions:
        events = {
            event: np.array([dist.mean])
            for event, dist in prediction.events.items() if event in allowed
        }
        rows.append(float(scorer.score_samples(events, is_forward=prediction.is_forward)[0]))
    return np.asarray(rows)


def paired_bootstrap_difference(
    actual: np.ndarray, candidate: np.ndarray, incumbent: np.ndarray,
    *, n_boot: int = 2000, seed: int = 17,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(actual)
    if not n:
        return {"mean": np.nan, "p05": np.nan, "p95": np.nan}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.mean(np.abs(candidate[idx] - actual[idx])) - np.mean(
            np.abs(incumbent[idx] - actual[idx])
        )
    return {
        "mean": float(diffs.mean()),
        "p05": float(np.quantile(diffs, 0.05)),
        "p95": float(np.quantile(diffs, 0.95)),
    }
