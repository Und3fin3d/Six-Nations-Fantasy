"""Competition-normalized evaluation and strict promotion gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .contracts import RawPrediction
from .schema import EVENTS
from .scoring import CompetitionScorer


def tie_aware_top_n(predicted: pd.Series, actual: pd.Series, n: int) -> tuple[float, float]:
    """Return fractional top-N overlap and actual-point capture."""
    n = min(n, len(actual))
    selected = set(predicted.nlargest(n).index)
    cutoff = actual.nlargest(n).iloc[-1]
    above = set(actual[actual > cutoff].index)
    tied = set(actual[actual == cutoff].index)
    remaining = n - len(above)
    tie_weight = remaining / max(len(tied), 1)
    hits = len(selected & above) + tie_weight * len(selected & tied)
    capture = float(actual.loc[list(selected)].sum() / max(actual.nlargest(n).sum(), 1e-9))
    return float(hits / n), capture


def actual_points(frame: pd.DataFrame, scorer: CompetitionScorer) -> np.ndarray:
    rows = []
    for row in frame.itertuples(index=False):
        events = {
            event: np.array([float(getattr(row, event))])
            for event in EVENTS if hasattr(row, event) and pd.notna(getattr(row, event))
        }
        rows.append(float(scorer.score_samples(events, is_forward=bool(row.is_forward))[0]))
    return np.asarray(rows)


def evaluate_predictions(
    frame: pd.DataFrame, predictions: list[RawPrediction], scorer: CompetitionScorer,
    top_ns: tuple[int, ...] = (10, 25, 50, 100),
) -> dict[str, float]:
    if len(frame) != len(predictions):
        raise ValueError("frame and predictions must have the same length")
    predicted = np.array([scorer.score_prediction(p, n=1200, seed=31 + i).mean
                          for i, p in enumerate(predictions)])
    actual = actual_points(frame, scorer)
    metrics: dict[str, float] = {
        "n": float(len(frame)), "points_mae": float(np.mean(np.abs(predicted - actual))),
        "spearman": float(pd.Series(predicted).corr(pd.Series(actual), method="spearman")),
    }
    pred_s, actual_s = pd.Series(predicted), pd.Series(actual)
    for n in top_ns:
        if len(frame) >= n:
            overlap, capture = tie_aware_top_n(pred_s, actual_s, n)
            metrics[f"top_{n}_overlap"] = overlap
            metrics[f"top_{n}_capture"] = capture
    for event in EVENTS:
        available = frame.get(f"available__{event}", pd.Series(False, index=frame.index)).astype(bool)
        predicted_event = np.array([p.events[event].mean if event in p.events else np.nan for p in predictions])
        valid = available.to_numpy() & np.isfinite(predicted_event)
        if valid.any():
            metrics[f"event_mae__{event}"] = float(np.mean(np.abs(
                predicted_event[valid] - frame.loc[valid, event].to_numpy(float))))
    minutes = np.array([p.minutes.mean for p in predictions])
    valid_min = frame["available__minutes"].astype(bool).to_numpy()
    metrics["minutes_mae"] = float(np.mean(np.abs(minutes[valid_min] - frame.loc[valid_min, "minutes"])))
    return metrics


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    reasons: tuple[str, ...]


def promotion_gate(
    candidate: dict[str, dict[str, float]], incumbent: dict[str, dict[str, float]],
    *, noninferiority_margin: float = .02, required_capture_gain: float = .03,
) -> PromotionDecision:
    """Require non-inferiority in every competition and aggregate rank gain."""
    reasons = []
    gains = []
    for competition in sorted(incumbent):
        if competition not in candidate:
            reasons.append(f"missing candidate results for {competition}")
            continue
        c, b = candidate[competition], incumbent[competition]
        if c["points_mae"] > b["points_mae"] * (1 + noninferiority_margin):
            reasons.append(f"{competition} points MAE regressed by more than {noninferiority_margin:.0%}")
        for key in ("top_10_capture", "top_25_capture", "top_50_capture", "top_100_capture"):
            if key in b and key in c:
                if c[key] < b[key] - noninferiority_margin:
                    reasons.append(f"{competition} {key} regressed by more than {noninferiority_margin:.0%}")
                gains.append(c[key] - b[key])
    if not gains or float(np.mean(gains)) < required_capture_gain:
        reasons.append(f"mean top-N point-capture gain is below {required_capture_gain:.0%}")
    return PromotionDecision(not reasons, tuple(reasons))
