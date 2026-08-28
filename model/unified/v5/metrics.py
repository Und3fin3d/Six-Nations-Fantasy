"""Evaluation helpers for the v5 diagnostics (plan §8, E0a/E1/E3).

Pure pandas/numpy; no model imports. These implement the precommitted
*diagnostics* (never gates beyond the existing safe-overall rule):
position-level bias tables, low-history slices, matched-only cohorts, and the
E1 scorer-ablation counterfactual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import RawPrediction
from ..scoring import scorer_for


def low_history_mask(
    frame: pd.DataFrame, career_col: str = "career_matches", threshold: float = 5.0,
) -> pd.Series:
    """Rows whose prior-history count is below ``threshold`` (plan gate cohort)."""
    return (
        pd.to_numeric(frame[career_col], errors="coerce").fillna(0.0) < threshold
    )


def first_cap_club_rich_mask(
    frame: pd.DataFrame,
    intl_col: str = "hist_intl_matches",
    club_col: str = "hist_club_matches",
    min_club_matches: float = 3.0,
) -> pd.Series:
    """The E3 slice (i): zero prior internationals but enough prior club rows.

    Counts are the ewm-weighted experience measures emitted by
    `v5.shrinkage` (most recent prior match weight 1), so a threshold of 3 is
    roughly four raw club matches.
    """
    intl = pd.to_numeric(frame[intl_col], errors="coerce").fillna(0.0)
    club = pd.to_numeric(frame[club_col], errors="coerce").fillna(0.0)
    return intl.eq(0) & club.ge(min_club_matches)


def position_bias(
    frame: pd.DataFrame, predicted_col: str, actual_col: str = "official_pts",
    position_col: str = "position",
) -> pd.DataFrame:
    """Per-position error decomposition (the E0a/E1 diagnostic table)."""
    rows = []
    for position, block in frame.groupby(position_col):
        pred = pd.to_numeric(block[predicted_col], errors="coerce")
        actual = pd.to_numeric(block[actual_col], errors="coerce")
        valid = pred.notna() & actual.notna()
        if not valid.any():
            continue
        error = pred[valid] - actual[valid]
        rows.append({
            "position": position, "n": int(valid.sum()),
            "mae": float(error.abs().mean()), "bias": float(error.mean()),
            "mean_predicted": float(pred[valid].mean()),
            "mean_actual": float(actual[valid].mean()),
        })
    return pd.DataFrame(rows).sort_values("bias").reset_index(drop=True)


def subgroup_mae(
    frame: pd.DataFrame, predicted_col: str, actual_col: str, mask: pd.Series,
) -> float:
    pred = pd.to_numeric(frame.loc[mask, predicted_col], errors="coerce")
    actual = pd.to_numeric(frame.loc[mask, actual_col], errors="coerce")
    valid = pred.notna() & actual.notna()
    if not valid.any():
        return float("nan")
    return float((pred[valid] - actual[valid]).abs().mean())


def ablated_expected_points(
    predictions: list[RawPrediction], competition: str,
    drop_events: tuple[str, ...] = (),
) -> np.ndarray:
    """Expected points with selected event means zeroed (E1 counterfactual).

    Scores the mean event vector through the deterministic scorer — identical
    to the sampling mean for the linear NCR scorer; the 6N scorer's
    ``floor(metres/10)`` term makes it a close approximation there. Used to
    measure how much of a capture/MAE gap is scorer-attribution bias without
    retraining (plan experiment E1).
    """
    scorer = scorer_for(competition)
    drop = set(drop_events)
    rows = []
    for prediction in predictions:
        events = {
            name: np.array([0.0 if name in drop else dist.mean])
            for name, dist in prediction.events.items()
        }
        rows.append(float(scorer.score_samples(events, is_forward=prediction.is_forward)[0]))
    return np.asarray(rows)


def paired_bootstrap_ci(
    actual: np.ndarray, candidate: np.ndarray, reference: np.ndarray,
    *, n_boot: int = 2000, seed: int = 17,
) -> dict[str, float]:
    """Paired bootstrap of candidate-minus-reference MAE difference."""
    rng = np.random.default_rng(seed)
    n = len(actual)
    if not n:
        return {"mean": float("nan"), "p05": float("nan"), "p95": float("nan")}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.mean(np.abs(candidate[idx] - actual[idx])) - np.mean(
            np.abs(reference[idx] - actual[idx])
        )
    return {
        "mean": float(diffs.mean()),
        "p05": float(np.quantile(diffs, 0.05)),
        "p95": float(np.quantile(diffs, 0.95)),
    }
