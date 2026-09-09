"""Reconstructed-rubric metrics for all-rugby blend candidates.

Standing project requirement: never report a raw score without the
reconstructed MAE for BOTH rubrics. This runs the unmodified
``ranking_metrics`` over blended component predictions, so the numbers are
directly comparable to the frozen v1 ledger.

Rules are supplied per fold, so a cross-fitted candidate is scored on each fold
with the weights that were fitted *without* that fold -- the rubric comparison
is out-of-sample exactly like the raw score it accompanies.

``mae_calibrated`` applies a leave-one-fold-out affine recalibration of
predicted points onto actual points: the slope and intercept applied to a fold
are fitted on every *other* fold's rows, so it is never in-sample either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import allrugby as A
from .allrugby_blend import BlendRule
from .config import OUT, STABLE_EVENTS, TOP_NS
from .folds import build_folds
from .metrics import (
    NationsChampionshipScorer, SixNationsScorer, _score_rows, _scorer_events,
    ranking_metrics,
)


def _mean_capture(frame: pd.DataFrame) -> pd.Series:
    columns = [f"top_{n}_capture" for n in TOP_NS if f"top_{n}_capture" in frame]
    return frame[columns].mean(axis=1, skipna=True)


def collect(rule_for: dict[str, callable]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``rule_for[name](fold_label) -> BlendRule``. Returns (rankings, points)."""
    store = A._store()
    rank_frames, point_frames = [], []
    for fold in build_folds(store):
        evaluation, _ = A.evaluation_context(store, fold)
        empirical = A.read_predictions(A._component_path(fold.label, "empirical"))
        v4 = A.read_predictions(A._component_path(fold.label, "v4"))
        for name, factory in rule_for.items():
            predictions = factory(fold.label).apply(empirical, v4)
            rank_frames.append(ranking_metrics(
                evaluation, predictions, engine=name, fold=fold.label,
            ).assign(tournament=fold.tournament, fold_hemisphere=fold.hemisphere))
            for rubric, scorer in (
                ("six_nations", SixNationsScorer()), ("ncr", NationsChampionshipScorer()),
            ):
                allowed = _scorer_events(scorer, STABLE_EVENTS)
                for slate_id, slate in evaluation.groupby("slate_id", sort=True):
                    local_predictions = [
                        predictions[int(index)] for index in slate.index.to_numpy()
                    ]
                    _, actual, predicted = _score_rows(
                        slate.reset_index(drop=True), local_predictions, scorer, allowed,
                    )
                    if not len(actual):
                        continue
                    point_frames.append(pd.DataFrame({
                        "engine": name, "fold": fold.label, "rubric": rubric,
                        "slate_id": str(slate_id), "actual": actual, "predicted": predicted,
                    }))
        print(f"[{fold.label}] rubric metrics for {len(rule_for)} rule(s)", flush=True)
    return (
        pd.concat(rank_frames, ignore_index=True, sort=False),
        pd.concat(point_frames, ignore_index=True, sort=False),
    )


def summarise(rankings: pd.DataFrame, tier: str = "stable") -> pd.DataFrame:
    block = rankings[rankings["tier"].eq(tier)].copy()
    block["mean_capture"] = _mean_capture(block)
    return block.groupby(["engine", "rubric"], as_index=False).agg(
        mae=("mae", "mean"), spearman=("spearman", "mean"),
        mean_capture=("mean_capture", "mean"), slates=("slate_id", "nunique"),
    )


def calibrated_mae(points: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-fold-out affine recalibration, then per-slate MAE."""
    rows = []
    for (engine, rubric), block in points.groupby(["engine", "rubric"], sort=True):
        for fold in sorted(block["fold"].unique()):
            held = block[block["fold"].eq(fold)]
            other = block[~block["fold"].eq(fold)]
            if len(other) < 2 or float(np.var(other["predicted"])) < 1e-12:
                slope, intercept = 1.0, 0.0
            else:
                slope, intercept = np.polyfit(other["predicted"], other["actual"], 1)
            frame = held.assign(
                calibrated=intercept + slope * held["predicted"].to_numpy(float),
            )
            for slate_id, slate in frame.groupby("slate_id", sort=True):
                rows.append({
                    "engine": engine, "rubric": rubric, "fold": fold, "slate_id": slate_id,
                    "mae_raw": float(np.mean(np.abs(slate["actual"] - slate["predicted"]))),
                    "mae_calibrated": float(np.mean(np.abs(slate["actual"] - slate["calibrated"]))),
                })
    frame = pd.DataFrame(rows)
    return frame.groupby(["engine", "rubric"], as_index=False).agg(
        mae_raw=("mae_raw", "mean"), mae_calibrated=("mae_calibrated", "mean"),
        slates=("slate_id", "nunique"),
    )


def paired_by_slate(rankings: pd.DataFrame, candidate: str, baseline: str, metric: str,
                    tier: str = "stable") -> dict[str, dict]:
    """Paired-by-slate bootstrap of a rubric metric, candidate minus baseline."""
    block = rankings[rankings["tier"].eq(tier)].copy()
    block["mean_capture"] = _mean_capture(block)
    out = {}
    for rubric in sorted(block["rubric"].dropna().unique()):
        pivot = block[block["rubric"].eq(rubric)].pivot_table(
            index="slate_id", columns="engine", values=metric,
        )
        if candidate not in pivot or baseline not in pivot:
            continue
        paired = pivot[[candidate, baseline]].dropna()
        out[rubric] = A.bootstrap_difference(
            paired[candidate].to_numpy(float) - paired[baseline].to_numpy(float)
        )
    return out


def frozen_reference(tier: str = "stable") -> pd.DataFrame:
    """The frozen v1 ledger's own rubric aggregates, for cross-checking."""
    return summarise(pd.read_csv(OUT / "ranking_metrics.csv"), tier)
