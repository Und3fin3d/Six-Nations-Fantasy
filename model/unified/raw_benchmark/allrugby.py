"""All-rugby hill-climb harness for the P3 unified event model.

Extracts the empirical and v4 component predictions out of the self-contained
frozen ``p3_event_50`` blend artifacts, so arbitrary blend rules can be scored
offline against the Historical Raw Benchmark v1 metric without refitting
anything. The frozen v1 ledger is never written to.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import RawPrediction
from .blend import EventBlend50
from .config import EXTENDED_EVENTS, OUT, STABLE_EVENTS
from .features import build_frozen_feature_frames
from .folds import (
    build_folds, evaluation_frame, masked_candidates, strict_training_frame,
)
from .metrics import NaiveComparator, event_metrics, ranking_metrics

WORK = OUT.parent / "allrugby_p3"
COMPONENTS = ("empirical", "v4")


def _store() -> pd.DataFrame:
    return pd.read_csv(
        OUT / "player_match.csv", low_memory=False, parse_dates=["date", "match_at"],
    )


def _component_path(fold_label: str, component: str) -> Path:
    return WORK / "components" / component / f"{fold_label}.jsonl"


def _write_predictions(path: Path, predictions: list[RawPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(prediction.to_dict(), sort_keys=True) + "\n"
        for prediction in predictions
    ))


def read_predictions(path: Path) -> list[RawPrediction]:
    return [
        RawPrediction.from_dict(json.loads(line))
        for line in path.read_text().splitlines() if line
    ]


def extract_components(fold_labels: tuple[str, ...] | None = None) -> None:
    """Recover per-fold empirical and v4 predictions from the frozen P3 blends."""
    store = _store()
    folds = build_folds(store)
    if fold_labels:
        folds = [fold for fold in folds if fold.label in set(fold_labels)]
    for fold in folds:
        targets = {
            component: _component_path(fold.label, component)
            for component in COMPONENTS
        }
        if all(path.exists() for path in targets.values()):
            print(f"[{fold.label}] components cached", flush=True)
            continue
        evaluation = evaluation_frame(store, fold).reset_index(drop=True)
        train = strict_training_frame(store, fold)
        candidates = masked_candidates(evaluation)
        print(f"[{fold.label}] building v4 features ({len(train):,} train rows)", flush=True)
        _, candidate_features = build_frozen_feature_frames(train, candidates, v4=True)
        blend = EventBlend50.load(OUT / "models" / "p3_event_50" / f"{fold.label}.pkl")
        for component in COMPONENTS:
            model = getattr(blend, component)
            print(f"[{fold.label}] predicting {component}", flush=True)
            _write_predictions(targets[component], model.predict_frame(candidate_features))


def evaluation_context(store: pd.DataFrame, fold) -> tuple[pd.DataFrame, NaiveComparator]:
    evaluation = evaluation_frame(store, fold).reset_index(drop=True)
    train = strict_training_frame(store, fold)
    history = train.groupby(train["player_id"].astype(str))["fixture_id"].nunique()
    evaluation["career_matches"] = (
        evaluation["player_id"].astype(str).map(history).fillna(0)
    )
    naive = NaiveComparator.fit(train, ("minutes", *STABLE_EVENTS, *EXTENDED_EVENTS))
    return evaluation, naive


def score_predictions(
    evaluation: pd.DataFrame, predictions: list[RawPrediction], naive: NaiveComparator,
    *, engine: str, fold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = event_metrics(
        evaluation, predictions, naive, engine=engine, fold=fold.label,
    ).assign(
        tournament=fold.tournament, calendar_year=fold.calendar_year,
        fold_hemisphere=fold.hemisphere,
    )
    rankings = ranking_metrics(
        evaluation, predictions, engine=engine, fold=fold.label,
    ).assign(
        tournament=fold.tournament, calendar_year=fold.calendar_year,
        fold_hemisphere=fold.hemisphere,
    )
    return events, rankings


def stable_score(events: pd.DataFrame, cohort: str = "all") -> pd.Series:
    """Competition-balanced stable score per engine (mean over folds)."""
    selected = events[
        events["cohort"].eq(cohort) & events["tier"].isin(["stable", "minutes"])
    ]
    per_fold = selected.groupby(["engine", "fold"], as_index=False)["relative_loss"].mean()
    return per_fold.groupby("engine")["relative_loss"].mean()


def per_fold_stable(events: pd.DataFrame, cohort: str = "all") -> pd.DataFrame:
    selected = events[
        events["cohort"].eq(cohort) & events["tier"].isin(["stable", "minutes"])
    ]
    return selected.groupby(
        ["engine", "fold", "tournament", "calendar_year"], as_index=False,
    )["relative_loss"].mean().rename(columns={"relative_loss": "stable_score"})


def bootstrap_difference(differences: np.ndarray, seed: int = 17) -> dict[str, float]:
    """Identical to report._bootstrap_difference (paired, by fold)."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return {"n_clusters": 0, "mean": np.nan, "p05": np.nan, "p95": np.nan}
    rng = np.random.default_rng(seed)
    samples = np.empty(2000)
    for index in range(len(samples)):
        chosen = rng.integers(0, len(differences), len(differences))
        samples[index] = float(np.mean(differences[chosen]))
    return {
        "n_clusters": int(len(differences)), "mean": float(np.mean(differences)),
        "p05": float(np.quantile(samples, 0.05)), "p95": float(np.quantile(samples, 0.95)),
    }
