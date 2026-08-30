"""Compact per-fold numeric cache for the all-rugby blend search.

``target_loss`` consumes only the predicted *mean*, so the competition-balanced
stable score is a pure function of the blended means. Caching
(actual, naive, empirical mean, v4 mean, validity, cohort masks) per fold and
target therefore makes the whole weight search a vectorised numpy computation
with no model or feature work in the loop.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import allrugby as A
from .allrugby_blend import ALL_TARGETS
from .config import STABLE_EVENTS
from .folds import build_folds
from .metrics import prediction_means, target_loss

CACHE_PATH = A.WORK / "fold_cache.pkl"
SCORED = ("minutes", *STABLE_EVENTS)


@dataclass
class FoldCache:
    label: str
    tournament: str
    calendar_year: int
    hemisphere: str
    actual: dict[str, np.ndarray]
    naive: dict[str, np.ndarray]
    empirical: dict[str, np.ndarray]
    v4: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    cohorts: dict[str, np.ndarray]
    n_rows: int


def build_cache() -> list[FoldCache]:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as handle:
            return pickle.load(handle)
    store = A._store()
    caches = []
    for fold in build_folds(store):
        evaluation, naive_model = A.evaluation_context(store, fold)
        empirical = A.read_predictions(A._component_path(fold.label, "empirical"))
        v4 = A.read_predictions(A._component_path(fold.label, "v4"))
        actual, naive, emp_means, v4_means, valid = {}, {}, {}, {}, {}
        for target in ALL_TARGETS:
            available = evaluation.get(
                f"available__{target}", pd.Series(False, index=evaluation.index),
            )
            actual_all = pd.to_numeric(evaluation[target], errors="coerce").to_numpy(float)
            emp_all = prediction_means(empirical, target)
            v4_all = prediction_means(v4, target)
            naive_all = naive_model.predict(evaluation, target)
            ok = available.fillna(False).astype(bool).to_numpy() & np.isfinite(actual_all)
            ok &= np.isfinite(emp_all) & np.isfinite(v4_all) & np.isfinite(naive_all)
            actual[target], naive[target] = actual_all, naive_all
            emp_means[target], v4_means[target], valid[target] = emp_all, v4_all, ok
        minutes = pd.to_numeric(evaluation["minutes"], errors="coerce").fillna(0).to_numpy(float)
        started = evaluation["started"].fillna(False).astype(bool).to_numpy()
        career = pd.to_numeric(
            evaluation.get("career_matches", 0), errors="coerce",
        ).fillna(0).to_numpy(float)
        cohorts = {
            "all": np.ones(len(evaluation), dtype=bool),
            "starter": started, "bench": ~started, "zero_minute": minutes <= 0,
            "low_history": career < 5,
            "north": evaluation["hemisphere"].eq("north").to_numpy(),
            "south": evaluation["hemisphere"].eq("south").to_numpy(),
        }
        caches.append(FoldCache(
            label=fold.label, tournament=fold.tournament,
            calendar_year=fold.calendar_year, hemisphere=fold.hemisphere,
            actual=actual, naive=naive, empirical=emp_means, v4=v4_means,
            valid=valid, cohorts=cohorts, n_rows=len(evaluation),
        ))
        print(f"[{fold.label}] cached {len(evaluation):,} rows", flush=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as handle:
        pickle.dump(caches, handle)
    return caches


def blended_mean(
    cache: FoldCache, target: str, weight: float, *, log_space: bool = False,
) -> np.ndarray:
    left, right = cache.v4[target], cache.empirical[target]
    if log_space:
        return np.expm1(
            weight * np.log1p(np.clip(left, 0, None))
            + (1.0 - weight) * np.log1p(np.clip(right, 0, None))
        )
    return weight * left + (1.0 - weight) * right


def relative_loss(
    cache: FoldCache, target: str, weight: float, *,
    cohort: str = "all", log_space: bool = False,
) -> float:
    mask = cache.valid[target] & cache.cohorts[cohort]
    if not mask.any():
        return np.nan
    predicted = blended_mean(cache, target, weight, log_space=log_space)[mask]
    actual = cache.actual[target][mask]
    loss = target_loss(target, actual, predicted)
    naive_loss = target_loss(target, actual, cache.naive[target][mask])
    return loss / max(naive_loss, 1e-12)


def fold_stable_score(
    cache: FoldCache, weights: dict[str, float], default: float = 0.5, *,
    cohort: str = "all", log_targets: frozenset[str] = frozenset(),
) -> float:
    values = [
        relative_loss(
            cache, target, float(weights.get(target, default)),
            cohort=cohort, log_space=target in log_targets,
        )
        for target in SCORED
    ]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def stable_score(
    caches: list[FoldCache], weights: dict[str, float], default: float = 0.5, *,
    cohort: str = "all", log_targets: frozenset[str] = frozenset(),
) -> float:
    per_fold = [
        fold_stable_score(
            cache, weights, default, cohort=cohort, log_targets=log_targets,
        )
        for cache in caches
    ]
    per_fold = [value for value in per_fold if np.isfinite(value)]
    return float(np.mean(per_fold))
