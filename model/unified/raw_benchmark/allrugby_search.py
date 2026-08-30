"""Diagnostics and the greedy hill-climb for the all-rugby P3 blend.

Acceptance is precommitted (see LEDGER.md). Any newly fitted parameter is
cross-fitted: the weight applied to a fold is never fitted on that fold.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import allrugby as A
from .allrugby_cache import (
    FoldCache, SCORED, blended_mean, build_cache, fold_stable_score, relative_loss,
)
from .config import SEED, STABLE_EVENTS
from .metrics import target_loss

FROZEN_BASELINE = 0.8864704364786675
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 4)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def per_target_table(caches: list[FoldCache]) -> pd.DataFrame:
    """Per-target relative loss at w=0 (empirical), 0.5 (P3) and 1 (v4)."""
    rows = []
    for target in SCORED:
        record = {"target": target}
        for name, weight in (("empirical", 0.0), ("p3_50", 0.5), ("v4", 1.0)):
            values = [relative_loss(cache, target, weight) for cache in caches]
            record[name] = float(np.nanmean(values))
        curve = [
            float(np.nanmean([relative_loss(cache, target, w) for cache in caches]))
            for w in GRID
        ]
        best = int(np.argmin(curve))
        record["best_w_insample"] = float(GRID[best])
        record["best_rel_insample"] = curve[best]
        record["gain_vs_p3"] = record["p3_50"] - curve[best]
        positive = [
            float(np.mean(cache.actual[target][cache.valid[target]] > 0))
            for cache in caches if cache.valid[target].any()
        ]
        record["positive_rate"] = float(np.mean(positive))
        rows.append(record)
    return pd.DataFrame(rows)


def per_fold_table(caches: list[FoldCache], weights=None, default=0.5) -> pd.DataFrame:
    weights = weights or {}
    rows = []
    for cache in caches:
        rows.append({
            "fold": cache.label, "tournament": cache.tournament,
            "hemisphere": cache.hemisphere,
            "stable": fold_stable_score(cache, weights, default),
            "north": fold_stable_score(cache, weights, default, cohort="north"),
            "south": fold_stable_score(cache, weights, default, cohort="south"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# fitting (always on a strict subset of folds)
# --------------------------------------------------------------------------

def fit_global_weight(train: list[FoldCache]) -> float:
    curve = [
        float(np.nanmean([fold_stable_score(cache, {}, w) for cache in train]))
        for w in GRID
    ]
    return float(GRID[int(np.argmin(curve))])


def fit_per_target_weights(train: list[FoldCache], shrink: float = 0.0) -> dict[str, float]:
    """Per-target weight, optionally shrunk toward the fitted global weight.

    ``shrink`` = 0 keeps the raw per-target minimum; 1 collapses to the global
    weight. The stable score is a plain mean over targets, so each target's
    weight can be optimised independently.
    """
    anchor = fit_global_weight(train)
    weights = {}
    for target in SCORED:
        curve = [
            float(np.nanmean([relative_loss(cache, target, w) for cache in train]))
            for w in GRID
        ]
        raw = float(GRID[int(np.argmin(curve))])
        weights[target] = (1.0 - shrink) * raw + shrink * anchor
    return weights


def fit_family_weights(train: list[FoldCache]) -> dict[str, float]:
    """One weight per loss family: minutes (MAE), counts (deviance), metres."""
    from ..schema import distribution_family
    groups: dict[str, list[str]] = {}
    for target in SCORED:
        if target == "minutes":
            key = "minutes"
        else:
            key = distribution_family(target)
        groups.setdefault(key, []).append(target)
    weights = {}
    for key, targets in groups.items():
        curve = [
            float(np.nanmean([
                np.nanmean([relative_loss(cache, target, w) for target in targets])
                for cache in train
            ]))
            for w in GRID
        ]
        best = float(GRID[int(np.argmin(curve))])
        for target in targets:
            weights[target] = best
    return weights


# --------------------------------------------------------------------------
# cross-fitted evaluation
# --------------------------------------------------------------------------

def cross_fitted_scores(
    caches: list[FoldCache], fitter, *, cohort: str = "all",
) -> tuple[pd.DataFrame, list[dict]]:
    """Leave-one-fold-out: the weight applied to a fold is fitted without it."""
    rows, fits = [], []
    for index, cache in enumerate(caches):
        train = [other for position, other in enumerate(caches) if position != index]
        fitted = fitter(train)
        weights, default = (
            (fitted, 0.5) if isinstance(fitted, dict) else ({}, float(fitted))
        )
        fits.append({
            "held_out_fold": cache.label,
            "weights": weights or {"__global__": default},
        })
        rows.append({
            "fold": cache.label, "tournament": cache.tournament,
            "hemisphere": cache.hemisphere,
            "stable": fold_stable_score(cache, weights, default, cohort=cohort),
            "north": fold_stable_score(cache, weights, default, cohort="north"),
            "south": fold_stable_score(cache, weights, default, cohort="south"),
        })
    return pd.DataFrame(rows), fits


def temporal_split(caches: list[FoldCache], fraction: float = 0.5):
    """Earliest folds fit, latest folds held out (folds are chronological)."""
    cut = int(len(caches) * fraction)
    return caches[:cut], caches[cut:]


def baseline_frame(caches: list[FoldCache]) -> pd.DataFrame:
    return per_fold_table(caches, {}, 0.5)


def evaluate_candidate(
    caches: list[FoldCache], candidate: pd.DataFrame, baseline: pd.DataFrame,
    *, name: str,
) -> dict:
    """Apply the precommitted acceptance rule to a cross-fitted candidate."""
    merged = baseline.merge(
        candidate, on=["fold", "tournament", "hemisphere"], suffixes=("_base", "_cand"),
    )
    if len(merged) != len(baseline):
        raise ValueError("candidate/baseline fold cohorts differ")
    score = float(merged["stable_cand"].mean())
    base_score = float(merged["stable_base"].mean())
    differences = merged["stable_cand"].to_numpy(float) - merged["stable_base"].to_numpy(float)
    boot = A.bootstrap_difference(differences, seed=SEED)
    reasons = []
    if score >= FROZEN_BASELINE:
        reasons.append(
            f"stable_score {score:.6f} did not improve on frozen {FROZEN_BASELINE:.6f}"
        )
    if not (boot["p05"] < 0 and boot["p95"] < 0):
        reasons.append(
            f"paired-by-fold bootstrap CI [{boot['p05']:.6f}, {boot['p95']:.6f}] includes 0"
        )
    for cohort in ("north", "south"):
        base = float(merged[f"{cohort}_base"].mean())
        cand = float(merged[f"{cohort}_cand"].mean())
        if cand > base * 1.02:
            reasons.append(f"{cohort} cohort regressed by more than 2%")
    tournaments = merged.groupby("tournament")[["stable_base", "stable_cand"]].mean()
    regressed = tournaments.index[
        tournaments["stable_cand"] > tournaments["stable_base"] * 1.02
    ].tolist()
    if regressed:
        reasons.append("tournament-family stable regression >2%: " + ", ".join(regressed))
    return {
        "name": name, "stable_score": score, "baseline_stable_score": base_score,
        "delta": score - base_score,
        "improvement_vs_frozen": 1.0 - score / FROZEN_BASELINE,
        "bootstrap": boot,
        "north": float(merged["north_cand"].mean()), "south": float(merged["south_cand"].mean()),
        "north_base": float(merged["north_base"].mean()),
        "south_base": float(merged["south_base"].mean()),
        "per_tournament": tournaments.to_dict(orient="index"),
        "accepted": not reasons, "reasons": reasons,
    }
