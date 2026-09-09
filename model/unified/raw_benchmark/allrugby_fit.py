"""Table-driven weight fitters and cross-fitted evaluation.

Every fitter returns one weight-grid index per scored target. Fitting is always
restricted to a subset of folds; the caller decides which. Because the fitters
are pure numpy over the precomputed loss table, nested cross-validation is
cheap enough to run honestly rather than being approximated away.
"""

from __future__ import annotations

import numpy as np

from .allrugby_table import COHORTS, GRID, LossTable
from ..schema import distribution_family


def _mean_over(table: LossTable, folds: np.ndarray, cohort: str = "all", *, log: bool = False):
    """Mean relative loss over the given folds -> array (n_targets, n_weights)."""
    block = (table.log_space if log else table.mean_space)[cohort][folds]
    usable = table.usable[cohort][folds]
    block = np.where(usable[:, :, None], block, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(block, axis=0)


def fit_global(table: LossTable, folds: np.ndarray, *, log: bool = False) -> np.ndarray:
    curve = np.nanmean(_mean_over(table, folds, log=log), axis=0)
    best = int(np.nanargmin(curve))
    return np.full(len(table.targets), best, dtype=int)


def fit_per_target(
    table: LossTable, folds: np.ndarray, *, shrink: float = 0.0, log: bool = False,
) -> np.ndarray:
    surface = _mean_over(table, folds, log=log)
    raw = np.nanargmin(surface, axis=1)
    if shrink <= 0.0:
        return raw.astype(int)
    anchor = fit_global(table, folds, log=log)[0]
    blended = (1.0 - shrink) * GRID[raw] + shrink * GRID[anchor]
    return np.abs(GRID[None, :] - blended[:, None]).argmin(axis=1).astype(int)


def fit_family(table: LossTable, folds: np.ndarray, *, log: bool = False) -> np.ndarray:
    surface = _mean_over(table, folds, log=log)
    groups: dict[str, list[int]] = {}
    for index, target in enumerate(table.targets):
        key = "minutes" if target == "minutes" else distribution_family(target)
        groups.setdefault(key, []).append(index)
    chosen = np.zeros(len(table.targets), dtype=int)
    for indices in groups.values():
        curve = np.nanmean(surface[indices], axis=0)
        chosen[indices] = int(np.nanargmin(curve))
    return chosen


def fit_density(
    table: LossTable, folds: np.ndarray, density: np.ndarray, *, log: bool = False,
) -> np.ndarray:
    """Two parameters: w(target) = clip(a + b * positive_rate, 0, 1)."""
    surface = _mean_over(table, folds, log=log)
    best, best_score = None, np.inf
    for a in np.round(np.arange(0.0, 0.901, 0.02), 3):
        for b in np.round(np.arange(-0.6, 0.901, 0.02), 3):
            weights = np.clip(a + b * density, 0.0, 1.0)
            indices = np.abs(GRID[None, :] - weights[:, None]).argmin(axis=1)
            score = float(np.nanmean(surface[np.arange(len(indices)), indices]))
            if score < best_score:
                best, best_score = indices, score
    return best.astype(int)


def fit_shrunk_per_target(table: LossTable, folds: np.ndarray, *, log: bool = False) -> np.ndarray:
    """Per-target weights with the shrinkage strength chosen by inner LOFO.

    The inner pass sees only ``folds``; the held-out fold of the outer loop
    never informs either the weights or the shrinkage strength.
    """
    lambdas = np.round(np.arange(0.0, 1.001, 0.05), 3)
    scores = []
    for value in lambdas:
        inner = []
        for position in range(len(folds)):
            inner_train = np.delete(folds, position)
            indices = fit_per_target(table, inner_train, shrink=float(value), log=log)
            inner.append(fold_score(table, folds[position], indices, log=log))
        scores.append(float(np.nanmean(inner)))
    chosen = float(lambdas[int(np.nanargmin(scores))])
    result = fit_per_target(table, folds, shrink=chosen, log=log)
    return result, chosen


def fold_score(
    table: LossTable, fold_index: int, weight_index: np.ndarray, *,
    cohort: str = "all", log: bool = False,
) -> float:
    block = (table.log_space if log else table.mean_space)[cohort]
    values = block[fold_index, np.arange(len(weight_index)), weight_index]
    values = np.where(table.usable[cohort][fold_index], values, np.nan)
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(values))


def all_fold_scores(
    table: LossTable, weight_index_per_fold, *, cohort: str = "all", log: bool = False,
) -> np.ndarray:
    """``weight_index_per_fold`` is a list of one weight-index array per fold."""
    return np.array([
        fold_score(table, index, weight_index_per_fold[index], cohort=cohort, log=log)
        for index in range(len(table.folds))
    ])


def cross_fit(table: LossTable, fitter, *, log: bool = False):
    """Leave-one-fold-out: the weights applied to a fold exclude that fold."""
    per_fold, fits = [], []
    for index in range(len(table.folds)):
        train = np.delete(np.arange(len(table.folds)), index)
        fitted = fitter(table, train)
        indices, extra = fitted if isinstance(fitted, tuple) else (fitted, None)
        per_fold.append(indices)
        fits.append({
            "held_out_fold": table.folds[index],
            "weights": {
                target: float(GRID[indices[position]])
                for position, target in enumerate(table.targets)
            },
            "hyperparameter": extra,
        })
    return per_fold, fits
