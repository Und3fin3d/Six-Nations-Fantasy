"""Precomputed relative-loss table over the blend-weight grid.

The stable score reads only the blended mean, and the naive comparator's loss
does not depend on the blend weight. So for every (fold, target, cohort) the
masked arrays and the naive denominator can be computed once and only the
numerator re-evaluated across the weight grid. That turns every weight fitter
into numpy indexing and makes nested cross-validation cheap enough to be
honest.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

from .allrugby import WORK
from .allrugby_cache import FoldCache, SCORED
from .metrics import target_loss

TABLE_PATH = WORK / "loss_table.pkl"
GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 4)
COHORTS = ("all", "north", "south")


@dataclass
class LossTable:
    grid: np.ndarray
    targets: tuple[str, ...]
    folds: tuple[str, ...]
    tournaments: tuple[str, ...]
    hemispheres: tuple[str, ...]
    #: cohort -> array of shape (n_folds, n_targets, n_weights)
    mean_space: dict[str, np.ndarray]
    log_space: dict[str, np.ndarray]
    #: cohort -> boolean array (n_folds, n_targets) marking usable cells
    usable: dict[str, np.ndarray]

    def stable(self, cohort: str, weight_index: np.ndarray, *, log: bool = False) -> np.ndarray:
        """Per-fold stable score for one weight index per target."""
        block = (self.log_space if log else self.mean_space)[cohort]
        chosen = np.take_along_axis(
            block, np.asarray(weight_index)[None, :, None], axis=2,
        )[:, :, 0]
        usable = self.usable[cohort]
        chosen = np.where(usable, chosen, np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(chosen, axis=1)

    def index_of(self, weight: float) -> int:
        return int(np.argmin(np.abs(self.grid - float(weight))))


def _fill(cache: FoldCache, target: str, cohort: str, log: bool) -> tuple[np.ndarray, bool]:
    mask = cache.valid[target] & cache.cohorts[cohort]
    if not mask.any():
        return np.full(len(GRID), np.nan), False
    actual = cache.actual[target][mask]
    left = np.clip(cache.v4[target][mask], 0, None)
    right = np.clip(cache.empirical[target][mask], 0, None)
    naive_loss = max(target_loss(target, actual, cache.naive[target][mask]), 1e-12)
    if log:
        log_left, log_right = np.log1p(left), np.log1p(right)
    values = np.empty(len(GRID))
    for index, weight in enumerate(GRID):
        if log:
            predicted = np.expm1(weight * log_left + (1.0 - weight) * log_right)
        else:
            predicted = weight * left + (1.0 - weight) * right
        values[index] = target_loss(target, actual, predicted) / naive_loss
    return values, True


def build_table(caches: list[FoldCache]) -> LossTable:
    if TABLE_PATH.exists():
        with TABLE_PATH.open("rb") as handle:
            return pickle.load(handle)
    n_folds, n_targets, n_weights = len(caches), len(SCORED), len(GRID)
    mean_space = {c: np.full((n_folds, n_targets, n_weights), np.nan) for c in COHORTS}
    log_space = {c: np.full((n_folds, n_targets, n_weights), np.nan) for c in COHORTS}
    usable = {c: np.zeros((n_folds, n_targets), dtype=bool) for c in COHORTS}
    for fold_index, cache in enumerate(caches):
        for target_index, target in enumerate(SCORED):
            for cohort in COHORTS:
                values, ok = _fill(cache, target, cohort, log=False)
                mean_space[cohort][fold_index, target_index] = values
                usable[cohort][fold_index, target_index] = ok
                log_values, _ = _fill(cache, target, cohort, log=True)
                log_space[cohort][fold_index, target_index] = log_values
        print(f"[{cache.label}] loss table filled", flush=True)
    table = LossTable(
        grid=GRID, targets=tuple(SCORED),
        folds=tuple(cache.label for cache in caches),
        tournaments=tuple(cache.tournament for cache in caches),
        hemispheres=tuple(cache.hemisphere for cache in caches),
        mean_space=mean_space, log_space=log_space, usable=usable,
    )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open("wb") as handle:
        pickle.dump(table, handle)
    return table
