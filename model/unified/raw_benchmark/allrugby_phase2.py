"""Phase-2 all-rugby search: component swaps and multi-component blends.

Phase 1 (``allrugby_p3/LEDGER.md``) fixed the two components and searched the
blend rule; that family terminated at C5, the shrunk per-target weight, at
0.879366. Every remaining phase-1 candidate was an alternative parameterisation
of the same one-dimensional family, so the next real lever is the components.

This module generalises the phase-1 machinery from a fixed (v4, empirical) pair
on a scalar weight grid to an arbitrary ordered tuple of components on a simplex
grid. With two components and the phase-1 grid it reproduces phase 1 exactly,
which is the control.

Nothing here writes to the frozen v1 ledger, and no candidate ever reads
NCR GW4-7.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from . import allrugby as A
from .allrugby_blend import ALL_TARGETS
from .config import EXTENDED_EVENTS, STABLE_EVENTS
from .folds import build_folds
from ..schema import distribution_family
from .metrics import prediction_means, target_loss

WORK = A.WORK / "phase2"
#: Pseudo-component: the benchmark's own training-only position x started x
#: competition_level stratum mean. It is a legitimate point-in-time model, and
#: blending toward it is ordinary shrinkage for targets where neither engine has
#: an edge. Because it is also the metric's denominator, any candidate that
#: leans on it must be judged on the reconstructed rubrics too, never on the raw
#: score alone -- a prediction shrunk to a stratum mean cannot rank players.
NAIVE = "naive"
SCORED = ("minutes", *STABLE_EVENTS)
COHORTS = ("all", "north", "south")
SEED = 17


# --------------------------------------------------------------------------
# per-fold numeric cache over N components
# --------------------------------------------------------------------------

@dataclass
class MultiCache:
    label: str
    tournament: str
    calendar_year: int
    hemisphere: str
    actual: dict[str, np.ndarray]
    naive: dict[str, np.ndarray]
    #: available & finite(actual) & finite(naive) -- independent of the components
    available: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]
    cohorts: dict[str, np.ndarray]
    #: component -> target -> per-row predicted mean
    means: dict[str, dict[str, np.ndarray]]
    position: np.ndarray
    started: np.ndarray
    n_rows: int


def _cache_path(components: tuple[str, ...]) -> "object":
    digest = hashlib.sha256("|".join(components).encode()).hexdigest()[:12]
    return WORK / f"cache_{digest}.pkl"


def build_multi_cache(
    components: tuple[str, ...], *, persist: bool = True,
    fold_labels: tuple[str, ...] | None = None,
) -> list[MultiCache]:
    """Cache actual/naive/component means per fold for the given components."""
    path = _cache_path(components)
    if fold_labels is None and path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    store = A._store()
    caches: list[MultiCache] = []
    wanted = set(fold_labels) if fold_labels else None
    for fold in build_folds(store):
        if wanted is not None and fold.label not in wanted:
            continue
        evaluation, naive_model = A.evaluation_context(store, fold)
        loaded = {
            component: A.read_predictions(A._component_path(fold.label, component))
            for component in components if component != NAIVE
        }
        actual, naive, valid, available_mask = {}, {}, {}, {}
        means = {component: {} for component in components}
        for target in ALL_TARGETS:
            available = evaluation.get(
                f"available__{target}", pd.Series(False, index=evaluation.index),
            )
            actual_all = pd.to_numeric(evaluation[target], errors="coerce").to_numpy(float)
            naive_all = naive_model.predict(evaluation, target)
            base = available.fillna(False).astype(bool).to_numpy() & np.isfinite(actual_all)
            base &= np.isfinite(naive_all)
            ok = base.copy()
            for component in components:
                component_all = (
                    naive_all if component == NAIVE
                    else prediction_means(loaded[component], target)
                )
                means[component][target] = component_all
                ok &= np.isfinite(component_all)
            available_mask[target] = base
            actual[target], naive[target], valid[target] = actual_all, naive_all, ok
        caches.append(MultiCache(
            label=fold.label, tournament=fold.tournament,
            calendar_year=fold.calendar_year, hemisphere=fold.hemisphere,
            actual=actual, naive=naive, available=available_mask, valid=valid,
            cohorts={
                "all": np.ones(len(evaluation), dtype=bool),
                "north": evaluation["hemisphere"].eq("north").to_numpy(),
                "south": evaluation["hemisphere"].eq("south").to_numpy(),
            },
            means=means,
            position=evaluation["position"].astype(str).to_numpy(),
            started=evaluation["started"].fillna(False).astype(bool).to_numpy(),
            n_rows=len(evaluation),
        ))
        print(f"[{fold.label}] phase2 cached {len(evaluation):,} rows", flush=True)
    if persist and fold_labels is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(caches, handle)
    return caches


def project_cache(caches: list[MultiCache], components: tuple[str, ...]) -> list[MultiCache]:
    """Restrict a cache built over a superset of components to ``components``.

    Building a cache re-derives each fold's strict training frame and refits the
    naive comparator over 32 targets, which dominates the run. Every candidate
    draws from the same fold geometry, so the union is built once and every
    component tuple is a projection of it.
    """
    projected = []
    for cache in caches:
        missing = [name for name in components if name not in cache.means]
        if missing:
            raise KeyError(f"cache lacks components {missing}")
        valid = {}
        for target, base in cache.available.items():
            mask = base.copy()
            for name in components:
                mask &= np.isfinite(cache.means[name][target])
            valid[target] = mask
        projected.append(MultiCache(
            label=cache.label, tournament=cache.tournament,
            calendar_year=cache.calendar_year, hemisphere=cache.hemisphere,
            actual=cache.actual, naive=cache.naive, available=cache.available,
            valid=valid, cohorts=cache.cohorts,
            means={name: cache.means[name] for name in components},
            position=cache.position, started=cache.started, n_rows=cache.n_rows,
        ))
    return projected


# --------------------------------------------------------------------------
# weight point sets
# --------------------------------------------------------------------------

def pair_grid(step: float = 0.005) -> np.ndarray:
    """Phase-1 compatible grid: column 0 is the first component's weight."""
    left = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    return np.stack([left, 1.0 - left], axis=1)


def cap_component(points: np.ndarray, index: int, cap: float) -> np.ndarray:
    """Restrict a point set to vectors whose ``index`` weight is at most ``cap``.

    Used to bound how far a candidate may shrink toward the stratum-mean
    comparator: a target driven fully to the comparator cannot rank players
    within a position/started stratum at all, however good its raw loss looks.
    """
    return points[points[:, index] <= cap + 1e-9]


def simplex_grid(n_components: int, step: float = 0.05) -> np.ndarray:
    """All non-negative weight vectors on ``step`` lattice summing to 1."""
    levels = int(round(1.0 / step))
    points = [
        combination
        for combination in product(range(levels + 1), repeat=n_components - 1)
        if sum(combination) <= levels
    ]
    rows = [
        [*(value / levels for value in combination),
         (levels - sum(combination)) / levels]
        for combination in points
    ]
    return np.round(np.asarray(rows, dtype=float), 6)


# --------------------------------------------------------------------------
# loss table
# --------------------------------------------------------------------------

@dataclass
class MultiTable:
    components: tuple[str, ...]
    points: np.ndarray                    # (n_points, n_components)
    targets: tuple[str, ...]
    folds: tuple[str, ...]
    tournaments: tuple[str, ...]
    #: cohort -> (n_folds, n_targets, n_points) relative loss
    loss: dict[str, np.ndarray]
    #: cohort -> (n_folds, n_targets) usable mask
    usable: dict[str, np.ndarray]

    def stable(self, cohort: str, point_index: np.ndarray) -> np.ndarray:
        block = self.loss[cohort]
        chosen = np.take_along_axis(
            block, np.asarray(point_index)[None, :, None], axis=2,
        )[:, :, 0]
        chosen = np.where(self.usable[cohort], chosen, np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(chosen, axis=1)

    def index_of(self, weights) -> int:
        weights = np.asarray(weights, dtype=float)
        return int(np.argmin(np.abs(self.points - weights[None, :]).sum(axis=1)))


def _table_path(components, points, targets) -> "object":
    key = "|".join(components) + f"#{len(points)}#{len(targets)}#{points.sum():.6f}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return WORK / f"table_{digest}.pkl"


def build_multi_table(
    caches: list[MultiCache], components: tuple[str, ...], points: np.ndarray,
    *, targets: tuple[str, ...] = SCORED, persist: bool = True,
) -> MultiTable:
    path = _table_path(components, points, targets)
    if persist and path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    shape = (len(caches), len(targets), len(points))
    loss = {cohort: np.full(shape, np.nan) for cohort in COHORTS}
    usable = {cohort: np.zeros(shape[:2], dtype=bool) for cohort in COHORTS}
    for fold_index, cache in enumerate(caches):
        for target_index, target in enumerate(targets):
            stacked = np.stack(
                [np.clip(cache.means[component][target], 0, None) for component in components],
                axis=1,
            )                                            # (n_rows, n_components)
            for cohort in COHORTS:
                mask = cache.valid[target] & cache.cohorts[cohort]
                if not mask.any():
                    continue
                actual = cache.actual[target][mask]
                naive_loss = max(target_loss(target, actual, cache.naive[target][mask]), 1e-12)
                blended = stacked[mask] @ points.T       # (n_rows, n_points)
                loss[cohort][fold_index, target_index] = (
                    row_losses(target, actual, blended).mean(axis=0) / naive_loss
                )
                usable[cohort][fold_index, target_index] = True
        print(f"[{cache.label}] phase2 loss table filled", flush=True)
    table = MultiTable(
        components=components, points=points, targets=tuple(targets),
        folds=tuple(cache.label for cache in caches),
        tournaments=tuple(cache.tournament for cache in caches),
        loss=loss, usable=usable,
    )
    if persist:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(table, handle)
    return table


# --------------------------------------------------------------------------
# fitters (generic over the point set)
# --------------------------------------------------------------------------

def _surface(table: MultiTable, folds: np.ndarray, cohort: str = "all") -> np.ndarray:
    block = table.loss[cohort][folds]
    block = np.where(table.usable[cohort][folds][:, :, None], block, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(block, axis=0)                 # (n_targets, n_points)


def fit_global(table: MultiTable, folds: np.ndarray) -> np.ndarray:
    curve = np.nanmean(_surface(table, folds), axis=0)
    return np.full(len(table.targets), int(np.nanargmin(curve)), dtype=int)


def fit_per_target(
    table: MultiTable, folds: np.ndarray, *, shrink: float = 0.0,
) -> np.ndarray:
    surface = _surface(table, folds)
    raw = np.nanargmin(surface, axis=1)
    if shrink <= 0.0:
        return raw.astype(int)
    anchor = table.points[fit_global(table, folds)[0]]
    blended = (1.0 - shrink) * table.points[raw] + shrink * anchor[None, :]
    return _snap(table.points, blended)


def _snap(points: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    return np.abs(points[None, :, :] - wanted[:, None, :]).sum(axis=2).argmin(axis=1).astype(int)


def fit_shrunk_per_target(table: MultiTable, folds: np.ndarray):
    """Per-target points with shrinkage chosen by an inner leave-one-fold-out.

    The inner pass sees only ``folds``, so the outer held-out fold informs
    neither the weights nor the shrinkage strength. Surfaces are computed once
    per inner training set and reused across the shrinkage grid.
    """
    lambdas = np.round(np.arange(0.0, 1.001, 0.05), 3)
    prepared = []
    for position in range(len(folds)):
        inner_train = np.delete(folds, position)
        surface = _surface(table, inner_train)
        raw = np.nanargmin(surface, axis=1)
        anchor = table.points[int(np.nanargmin(np.nanmean(surface, axis=0)))]
        prepared.append((folds[position], table.points[raw], anchor))
    scores = []
    for value in lambdas:
        inner = []
        for held, raw_points, anchor in prepared:
            wanted = (1.0 - value) * raw_points + value * anchor[None, :]
            inner.append(fold_score(table, held, _snap(table.points, wanted)))
        scores.append(float(np.nanmean(inner)))
    chosen = float(lambdas[int(np.nanargmin(scores))])
    return fit_per_target(table, folds, shrink=chosen), chosen


def fold_score(
    table: MultiTable, fold_index: int, point_index: np.ndarray, *, cohort: str = "all",
) -> float:
    values = table.loss[cohort][fold_index, np.arange(len(point_index)), point_index]
    values = np.where(table.usable[cohort][fold_index], values, np.nan)
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(values))


def cross_fit(table: MultiTable, fitter):
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
                target: table.points[indices[position]].tolist()
                for position, target in enumerate(table.targets)
            },
            "hyperparameter": extra,
        })
    return per_fold, fits


def cross_fitted_scores(
    table: MultiTable, per_fold, *, cohort: str = "all",
) -> np.ndarray:
    return np.array([
        fold_score(table, index, per_fold[index], cohort=cohort)
        for index in range(len(table.folds))
    ])


def fixed_scores(table: MultiTable, indices: np.ndarray, *, cohort: str = "all") -> np.ndarray:
    return np.array([
        fold_score(table, index, indices, cohort=cohort)
        for index in range(len(table.folds))
    ])


def bootstrap(differences: np.ndarray, seed: int = SEED) -> dict:
    return A.bootstrap_difference(np.asarray(differences, dtype=float), seed=seed)


# --------------------------------------------------------------------------
# stage-2 refinements: per-target calibration and row-varying weights
# --------------------------------------------------------------------------

CALIBRATION_GRID = np.round(np.arange(0.70, 1.3001, 0.01), 4)


def blended_rows(
    cache: MultiCache, components: tuple[str, ...], target: str, weights: np.ndarray,
) -> np.ndarray:
    """``weights`` is (n_components,) or (n_rows, n_components)."""
    stacked = np.stack(
        [np.clip(cache.means[component][target], 0, None) for component in components],
        axis=1,
    )
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        return stacked @ weights
    return (stacked * weights).sum(axis=1)


def target_relative_loss(
    cache: MultiCache, components: tuple[str, ...], target: str, weights: np.ndarray,
    *, scale: float = 1.0, cohort: str = "all",
) -> float:
    mask = cache.valid[target] & cache.cohorts[cohort]
    if not mask.any():
        return np.nan
    predicted = blended_rows(cache, components, target, weights)
    predicted = predicted[mask] if predicted.ndim == 1 else predicted[mask]
    actual = cache.actual[target][mask]
    naive_loss = max(target_loss(target, actual, cache.naive[target][mask]), 1e-12)
    return target_loss(target, actual, scale * predicted) / naive_loss


def fit_calibration(
    caches: list[MultiCache], components: tuple[str, ...], table: MultiTable,
    point_index: np.ndarray, folds: np.ndarray, *, grid: np.ndarray = CALIBRATION_GRID,
) -> np.ndarray:
    """One multiplicative scale per target, chosen on ``folds`` only."""
    scales = np.ones(len(table.targets))
    for position, target in enumerate(table.targets):
        weights = table.points[point_index[position]]
        curve = np.full(len(grid), np.nan)
        for index, scale in enumerate(grid):
            values = [
                target_relative_loss(
                    caches[fold], components, target, weights, scale=float(scale),
                )
                for fold in folds
            ]
            values = [value for value in values if np.isfinite(value)]
            curve[index] = float(np.mean(values)) if values else np.nan
        if np.isfinite(curve).any():
            scales[position] = float(grid[int(np.nanargmin(curve))])
    return scales


def calibrated_fold_score(
    cache: MultiCache, components: tuple[str, ...], table: MultiTable,
    point_index: np.ndarray, scales: np.ndarray, *, cohort: str = "all",
) -> float:
    values = [
        target_relative_loss(
            cache, components, target, table.points[point_index[position]],
            scale=float(scales[position]), cohort=cohort,
        )
        for position, target in enumerate(table.targets)
    ]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


POSITION_GROUPS = {
    "front_row": {"Prop", "Hooker", "Loosehead Prop", "Tighthead Prop"},
    "second_row": {"Lock", "Second Row"},
    "back_row": {"Flanker", "Number 8", "Number Eight", "Back Row",
                 "Blindside Flanker", "Openside Flanker"},
    "half_back": {"Scrum-Half", "Scrum Half", "Fly-Half", "Fly Half"},
    "back": {"Centre", "Wing", "Fullback", "Full Back", "Winger",
             "Inside Centre", "Outside Centre"},
}


def position_group_ids(cache: MultiCache) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(POSITION_GROUPS) + ("other",)
    lookup = {
        value: index
        for index, key in enumerate(POSITION_GROUPS)
        for value in POSITION_GROUPS[key]
    }
    ids = np.array([lookup.get(value, len(POSITION_GROUPS)) for value in cache.position])
    return ids, names


def fit_group_weights(
    caches: list[MultiCache], components: tuple[str, ...], target: str,
    group_ids: list[np.ndarray], n_groups: int, folds: np.ndarray,
    *, grid: np.ndarray, start: np.ndarray, rounds: int = 3,
) -> np.ndarray:
    """Coordinate descent over one weight vector per row group.

    ``grid`` is (n_points, n_components); ``start`` is (n_groups,) point indices.
    Competition-independent by construction: groups are row attributes, never
    the tournament.
    """
    chosen = np.asarray(start, dtype=int).copy()
    for _ in range(rounds):
        moved = False
        for group in range(n_groups):
            best, best_score = chosen[group], np.inf
            for index in range(len(grid)):
                trial = chosen.copy()
                trial[group] = index
                values = []
                for fold in folds:
                    cache = caches[fold]
                    row_weights = grid[trial[group_ids[fold]]]
                    values.append(target_relative_loss(
                        cache, components, target, row_weights,
                    ))
                values = [value for value in values if np.isfinite(value)]
                score = float(np.mean(values)) if values else np.inf
                if score < best_score - 1e-12:
                    best, best_score = index, score
            if best != chosen[group]:
                chosen[group], moved = best, True
        if not moved:
            break
    return chosen


# --------------------------------------------------------------------------
# row-group weights (exact, because every loss is a mean of per-row terms)
# --------------------------------------------------------------------------

def row_losses(target: str, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Per-row loss terms whose mean is ``metrics.target_loss``.

    Every loss in the benchmark is a mean of independent per-row terms, so a
    weight assigned to one group of rows cannot change any other group's loss.
    That makes per-group weights separable and exactly fittable by table lookup
    instead of coordinate descent.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 0, None)
    if actual.ndim == 1 and predicted.ndim == 2:
        actual = actual[:, None]
    if target == "minutes":
        return np.abs(actual - predicted)
    family = distribution_family(target)
    if family == "bernoulli":
        probability = np.clip(predicted, 1e-7, 1 - 1e-7)
        binary = (actual > 0).astype(float)
        return -(binary * np.log(probability) + (1 - binary) * np.log1p(-probability))
    if family == "lognormal":
        return (np.log1p(actual) - np.log1p(predicted)) ** 2
    clipped = np.clip(predicted, 1e-8, None)
    term = np.where(
        actual > 0, actual * np.log(np.clip(actual, 1e-8, None) / clipped), 0.0,
    )
    return 2.0 * (term - (actual - clipped))


@dataclass
class GroupTable:
    """Per (fold, target, group, point) mean-per-row loss, and naive references.

    ``loss[fold, target, group, point]`` is the SUM of row losses for that group
    divided by the target's total valid row count in the fold, so summing over
    groups reproduces the fold's mean loss exactly.
    """

    components: tuple[str, ...]
    points: np.ndarray
    targets: tuple[str, ...]
    folds: tuple[str, ...]
    tournaments: tuple[str, ...]
    groups: tuple[str, ...]
    loss: np.ndarray                       # (n_folds, n_targets, n_groups, n_points)
    naive: np.ndarray                      # (n_folds, n_targets)
    usable: np.ndarray                     # (n_folds, n_targets)
    present: np.ndarray                    # (n_folds, n_targets, n_groups)

    def relative(self, point_index: np.ndarray) -> np.ndarray:
        """Per-fold stable score. ``point_index`` is (n_targets, n_groups)."""
        chosen = np.take_along_axis(
            self.loss, np.asarray(point_index)[None, :, :, None], axis=3,
        )[:, :, :, 0]
        totals = np.where(self.present, chosen, 0.0).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            relative = np.where(self.usable, totals / self.naive, np.nan)
            return np.nanmean(relative, axis=1)

    def index_of(self, weights) -> int:
        weights = np.asarray(weights, dtype=float)
        return int(np.argmin(np.abs(self.points - weights[None, :]).sum(axis=1)))


def build_group_table(
    caches: list[MultiCache], components: tuple[str, ...], points: np.ndarray,
    group_fn, group_names: tuple[str, ...], *, targets: tuple[str, ...] = SCORED,
) -> GroupTable:
    n_groups = len(group_names)
    shape = (len(caches), len(targets), n_groups, len(points))
    loss = np.zeros(shape)
    naive = np.full(shape[:2], np.nan)
    usable = np.zeros(shape[:2], dtype=bool)
    present = np.zeros(shape[:3], dtype=bool)
    for fold_index, cache in enumerate(caches):
        ids = group_fn(cache)
        for target_index, target in enumerate(targets):
            mask = cache.valid[target]
            if not mask.any():
                continue
            actual = cache.actual[target][mask]
            stacked = np.stack(
                [np.clip(cache.means[component][target], 0, None)[mask]
                 for component in components], axis=1,
            )
            blended = stacked @ points.T                 # (n_rows, n_points)
            rows = row_losses(target, actual, blended)   # (n_rows, n_points)
            indicator = np.zeros((n_groups, len(actual)))
            indicator[ids[mask], np.arange(len(actual))] = 1.0
            loss[fold_index, target_index] = (indicator @ rows) / len(actual)
            present[fold_index, target_index] = indicator.sum(axis=1) > 0
            naive[fold_index, target_index] = max(
                target_loss(target, actual, cache.naive[target][mask]), 1e-12,
            )
            usable[fold_index, target_index] = True
        print(f"[{cache.label}] group table filled", flush=True)
    return GroupTable(
        components=components, points=points, targets=tuple(targets),
        folds=tuple(cache.label for cache in caches),
        tournaments=tuple(cache.tournament for cache in caches),
        groups=group_names, loss=loss, naive=naive, usable=usable, present=present,
    )


def _group_surface(table: GroupTable, folds: np.ndarray) -> np.ndarray:
    """Mean over folds of loss-share / naive -> (n_targets, n_groups, n_points)."""
    block = table.loss[folds]
    denominator = table.naive[folds][:, :, None, None]
    usable = table.usable[folds][:, :, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(usable, block / denominator, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(scaled, axis=0)


def fit_group_per_target(
    table: GroupTable, folds: np.ndarray, *, shrink: float = 0.0,
) -> np.ndarray:
    """Exact per-(target, group) fit; groups are additively separable."""
    surface = _group_surface(table, folds)               # (targets, groups, points)
    raw = np.nanargmin(surface, axis=2)
    if shrink <= 0.0:
        return raw.astype(int)
    pooled = np.nanargmin(np.nansum(surface, axis=1), axis=1)   # per-target anchor
    wanted = (
        (1.0 - shrink) * table.points[raw]
        + shrink * table.points[pooled][:, None, :]
    )
    flat = _snap(table.points, wanted.reshape(-1, table.points.shape[1]))
    return flat.reshape(raw.shape)


def fit_group_shrunk(table: GroupTable, folds: np.ndarray):
    """Per-(target, group) weights with shrinkage chosen by inner LOFO."""
    lambdas = np.round(np.arange(0.0, 1.001, 0.05), 3)
    prepared = []
    for position in range(len(folds)):
        inner_train = np.delete(folds, position)
        surface = _group_surface(table, inner_train)
        raw = table.points[np.nanargmin(surface, axis=2)]
        pooled = table.points[np.nanargmin(np.nansum(surface, axis=1), axis=1)]
        prepared.append((folds[position], raw, pooled))
    scores = []
    for value in lambdas:
        inner = []
        for held, raw, pooled in prepared:
            wanted = (1.0 - value) * raw + value * pooled[:, None, :]
            indices = _snap(
                table.points, wanted.reshape(-1, table.points.shape[1]),
            ).reshape(raw.shape[:2])
            inner.append(float(table.relative(indices)[held]))
        scores.append(float(np.nanmean(inner)))
    chosen = float(lambdas[int(np.nanargmin(scores))])
    return fit_group_per_target(table, folds, shrink=chosen), chosen


def group_fold_score(
    cache: MultiCache, components: tuple[str, ...], targets: tuple[str, ...],
    points: np.ndarray, indices: np.ndarray, group_ids: np.ndarray,
    *, cohort: str = "all",
) -> float:
    """Stable score for one fold under per-(target, group) weights, any cohort.

    The group table has no cohort axis -- adding one would multiply its build
    cost by three for a check that only ever runs on the chosen weights. This
    evaluates those weights directly instead.
    """
    values = []
    for position, target in enumerate(targets):
        row_weights = points[indices[position][group_ids]]
        values.append(target_relative_loss(
            cache, components, target, row_weights, cohort=cohort,
        ))
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def started_groups(cache: MultiCache) -> np.ndarray:
    return cache.started.astype(int)


STARTED_NAMES = ("bench", "starter")


def position_groups(cache: MultiCache) -> np.ndarray:
    return position_group_ids(cache)[0]


POSITION_NAMES = tuple(POSITION_GROUPS) + ("other",)


def single_group(cache: MultiCache) -> np.ndarray:
    return np.zeros(cache.n_rows, dtype=int)
