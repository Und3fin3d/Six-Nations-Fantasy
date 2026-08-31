"""Reconstructed-rubric metrics for phase-2 candidates, per tournament family.

Phase 1's rubric module is hardwired to the frozen (empirical, v4) pair. This
one takes an arbitrary ordered component tuple and a per-fold weight rule, so a
component-swapped or multi-component candidate can be scored under both the Six
Nations and Nations Championship rubrics.

Standing requirement: never report a raw score without the reconstructed MAE for
BOTH rubrics, raw and calibrated. The per-tournament breakdown is what answers
"is the competition-agnostic model the best one in each competition".
"""

from __future__ import annotations

import gc
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import allrugby as A
from . import allrugby_phase2 as P
from ..contracts import EventDistribution, RawPrediction
from ..schema import distribution_family
from .allrugby_blend import ALL_TARGETS
from .blend import _variance
from .config import STABLE_EVENTS, TOP_NS
from .folds import build_folds
from .metrics import (
    NationsChampionshipScorer, SixNationsScorer, _score_rows, _scorer_events,
    ranking_metrics,
)


def _family(target: str) -> str:
    return "lognormal" if target == "minutes" else distribution_family(target)


def blend_predictions(
    loaded: dict[str, list[RawPrediction]], components: tuple[str, ...],
    weight_for, scale_for=None,
) -> list[RawPrediction]:
    """Blend N component predictions row-wise. ``weight_for(target) -> vector``."""
    reference = loaded[components[0]]
    for component in components[1:]:
        other = loaded[component]
        if len(other) != len(reference):
            raise ValueError("component prediction lengths differ")
    output = []
    for index in range(len(reference)):
        rows = [loaded[component][index] for component in components]
        head = rows[0]
        for row in rows[1:]:
            if (row.fixture_id, row.player_id) != (head.fixture_id, head.player_id):
                raise ValueError("component rows are misaligned")
        events = {}
        names = sorted({name for row in rows for name in row.events})
        for target in names:
            events[target] = _combine(
                [row.events.get(target) for row in rows], target,
                weight_for(target), scale_for(target) if scale_for else 1.0,
            )
        minutes = _combine(
            [row.minutes for row in rows], "minutes",
            weight_for("minutes"), scale_for("minutes") if scale_for else 1.0,
        )
        output.append(RawPrediction(
            fixture_id=head.fixture_id, player_id=head.player_id,
            player_name=head.player_name, team=head.team, opponent=head.opponent,
            position=head.position, is_forward=head.is_forward,
            events=events, minutes=minutes, metadata={"model": "phase2"},
        ))
    return output


def _combine(parts, target: str, weights, scale: float) -> EventDistribution:
    present = [(part, weight) for part, weight in zip(parts, weights) if part is not None]
    if not present:
        return EventDistribution(_family(target), 0.0, 1.0)
    total = sum(weight for _, weight in present)
    if total <= 0:
        present = [(part, 1.0 / len(present)) for part, _ in present]
        total = 1.0
    present = [(part, weight / total) for part, weight in present]
    mean = scale * sum(weight * part.mean for part, weight in present)
    variance = sum(
        weight * (_variance(part) + (part.mean - mean) ** 2) for part, weight in present
    )
    family = _family(target)
    if family == "bernoulli":
        return EventDistribution(family, min(max(mean, 0.0), 1.0), 1.0)
    if family == "negative_binomial":
        dispersion = max(mean * mean / max(variance - mean, 1e-6), 0.05)
    else:
        dispersion = max(variance, 1e-6)
    return EventDistribution(family, max(mean, 0.0), dispersion)


@contextmanager
def _without_cyclic_gc():
    """Scoring builds millions of small acyclic objects; the collector dominates.

    Each fold materialises one ``EventDistribution`` per event per player per
    engine -- of the order of a million objects -- while the result frames stay
    live across folds. Every gen-2 collection then walks the whole heap, which
    turns a thirty-second fold into a twenty-minute one. The objects are plain
    dataclasses holding floats and strings, so reference counting alone reclaims
    them and the cycle collector has nothing to find.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def collect(
    components: tuple[str, ...], rule_for: dict[str, callable],
    fold_labels: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``rule_for[name](fold_label) -> (weight_for, scale_for or None)``."""
    store = A._store()
    rank_frames, point_frames = [], []
    wanted = set(fold_labels) if fold_labels else None
    for fold in build_folds(store):
        if wanted is not None and fold.label not in wanted:
            continue
        evaluation, naive = A.evaluation_context(store, fold)
        loaded = {
            component: A.read_predictions(A._component_path(fold.label, component))
            for component in components if component != P.NAIVE
        }
        if P.NAIVE in components:
            with _without_cyclic_gc():
                loaded[P.NAIVE] = naive_predictions(evaluation, naive)
        for name, factory in rule_for.items():
            weight_for, scale_for = factory(fold.label)
            with _without_cyclic_gc():
                predictions = blend_predictions(loaded, components, weight_for, scale_for)
            rank_frames.append(ranking_metrics(
                evaluation, predictions, engine=name, fold=fold.label,
            ).assign(tournament=fold.tournament, fold_hemisphere=fold.hemisphere))
            for rubric, scorer in (
                ("six_nations", SixNationsScorer()), ("ncr", NationsChampionshipScorer()),
            ):
                allowed = _scorer_events(scorer, STABLE_EVENTS)
                for slate_id, slate in evaluation.groupby("slate_id", sort=True):
                    local = [predictions[int(index)] for index in slate.index.to_numpy()]
                    _, actual, predicted = _score_rows(
                        slate.reset_index(drop=True), local, scorer, allowed,
                    )
                    if not len(actual):
                        continue
                    point_frames.append(pd.DataFrame({
                        "engine": name, "fold": fold.label, "tournament": fold.tournament,
                        "rubric": rubric, "slate_id": str(slate_id),
                        "actual": actual, "predicted": predicted,
                    }))
        print(f"[{fold.label}] phase2 rubric for {len(rule_for)} rule(s)", flush=True)
    return (
        pd.concat(rank_frames, ignore_index=True, sort=False),
        pd.concat(point_frames, ignore_index=True, sort=False),
    )


def naive_predictions(evaluation: pd.DataFrame, naive) -> list[RawPrediction]:
    """The benchmark's own stratum-mean comparator, as RawPredictions."""
    means = {
        target: naive.predict(evaluation, target)
        for target in ALL_TARGETS if target in naive.group_means
    }
    output = []
    for index, row in enumerate(evaluation.itertuples(index=False)):
        events = {
            target: EventDistribution(
                _family(target), max(float(values[index]), 0.0), 1.0,
            )
            for target, values in means.items() if target != "minutes"
        }
        minutes_mean = float(means["minutes"][index]) if "minutes" in means else 0.0
        output.append(RawPrediction(
            fixture_id=str(row.fixture_id), player_id=str(row.player_id),
            player_name=str(row.player_name), team=str(row.team),
            opponent=str(row.opponent), position=str(row.position),
            is_forward=bool(row.is_forward), events=events,
            minutes=EventDistribution("lognormal", max(minutes_mean, 0.0), 25.0),
            metadata={"model": "naive"},
        ))
    return output


def _mean_capture(frame: pd.DataFrame) -> pd.Series:
    columns = [f"top_{n}_capture" for n in TOP_NS if f"top_{n}_capture" in frame]
    return frame[columns].mean(axis=1, skipna=True)


def summarise(
    rankings: pd.DataFrame, tier: str = "stable", by_tournament: bool = False,
    by_fold: bool = False,
):
    block = rankings[rankings["tier"].eq(tier)].copy()
    block["mean_capture"] = _mean_capture(block)
    keys = ["engine", "rubric"]
    if by_fold:
        keys += ["tournament", "fold"]
    elif by_tournament:
        keys += ["tournament"]
    return block.groupby(keys, as_index=False).agg(
        mae=("mae", "mean"), spearman=("spearman", "mean"),
        mean_capture=("mean_capture", "mean"), slates=("slate_id", "nunique"),
    )


def calibrated_mae(points: pd.DataFrame, by_tournament: bool = False) -> pd.DataFrame:
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
                    "engine": engine, "rubric": rubric, "fold": fold,
                    "tournament": slate["tournament"].iloc[0], "slate_id": slate_id,
                    "mae_raw": float(np.mean(np.abs(slate["actual"] - slate["predicted"]))),
                    "mae_calibrated": float(np.mean(np.abs(slate["actual"] - slate["calibrated"]))),
                })
    frame = pd.DataFrame(rows)
    keys = ["engine", "rubric"] + (["tournament"] if by_tournament else [])
    return frame.groupby(keys, as_index=False).agg(
        mae_raw=("mae_raw", "mean"), mae_calibrated=("mae_calibrated", "mean"),
        slates=("slate_id", "nunique"),
    )


def paired_by_slate(rankings, candidate: str, baseline: str, metric: str,
                    tier: str = "stable") -> dict[str, dict]:
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
