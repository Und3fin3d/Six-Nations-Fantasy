"""Per-competition scoreboard for the competition-agnostic model.

The raw score answers "is this the better model overall". This answers the
question the project actually cares about: **in each competition separately, is
the one agnostic model the best model available?** Every engine here is scored
on the same 21 exact-kickoff tournament holdouts, under both the Six Nations and
Nations Championship rubrics, so the per-tournament rows are like-for-like.

Reference engines are the single components the blend is built from -- the v4
GBDT, the frozen empirical engine (which is the NCR incumbent's engine family)
and the benchmark's own stratum-mean comparator -- plus the frozen P3 blend and
the phase-1 C5 incumbent. The Six Nations champion is not on these folds (it
only has 2-fold, 13-target coverage); its comparison lives in
``allrugby_sixnations/LEDGER.md``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import allrugby_phase2 as P
from . import allrugby_phase2_rubric as R

WORK = P.WORK


def constant_rule(weights: dict[str, list[float]] | list[float], n_components: int):
    """Factory producing (weight_for, scale_for) for a fold-independent rule."""
    def factory(_fold_label: str):
        def weight_for(target: str):
            if isinstance(weights, dict):
                return np.asarray(weights.get(target, [1.0 / n_components] * n_components))
            return np.asarray(weights)
        return weight_for, None
    return factory


def per_fold_rule(weights_by_fold: dict[str, dict[str, list[float]]], n_components: int):
    """Cross-fitted rule: each fold is scored with weights fitted without it."""
    def factory(fold_label: str):
        table = weights_by_fold[fold_label]

        def weight_for(target: str):
            return np.asarray(table.get(target, [1.0 / n_components] * n_components))
        return weight_for, None
    return factory


def one_hot(components: tuple[str, ...], component: str) -> list[float]:
    return [1.0 if name == component else 0.0 for name in components]


def collect_shard(
    components: tuple[str, ...], rules: dict[str, callable], out_prefix: str,
    fold_labels: tuple[str, ...],
) -> None:
    """Score one slice of folds and park the raw rows for a later merge.

    Scoring every engine on every fold is the slowest step in the pipeline and
    the folds are independent, so it is sharded across processes.
    """
    rankings, points = R.collect(components, rules, fold_labels)
    shard = WORK / "shards"
    shard.mkdir(parents=True, exist_ok=True)
    key = "-".join(sorted(fold_labels))[:60]
    rankings.to_csv(shard / f"{out_prefix}_rankings_{key}.csv", index=False)
    points.to_csv(shard / f"{out_prefix}_points_{key}.csv", index=False)


def merge_shards(out_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    shard = WORK / "shards"
    rankings = pd.concat(
        [pd.read_csv(path) for path in sorted(shard.glob(f"{out_prefix}_rankings_*.csv"))],
        ignore_index=True, sort=False,
    )
    points = pd.concat(
        [pd.read_csv(path) for path in sorted(shard.glob(f"{out_prefix}_points_*.csv"))],
        ignore_index=True, sort=False,
    )
    return rankings, points


def build(
    components: tuple[str, ...], rules: dict[str, callable], out_prefix: str,
    *, from_shards: bool = False,
):
    if from_shards:
        rankings, points = merge_shards(out_prefix)
    else:
        rankings, points = R.collect(components, rules)
    WORK.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(WORK / f"{out_prefix}_rankings.csv", index=False)
    overall = R.summarise(rankings)
    by_tournament = R.summarise(rankings, by_tournament=True)
    calibrated = R.calibrated_mae(points)
    calibrated_by_tournament = R.calibrated_mae(points, by_tournament=True)
    overall = overall.merge(calibrated, on=["engine", "rubric"], suffixes=("", "_cal"))
    by_tournament = by_tournament.merge(
        calibrated_by_tournament, on=["engine", "rubric", "tournament"], suffixes=("", "_cal"),
    )
    overall.to_csv(WORK / f"{out_prefix}_overall.csv", index=False)
    by_tournament.to_csv(WORK / f"{out_prefix}_by_tournament.csv", index=False)
    # Per fold as well: the Six Nations champion only has two folds of coverage,
    # so the only like-for-like comparison against it is a two-fold slice.
    by_fold = R.summarise(rankings, by_fold=True)
    by_fold.to_csv(WORK / f"{out_prefix}_by_fold.csv", index=False)
    return overall, by_tournament


def winner_table(by_tournament: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    """Which engine wins each (rubric, tournament) cell."""
    rows = []
    for (rubric, tournament), block in by_tournament.groupby(["rubric", "tournament"]):
        ordered = block.sort_values(metric, ascending=not higher_is_better)
        rows.append({
            "rubric": rubric, "tournament": tournament,
            "winner": ordered.iloc[0]["engine"], "best": float(ordered.iloc[0][metric]),
            "runner_up": ordered.iloc[1]["engine"] if len(ordered) > 1 else None,
            "runner_up_value": float(ordered.iloc[1][metric]) if len(ordered) > 1 else np.nan,
        })
    return pd.DataFrame(rows)
