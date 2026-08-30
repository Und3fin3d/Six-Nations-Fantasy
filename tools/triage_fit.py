"""Fit-time triage for the empirical engine.

Refits the candidate per fold for configs that change fitted state (minutes
empirical-Bayes tables, position priors, per-event shrinkage), scoring them on
whatever folds the benchmark runner has already cached a v1 comparison for.
Prioritisation only -- reported trial numbers come from `raw_benchmark.cli`.

Usage: python -m tools.triage_fit <variant> [<variant> ...]
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "unified" / "raw_benchmark" / "allrugby_empirical"


def configs() -> dict:
    from model.unified.raw_benchmark.empirical import BASE

    eb = replace(BASE, minutes_mode="eb")
    eb_off = replace(eb, matchup_mode="off")
    return {
        "base": BASE,
        "eb": eb,
        "eb_off": eb_off,
        "eb_off_prior": replace(eb_off, position_prior_mode="exposure_weighted"),
        "eb_off_recency": replace(eb_off, position_prior_recency=True),
        "eb_off_ebk": replace(eb_off, shrinkage_mode="empirical_bayes"),
        "eb_off_median": replace(eb_off, minutes_statistic="median"),
        "pos_off": replace(BASE, minutes_mode="position", matchup_mode="off"),
        "eb_damped": replace(eb, matchup_mode="damped", matchup_damping=0.25),
        "eb_off_head": replace(eb_off, minutes_head_statistic="median"),
        "eb_off_ebk_head": replace(
            eb_off, shrinkage_mode="empirical_bayes", minutes_head_statistic="median",
        ),
    }


def main() -> None:
    from model.unified.raw_benchmark.config import EXTENDED_EVENTS, STABLE_EVENTS
    from model.unified.raw_benchmark.empirical import EmpiricalEventModel
    from model.unified.raw_benchmark.folds import (
        build_folds, evaluation_frame, masked_candidates, strict_training_frame,
    )
    from model.unified.raw_benchmark.metrics import NaiveComparator, event_metrics

    wanted = sys.argv[1:] or ["base", "eb", "eb_off"]
    chosen = {name: configs()[name] for name in wanted}

    store = pd.read_csv(
        WORK / "player_match.csv", low_memory=False, parse_dates=["date", "match_at"],
    )
    cached = {path.stem for path in (WORK / "metrics" / "events" / "v1").glob("*.csv")}
    folds = [fold for fold in build_folds(store) if fold.label in cached]
    print(f"fit-triage {len(folds)} folds x {len(chosen)} configs", flush=True)

    rows = []
    for fold in folds:
        evaluation = evaluation_frame(store, fold).reset_index(drop=True)
        train = strict_training_frame(store, fold)
        history = train.groupby(train["player_id"].astype(str))["fixture_id"].nunique()
        evaluation["career_matches"] = evaluation["player_id"].astype(str).map(history).fillna(0)
        naive = NaiveComparator.fit(train, ("minutes", *STABLE_EVENTS, *EXTENDED_EVENTS))
        candidates = masked_candidates(evaluation)
        for name, config in chosen.items():
            model = EmpiricalEventModel(asof=fold.cutoff, config=config).fit(train)
            predictions = model.predict_frame(candidates)
            rows.append(
                event_metrics(evaluation, predictions, naive, engine=name, fold=fold.label)
                .assign(tournament=fold.tournament)
            )
        # v1 reference for the same folds, straight from the runner's cache.
        rows.append(pd.read_csv(WORK / "metrics" / "events" / "v1" / f"{fold.label}.csv"))
        print(f"  {fold.label} done", flush=True)

    events = pd.concat(rows, ignore_index=True, sort=False)
    stable = events[events["cohort"].eq("all") & events["tier"].isin(["stable", "minutes"])]
    per_fold = stable.groupby(["engine", "fold"], as_index=False)["relative_loss"].mean()
    overall = per_fold.groupby("engine")["relative_loss"].mean().sort_values()
    print("\n=== fit-triage stable_score (cached folds only) ===")
    print(overall.round(5).to_string())
    print("\n=== per-target relative_loss ===")
    print(stable.pivot_table(index="target", columns="engine", values="relative_loss").round(3).to_string())
    print("\n=== per-fold ===")
    print(per_fold.pivot_table(index="fold", columns="engine", values="relative_loss").round(4).to_string())
    (WORK / "triage_fit.json").write_text(json.dumps({
        "folds": [fold.label for fold in folds],
        "stable_score": overall.to_dict(),
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
