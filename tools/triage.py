"""Fast predict-time triage for the empirical engine.

Reuses the fitted models already cached by the benchmark runner, so
prediction-time knobs (the opponent-strength multiplier and its event
grouping) can be scored across folds without refitting anything. This is a
prioritisation tool only -- every number reported as a trial result comes from
the standard `raw_benchmark.cli` pipeline.

Usage: python -m tools.triage <variant> [<variant> ...]
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "unified" / "raw_benchmark" / "allrugby_empirical"


def _configs() -> dict:
    from model.unified.raw_benchmark.empirical import BASE

    return {
        "base": BASE,
        "matchup_off": replace(BASE, matchup_mode="off"),
        "matchup_half": replace(BASE, matchup_mode="damped", matchup_damping=0.5),
        "matchup_quarter": replace(BASE, matchup_mode="damped", matchup_damping=0.25),
        "min_position": replace(BASE, minutes_mode="position"),
        "min_shrunk": replace(BASE, minutes_mode="shrunk", minutes_shrinkage=0.5),
        "min_pos_matchup_off": replace(BASE, minutes_mode="position", matchup_mode="off"),
        "min_shrunk_mq": replace(
            BASE, minutes_mode="shrunk", minutes_shrinkage=0.5,
            matchup_mode="damped", matchup_damping=0.25,
        ),
    }


def main() -> None:
    from model.unified.raw_benchmark.config import EXTENDED_EVENTS, STABLE_EVENTS
    from model.unified.raw_benchmark.empirical import EmpiricalEventModel
    from model.unified.raw_benchmark.folds import (
        build_folds, evaluation_frame, masked_candidates, strict_training_frame,
    )
    from model.unified.raw_benchmark.metrics import NaiveComparator, event_metrics

    wanted = sys.argv[1:] or sorted(_configs())
    configs = {name: _configs()[name] for name in wanted}

    store = pd.read_csv(
        WORK / "player_match.csv", low_memory=False, parse_dates=["date", "match_at"],
    )
    available = {path.stem for path in (WORK / "models" / "empirical_event").glob("*.pkl")}
    folds = [fold for fold in build_folds(store) if fold.label in available]
    print(f"triaging {len(folds)} cached folds: {[f.label for f in folds]}", flush=True)

    rows = []
    for fold in folds:
        evaluation = evaluation_frame(store, fold).reset_index(drop=True)
        train = strict_training_frame(store, fold)
        history = train.groupby(train["player_id"].astype(str))["fixture_id"].nunique()
        evaluation["career_matches"] = evaluation["player_id"].astype(str).map(history).fillna(0)
        naive = NaiveComparator.fit(train, ("minutes", *STABLE_EVENTS, *EXTENDED_EVENTS))
        candidates = masked_candidates(evaluation)
        model = EmpiricalEventModel.load(WORK / "models" / "empirical_event" / f"{fold.label}.pkl")
        for name, config in configs.items():
            model.config = config
            predictions = model.predict_frame(candidates)
            frame = event_metrics(
                evaluation, predictions, naive, engine=name, fold=fold.label,
            )
            rows.append(frame.assign(tournament=fold.tournament))
        print(f"  {fold.label} done", flush=True)

    events = pd.concat(rows, ignore_index=True, sort=False)
    stable = events[events["cohort"].eq("all") & events["tier"].isin(["stable", "minutes"])]
    per_fold = stable.groupby(["engine", "fold"], as_index=False)["relative_loss"].mean()
    overall = per_fold.groupby("engine")["relative_loss"].mean().sort_values()
    print("\n=== triage stable_score (cached folds only) ===")
    print(overall.round(5).to_string())
    per_target = stable.pivot_table(index="target", columns="engine", values="relative_loss")
    print("\n=== per-target relative_loss ===")
    print(per_target.round(3).to_string())
    (WORK / "triage.json").write_text(json.dumps({
        "folds": [fold.label for fold in folds],
        "stable_score": overall.to_dict(),
        "per_target": per_target.to_dict(),
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
