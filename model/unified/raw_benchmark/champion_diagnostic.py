"""Honest partial raw-component diagnostic for the Six Nations champion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model.baselines import SCORED
from model.data import load as load_six_nations_store
from model.research import PROMOTED_CONFIG, _predict_config, config_from_dict

from ..contracts import EventDistribution, RawPrediction
from ..schema import distribution_family
from .config import EXTENDED_EVENTS, OUT, STABLE_EVENTS
from .folds import build_folds, evaluation_frame, strict_training_frame
from .metrics import NaiveComparator, event_metrics
from .runner import _write_table

ENGINE = "six_nations_champion_components"
SEASONS = (2025, 2026)
RAW_EVENTS = tuple(event for event in SCORED if event in STABLE_EVENTS + EXTENDED_EVENTS)
SHARED_STABLE_TARGETS = ("minutes", *(event for event in RAW_EVENTS if event in STABLE_EVENTS))


def _raw_predictions(frame: pd.DataFrame, predicted: pd.DataFrame) -> list[RawPrediction]:
    table = predicted.copy()
    table["fixture_id"] = table["fixture_id"].astype(str)
    table["player_id"] = table["player_id"].astype(str)
    by_key = table.set_index(["fixture_id", "player_id"], drop=False)
    output = []
    missing = []
    for row in frame.itertuples(index=False):
        key = (str(row.fixture_id), str(row.player_id))
        if key not in by_key.index:
            missing.append(key)
            continue
        item = by_key.loc[key]
        if isinstance(item, pd.DataFrame):
            raise ValueError(f"champion prediction key is ambiguous: {key}")
        events = {}
        for event in RAW_EVENTS:
            mean = max(float(item[f"hat_{event}"]), 0.0)
            family = distribution_family(event)
            if family == "bernoulli":
                mean = min(mean, 1.0)
            events[event] = EventDistribution(family, mean, 1.0)
        output.append(RawPrediction(
            fixture_id=key[0], player_id=key[1], player_name=str(row.player_name),
            team=str(row.team), opponent=str(row.opponent), position=str(row.position),
            is_forward=bool(row.is_forward), events=events,
            minutes=EventDistribution("lognormal", max(float(item["minutes_hat"]), 0.0), 1.0),
            metadata={"model": ENGINE, "scope": "native_six_nations_partial_components"},
        ))
    if missing:
        raise ValueError(f"champion missed {len(missing)} benchmark players: {missing[:5]}")
    return output


def run_champion_diagnostic(output_dir: Path = OUT) -> dict:
    benchmark = pd.read_csv(
        output_dir / "player_match.csv", low_memory=False, parse_dates=["date", "match_at"],
    )
    folds = {fold.label: fold for fold in build_folds(benchmark)}
    native = load_six_nations_store()
    config = config_from_dict(json.loads(PROMOTED_CONFIG.read_text()))
    frames = []
    for season in SEASONS:
        fold = folds[f"six_nations_{season}"]
        evaluation = evaluation_frame(benchmark, fold).reset_index(drop=True)
        train = strict_training_frame(benchmark, fold)
        history = train.groupby(train["player_id"].astype(str))["fixture_id"].nunique()
        evaluation["career_matches"] = evaluation["player_id"].astype(str).map(history).fillna(0)
        predicted, _, _ = _predict_config(native, config, season)
        raw = _raw_predictions(evaluation, predicted)
        naive = NaiveComparator.fit(train, ("minutes", *STABLE_EVENTS, *EXTENDED_EVENTS))
        metrics = event_metrics(
            evaluation, raw, naive, engine=ENGINE, fold=fold.label,
        ).assign(tournament=fold.tournament, calendar_year=fold.calendar_year,
                 fold_hemisphere=fold.hemisphere)
        path = output_dir / "metrics" / "events" / ENGINE / f"{fold.label}.csv"
        _write_table(path, metrics)
        frames.append(metrics)
    champion = pd.concat(frames, ignore_index=True, sort=False)
    base = pd.read_csv(output_dir / "event_metrics.csv")
    folds_wanted = {f"six_nations_{season}" for season in SEASONS}
    comparison = pd.concat([base, champion], ignore_index=True, sort=False)
    comparison = comparison[
        comparison["fold"].isin(folds_wanted)
        & comparison["cohort"].eq("all")
        & comparison["target"].isin(SHARED_STABLE_TARGETS)
        & comparison["engine"].isin(["v1", "p3_event_50", ENGINE])
    ]
    by_fold = comparison.groupby(["engine", "fold"], as_index=False).agg(
        partial_stable_score=("relative_loss", "mean"), targets=("target", "nunique"),
    )
    summary = by_fold.groupby("engine", as_index=False).agg(
        partial_stable_score=("partial_stable_score", "mean"),
        folds=("fold", "nunique"), targets=("targets", "min"),
    ).sort_values("partial_stable_score")
    _write_table(output_dir / "champion_partial_comparison.csv", summary)
    by_event = comparison.groupby(["target", "engine"])["relative_loss"].mean().unstack()
    by_event["champion_vs_v1_pct"] = (
        by_event[ENGINE] / by_event["v1"] - 1.0
    ) * 100.0
    by_event = by_event.reset_index().sort_values("champion_vs_v1_pct")
    _write_table(output_dir / "champion_partial_by_event.csv", by_event)
    return {
        "engine": ENGINE, "seasons": list(SEASONS),
        "events": list(RAW_EVENTS), "shared_stable_targets": list(SHARED_STABLE_TARGETS),
        "comparison": summary.to_dict("records"),
        "by_event": by_event.to_dict("records"),
        "eligible_for_promotion": False,
        "reason": "only two Six Nations folds and 12 of 23 stable events plus minutes",
    }
