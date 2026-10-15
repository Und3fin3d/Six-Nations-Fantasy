from __future__ import annotations

import json
import os

import pandas as pd

from ..contracts import RawPrediction
from ..v4.gbdt import V4GBDT
from .blend import EventBlend50
from .config import EXTENDED_EVENTS, OUT, STABLE_EVENTS
from .empirical import EmpiricalEventModel
from .folds import HistoricalFold, build_folds, evaluation_frame, masked_candidates, strict_training_frame, write_fold_manifest
from .features import build_frozen_feature_frames
from .metrics import NaiveComparator, event_metrics, ranking_metrics

ENGINES = ("v4", "empirical_event", "p3_event_50")


def _load_model(engine: str, path: str):
    return {"v4": V4GBDT, "empirical_event": EmpiricalEventModel, "p3_event_50": EventBlend50}[engine].load(path)


def _predict(model, engine: str, evaluation: pd.DataFrame, v4_evaluation: pd.DataFrame) -> list[RawPrediction]:
    if engine != "empirical_event":
        return model.predict_frame(v4_evaluation)
    output: dict[int, RawPrediction] = {}
    for _, block in evaluation.groupby("fixture_id", sort=False):
        by_key = {(p.fixture_id, p.player_id, p.team): p for p in model.predict_frame(masked_candidates(block))}
        for row in block.itertuples(index=False):
            output[int(row.benchmark_row_id)] = by_key[(str(row.fixture_id), str(row.player_id), str(row.team))]
    return [output[int(row.benchmark_row_id)] for row in evaluation.itertuples(index=False)]


def _prediction_path(output_dir: str, engine: str, fold: HistoricalFold) -> str:
    return f"{output_dir}/predictions/{engine}/{fold.label}.jsonl"


def _write_predictions(path: str, predictions: list[RawPrediction]) -> None:
    content = "".join(json.dumps(prediction.to_dict(), sort_keys=True) + "\n" for prediction in predictions)
    if os.path.exists(path):
        if open(path).read() != content:
            raise FileExistsError(f"immutable predictions differ: {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(content)


def _read_predictions(path: str) -> list[RawPrediction]:
    return [RawPrediction.from_dict(json.loads(line)) for line in open(path).read().splitlines() if line]


def _write_table(path: str, frame: pd.DataFrame) -> None:
    content = frame.to_csv(index=False)
    if os.path.exists(path):
        if open(path).read() != content:
            raise FileExistsError(f"immutable aggregate differs: {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(content)


def run_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    store_path = f"{OUT}/player_match.csv"
    if not os.path.exists(store_path):
        raise FileNotFoundError("run raw-benchmark audit before raw-benchmark run")
    store = pd.read_csv(store_path, low_memory=False, parse_dates=["date", "match_at"])
    event_frames, ranking_frames = [], []
    for fold in build_folds(store):
        evaluation = evaluation_frame(store, fold).reset_index(drop=True)
        train = strict_training_frame(store, fold)
        evaluation["career_matches"] = evaluation["player_id"].astype(str).map(train.groupby(train["player_id"].astype(str))["fixture_id"].nunique()).fillna(0)
        naive = NaiveComparator.fit(train, ("minutes", *STABLE_EVENTS, *EXTENDED_EVENTS))
        candidates = masked_candidates(evaluation)
        fitted: dict[str, object] = {}
        force_blend_dependencies = not os.path.exists(f"{OUT}/models/p3_event_50/{fold.label}.pkl")

        def needs_model_work(name: str) -> bool:
            return (
                not os.path.exists(f"{OUT}/models/{name}/{fold.label}.pkl")
                or not os.path.exists(_prediction_path(OUT, name, fold))
                or (force_blend_dependencies and name in {"empirical_event", "v4"})
            )

        v4_train = v4_evaluation = None
        if needs_model_work("v4") or needs_model_work("p3_event_50"):
            v4_train, v4_evaluation = build_frozen_feature_frames(train, candidates, v4=True)
        for engine in ENGINES:
            artifact, manifest, prediction_path = (
                f"{OUT}/models/{engine}/{fold.label}.pkl", f"{OUT}/manifests/{engine}/{fold.label}.json", _prediction_path(OUT, engine, fold),
            )
            if os.path.exists(artifact) != os.path.exists(manifest):
                raise RuntimeError(f"partial immutable artifact pair for {engine}/{fold.label}")
            if os.path.exists(artifact):
                model = _load_model(engine, artifact) if needs_model_work(engine) else None
            elif engine == "p3_event_50":
                model = EventBlend50(fitted["empirical_event"], fitted["v4"])
                model.save(artifact)
            else:
                print(f"[{fold.label}] fitting {engine} on {len(train):,} rows", flush=True)
                model = (
                    V4GBDT(events=(*STABLE_EVENTS, *EXTENDED_EVENTS), weighting="natural", pool_player_id=True, player_effects=True).fit(v4_train)
                    if engine == "v4"
                    else EmpiricalEventModel(asof=fold.cutoff).fit(train)
                )
                model.save(artifact)
            if model is not None:
                fitted[engine] = model
            if os.path.exists(prediction_path):
                predictions = _read_predictions(prediction_path)
            else:
                predictions = _predict(model, engine, evaluation, v4_evaluation)
                _write_predictions(prediction_path, predictions)
            write_fold_manifest(manifest, fold, train, evaluation, store_path=store_path, artifact_paths=(artifact, prediction_path), engine=engine)
            tags = dict(tournament=fold.tournament, calendar_year=fold.calendar_year, fold_hemisphere=fold.hemisphere)
            event_path, ranking_path = f"{OUT}/metrics/events/{engine}/{fold.label}.csv", f"{OUT}/metrics/rankings/{engine}/{fold.label}.csv"
            if os.path.exists(event_path):
                fold_events = pd.read_csv(event_path)
            else:
                print(f"[{fold.label}] metrics {engine}", flush=True)
                fold_events = event_metrics(evaluation, predictions, naive, engine=engine, fold=fold.label).assign(**tags)
                _write_table(event_path, fold_events)
            if os.path.exists(ranking_path):
                fold_rankings = pd.read_csv(ranking_path)
            else:
                fold_rankings = ranking_metrics(evaluation, predictions, engine=engine, fold=fold.label).assign(**tags)
                _write_table(ranking_path, fold_rankings)
            event_frames.append(fold_events)
            ranking_frames.append(fold_rankings)
    events, rankings = pd.concat(event_frames, ignore_index=True, sort=False), pd.concat(ranking_frames, ignore_index=True, sort=False)
    _write_table(f"{OUT}/event_metrics.csv", events)
    _write_table(f"{OUT}/ranking_metrics.csv", rankings)
    return events, rankings
