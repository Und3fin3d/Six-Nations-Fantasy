from __future__ import annotations

import hashlib
import json
import os

import pandas as pd

from ..contracts import RawPrediction
from ..v4.gbdt import V4GBDT
from .blend import EventBlend50
from .config import CORE_ENGINE_ORDER, ENGINE_ORDER, EXTENDED_EVENTS, OUT, STABLE_EVENTS
from .empirical import EmpiricalEventModel
from .folds import (
    HistoricalFold, build_folds, evaluation_frame, masked_candidates,
    strict_training_frame, write_fold_manifest,
)
from .features import build_frozen_feature_frames
from .metrics import NaiveComparator, event_metrics, ranking_metrics


def _fit_engine(
    engine: str, train: pd.DataFrame, fold: HistoricalFold,
    prepared_train: pd.DataFrame | None = None,
):
    if engine == "v4":
        features = prepared_train if prepared_train is not None else build_frozen_feature_frames(
            train, train.iloc[0:0], v4=True,
        )[0]
        return V4GBDT(
            events=(*STABLE_EVENTS, *EXTENDED_EVENTS), weighting="natural",
            pool_player_id=True, player_effects=True,
        ).fit(features)
    return EmpiricalEventModel(asof=fold.cutoff).fit(train)


def _save_model(model, path: str) -> None:
    model.save(path)


def _load_model(engine: str, path: str):
    if engine == "v4":
        return V4GBDT.load(path)
    if engine == "empirical_event":
        return EmpiricalEventModel.load(path)
    return EventBlend50.load(path)


def _predict_fixturewise(
    model, engine: str, train: pd.DataFrame, evaluation: pd.DataFrame,
    prepared_evaluation: pd.DataFrame | None = None,
) -> list[RawPrediction]:
    if prepared_evaluation is not None:
        if engine in {"v4", "p3_event_50"}:
            return model.predict_frame(prepared_evaluation)
        return model.predict_frame(masked_candidates(evaluation))
    output: dict[int, RawPrediction] = {}
    for fixture_id, block in evaluation.groupby("fixture_id", sort=False):
        candidates = masked_candidates(block)
        if engine in {"v4", "p3_event_50"}:
            _, candidate_features = build_frozen_feature_frames(
                train, candidates, v4=True,
            )
            predicted = model.predict_frame(candidate_features)
        else:
            predicted = model.predict_frame(candidates)
        by_key = {
            (prediction.fixture_id, prediction.player_id, prediction.team): prediction
            for prediction in predicted
        }
        for row in block.itertuples(index=False):
            key = (str(row.fixture_id), str(row.player_id), str(row.team))
            output[int(row.benchmark_row_id)] = by_key[key]
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


def run_benchmark(
    *, output_dir: str = OUT, engines: tuple[str, ...] = ENGINE_ORDER,
    fold_labels: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    store_path = f"{output_dir}/player_match.csv"
    if not os.path.exists(store_path):
        raise FileNotFoundError("run raw-benchmark audit before raw-benchmark run")
    store = pd.read_csv(store_path, low_memory=False, parse_dates=["date", "match_at"])
    folds = build_folds(store)
    if fold_labels:
        wanted = set(fold_labels)
        folds = [fold for fold in folds if fold.label in wanted]

    event_frames, ranking_frames = [], []
    for fold in folds:
        evaluation = evaluation_frame(store, fold).reset_index(drop=True)
        train = strict_training_frame(store, fold)
        history = train.groupby(train["player_id"].astype(str))["fixture_id"].nunique()
        evaluation["career_matches"] = evaluation["player_id"].astype(str).map(history).fillna(0)
        naive = NaiveComparator.fit(train, ("minutes", *STABLE_EVENTS, *EXTENDED_EVENTS))
        candidates = masked_candidates(evaluation)
        fitted: dict[str, object] = {}
        requested = list(dict.fromkeys(engines))
        if "p3_event_50" in requested:
            for dependency in ("empirical_event", "v4"):
                if dependency not in requested:
                    requested.insert(0, dependency)
        p3_artifact = f"{output_dir}/models/p3_event_50/{fold.label}.pkl"
        force_blend_dependencies = "p3_event_50" in requested and not os.path.exists(p3_artifact)

        def needs_model_work(name: str) -> bool:
            artifact_path = f"{output_dir}/models/{name}/{fold.label}.pkl"
            prediction_path = _prediction_path(output_dir, name, fold)
            return (
                not os.path.exists(artifact_path)
                or not os.path.exists(prediction_path)
                or (force_blend_dependencies and name in {"empirical_event", "v4"})
            )

        v4_train = v4_evaluation = None
        if any(needs_model_work(name) for name in set(requested) & {"v4", "p3_event_50"}):
            v4_train, v4_evaluation = build_frozen_feature_frames(
                train, candidates, v4=True,
            )
        for engine in requested:
            artifact = f"{output_dir}/models/{engine}/{fold.label}.pkl"
            manifest = f"{output_dir}/manifests/{engine}/{fold.label}.json"
            prediction_path = _prediction_path(output_dir, engine, fold)
            if os.path.exists(artifact) != os.path.exists(manifest):
                raise RuntimeError(f"partial immutable artifact pair for {engine}/{fold.label}")
            if os.path.exists(artifact):
                model = _load_model(engine, artifact) if needs_model_work(engine) else None
            elif engine == "p3_event_50":
                model = EventBlend50(fitted["empirical_event"], fitted["v4"])
                _save_model(model, artifact)
            else:
                print(f"[{fold.label}] fitting {engine} on {len(train):,} rows", flush=True)
                prepared_train = v4_train if engine == "v4" else None
                model = _fit_engine(engine, train, fold, prepared_train=prepared_train)
                _save_model(model, artifact)
            if model is not None:
                fitted[engine] = model
            if os.path.exists(prediction_path):
                predictions = _read_predictions(prediction_path)
            else:
                prepared_evaluation = (
                    v4_evaluation if engine in {"v4", "p3_event_50"} else None
                )
                predictions = _predict_fixturewise(
                    model, engine, train, evaluation,
                    prepared_evaluation=prepared_evaluation,
                )
                _write_predictions(prediction_path, predictions)
            write_fold_manifest(
                manifest, fold, train, evaluation, store_path=store_path,
                artifact_paths=(artifact, prediction_path), engine=engine,
            )
            if engine not in engines:
                continue
            event_metric_path = f"{output_dir}/metrics/events/{engine}/{fold.label}.csv"
            ranking_metric_path = f"{output_dir}/metrics/rankings/{engine}/{fold.label}.csv"
            if os.path.exists(event_metric_path):
                fold_events = pd.read_csv(event_metric_path)
            else:
                print(f"[{fold.label}] metrics {engine}", flush=True)
                fold_events = event_metrics(
                    evaluation, predictions, naive, engine=engine, fold=fold.label,
                ).assign(tournament=fold.tournament, calendar_year=fold.calendar_year,
                         fold_hemisphere=fold.hemisphere)
                _write_table(event_metric_path, fold_events)
            if os.path.exists(ranking_metric_path):
                fold_rankings = pd.read_csv(ranking_metric_path)
            else:
                fold_rankings = ranking_metrics(
                    evaluation, predictions, engine=engine, fold=fold.label,
                ).assign(tournament=fold.tournament, calendar_year=fold.calendar_year,
                         fold_hemisphere=fold.hemisphere)
                _write_table(ranking_metric_path, fold_rankings)
            event_frames.append(fold_events)
            ranking_frames.append(fold_rankings)

    events = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    rankings = pd.concat(ranking_frames, ignore_index=True, sort=False) if ranking_frames else pd.DataFrame()
    if fold_labels or tuple(engines) != tuple(CORE_ENGINE_ORDER):
        partial_key = json.dumps({
            "folds": sorted(fold_labels or (fold.label for fold in folds)),
            "engines": list(engines),
        }, sort_keys=True).encode()
        suffix = "_partial_" + hashlib.sha256(partial_key).hexdigest()[:16]
    else:
        suffix = ""
    _write_table(f"{output_dir}/event_metrics{suffix}.csv", events)
    _write_table(f"{output_dir}/ranking_metrics{suffix}.csv", rankings)
    return events, rankings
