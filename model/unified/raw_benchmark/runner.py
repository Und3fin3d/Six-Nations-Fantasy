"""Train every raw candidate on immutable historical tournament folds."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import RawPrediction
from ..gbdt import UniversalGBDT
from ..neural import NeuralTrainingConfig, UniversalNeuralModel
from ..v3.config import BaselineConfig, GBDTV3Config
from ..v3.gbdt import ExposureRateGBDT
from ..v4.gbdt import V4GBDT
from ..v5.config import V5Config
from ..v5.model import ShrunkFormGBDT
from .blend import EventBlend50
from .config import CORE_ENGINE_ORDER, ENGINE_ORDER, EXTENDED_EVENTS, OUT, STABLE_EVENTS
from .coverage import sha256
from .empirical import EmpiricalEventModel
from .folds import (
    HistoricalFold, build_folds, evaluation_frame, masked_candidates,
    strict_training_frame, write_fold_manifest,
)
from .features import build_frozen_feature_frames
from .metrics import NaiveComparator, event_metrics, ranking_metrics


def _load_configs() -> tuple[BaselineConfig, GBDTV3Config]:
    search = OUT.parents[1] / "v3" / "search"
    baseline_path = search / "best_baseline.json"
    gbdt_path = search / "best_gbdt.json"
    baseline = (
        BaselineConfig.from_dict(json.loads(baseline_path.read_text()))
        if baseline_path.exists() else BaselineConfig()
    )
    gbdt = (
        GBDTV3Config.from_dict(json.loads(gbdt_path.read_text()))
        if gbdt_path.exists() else GBDTV3Config(context_blocks=())
    )
    # Historical v1 contract has no external context features.
    gbdt.context_blocks = ()
    return baseline, gbdt


def _fit_engine(
    engine: str, train: pd.DataFrame, fold: HistoricalFold,
    prepared_train: pd.DataFrame | None = None,
):
    baseline_config, gbdt_config = _load_configs()
    if engine == "v1":
        features = prepared_train if prepared_train is not None else build_frozen_feature_frames(
            train, train.iloc[0:0], v4=False,
        )[0]
        return UniversalGBDT(
            events=(*STABLE_EVENTS, *EXTENDED_EVENTS),
            random_state=baseline_config.seed, weighting=baseline_config.weighting,
            time_half_life_days=baseline_config.time_half_life_days,
            n_estimators=baseline_config.n_estimators, num_leaves=baseline_config.num_leaves,
        ).fit(features)
    if engine == "gbdt_v3":
        features = prepared_train if prepared_train is not None else build_frozen_feature_frames(
            train, train.iloc[0:0], v4=False,
        )[0]
        return ExposureRateGBDT(
            config=gbdt_config, events=(*STABLE_EVENTS, *EXTENDED_EVENTS),
        ).fit(features)
    if engine == "v1_neural":
        features = prepared_train if prepared_train is not None else build_frozen_feature_frames(
            train, train.iloc[0:0], v4=False,
        )[0]
        dates = pd.to_datetime(features["date"], errors="coerce")
        unique_dates = sorted(dates.dropna().unique())
        if len(unique_dates) < 2:
            fit, validation = features, None
        else:
            validation_start = unique_dates[max(1, int(len(unique_dates) * 0.85)) - 1]
            fit = features[dates.lt(validation_start)].copy()
            validation = features[dates.ge(validation_start)].copy()
        return UniversalNeuralModel(
            events=(*STABLE_EVENTS, *EXTENDED_EVENTS),
            config=NeuralTrainingConfig(),
        ).fit(fit, validation)
    if engine == "v4":
        features = prepared_train if prepared_train is not None else build_frozen_feature_frames(
            train, train.iloc[0:0], v4=True,
        )[0]
        return V4GBDT(
            events=(*STABLE_EVENTS, *EXTENDED_EVENTS), weighting="natural",
            pool_player_id=True, player_effects=True,
        ).fit(features)
    if engine == "v5_t":
        config = V5Config(
            use_eb_features=False, use_slot_minutes=False,
            mask_attribution_corrupted=True, attribution_events=("lineouts_won",),
        )
        model = ShrunkFormGBDT(
            config=config, events=(*STABLE_EVENTS, *EXTENDED_EVENTS),
        )
        return model.fit(train)
    if engine == "empirical_event":
        return EmpiricalEventModel(asof=fold.cutoff).fit(train)
    raise ValueError(engine)


def _save_model(model, path: Path) -> None:
    model.save(path)


def _load_model(engine: str, path: Path):
    if engine == "v1":
        return UniversalGBDT.load(path)
    if engine == "gbdt_v3":
        return ExposureRateGBDT.load(path)
    if engine == "v1_neural":
        return UniversalNeuralModel.load(path)
    if engine == "v4":
        with path.open("rb") as handle:
            import pickle
            model = pickle.load(handle)
        if not isinstance(model, V4GBDT):
            raise TypeError(path)
        return model
    if engine == "v5_t":
        return ShrunkFormGBDT.load(path)
    if engine == "empirical_event":
        return EmpiricalEventModel.load(path)
    if engine == "p3_event_50":
        return EventBlend50.load(path)
    raise ValueError(engine)


def _predict_fixturewise(
    model, engine: str, train: pd.DataFrame, evaluation: pd.DataFrame,
    prepared_evaluation: pd.DataFrame | None = None,
) -> list[RawPrediction]:
    if prepared_evaluation is not None:
        if engine == "v5_t":
            return model.predict_features(prepared_evaluation)
        if engine in {"v1", "v1_neural", "gbdt_v3", "v4", "p3_event_50"}:
            return model.predict_frame(prepared_evaluation)
        return model.predict_frame(masked_candidates(evaluation))
    output: dict[int, RawPrediction] = {}
    for fixture_id, block in evaluation.groupby("fixture_id", sort=False):
        candidates = masked_candidates(block)
        if engine in {"v1", "v1_neural", "gbdt_v3", "v4", "p3_event_50"}:
            _, candidate_features = build_frozen_feature_frames(
                train, candidates, v4=engine in {"v4", "p3_event_50"},
            )
            predicted = model.predict_frame(candidate_features)
        elif engine == "v5_t":
            train_features, candidate_features = build_frozen_feature_frames(
                train, candidates, v4=False,
            )
            predicted = model.predict_features(candidate_features)
        else:
            predicted = model.predict_frame(candidates)
        by_key = {
            (prediction.fixture_id, prediction.player_id, prediction.team): prediction
            for prediction in predicted
        }
        for row in block.itertuples(index=False):
            key = (str(row.fixture_id), str(row.player_id), str(row.team))
            if key not in by_key:
                raise ValueError(f"{engine} missing raw prediction for {key}")
            output[int(row.benchmark_row_id)] = by_key[key]
    return [output[int(row.benchmark_row_id)] for row in evaluation.itertuples(index=False)]


def _prediction_path(output_dir: Path, engine: str, fold: HistoricalFold) -> Path:
    return output_dir / "predictions" / engine / f"{fold.label}.jsonl"


def _write_predictions(path: Path, predictions: list[RawPrediction]) -> None:
    content = "".join(json.dumps(prediction.to_dict(), sort_keys=True) + "\n" for prediction in predictions)
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"immutable predictions differ: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _read_predictions(path: Path) -> list[RawPrediction]:
    return [RawPrediction.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    content = frame.to_csv(index=False)
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"immutable aggregate differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def run_benchmark(
    *, output_dir: Path = OUT, engines: tuple[str, ...] = ENGINE_ORDER,
    fold_labels: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    store_path = output_dir / "player_match.csv"
    if not store_path.exists():
        raise FileNotFoundError("run raw-benchmark audit before raw-benchmark run")
    store = pd.read_csv(store_path, low_memory=False, parse_dates=["date", "match_at"])
    folds = build_folds(store)
    if fold_labels:
        wanted = set(fold_labels)
        folds = [fold for fold in folds if fold.label in wanted]
        missing = wanted - {fold.label for fold in folds}
        if missing:
            raise ValueError(f"unknown fold labels: {sorted(missing)}")
    unknown = set(engines) - set(ENGINE_ORDER)
    if unknown:
        raise ValueError(f"unknown raw benchmark engines: {sorted(unknown)}")

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
        p3_artifact = output_dir / "models" / "p3_event_50" / f"{fold.label}.pkl"
        force_blend_dependencies = "p3_event_50" in requested and not p3_artifact.exists()

        def needs_model_work(name: str) -> bool:
            artifact_path = output_dir / "models" / name / f"{fold.label}.pkl"
            prediction_path = _prediction_path(output_dir, name, fold)
            return (
                not artifact_path.exists()
                or not prediction_path.exists()
                or (force_blend_dependencies and name in {"empirical_event", "v4"})
            )

        base_train = base_evaluation = v4_train = v4_evaluation = None
        if any(needs_model_work(name) for name in set(requested) & {"v1", "v1_neural", "gbdt_v3", "v5_t"}):
            base_train, base_evaluation = build_frozen_feature_frames(
                train, candidates, v4=False,
            )
        if any(needs_model_work(name) for name in set(requested) & {"v4", "p3_event_50"}):
            v4_train, v4_evaluation = build_frozen_feature_frames(
                train, candidates, v4=True,
            )
        for engine in requested:
            artifact = output_dir / "models" / engine / f"{fold.label}.pkl"
            manifest = output_dir / "manifests" / engine / f"{fold.label}.json"
            prediction_path = _prediction_path(output_dir, engine, fold)
            if artifact.exists() != manifest.exists():
                raise RuntimeError(f"partial immutable artifact pair for {engine}/{fold.label}")
            if artifact.exists():
                model = _load_model(engine, artifact) if needs_model_work(engine) else None
            elif engine == "p3_event_50":
                model = EventBlend50(fitted["empirical_event"], fitted["v4"])
                _save_model(model, artifact)
            else:
                print(f"[{fold.label}] fitting {engine} on {len(train):,} rows", flush=True)
                prepared_train = (
                    v4_train if engine in {"v4", "p3_event_50"}
                    else base_train if engine in {"v1", "v1_neural", "gbdt_v3"}
                    else None
                )
                model = _fit_engine(engine, train, fold, prepared_train=prepared_train)
                _save_model(model, artifact)
            if model is not None:
                fitted[engine] = model
            if prediction_path.exists():
                predictions = _read_predictions(prediction_path)
            else:
                if model is None:
                    raise RuntimeError(f"{engine}/{fold.label}: prediction missing without model")
                prepared_evaluation = (
                    v4_evaluation if engine in {"v4", "p3_event_50"}
                    else base_evaluation if engine in {"v1", "v1_neural", "gbdt_v3", "v5_t"}
                    else None
                )
                predictions = _predict_fixturewise(
                    model, engine, train, evaluation,
                    prepared_evaluation=prepared_evaluation,
                )
                _write_predictions(prediction_path, predictions)
            if len(predictions) != len(evaluation):
                raise ValueError(f"{engine}/{fold.label}: prediction cohort mismatch")
            write_fold_manifest(
                manifest, fold, train, evaluation, store_path=store_path,
                artifact_paths=(artifact, prediction_path), engine=engine,
            )
            if engine not in engines:
                continue
            event_metric_path = output_dir / "metrics" / "events" / engine / f"{fold.label}.csv"
            ranking_metric_path = output_dir / "metrics" / "rankings" / engine / f"{fold.label}.csv"
            if event_metric_path.exists():
                fold_events = pd.read_csv(event_metric_path)
            else:
                print(f"[{fold.label}] metrics {engine}", flush=True)
                fold_events = event_metrics(
                    evaluation, predictions, naive, engine=engine, fold=fold.label,
                ).assign(tournament=fold.tournament, calendar_year=fold.calendar_year,
                         fold_hemisphere=fold.hemisphere)
                _write_table(event_metric_path, fold_events)
            if ranking_metric_path.exists():
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
    _write_table(output_dir / f"event_metrics{suffix}.csv", events)
    _write_table(output_dir / f"ranking_metrics{suffix}.csv", rankings)
    return events, rankings
