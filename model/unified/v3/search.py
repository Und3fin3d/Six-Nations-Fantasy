"""Bounded rolling-origin search for GBDT-v3 and the 20 neural challengers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import ROOT
from ..features import build_pit_features
from ..gbdt import UniversalGBDT
from .blend import blend_predictions
from .config import BaselineConfig, GBDTV3Config, NeuralV3Config, SearchConfig
from .context import augment_context
from .gbdt import ExposureRateGBDT
from .harness import OUT, attach_match_timestamps
from .metrics import (
    observable_events, observable_points_actual, observable_points_predicted,
    raw_event_deviance, raw_residual_vector,
)
from .neural import ExposureRateNeural

DATA = ROOT / "data"


def rolling_folds(frame: pd.DataFrame, years: tuple[int, ...] = (2022, 2023, 2024)):
    match_at = pd.to_datetime(frame["match_at"], utc=True)
    for year in years:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        train = frame[match_at < start].copy()
        validation = frame[
            (match_at >= start) & (match_at < end)
            & frame["competition_level"].eq("international")
        ].copy()
        if len(train) and len(validation):
            yield year, train, validation


def _objective(train: pd.DataFrame, validation: pd.DataFrame, model,
               observable_enabled: bool) -> dict[str, float]:
    predictions = model.predict_frame(validation)
    metrics = raw_event_deviance(validation, predictions)
    observable_maes = []
    if observable_enabled:
        allowed = observable_events(train[train["competition_level"].eq("international")])
        if allowed:
            for competition in ("six_nations", "ncr"):
                actual = observable_points_actual(validation, competition, allowed)
                predicted = observable_points_predicted(predictions, competition, allowed)
                observable_maes.append(float(np.mean(np.abs(predicted - actual))))
    observable_mae = float(np.mean(observable_maes)) if observable_maes else np.nan
    metrics["observable_points_mae"] = observable_mae
    metrics["objective"] = metrics["raw_deviance"] + (
        observable_mae / 25.0 if np.isfinite(observable_mae) else 0.0
    )
    return metrics


def _load_base_store() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "unified" / "player_match.csv", low_memory=False, parse_dates=["date"])
    return attach_match_timestamps(raw)


def _feature_store(base: pd.DataFrame, blocks: tuple[str, ...]) -> pd.DataFrame:
    return build_pit_features(augment_context(base, blocks))


def default_gbdt_candidates() -> list[GBDTV3Config]:
    """Twelve predeclared candidates spanning domain, time, capacity and context."""
    specs = [
        ("natural", None, 23, 2.0, ()),
        ("level_balanced", None, 23, 4.0, ()),
        ("level_balanced", 1095.0, 31, 4.0, ()),
        ("level_balanced", 1095.0, 31, 4.0, ("wr",)),
        ("level_balanced", 547.5, 31, 8.0, ("wr",)),
        ("international_x2", None, 31, 2.0, ()),
        ("international_x2", None, 31, 2.0, ("wr",)),
        ("international_x2", 1095.0, 31, 6.0, ()),
        ("international_x2", 1095.0, 31, 6.0, ("wr",)),
        ("international_x2", 1095.0, 31, 6.0, ("wr", "roles")),
        ("international_x2", 1095.0, 31, 6.0, ("wr", "style")),
        ("international_x4", 1095.0, 63, 8.0, ("wr",)),
    ]
    return [
        GBDTV3Config(
            weighting=weighting, time_half_life_days=half_life,
            num_leaves=leaves, reg_lambda=reg_lambda, context_blocks=blocks,
            min_child_samples=35 if leaves > 31 else 45,
        )
        for weighting, half_life, leaves, reg_lambda, blocks in specs
    ]


def default_baseline_candidates() -> list[BaselineConfig]:
    return [
        BaselineConfig(weighting=weighting, time_half_life_days=half_life,
                       n_estimators=trees, num_leaves=leaves)
        for weighting, half_life, trees, leaves in (
            ("natural", None, 180, 23),
            ("natural", 1095.0, 220, 31),
            ("level_balanced", None, 180, 23),
            ("level_balanced", 1095.0, 220, 31),
            ("level_balanced", 547.5, 220, 31),
            ("international_x2", None, 220, 31),
            ("international_x2", 1095.0, 220, 31),
            ("international_x4", 1095.0, 260, 31),
        )
    ]


def neural_candidates(
    count: int = 20, seed: int = 17,
    admitted_blocks: tuple[str, ...] = ("wr", "roles", "style"),
) -> list[NeuralV3Config]:
    """Exactly 20 deterministic unique candidates for successive halving."""
    rng = np.random.default_rng(seed)
    candidates = []
    seen = set()
    weightings = ("natural", "level_balanced", "international_x2", "international_x4")
    base = ("wr",) if "wr" in admitted_blocks else ()
    block_options = [()]
    if base:
        block_options.append(base)
    if "roles" in admitted_blocks:
        block_options.append(("roles",))
        if base:
            block_options.append(tuple(dict.fromkeys((*base, "roles"))))
    if "style" in admitted_blocks:
        block_options.append(("style",))
        if base:
            block_options.append(tuple(dict.fromkeys((*base, "style"))))
    blocks = tuple(dict.fromkeys(block_options))
    while len(candidates) < count:
        config = NeuralV3Config(
            hidden=int(rng.choice([128, 256])),
            depth=int(rng.choice([2, 3])),
            dropout=float(rng.uniform(0.1, 0.3)),
            embedding_scale=float(rng.choice([0.5, 1.0, 2.0])),
            learning_rate=float(np.exp(rng.uniform(np.log(3e-4), np.log(2e-3)))),
            weight_decay=float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-3)))),
            weighting=str(rng.choice(weightings)),
            time_half_life_days=float(rng.choice([547.5, 1095.0, 1825.0])),
            context_blocks=blocks[int(rng.integers(0, len(blocks)))],
            seed=seed + len(candidates),
        )
        key = (
            config.hidden, config.depth, round(config.dropout, 4),
            config.embedding_scale, round(config.learning_rate, 7),
            round(config.weight_decay, 8), config.weighting,
            config.time_half_life_days, config.context_blocks,
        )
        if key not in seen:
            seen.add(key)
            candidates.append(config)
    return candidates


def _audit_observable_enabled() -> bool:
    path = OUT / "audit" / "audit.json"
    if not path.exists():
        return False
    return bool(json.loads(path.read_text()).get("observable_points_enabled"))


def _audit_appearance_enabled() -> bool:
    path = OUT / "audit" / "audit.json"
    if not path.exists():
        return False
    return bool(
        json.loads(path.read_text()).get("zero_minutes", {})
        .get("appearance_head_enabled")
    )


def _gbdt_reference(output_dir: Path, years: tuple[int, ...]) -> dict[str, float] | None:
    path = output_dir / "gbdt_candidates.csv"
    if not path.exists():
        return None
    rows = pd.read_csv(path)
    rows = rows[rows["year"].isin(years)]
    if rows.empty:
        return None
    summary = (
        rows.groupby("candidate_id", as_index=False)
        .agg(raw_deviance=("raw_deviance", "mean"),
             observable_points_mae=("observable_points_mae", "mean"))
        .sort_values("raw_deviance")
    )
    return summary.iloc[0].to_dict()


def tune_gbdt(output_dir: Path = OUT / "search") -> tuple[GBDTV3Config, pd.DataFrame]:
    base = _load_base_store()
    observable_enabled = _audit_observable_enabled()
    stores: dict[tuple[str, ...], pd.DataFrame] = {}
    rows = []
    candidates = [
        replace(candidate, use_appearance=_audit_appearance_enabled())
        for candidate in default_gbdt_candidates()
    ]
    for candidate_id, config in enumerate(candidates):
        if config.context_blocks not in stores:
            stores[config.context_blocks] = _feature_store(base, config.context_blocks)
        store = stores[config.context_blocks]
        for year, train, validation in rolling_folds(store):
            print(f"[gbdt {candidate_id + 1:02d}/{len(candidates)}] fold {year}", flush=True)
            model = ExposureRateGBDT(config=config).fit(train)
            metrics = _objective(train, validation, model, observable_enabled)
            rows.append({
                "candidate_id": candidate_id, "year": year,
                **config.to_dict(), **metrics,
            })
    results = pd.DataFrame(rows)
    summary = (
        results.groupby("candidate_id", as_index=False)
        .agg(objective=("objective", "mean"), raw_deviance=("raw_deviance", "mean"),
             observable_points_mae=("observable_points_mae", "mean"))
        .sort_values("objective")
    )
    mean_objective = summary.set_index("candidate_id")["objective"]
    ablations = {
        "wr": float(np.mean([
            mean_objective.loc[3] - mean_objective.loc[2],
            mean_objective.loc[6] - mean_objective.loc[5],
            mean_objective.loc[8] - mean_objective.loc[7],
        ])),
        "roles": float(mean_objective.loc[9] - mean_objective.loc[8]),
        "style": float(mean_objective.loc[10] - mean_objective.loc[8]),
        "weather": None,
    }
    admitted = {
        "wr": ablations["wr"] < 0,
        "roles": ablations["roles"] < 0,
        "style": ablations["style"] < 0,
        "weather": False,
    }
    allowed_ids = [
        i for i, candidate in enumerate(candidates)
        if all(admitted.get(block, False) for block in candidate.context_blocks)
    ]
    eligible_summary = summary[summary["candidate_id"].isin(allowed_ids)]
    best_id = int(eligible_summary.iloc[0]["candidate_id"])
    best = candidates[best_id]
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "gbdt_candidates.csv", index=False)
    summary.to_csv(output_dir / "gbdt_summary.csv", index=False)
    (output_dir / "best_gbdt.json").write_text(json.dumps(best.to_dict(), indent=2) + "\n")
    (output_dir / "feature_admission.json").write_text(json.dumps({
        "admitted": admitted, "objective_delta_with_block": ablations,
        "rule": "admit only when the paired rolling-fold objective delta is negative",
    }, indent=2) + "\n")
    return best, results


def tune_baseline(output_dir: Path = OUT / "search") -> tuple[BaselineConfig, pd.DataFrame]:
    store = _feature_store(_load_base_store(), ())
    observable_enabled = _audit_observable_enabled()
    candidates = default_baseline_candidates()
    rows = []
    for candidate_id, config in enumerate(candidates):
        for year, train, validation in rolling_folds(store):
            print(
                f"[baseline {candidate_id + 1:02d}/{len(candidates)}] fold {year}",
                flush=True,
            )
            model = UniversalGBDT(
                random_state=config.seed, weighting=config.weighting,
                time_half_life_days=config.time_half_life_days,
                n_estimators=config.n_estimators, num_leaves=config.num_leaves,
            ).fit(train)
            metrics = _objective(train, validation, model, observable_enabled)
            rows.append({
                "candidate_id": candidate_id, "year": year,
                **config.to_dict(), **metrics,
            })
    results = pd.DataFrame(rows)
    summary = (
        results.groupby("candidate_id", as_index=False)
        .agg(objective=("objective", "mean"), raw_deviance=("raw_deviance", "mean"),
             observable_points_mae=("observable_points_mae", "mean"))
        .sort_values("objective")
    )
    best_id = int(summary.iloc[0]["candidate_id"])
    best = candidates[best_id]
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "baseline_candidates.csv", index=False)
    summary.to_csv(output_dir / "baseline_summary.csv", index=False)
    (output_dir / "best_baseline.json").write_text(
        json.dumps(best.to_dict(), indent=2) + "\n"
    )
    return best, results


def tune_neural(output_dir: Path = OUT / "search",
                search: SearchConfig = SearchConfig()) -> tuple[NeuralV3Config, pd.DataFrame]:
    base = _load_base_store()
    observable_enabled = _audit_observable_enabled()
    admission_path = output_dir / "feature_admission.json"
    admission = (
        json.loads(admission_path.read_text()).get("admitted", {})
        if admission_path.exists() else {"wr": True, "roles": False, "style": False}
    )
    admitted_blocks = tuple(block for block, allowed in admission.items() if allowed)
    candidates = neural_candidates(
        search.neural_candidates, search.seed, admitted_blocks=admitted_blocks,
    )
    candidates = [
        replace(candidate, use_appearance=_audit_appearance_enabled())
        for candidate in candidates
    ]
    active = list(range(len(candidates)))
    rows = []
    stores: dict[tuple[str, ...], pd.DataFrame] = {}
    stopped_early = False
    stop_reason = None
    for rung_index, epochs in enumerate(search.neural_rungs):
        rung_rows = []
        fold_years = (2024,) if rung_index < len(search.neural_rungs) - 1 else search.rolling_years
        for candidate_id in active:
            config = replace(candidates[candidate_id], epochs=epochs)
            if config.context_blocks not in stores:
                stores[config.context_blocks] = _feature_store(base, config.context_blocks)
            store = stores[config.context_blocks]
            fold_metrics = []
            for year, train, validation in rolling_folds(store, fold_years):
                print(
                    f"[neural rung {rung_index + 1}/{len(search.neural_rungs)} "
                    f"candidate {candidate_id + 1:02d}] fold {year}, {epochs} epochs",
                    flush=True,
                )
                model = ExposureRateNeural(config=config).fit(train, validation)
                fold_metrics.append(_objective(train, validation, model, observable_enabled))
            aggregate = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
            row = {
                "candidate_id": candidate_id, "rung": rung_index + 1,
                "epochs": epochs, **config.to_dict(), **aggregate,
            }
            rows.append(row)
            rung_rows.append(row)
        reference = _gbdt_reference(output_dir, tuple(fold_years))
        if reference is not None:
            materially_worse = []
            for row in rung_rows:
                raw_worse = row["raw_deviance"] > reference["raw_deviance"] * 1.03
                observable_worse = (
                    not observable_enabled
                    or (
                        np.isfinite(row.get("observable_points_mae", np.nan))
                        and row["observable_points_mae"]
                        > reference["observable_points_mae"] * 1.03
                    )
                )
                materially_worse.append(raw_worse and observable_worse)
            if materially_worse and all(materially_worse):
                stopped_early = True
                stop_reason = (
                    f"all {len(rung_rows)} configurations exceeded the same-fold "
                    "GBDT-v3 raw deviance and observable-points MAE by more than 3%"
                )
                break
        if rung_index < len(search.neural_survivors):
            keep = search.neural_survivors[rung_index]
            active = [
                int(row["candidate_id"])
                for row in sorted(rung_rows, key=lambda item: item["objective"])[:keep]
            ]
    results = pd.DataFrame(rows)
    completed_rung = int(results["rung"].max())
    final_rung = results[results["rung"].eq(completed_rung)].sort_values("objective")
    best_id = int(final_rung.iloc[0]["candidate_id"])
    best = candidates[best_id]
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "neural_candidates.csv", index=False)
    (output_dir / "best_neural.json").write_text(json.dumps(best.to_dict(), indent=2) + "\n")
    (output_dir / "neural_track.json").write_text(json.dumps({
        "status": "stopped" if stopped_early else "completed",
        "completed_rung": completed_rung,
        "surviving_candidate_id": best_id,
        "reason": stop_reason,
    }, indent=2) + "\n")
    return best, results


def assess_blend(output_dir: Path = OUT / "search") -> dict:
    """Evaluate the sole permitted global event-level blend on tuning folds."""
    track_path = output_dir / "neural_track.json"
    if track_path.exists() and json.loads(track_path.read_text()).get("status") == "stopped":
        payload = {
            "eligible": False,
            "reason": "neural track stopped before completing the bounded search",
        }
        (output_dir / "blend.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    g_path, n_path = output_dir / "best_gbdt.json", output_dir / "best_neural.json"
    if not g_path.exists() or not n_path.exists():
        raise FileNotFoundError("run both GBDT and neural tuning before blend assessment")
    g_config = GBDTV3Config.from_dict(json.loads(g_path.read_text()))
    n_config = NeuralV3Config.from_dict(json.loads(n_path.read_text()))
    base = _load_base_store()
    g_store = _feature_store(base, g_config.context_blocks)
    n_store = _feature_store(base, n_config.context_blocks)
    observable_enabled = _audit_observable_enabled()
    rows = []
    g_residuals, n_residuals = [], []
    for year in SearchConfig().rolling_years:
        g_fold = list(rolling_folds(g_store, (year,)))
        n_fold = list(rolling_folds(n_store, (year,)))
        if not g_fold or not n_fold:
            continue
        _, g_train, g_validation = g_fold[0]
        _, n_train, n_validation = n_fold[0]
        print(f"[blend eligibility] fold {year}", flush=True)
        g_model = ExposureRateGBDT(config=g_config).fit(g_train)
        n_model = ExposureRateNeural(config=n_config).fit(n_train, n_validation)
        g_predictions = g_model.predict_frame(g_validation)
        n_predictions = n_model.predict_frame(n_validation)
        g_metrics = raw_event_deviance(g_validation, g_predictions)
        n_metrics = raw_event_deviance(g_validation, n_predictions)
        common_events = sorted(
            set(g_predictions[0].events) & set(n_predictions[0].events)
        )
        g_residuals.append(raw_residual_vector(g_validation, g_predictions, common_events))
        n_residuals.append(raw_residual_vector(n_validation, n_predictions, common_events))
        allowed = observable_events(g_train[g_train["competition_level"].eq("international")])
        for weight in (0.25, 0.5, 0.75):
            blended = blend_predictions(g_predictions, n_predictions, weight)
            metrics = raw_event_deviance(g_validation, blended)
            observable_maes = []
            if observable_enabled and allowed:
                for competition in ("six_nations", "ncr"):
                    actual = observable_points_actual(g_validation, competition, allowed)
                    predicted = observable_points_predicted(blended, competition, allowed)
                    observable_maes.append(float(np.mean(np.abs(predicted - actual))))
            observable_mae = (
                float(np.mean(observable_maes)) if observable_maes else np.nan
            )
            rows.append({
                "year": year, "neural_weight": weight,
                "gbdt_raw_deviance": g_metrics["raw_deviance"],
                "neural_raw_deviance": n_metrics["raw_deviance"],
                "raw_deviance": metrics["raw_deviance"],
                "observable_points_mae": observable_mae,
                "objective": metrics["raw_deviance"] + (
                    observable_mae / 25.0 if np.isfinite(observable_mae) else 0.0
                ),
            })
    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("no rolling folds available for blend assessment")
    g_residual = np.concatenate(g_residuals)
    n_residual = np.concatenate(n_residuals)
    residual_correlation = float(np.corrcoef(g_residual, n_residual)[0, 1])
    g_deviance = float(results["gbdt_raw_deviance"].mean())
    n_deviance = float(results["neural_raw_deviance"].mean())
    eligible = bool(
        n_deviance <= g_deviance * 1.03
        and np.isfinite(residual_correlation)
        and residual_correlation < 0.95
    )
    summary = (
        results.groupby("neural_weight", as_index=False)
        .agg(objective=("objective", "mean"), raw_deviance=("raw_deviance", "mean"),
             observable_points_mae=("observable_points_mae", "mean"))
        .sort_values("objective")
    )
    best_weight = float(summary.iloc[0]["neural_weight"]) if eligible else None
    payload = {
        "eligible": eligible, "neural_weight": best_weight,
        "gbdt_raw_deviance": g_deviance, "neural_raw_deviance": n_deviance,
        "neural_relative_deviance": n_deviance / max(g_deviance, 1e-12) - 1,
        "residual_correlation": residual_correlation,
        "rules": {"max_neural_regression": 0.03, "max_residual_correlation": 0.95},
        "reason": (
            None if eligible else
            "neural must be within 3% raw deviance and residual correlation below 0.95"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "blend_folds.csv", index=False)
    summary.to_csv(output_dir / "blend_summary.csv", index=False)
    (output_dir / "blend.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
