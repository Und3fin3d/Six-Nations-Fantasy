"""Reproduce the guarded NCR GW1-3 P3 blend checkpoint.

This module performs a retrospective reachability search. Official NCR GW1-3
labels enter the search, so its result is not prospective model evidence. The
module reads frozen P3 components and writes only to a new output directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from ..schema import EVENTS
from ..scoring import NationsChampionshipScorer, SixNationsScorer
from .blend import EventBlend50
from .config import STABLE_EVENTS
from .p3_hillclimb import (
    BASELINE_ENGINE,
    BASELINE_WEIGHT,
    CANDIDATE_ENGINE,
    DEVELOPMENT_YEARS,
    SELECTION_YEARS,
    TargetEvidence,
    _manifest_key,
    _sha256,
    build_run_manifest,
    infer_left_mean,
    load_evidence,
    official_ncr_proof,
    target_loss,
    weighted_mean,
)

LABEL = "NCR-GW1-3-retrospective-optimized"
SCHEMA_VERSION = 1
TARGET_TEAM_POINTS = 1595.0
DEVELOPMENT_RAW_LIMIT = 0.884331824
SELECTION_RAW_LIMIT = 0.898153286
MAX_FOLD_REGRESSION = 0.010
MAX_OFFICIAL_EVALUATIONS = 3000
MAX_COORDINATE_PASSES = 8
DUAL_RUBRIC_RIDGE = 0.05

RAW_WEIGHT_GRID = tuple(round(index / 100.0, 2) for index in range(101))
PATH_ALPHAS = tuple(round(index / 100.0, 2) for index in range(1, 101))
COORDINATE_GRID = tuple(round(index / 20.0, 2) for index in range(21))
COMPLETE_TARGETS = ("minutes", *EVENTS)
PRIMARY_RAW_TARGETS = ("minutes", *STABLE_EVENTS)

# The order is part of the checkpoint contract. Every coordinate is global and
# competition-independent. NCR labels select the checkpoint retrospectively.
OFFICIAL_COORDINATES = (
    "tries",
    "try_assists",
    "conversion_goals",
    "missed_conversion_goals",
    "penalty_goals",
    "missed_penalty_goals",
    "drop_goals_converted",
    "defenders_beaten",
    "offload",
    "clean_breaks",
    "tackles",
    "missed_tackles",
    "tackle_turnover",
    "turnovers_conceded",
    "penalties_conceded",
    "yellow_cards",
    "red_cards",
    "lineout_steals",
    "potm",
    "scrums_won",
)

RESULT_FILES = (
    "config.json",
    "dual_rubric_seed.json",
    "official_direct_metrics.csv",
    "official_direct_teams.csv",
    "official_direct_verification.json",
    "raw_grid.csv",
    "raw_summary.json",
    "REPORT.md",
    "summary.json",
    "trials.jsonl",
)


def _snap(value: float, step: float = 0.01) -> float:
    return float(np.clip(np.rint(float(value) / step) * step, 0.0, 1.0))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _load_complete_weights(path: Path) -> tuple[dict, dict[str, float]]:
    config = json.loads(path.read_text())
    if config.get("base_model") != BASELINE_ENGINE:
        raise ValueError(f"{path} does not use {BASELINE_ENGINE}")
    default = float(config.get("default_weight_v4", BASELINE_WEIGHT))
    if not np.isclose(default, BASELINE_WEIGHT, rtol=0.0, atol=1e-12):
        raise ValueError("the checkpoint requires the frozen P3 default weight 0.5")
    supplied = {
        str(target): float(weight)
        for target, weight in config.get("event_weights_v4", {}).items()
    }
    unknown = set(supplied) - set(COMPLETE_TARGETS)
    if unknown:
        raise ValueError(f"the base config has unknown targets: {sorted(unknown)}")
    invalid = {
        target: weight for target, weight in supplied.items()
        if not 0.0 <= weight <= 1.0
    }
    if invalid:
        raise ValueError(f"the base config has invalid weights: {invalid}")
    complete = {target: default for target in COMPLETE_TARGETS}
    complete.update(supplied)
    return config, complete


def _sorted_weights(weights: Mapping[str, float]) -> dict[str, float]:
    return {target: float(weights[target]) for target in sorted(weights)}


def _sparse_weights(weights: Mapping[str, float]) -> dict[str, float]:
    return {
        target: float(weight) for target, weight in weights.items()
        if not np.isclose(weight, BASELINE_WEIGHT, rtol=0.0, atol=1e-12)
    }


def _prediction_mean(predictions, target: str) -> np.ndarray:
    return np.asarray([
        prediction.minutes.mean if target == "minutes"
        else prediction.events[target].mean
        for prediction in predictions
    ], dtype=float)


def _add_optimized_extended_evidence(
    evidence: Iterable[TargetEvidence], bundles: Iterable,
    benchmark_dir: Path,
) -> list[TargetEvidence]:
    """Add raw evidence for every optimized event outside the stable tier."""
    output = list(evidence)
    extended_targets = tuple(
        target for target in OFFICIAL_COORDINATES
        if target not in PRIMARY_RAW_TARGETS
    )
    metrics = pd.read_csv(benchmark_dir / "event_metrics.csv")
    metrics = metrics[
        metrics["engine"].eq(BASELINE_ENGINE)
        & metrics["cohort"].eq("all")
        & metrics["target"].isin(extended_targets)
    ].copy()
    if metrics.duplicated(["fold", "target"]).any():
        raise ValueError("extended raw metrics contain duplicate fold targets")
    metric_rows = metrics.set_index(["fold", "target"])
    for bundle in bundles:
        for target in extended_targets:
            key = (bundle.fold, target)
            if key not in metric_rows.index:
                continue
            actual = pd.to_numeric(
                bundle.evaluation[target], errors="coerce",
            ).to_numpy(float)
            available = bundle.evaluation[
                f"available__{target}"
            ].fillna(False).to_numpy(bool)
            empirical_mean = _prediction_mean(bundle.empirical, target)
            baseline_mean = _prediction_mean(bundle.baseline, target)
            valid = available & np.isfinite(actual) & np.isfinite(empirical_mean)
            valid &= np.isfinite(baseline_mean)
            if not valid.any():
                raise ValueError(f"{bundle.fold}/{target}: no valid extended rows")
            v4_mean = infer_left_mean(
                baseline_mean[valid], empirical_mean[valid],
            )
            rebuilt = weighted_mean(
                empirical_mean[valid], v4_mean, BASELINE_WEIGHT,
            )
            if not np.allclose(
                rebuilt, baseline_mean[valid], rtol=0.0, atol=1e-10,
            ):
                raise AssertionError(
                    f"{bundle.fold}/{target}: reconstructed P3 means differ"
                )
            row = metric_rows.loc[key]
            naive_loss = float(row["naive_loss"])
            rebuilt_score = (
                target_loss(target, actual[valid], rebuilt) / naive_loss
            )
            if not np.isclose(
                rebuilt_score, float(row["relative_loss"]),
                rtol=0.0, atol=1e-10,
            ):
                raise AssertionError(
                    f"{bundle.fold}/{target}: extended raw score differs"
                )
            output.append(TargetEvidence(
                fold=bundle.fold,
                tournament=bundle.tournament,
                calendar_year=bundle.calendar_year,
                hemisphere=str(bundle.evaluation["hemisphere"].iloc[0]),
                target=target,
                actual=actual[valid],
                empirical=empirical_mean[valid],
                v4=v4_mean,
                naive_loss=naive_loss,
            ))
    covered = {item.target for item in output}
    missing = sorted(set(OFFICIAL_COORDINATES) - covered)
    if missing:
        raise ValueError(f"optimized coordinates lack raw evidence: {missing}")
    return output


@dataclass(frozen=True)
class RawGrid:
    evidence: tuple[TargetEvidence, ...]
    values: np.ndarray

    @classmethod
    def build(cls, evidence: Iterable[TargetEvidence]) -> "RawGrid":
        items = tuple(evidence)
        values = np.empty((len(items), len(RAW_WEIGHT_GRID)), dtype=float)
        for row, item in enumerate(items):
            for column, weight in enumerate(RAW_WEIGHT_GRID):
                prediction = weighted_mean(item.empirical, item.v4, weight)
                values[row, column] = (
                    target_loss(item.target, item.actual, prediction) / item.naive_loss
                )
        return cls(evidence=items, values=values)

    def summary(self, weights: Mapping[str, float]) -> dict:
        columns = np.asarray([
            int(np.rint(float(weights[item.target]) * 100.0))
            for item in self.evidence
        ])
        if ((columns < 0) | (columns >= len(RAW_WEIGHT_GRID))).any():
            raise ValueError("raw weights must be on the 0.01 grid in [0, 1]")
        candidate = self.values[np.arange(len(self.evidence)), columns]
        baseline = self.values[:, 50]
        years = np.asarray([item.calendar_year for item in self.evidence])
        folds = np.asarray([item.fold for item in self.evidence], dtype=object)
        targets = np.asarray([item.target for item in self.evidence], dtype=object)
        primary = np.isin(targets, PRIMARY_RAW_TARGETS)
        development = np.isin(years, DEVELOPMENT_YEARS) & primary
        selection = np.isin(years, SELECTION_YEARS) & primary
        fold_scores: dict[str, float] = {}
        baseline_fold_scores: dict[str, float] = {}
        fold_deltas: dict[str, float] = {}
        for fold in sorted(set(folds)):
            mask = folds == fold
            fold_scores[fold] = float(candidate[mask].mean())
            baseline_fold_scores[fold] = float(baseline[mask].mean())
            fold_deltas[fold] = fold_scores[fold] - baseline_fold_scores[fold]
        limiting_fold = max(fold_deltas, key=lambda name: (fold_deltas[name], name))
        development_score = float(candidate[development].mean())
        selection_score = float(candidate[selection].mean())
        maximum_regression = float(fold_deltas[limiting_fold])
        failures = []
        if development_score > DEVELOPMENT_RAW_LIMIT + 1e-12:
            failures.append("development_raw_limit")
        if selection_score > SELECTION_RAW_LIMIT + 1e-12:
            failures.append("selection_raw_limit")
        if maximum_regression > MAX_FOLD_REGRESSION + 1e-12:
            failures.append("maximum_fold_regression")
        return {
            "development_score": development_score,
            "selection_score": selection_score,
            "all_fold_score": float(candidate[primary].mean()),
            "guarded_targets_all_score": float(candidate.mean()),
            "baseline_development_score": float(baseline[development].mean()),
            "baseline_selection_score": float(baseline[selection].mean()),
            "baseline_all_fold_score": float(baseline[primary].mean()),
            "baseline_guarded_targets_all_score": float(baseline.mean()),
            "fold_scores": fold_scores,
            "baseline_fold_scores": baseline_fold_scores,
            "fold_deltas": fold_deltas,
            "max_fold_regression": maximum_regression,
            "max_fold_regression_fold": limiting_fold,
            "raw_feedback_years": sorted(int(year) for year in set(years)),
            "uses_2026_raw_feedback": bool(2026 in set(years)),
            "optimized_extended_targets": sorted(
                set(targets) - set(PRIMARY_RAW_TARGETS)
            ),
            "passes": not failures,
            "failures": failures,
        }

    def frame(self) -> pd.DataFrame:
        rows = []
        for item, values in zip(self.evidence, self.values):
            row = {
                "fold": item.fold,
                "tournament": item.tournament,
                "calendar_year": item.calendar_year,
                "hemisphere": item.hemisphere,
                "target": item.target,
                "tier": (
                    "primary" if item.target in PRIMARY_RAW_TARGETS
                    else "optimized_extended"
                ),
                "naive_loss": item.naive_loss,
            }
            row.update({
                f"weight_{weight:.2f}": float(value)
                for weight, value in zip(RAW_WEIGHT_GRID, values)
            })
            rows.append(row)
        return pd.DataFrame(rows)


def _rubric_coefficients(target: str) -> list[float]:
    ncr = dict(NationsChampionshipScorer.weights)
    ncr["scrums_won"] = 2.0
    six = dict(SixNationsScorer.weights)
    # The Six Nations adapter has position-dependent tries and stepwise metres.
    # These two linear approximations define this historical seed only.
    six["tries"] = 12.5
    six["metres"] = 0.1
    return [
        abs(float(coefficient))
        for coefficient in (ncr.get(target, 0.0), six.get(target, 0.0))
        if coefficient
    ]


def _dual_rubric_seed(
    evidence: Iterable[TargetEvidence], prior: Mapping[str, float],
) -> tuple[dict[str, float], dict]:
    items = tuple(evidence)
    seed = dict(prior)
    details = {}
    for target in COMPLETE_TARGETS:
        coefficients = _rubric_coefficients(target)
        target_items = [
            item for item in items
            if item.target == target and item.calendar_year in DEVELOPMENT_YEARS
        ]
        if not coefficients or not target_items:
            continue
        delta = np.concatenate([item.v4 - item.empirical for item in target_items])
        residual = np.concatenate([
            item.actual - item.empirical for item in target_items
        ])
        scale = float(sum(value * value for value in coefficients))
        numerator = scale * float(np.mean(delta * residual))
        denominator = scale * float(np.mean(delta * delta))
        unsnapped = (
            numerator + DUAL_RUBRIC_RIDGE * float(prior[target])
        ) / (denominator + DUAL_RUBRIC_RIDGE)
        snapped = _snap(unsnapped)
        seed[target] = snapped
        details[target] = {
            "rubric_coefficients": coefficients,
            "weight_v4_unsnapped": float(unsnapped),
            "weight_v4": snapped,
            "observations": int(len(delta)),
        }
    return seed, {
        "origin": "2022-2024 dual-rubric bounded least squares",
        "years": list(DEVELOPMENT_YEARS),
        "rubrics": ["ncr", "six_nations"],
        "ridge_lambda": DUAL_RUBRIC_RIDGE,
        "prior": "base p3_event_weighted config",
        "event_weights_v4": _sorted_weights(seed),
        "details": details,
    }


@dataclass
class OfficialEvaluator:
    matrix: np.ndarray
    constant: np.ndarray
    labels: np.ndarray
    gameweeks: np.ndarray
    fantasy_ids: np.ndarray
    expected_minutes: np.ndarray
    projections: dict[int, pd.DataFrame]
    actuals: dict[int, Mapping]
    gw_exclusions: Mapping[int, tuple[str, ...]]

    @classmethod
    def build(
        cls, benchmark_dir: Path, base_weights: Mapping[str, float],
    ) -> "OfficialEvaluator":
        from model.ncr_eval import load_actuals as load_team_actuals

        from ..ncr_gw_eval import GW_EXCLUSIONS
        from ..v3.shadow import ncr_candidates
        from .features import build_frozen_feature_frames
        from .folds import build_folds, masked_candidates, strict_training_frame

        data = benchmark_dir.parents[2]
        ncr = data / "ncr"
        corrected_gw3 = (
            data / "unified" / "ncr_gw1_3_eval"
            / "corrected_gw3_incumbent_projections.csv"
        )
        sources = []
        projections = {}
        actuals = {}
        for gameweek in (1, 2, 3):
            if gameweek == 3:
                projection = pd.read_csv(corrected_gw3)
                source = ncr_candidates(gameweek, projection=projection)
            else:
                projection = pd.read_csv(ncr / f"ncr_gw{gameweek}_projections.csv")
                source = ncr_candidates(gameweek)
            source = source.copy()
            source["gw"] = gameweek
            sources.append(source)
            projections[gameweek] = projection
            actuals[gameweek] = load_team_actuals(
                ncr / "feeds" / f"players_gw{gameweek}.json"
            )
        candidates = masked_candidates(
            pd.concat(sources, ignore_index=True, sort=False)
        )
        store = pd.read_csv(
            benchmark_dir / "player_match.csv", low_memory=False,
            parse_dates=["date", "match_at"],
        )
        fold = next(
            item for item in build_folds(store)
            if item.label == "nations_championship_2026"
        )
        train = strict_training_frame(store, fold)
        _, features = build_frozen_feature_frames(train, candidates, v4=True)
        model = EventBlend50.load(
            benchmark_dir / "models" / BASELINE_ENGINE / f"{fold.label}.pkl"
        )
        baseline = tuple(model.predict_frame(features))
        empirical = tuple(model.empirical.predict_frame(features))
        if len(baseline) != len(candidates) or len(empirical) != len(candidates):
            raise ValueError("official component prediction lengths differ")

        scoring = dict(NationsChampionshipScorer.weights)
        scoring["scrums_won"] = 2.0
        coordinate_index = {
            target: index for index, target in enumerate(OFFICIAL_COORDINATES)
        }
        matrix = np.zeros((len(candidates), len(OFFICIAL_COORDINATES)), dtype=float)
        constant = np.zeros(len(candidates), dtype=float)
        expected_minutes = np.empty(len(candidates), dtype=float)
        for row_index, (source, base, empirical_prediction) in enumerate(zip(
            candidates.itertuples(index=False), baseline, empirical,
        )):
            expected_key = (str(source.player_id), str(source.team))
            if (base.player_id, base.team) != expected_key:
                raise ValueError("official baseline prediction order changed")
            if (empirical_prediction.player_id, empirical_prediction.team) != expected_key:
                raise ValueError("official empirical prediction order changed")
            expected_minutes[row_index] = base.minutes.mean
            for target, coefficient in scoring.items():
                base_distribution = base.events.get(target)
                empirical_distribution = empirical_prediction.events.get(target)
                if base_distribution is None:
                    continue
                if empirical_distribution is None:
                    constant[row_index] += coefficient * base_distribution.mean
                    continue
                empirical_mean = float(empirical_distribution.mean)
                v4_mean = float(infer_left_mean(
                    np.asarray([base_distribution.mean]),
                    np.asarray([empirical_mean]),
                )[0])
                if target in coordinate_index:
                    constant[row_index] += coefficient * empirical_mean
                    matrix[row_index, coordinate_index[target]] += (
                        coefficient * (v4_mean - empirical_mean)
                    )
                else:
                    mean = float(weighted_mean(
                        np.asarray([empirical_mean]), np.asarray([v4_mean]),
                        float(base_weights.get(target, BASELINE_WEIGHT)),
                    )[0])
                    constant[row_index] += coefficient * mean

        labels = []
        for source in candidates.itertuples(index=False):
            values = actuals[int(source.gw)].get(int(source.fantasy_id))
            if values is None:
                values = actuals[int(source.gw)].get(str(source.fantasy_id))
            if values is None:
                raise ValueError("official NCR proof is missing fantasy labels")
            labels.append(float(values[0]))
        evaluator = cls(
            matrix=matrix,
            constant=constant,
            labels=np.asarray(labels, dtype=float),
            gameweeks=candidates["gw"].to_numpy(int),
            fantasy_ids=candidates["fantasy_id"].to_numpy(int),
            expected_minutes=expected_minutes,
            projections=projections,
            actuals=actuals,
            gw_exclusions=GW_EXCLUSIONS,
        )
        return evaluator

    def oracle_seed(self, prior: Mapping[str, float]) -> tuple[dict[str, float], dict]:
        solution = lsq_linear(
            self.matrix, self.labels - self.constant, bounds=(0.0, 1.0),
            method="trf", tol=1e-12, lsmr_tol=1e-12,
        )
        if not solution.success:
            raise RuntimeError(f"official bounded least squares failed: {solution.message}")
        seed = dict(prior)
        for target, weight in zip(OFFICIAL_COORDINATES, solution.x):
            seed[target] = _snap(float(weight))
        return seed, {
            "origin": "official NCR GW1-3 bounded least squares",
            "retrospective_oracle": True,
            "cost": float(solution.cost),
            "optimality": float(solution.optimality),
            "iterations": None if solution.nit is None else int(solution.nit),
            "event_weights_v4": {
                target: seed[target] for target in OFFICIAL_COORDINATES
            },
        }

    def proxy(self, vector: np.ndarray) -> float:
        prediction = self.constant + self.matrix @ vector
        values = []
        for gameweek in (1, 2, 3):
            mask = self.gameweeks == gameweek
            residual = prediction[mask] - self.labels[mask]
            residual = residual - residual.mean()
            values.append(float(np.mean(residual * residual)))
        return -float(np.mean(values))

    def evaluate(self, weights: Mapping[str, float]) -> dict:
        from model.ncr_eval import team_points
        from model.ncr_project import optimise

        vector = np.asarray([weights[target] for target in OFFICIAL_COORDINATES])
        predicted = self.constant + self.matrix @ vector
        rounds = []
        teams = {}
        for gameweek in (1, 2, 3):
            mask = self.gameweeks == gameweek
            prediction = pd.DataFrame({
                "fantasy_id": self.fantasy_ids[mask],
                "predicted_points": predicted[mask],
                "expected_minutes": self.expected_minutes[mask],
            })
            pool = self.projections[gameweek].merge(
                prediction, left_on="id", right_on="fantasy_id",
                validate="one_to_one",
            )
            exclusions = self.gw_exclusions[gameweek]
            if exclusions:
                pool = pool[~pool["team"].isin(exclusions)].copy()
            pool["starter_exp"] = pool["predicted_points"]
            pool["supersub_exp"] = 3.0 * pool["predicted_points"]
            pool["exp_min"] = pool["expected_minutes"]
            squad, _, _ = optimise(pool)
            points = float(team_points(squad, self.actuals[gameweek]))
            rounds.append(points)
            teams[str(gameweek)] = {
                "team_points": points,
                "captain": str(squad.loc[squad["is_capt"], "name"].iloc[0]),
                "super_sub": str(squad.loc[squad["is_sub"], "name"].iloc[0]),
                "players": sorted(str(name) for name in squad["name"]),
            }
        return {
            "total": float(sum(rounds)),
            "rounds": rounds,
            "proxy": self.proxy(vector),
            "teams": teams,
        }


@dataclass
class SearchState:
    raw_grid: RawGrid
    official: OfficialEvaluator
    ledger: list[dict]
    official_cache: dict[tuple[float, ...], dict]
    official_evaluations: int = 0

    def evaluate(self, weights: Mapping[str, float], origin: str) -> dict:
        complete = {
            target: _snap(float(weights[target])) for target in COMPLETE_TARGETS
        }
        raw = self.raw_grid.summary(complete)
        key = tuple(complete[target] for target in OFFICIAL_COORDINATES)
        cached = False
        official_result = None
        if raw["passes"]:
            if key in self.official_cache:
                cached = True
                official_result = self.official_cache[key]
            else:
                if self.official_evaluations >= MAX_OFFICIAL_EVALUATIONS:
                    raise RuntimeError(
                        f"official evaluation limit {MAX_OFFICIAL_EVALUATIONS} reached"
                    )
                official_result = self.official.evaluate(complete)
                self.official_cache[key] = official_result
                self.official_evaluations += 1
        record = {
            "trial_id": len(self.ledger) + 1,
            "origin": origin,
            "event_weights_v4": _sorted_weights(complete),
            "raw": raw,
            "official_evaluated": official_result is not None,
            "official_cache_hit": cached,
            "official_evaluation_number": (
                self.official_evaluations
                if official_result is not None and not cached else None
            ),
            "official_cache_key": list(key),
            "official": official_result,
            "accepted_as_incumbent": False,
        }
        self.ledger.append(record)
        return record


def _objective(record: Mapping) -> tuple[float, float]:
    official = record.get("official")
    if official is None:
        return (-np.inf, -np.inf)
    return float(official["total"]), float(official["proxy"])


def _mark_incumbent(record: dict) -> dict:
    record["accepted_as_incumbent"] = True
    return record


def _threshold_reached(record: Mapping) -> bool:
    official = record.get("official")
    return official is not None and float(official["total"]) >= TARGET_TEAM_POINTS


def _interpolate(
    start: Mapping[str, float], end: Mapping[str, float], alpha: float,
) -> dict[str, float]:
    return {
        target: _snap(
            float(start[target]) + alpha * (float(end[target]) - float(start[target]))
        )
        for target in COMPLETE_TARGETS
    }


def _run_search(
    raw_grid: RawGrid, official: OfficialEvaluator,
    base: Mapping[str, float], dual: Mapping[str, float],
    oracle: Mapping[str, float],
) -> tuple[dict, list[dict], int]:
    state = SearchState(raw_grid, official, [], {})
    incumbent = _mark_incumbent(state.evaluate(base, "seed_current_weighted"))
    if _threshold_reached(incumbent):
        return incumbent, state.ledger, state.official_evaluations

    for origin, seed in (
        ("seed_dual_rubric_2022_2024", dual),
        ("seed_official_bounded_ls_oracle", oracle),
    ):
        record = state.evaluate(seed, origin)
        if _objective(record) > _objective(incumbent):
            incumbent = _mark_incumbent(record)
        if _threshold_reached(record):
            return _mark_incumbent(record), state.ledger, state.official_evaluations

    for path_name, endpoint in (
        ("current_to_official_oracle", oracle),
        ("current_to_dual_rubric", dual),
    ):
        for alpha in PATH_ALPHAS:
            record = state.evaluate(
                _interpolate(base, endpoint, alpha),
                f"path_{path_name}_alpha_{alpha:.2f}",
            )
            if _objective(record) > _objective(incumbent):
                incumbent = _mark_incumbent(record)
            if _threshold_reached(record):
                return _mark_incumbent(record), state.ledger, state.official_evaluations

    for pass_number in range(1, MAX_COORDINATE_PASSES + 1):
        pass_improved = False
        for target in OFFICIAL_COORDINATES:
            coordinate_start = dict(incumbent["event_weights_v4"])
            coordinate_best = incumbent
            for weight in COORDINATE_GRID:
                candidate = dict(coordinate_start)
                candidate[target] = weight
                record = state.evaluate(
                    candidate,
                    f"coordinate_{pass_number}_{target}_{weight:.2f}",
                )
                if _threshold_reached(record):
                    return _mark_incumbent(record), state.ledger, state.official_evaluations
                if _objective(record) > _objective(coordinate_best):
                    coordinate_best = record
            if coordinate_best is not incumbent:
                incumbent = _mark_incumbent(coordinate_best)
                pass_improved = True
        if not pass_improved:
            break
    return incumbent, state.ledger, state.official_evaluations


def _direct_verification(
    weights: Mapping[str, float], benchmark_dir: Path, search_result: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics, teams = official_ncr_proof(_sparse_weights(weights), benchmark_dir)
    candidate = teams[teams["engine"].eq(CANDIDATE_ENGINE)]
    direct_rounds = [
        float(candidate.loc[candidate["gw"].eq(gameweek), "team_points"].iloc[0])
        for gameweek in (1, 2, 3)
    ]
    direct_total = float(candidate.loc[candidate["gw"].eq(0), "team_points"].iloc[0])
    search_rounds = [float(value) for value in search_result["rounds"]]
    search_total = float(search_result["total"])
    if direct_rounds != search_rounds or direct_total != search_total:
        raise AssertionError(
            "the direct weighted model does not reproduce the linear checkpoint proof"
        )
    return metrics, teams, {
        "search_total": search_total,
        "search_rounds": search_rounds,
        "direct_total": direct_total,
        "direct_rounds": direct_rounds,
        "totals_match": True,
        "direct_path": "p3_hillclimb.official_ncr_proof",
    }


def _report(summary: Mapping) -> str:
    raw = summary["raw"]
    official = summary["official"]
    rounds = official["rounds"]
    reached = "reaches" if summary["threshold_reached"] else "does not reach"
    return "\n".join([
        f"# {LABEL}",
        "",
        "This checkpoint is a retrospective reachability result.",
        "Official NCR GW1-3 labels enter the weight search.",
        "The raw fold guard also uses available 2026 event labels.",
        "The checkpoint does not change an active model or projection artifact.",
        "",
        "## Result",
        "",
        f"The global event-weight vector {reached} {TARGET_TEAM_POINTS:.0f} points.",
        f"The verified total is {official['total']:.0f} points.",
        f"The round totals are {rounds[0]:.0f}, {rounds[1]:.0f}, and {rounds[2]:.0f}.",
        f"The search used {summary['official_evaluations']} official MILP evaluations.",
        "",
        "## Raw guards",
        "",
        f"The development raw score is {raw['development_score']:.12f}.",
        f"The 2025 raw score is {raw['selection_score']:.12f}.",
        f"The all-fold raw score is {raw['all_fold_score']:.12f}.",
        "The all-fold score uses the primary stable raw-event target set.",
        f"The maximum fold regression is {raw['max_fold_regression']:+.12f}.",
        f"The limiting fold is `{raw['max_fold_regression_fold']}`.",
        "The fold guard includes every optimized extended event with raw evidence.",
        "",
        "## Files",
        "",
        "`config.json` contains the complete merged vector.",
        "`trials.jsonl` contains every attempted trial.",
        "`raw_grid.csv` contains the precomputed 0.01 raw-loss grid.",
        "`official_direct_verification.json` records the direct model check.",
        "`manifest.json` records source and result SHA-256 hashes.",
        "",
    ])


def _source_manifest(
    benchmark_dir: Path, base_config_path: Path, output_dir: Path,
) -> dict:
    manifest = build_run_manifest(benchmark_dir, include_official_ncr=True)
    repository = Path(__file__).resolve().parents[3]
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["checkpoint_label"] = LABEL
    manifest["input_sha256"] = dict(manifest["input_sha256"])
    manifest["input_sha256"][
        _manifest_key(base_config_path, repository)
    ] = _sha256(base_config_path)
    manifest["implementation_sha256"] = dict(manifest["implementation_sha256"])
    manifest["implementation_sha256"][
        _manifest_key(Path(__file__), repository)
    ] = _sha256(Path(__file__))
    cli_path = Path(__file__).resolve().with_name("cli.py")
    manifest["implementation_sha256"][
        _manifest_key(cli_path, repository)
    ] = _sha256(cli_path)
    manifest["invocation"] = {
        "command": "python -m model.unified.cli raw-benchmark p3-checkpoint",
        "benchmark": _manifest_key(benchmark_dir, repository),
        "base_config": _manifest_key(base_config_path, repository),
        "output": _manifest_key(output_dir, repository),
    }
    return manifest


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def run_checkpoint(
    benchmark_dir: Path, base_config_path: Path, output_dir: Path,
) -> dict:
    """Run and persist the guarded retrospective checkpoint search.

    The function requires a new output directory. It never overwrites an
    existing directory, the base config, fitted models, predictions, or cache.
    """
    benchmark_dir = Path(benchmark_dir)
    base_config_path = Path(base_config_path)
    output_dir = Path(output_dir)
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(benchmark_dir)
    if not base_config_path.is_file():
        raise FileNotFoundError(base_config_path)
    if output_dir.exists():
        raise FileExistsError(f"checkpoint output must be a new directory: {output_dir}")
    resolved_output = output_dir.resolve()
    repository = Path(__file__).resolve().parents[3]
    forbidden_outputs = {
        "benchmark artifacts": benchmark_dir.resolve(),
        "base config artifacts": base_config_path.resolve().parent,
        "API cache": (repository / "data" / "cache").resolve(),
    }
    for description, forbidden in forbidden_outputs.items():
        if _is_within(resolved_output, forbidden):
            raise ValueError(
                f"checkpoint output cannot be inside {description}: {output_dir}"
            )

    source_manifest = _source_manifest(
        benchmark_dir, base_config_path, output_dir,
    )
    base_config, base_weights = _load_complete_weights(base_config_path)
    evidence, bundles = load_evidence(benchmark_dir)
    evidence = _add_optimized_extended_evidence(
        evidence, bundles, benchmark_dir,
    )
    raw_grid = RawGrid.build(evidence)
    baseline_raw = raw_grid.summary({
        target: BASELINE_WEIGHT for target in COMPLETE_TARGETS
    })
    frozen = pd.read_csv(benchmark_dir / "event_metrics.csv")
    expected_baseline = float(frozen[
        frozen["engine"].eq(BASELINE_ENGINE)
        & frozen["cohort"].eq("all")
        & frozen["tier"].isin(["minutes", "stable"])
    ]["relative_loss"].mean())
    if not np.isclose(
        baseline_raw["all_fold_score"], expected_baseline, rtol=0.0, atol=1e-10,
    ):
        raise AssertionError("the 0.01 raw grid does not reproduce frozen P3")

    dual_seed, dual_details = _dual_rubric_seed(evidence, base_weights)
    official = OfficialEvaluator.build(benchmark_dir, base_weights)
    oracle_seed, oracle_details = official.oracle_seed(base_weights)
    winner, ledger, official_evaluations = _run_search(
        raw_grid, official, base_weights, dual_seed, oracle_seed,
    )
    if winner["official"] is None:
        raise RuntimeError("the guarded search did not produce an official candidate")
    final_weights = dict(winner["event_weights_v4"])
    raw_summary = raw_grid.summary(final_weights)
    if not raw_summary["passes"]:
        raise AssertionError("the selected checkpoint does not pass the raw guards")
    direct_metrics, direct_teams, direct_verification = _direct_verification(
        final_weights, benchmark_dir, winner["official"],
    )
    threshold_reached = float(winner["official"]["total"]) >= TARGET_TEAM_POINTS
    config = {
        "schema_version": SCHEMA_VERSION,
        "label": LABEL,
        "model": CANDIDATE_ENGINE,
        "base_model": BASELINE_ENGINE,
        "default_weight_v4": BASELINE_WEIGHT,
        "event_weights_v4": _sorted_weights(final_weights),
        "model_form_competition_independent": True,
        "selection_feedback_competition_specific": True,
        "retrospective_optimized_on": "official NCR GW1-3",
        "retrospective_raw_feedback_includes_2026": True,
        "active_artifact": False,
        "base_config": _manifest_key(base_config_path, repository),
        "base_config_schema_version": base_config.get("schema_version"),
    }
    dual_details["raw"] = raw_grid.summary(dual_seed)
    dual_record = next(
        (
            record for record in ledger
            if record["origin"] == "seed_dual_rubric_2022_2024"
        ),
        None,
    )
    dual_details["official"] = None if dual_record is None else dual_record["official"]
    raw_payload = {
        "schema_version": SCHEMA_VERSION,
        "guards": {
            "development_score_maximum": DEVELOPMENT_RAW_LIMIT,
            "selection_2025_score_maximum": SELECTION_RAW_LIMIT,
            "fold_regression_maximum": MAX_FOLD_REGRESSION,
        },
        "baseline": baseline_raw,
        "checkpoint": raw_summary,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "label": LABEL,
        "retrospective_oracle": True,
        "threshold": TARGET_TEAM_POINTS,
        "threshold_reached": threshold_reached,
        "origin": winner["origin"],
        "event_weights_v4": _sorted_weights(final_weights),
        "raw": raw_summary,
        "official": winner["official"],
        "official_evaluations": official_evaluations,
        "trial_records": len(ledger),
        "search": {
            "stochastic_steps": False,
            "deterministic_fixed_order": True,
            "raw_weight_grid": list(RAW_WEIGHT_GRID),
            "path_alphas": list(PATH_ALPHAS),
            "coordinate_grid": list(COORDINATE_GRID),
            "coordinate_order": list(OFFICIAL_COORDINATES),
            "maximum_coordinate_passes": MAX_COORDINATE_PASSES,
            "maximum_official_evaluations": MAX_OFFICIAL_EVALUATIONS,
            "raw_invalid_candidates_reach_official_milp": False,
        },
        "dual_rubric_seed": dual_details,
        "official_oracle_seed": oracle_details,
        "official_direct_verification": direct_verification,
    }

    if _source_manifest(
        benchmark_dir, base_config_path, output_dir,
    ) != source_manifest:
        raise RuntimeError("checkpoint sources changed during the search")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "dual_rubric_seed.json", dual_details)
    direct_metrics.to_csv(
        output_dir / "official_direct_metrics.csv", index=False, float_format="%.15g",
    )
    direct_teams.to_csv(
        output_dir / "official_direct_teams.csv", index=False, float_format="%.15g",
    )
    _write_json(
        output_dir / "official_direct_verification.json", direct_verification,
    )
    raw_grid.frame().to_csv(
        output_dir / "raw_grid.csv", index=False, float_format="%.15g",
    )
    _write_json(output_dir / "raw_summary.json", raw_payload)
    (output_dir / "REPORT.md").write_text(_report(summary))
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "trials.jsonl").write_text("".join(
        json.dumps(_jsonable(record), sort_keys=True) + "\n" for record in ledger
    ))
    completed_manifest = dict(source_manifest)
    completed_manifest["result_sha256"] = {
        name: _sha256(output_dir / name) for name in RESULT_FILES
    }
    _write_json(output_dir / "manifest.json", completed_manifest)
    return summary


__all__ = ["run_checkpoint"]
