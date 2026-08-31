"""Local hill climb for P3 raw-event blend weights.

The search changes one raw-event blend coordinate at a time. It uses 2022-2024
for optimisation and 2025 for selection. It does not use 2026 in weight
selection. The existing 2026 results are retrospective confirmation evidence.
Fantasy scoring adapters are secondary diagnostics.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..contracts import EventDistribution, RawPrediction
from .blend import EventBlend50, EventWeightedBlend
from .config import OUT, STABLE_EVENTS, TOP_NS
from .folds import build_folds, evaluation_frame, masked_candidates
from .metrics import ranking_metrics, target_loss

BASELINE_ENGINE = "p3_event_50"
CANDIDATE_ENGINE = "p3_event_weighted"
BASELINE_WEIGHT = 0.5
TARGETS = ("minutes", *STABLE_EVENTS)
DEVELOPMENT_YEARS = (2022, 2023, 2024)
SELECTION_YEARS = (2025,)
CONFIRMATION_YEARS = (2026,)
WEIGHT_GRID = tuple(round(index / 20.0, 2) for index in range(21))
MIN_TARGET_DEVELOPMENT_GAIN = 0.00005
MAX_TARGET_SELECTION_REGRESSION = 0.002
DEFAULT_OUTPUT = OUT.parents[1] / "p3_hillclimb"
MANIFEST_NAME = "manifest.json"
RESULT_FILES = (
    "trials.csv", "raw_metrics.csv", "ranking_metrics.csv",
    "official_ncr_metrics.csv", "official_ncr_team_metrics.csv",
    "ledger.jsonl", "config.json", "summary.json", "REPORT.md",
)


@dataclass(frozen=True)
class TargetEvidence:
    fold: str
    tournament: str
    calendar_year: int
    hemisphere: str
    target: str
    actual: np.ndarray
    empirical: np.ndarray
    v4: np.ndarray
    naive_loss: float


@dataclass(frozen=True)
class FoldBundle:
    fold: str
    tournament: str
    calendar_year: int
    evaluation: pd.DataFrame
    empirical: tuple[RawPrediction, ...]
    baseline: tuple[RawPrediction, ...]


def infer_left_mean(
    blended: np.ndarray, right: np.ndarray, weight_left: float = BASELINE_WEIGHT,
) -> np.ndarray:
    """Recover the v4 mean from the stored blend and empirical mean."""
    if not 0.0 < weight_left <= 1.0:
        raise ValueError("weight_left must be in (0, 1]")
    value = (np.asarray(blended) - (1.0 - weight_left) * np.asarray(right)) / weight_left
    return np.clip(value.astype(float), 0.0, None)


def weighted_mean(
    empirical: np.ndarray, v4: np.ndarray, weight_v4: float,
) -> np.ndarray:
    if not 0.0 <= weight_v4 <= 1.0:
        raise ValueError("weight_v4 must be in [0, 1]")
    return weight_v4 * np.asarray(v4) + (1.0 - weight_v4) * np.asarray(empirical)


def select_coordinate(
    trials: pd.DataFrame, *, current_weight: float = BASELINE_WEIGHT,
    min_development_gain: float = MIN_TARGET_DEVELOPMENT_GAIN,
    max_selection_regression: float = MAX_TARGET_SELECTION_REGRESSION,
) -> dict:
    """Select one coordinate without consulting confirmation results."""
    required = {"weight_v4", "development_score", "selection_score"}
    missing = required - set(trials)
    if missing:
        raise ValueError(f"coordinate trials are missing columns: {sorted(missing)}")
    current = trials[np.isclose(trials["weight_v4"], current_weight)]
    if len(current) != 1:
        raise ValueError("coordinate trials must contain the current weight exactly once")
    ordered = trials.assign(
        distance=(trials["weight_v4"] - current_weight).abs(),
    ).sort_values(["development_score", "distance", "weight_v4"])
    best = ordered.iloc[0]
    current = current.iloc[0]
    development_gain = float(current["development_score"] - best["development_score"])
    selection_delta = float(best["selection_score"] - current["selection_score"])
    accepted = bool(
        not np.isclose(float(best["weight_v4"]), current_weight)
        and development_gain >= min_development_gain
        and selection_delta <= max_selection_regression
    )
    if not accepted and np.isclose(float(best["weight_v4"]), current_weight):
        reason = "incumbent coordinate is development optimum"
    elif not accepted and development_gain < min_development_gain:
        reason = "development gain is below the minimum"
    elif not accepted:
        reason = "selection raw loss regresses beyond the guard"
    else:
        reason = "development winner passes the soft selection guard"
    return {
        "accepted": accepted,
        "selected_weight_v4": float(best["weight_v4"] if accepted else current_weight),
        "best_development_weight_v4": float(best["weight_v4"]),
        "development_gain": development_gain,
        "selection_delta": selection_delta,
        "reason": reason,
        "confirmation_consulted": False,
    }


def _prediction_mean(predictions: Iterable[RawPrediction], target: str) -> np.ndarray:
    if target == "minutes":
        return np.asarray([prediction.minutes.mean for prediction in predictions], dtype=float)
    return np.asarray([prediction.events[target].mean for prediction in predictions], dtype=float)


def _read_predictions(path: Path) -> tuple[RawPrediction, ...]:
    return tuple(
        RawPrediction.from_dict(json.loads(line))
        for line in path.read_text().splitlines() if line
    )


def load_evidence(
    benchmark_dir: Path = OUT,
) -> tuple[list[TargetEvidence], list[FoldBundle]]:
    """Load frozen components without refitting either P3 component."""
    store = pd.read_csv(
        benchmark_dir / "player_match.csv", low_memory=False,
        parse_dates=["date", "match_at"],
    )
    metrics = pd.read_csv(benchmark_dir / "event_metrics.csv")
    naive = metrics[
        metrics["engine"].eq(BASELINE_ENGINE)
        & metrics["cohort"].eq("all")
        & metrics["tier"].isin(["minutes", "stable"])
    ].set_index(["fold", "target"])["naive_loss"].to_dict()
    evidence: list[TargetEvidence] = []
    bundles: list[FoldBundle] = []
    for fold in build_folds(store):
        evaluation = evaluation_frame(store, fold).reset_index(drop=True)
        artifact_path = benchmark_dir / "models" / BASELINE_ENGINE / f"{fold.label}.pkl"
        prediction_path = benchmark_dir / "predictions" / BASELINE_ENGINE / f"{fold.label}.jsonl"
        model = EventBlend50.load(artifact_path)
        if not np.isclose(model.weight_v4, BASELINE_WEIGHT):
            raise ValueError(f"{fold.label}: expected frozen P3 weight 0.5")
        baseline = _read_predictions(prediction_path)
        empirical = tuple(model.empirical.predict_frame(masked_candidates(evaluation)))
        if len(baseline) != len(evaluation) or len(empirical) != len(evaluation):
            raise ValueError(f"{fold.label}: prediction cohort length differs")
        for row, base_prediction, empirical_prediction in zip(
            evaluation.itertuples(index=False), baseline, empirical,
        ):
            expected = (str(row.fixture_id), str(row.player_id), str(row.team))
            actual_base = (
                base_prediction.fixture_id, base_prediction.player_id, base_prediction.team,
            )
            actual_empirical = (
                empirical_prediction.fixture_id, empirical_prediction.player_id,
                empirical_prediction.team,
            )
            if actual_base != expected or actual_empirical != expected:
                raise ValueError(f"{fold.label}: component prediction order changed")
        for target in TARGETS:
            actual = pd.to_numeric(evaluation[target], errors="coerce").to_numpy(float)
            available = evaluation[f"available__{target}"].fillna(False).to_numpy(bool)
            empirical_mean = _prediction_mean(empirical, target)
            baseline_mean = _prediction_mean(baseline, target)
            valid = available & np.isfinite(actual) & np.isfinite(empirical_mean)
            valid &= np.isfinite(baseline_mean)
            if not valid.any():
                raise ValueError(f"{fold.label}/{target}: no valid evaluation rows")
            v4_mean = infer_left_mean(
                baseline_mean[valid], empirical_mean[valid],
            )
            rebuilt_mean = weighted_mean(
                empirical_mean[valid], v4_mean, BASELINE_WEIGHT,
            )
            if not np.allclose(
                rebuilt_mean, baseline_mean[valid], rtol=0.0, atol=1e-10,
            ):
                raise AssertionError(
                    f"{fold.label}/{target}: reconstructed P3 row means differ"
                )
            evidence.append(TargetEvidence(
                fold=fold.label, tournament=fold.tournament,
                calendar_year=fold.calendar_year, hemisphere=fold.hemisphere,
                target=target, actual=actual[valid], empirical=empirical_mean[valid],
                v4=v4_mean,
                naive_loss=float(naive[(fold.label, target)]),
            ))
        bundles.append(FoldBundle(
            fold=fold.label, tournament=fold.tournament,
            calendar_year=fold.calendar_year, evaluation=evaluation,
            empirical=empirical, baseline=baseline,
        ))
        print(f"loaded frozen P3 components: {fold.label}", flush=True)
    return evidence, bundles


def target_score(
    evidence: Iterable[TargetEvidence], target: str, weight_v4: float,
    years: Iterable[int],
) -> float:
    wanted = set(years)
    values = []
    for item in evidence:
        if item.target != target or item.calendar_year not in wanted:
            continue
        predicted = weighted_mean(item.empirical, item.v4, weight_v4)
        values.append(target_loss(target, item.actual, predicted) / item.naive_loss)
    if not values:
        raise ValueError(f"no evidence for {target} in years {sorted(wanted)}")
    return float(np.mean(values))


def global_score(
    evidence: Iterable[TargetEvidence], weights: Mapping[str, float],
    years: Iterable[int],
) -> float:
    wanted = set(years)
    values = []
    for item in evidence:
        if item.calendar_year not in wanted:
            continue
        weight = float(weights.get(item.target, BASELINE_WEIGHT))
        predicted = weighted_mean(item.empirical, item.v4, weight)
        values.append(target_loss(item.target, item.actual, predicted) / item.naive_loss)
    if not values:
        raise ValueError(f"no evidence in years {sorted(wanted)}")
    return float(np.mean(values))


def run_coordinate_search(
    evidence: list[TargetEvidence],
) -> tuple[dict[str, float], pd.DataFrame, list[dict]]:
    trials, ledger, weights = [], [], {}
    for target in TARGETS:
        rows = []
        for weight in WEIGHT_GRID:
            row = {
                "target": target,
                "weight_v4": weight,
                "development_score": target_score(
                    evidence, target, weight, DEVELOPMENT_YEARS,
                ),
                "selection_score": target_score(
                    evidence, target, weight, SELECTION_YEARS,
                ),
            }
            rows.append(row)
            trials.append(row)
        coordinate = pd.DataFrame(rows)
        decision = select_coordinate(coordinate)
        decision["target"] = target
        ledger.append(decision)
        if decision["accepted"]:
            weights[target] = float(decision["selected_weight_v4"])
    return weights, pd.DataFrame(trials), ledger


def raw_metric_table(
    evidence: Iterable[TargetEvidence], weights: Mapping[str, float],
) -> pd.DataFrame:
    rows = []
    for item in evidence:
        baseline = weighted_mean(item.empirical, item.v4, BASELINE_WEIGHT)
        candidate = weighted_mean(
            item.empirical, item.v4, weights.get(item.target, BASELINE_WEIGHT),
        )
        baseline_score = target_loss(item.target, item.actual, baseline) / item.naive_loss
        candidate_score = target_loss(item.target, item.actual, candidate) / item.naive_loss
        split = (
            "development" if item.calendar_year in DEVELOPMENT_YEARS
            else "selection" if item.calendar_year in SELECTION_YEARS
            else "confirmation"
        )
        rows.append({
            "split": split, "fold": item.fold, "tournament": item.tournament,
            "calendar_year": item.calendar_year, "hemisphere": item.hemisphere,
            "target": item.target, "weight_v4": weights.get(item.target, BASELINE_WEIGHT),
            "baseline_score": baseline_score, "candidate_score": candidate_score,
            "delta": candidate_score - baseline_score,
        })
    return pd.DataFrame(rows)


def _replace_mean(distribution: EventDistribution, mean: float) -> EventDistribution:
    if distribution.family == "bernoulli":
        mean = min(max(mean, 0.0), 1.0)
    else:
        mean = max(mean, 0.0)
    return replace(distribution, mean=float(mean))


def reweight_saved_predictions(
    empirical: tuple[RawPrediction, ...], baseline: tuple[RawPrediction, ...],
    weights: Mapping[str, float],
) -> list[RawPrediction]:
    """Reweight stored expected values for deterministic historical diagnostics."""
    if len(empirical) != len(baseline):
        raise ValueError("stored P3 component prediction lengths differ")
    output = []
    for empirical_prediction, baseline_prediction in zip(empirical, baseline):
        empirical_key = (
            empirical_prediction.fixture_id,
            empirical_prediction.player_id,
            empirical_prediction.team,
        )
        baseline_key = (
            baseline_prediction.fixture_id,
            baseline_prediction.player_id,
            baseline_prediction.team,
        )
        if empirical_key != baseline_key:
            raise ValueError("stored P3 component prediction rows are misaligned")
        events = dict(baseline_prediction.events)
        for target, weight in weights.items():
            if target == "minutes" or target not in events:
                continue
            empirical_distribution = empirical_prediction.events[target]
            baseline_distribution = baseline_prediction.events[target]
            v4_mean = infer_left_mean(
                np.asarray([baseline_distribution.mean]),
                np.asarray([empirical_distribution.mean]),
            )[0]
            mean = float(weighted_mean(
                np.asarray([empirical_distribution.mean]), np.asarray([v4_mean]), weight,
            )[0])
            events[target] = _replace_mean(baseline_distribution, mean)
        minutes = baseline_prediction.minutes
        if "minutes" in weights:
            empirical_mean = empirical_prediction.minutes.mean
            v4_mean = infer_left_mean(
                np.asarray([minutes.mean]), np.asarray([empirical_mean]),
            )[0]
            mean = weighted_mean(
                np.asarray([empirical_mean]), np.asarray([v4_mean]), weights["minutes"],
            )[0]
            minutes = _replace_mean(minutes, float(mean))
        output.append(replace(
            baseline_prediction, events=events, minutes=minutes,
            metadata={
                "model": CANDIDATE_ENGINE,
                "weight_v4": BASELINE_WEIGHT,
                "event_weights_v4": dict(sorted(weights.items())),
                "expected_value_diagnostic": True,
            },
        ))
    return output


def ranking_table(
    bundles: list[FoldBundle], weights: Mapping[str, float],
    benchmark_dir: Path = OUT,
) -> pd.DataFrame:
    candidate_frames = []
    for bundle in bundles:
        candidate = reweight_saved_predictions(bundle.empirical, bundle.baseline, weights)
        frame = ranking_metrics(
            bundle.evaluation, candidate, engine=CANDIDATE_ENGINE, fold=bundle.fold,
        ).assign(
            tournament=bundle.tournament, calendar_year=bundle.calendar_year,
        )
        candidate_frames.append(frame)
    baseline = pd.read_csv(benchmark_dir / "ranking_metrics.csv")
    baseline = baseline[baseline["engine"].eq(BASELINE_ENGINE)]
    return pd.concat([baseline, *candidate_frames], ignore_index=True, sort=False)


def _mean_capture(frame: pd.DataFrame) -> pd.Series:
    return frame[[f"top_{n}_capture" for n in TOP_NS]].mean(axis=1, skipna=True)


def ranking_summary(rankings: pd.DataFrame) -> list[dict]:
    stable = rankings[rankings["tier"].eq("stable")].copy()
    stable["mean_capture"] = _mean_capture(stable)
    stable["split"] = np.select(
        [
            stable["calendar_year"].isin(DEVELOPMENT_YEARS),
            stable["calendar_year"].isin(SELECTION_YEARS),
            stable["calendar_year"].isin(CONFIRMATION_YEARS),
        ],
        ["development", "selection", "confirmation"],
        default="other",
    )
    rows = []
    for (split, rubric, engine), group in stable.groupby(
        ["split", "rubric", "engine"], sort=True,
    ):
        rows.append({
            "split": split, "rubric": rubric, "engine": engine,
            "slates": int(group["slate_id"].nunique()),
            "mae": float(group["mae"].mean()),
            "spearman": float(group["spearman"].mean()),
            "mean_capture": float(group["mean_capture"].mean()),
        })
    all_stable = stable[stable["split"].ne("other")]
    for (rubric, engine), group in all_stable.groupby(["rubric", "engine"], sort=True):
        rows.append({
            "split": "all", "rubric": rubric, "engine": engine,
            "slates": int(group["slate_id"].nunique()),
            "mae": float(group["mae"].mean()),
            "spearman": float(group["spearman"].mean()),
            "mean_capture": float(group["mean_capture"].mean()),
        })
    return rows


def official_ncr_proof(
    weights: Mapping[str, float], benchmark_dir: Path = OUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the frozen candidate on official NCR GW1-3 as a soft diagnostic."""
    from model.ncr_eval import load_actuals as load_team_actuals
    from model.ncr_eval import team_points
    from model.ncr_project import optimise

    from ..benchmark_v2 import _group_metrics
    from ..ncr_gw_eval import GW_EXCLUSIONS, expected_ncr_points
    from ..v3.shadow import ncr_candidates
    from .features import build_frozen_feature_frames
    from .folds import strict_training_frame

    data = benchmark_dir.parents[2]
    ncr = data / "ncr"
    corrected_gw3 = data / "unified" / "ncr_gw1_3_eval" / "corrected_gw3_incumbent_projections.csv"
    pieces = []
    projections: dict[int, pd.DataFrame] = {}
    for gw in (1, 2, 3):
        if gw == 3:
            projection = pd.read_csv(corrected_gw3)
            source = ncr_candidates(gw, projection=projection)
        else:
            projection = pd.read_csv(ncr / f"ncr_gw{gw}_projections.csv")
            source = ncr_candidates(gw)
        source = source.copy()
        source["gw"] = gw
        projections[gw] = projection
        pieces.append(source)
    source = pd.concat(pieces, ignore_index=True, sort=False)
    candidates = masked_candidates(source)

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
    base_model = EventBlend50.load(
        benchmark_dir / "models" / BASELINE_ENGINE / f"{fold.label}.pkl"
    )
    baseline_predictions = tuple(base_model.predict_frame(features))
    empirical_predictions = tuple(base_model.empirical.predict_frame(features))
    candidate_model = EventWeightedBlend(
        base_model.empirical, base_model.v4,
        weight_v4=BASELINE_WEIGHT, event_weights_v4=weights,
    )
    candidate_predictions = tuple(candidate_model.predict_frame(features))
    reconstructed_predictions = tuple(reweight_saved_predictions(
        empirical_predictions, baseline_predictions, weights,
    ))
    for direct, reconstructed in zip(candidate_predictions, reconstructed_predictions):
        if not np.isclose(
            direct.minutes.mean, reconstructed.minutes.mean, rtol=0.0, atol=1e-10,
        ):
            raise AssertionError("direct weighted P3 minutes mean differs from reconstruction")
        for event in direct.events:
            if not np.isclose(
                direct.events[event].mean, reconstructed.events[event].mean,
                rtol=0.0, atol=1e-10,
            ):
                raise AssertionError(
                    f"direct weighted P3 {event} mean differs from reconstruction"
                )

    def prediction_rows(engine: str, predictions: tuple[RawPrediction, ...]) -> pd.DataFrame:
        rows = []
        for row, prediction in zip(candidates.itertuples(index=False), predictions):
            if str(row.player_id) != prediction.player_id:
                raise ValueError(f"{engine}: NCR proof prediction order changed")
            rows.append({
                "gw": int(row.gw), "fantasy_id": int(row.fantasy_id),
                "player_name": prediction.player_name, "team": prediction.team,
                "status": "P" if bool(row.started) else "B", "engine": engine,
                "predicted_points": expected_ncr_points(prediction),
                "expected_minutes": prediction.minutes.mean,
            })
        return pd.DataFrame(rows)

    predicted = pd.concat([
        prediction_rows(BASELINE_ENGINE, baseline_predictions),
        prediction_rows(CANDIDATE_ENGINE, candidate_predictions),
    ], ignore_index=True)
    labels = []
    for gw in (1, 2, 3):
        actuals = load_team_actuals(ncr / "feeds" / f"players_gw{gw}.json")
        labels.extend(
            {"gw": gw, "fantasy_id": int(player_id), "official_points": float(values[0])}
            for player_id, values in actuals.items()
        )
    labels = pd.DataFrame(labels).drop_duplicates(["gw", "fantasy_id"])
    predicted = predicted.merge(
        labels, on=["gw", "fantasy_id"], how="left", validate="many_to_one",
    )
    if predicted["official_points"].isna().any():
        raise ValueError("official NCR proof is missing fantasy labels")

    metric_rows = []
    for engine in (BASELINE_ENGINE, CANDIDATE_ENGINE):
        for gw in (1, 2, 3):
            block = predicted[
                predicted["engine"].eq(engine) & predicted["gw"].eq(gw)
            ]
            values = _group_metrics(
                block, "predicted_points", actual_col="official_points",
            )
            metric_rows.append({"scope": f"GW{gw}", "gw": gw, "engine": engine, **values})
    metrics = pd.DataFrame(metric_rows)
    for engine in (BASELINE_ENGINE, CANDIDATE_ENGINE):
        block = metrics[metrics["engine"].eq(engine)]
        values = {
            column: float(block[column].mean())
            for column in block.select_dtypes("number")
            if column not in {"gw", "n"}
        }
        metric_rows.append({
            "scope": "GW1-3 mean", "gw": np.nan, "engine": engine,
            "n": int(block["n"].sum()), **values,
        })
    metrics = pd.DataFrame(metric_rows)
    metrics["mean_capture"] = metrics[
        [f"top_{n}_capture" for n in TOP_NS]
    ].mean(axis=1, skipna=True)

    team_rows = []
    for gw in (1, 2, 3):
        projection = projections[gw]
        actuals = load_team_actuals(ncr / "feeds" / f"players_gw{gw}.json")
        for engine in (BASELINE_ENGINE, CANDIDATE_ENGINE):
            engine_predictions = predicted[
                predicted["engine"].eq(engine) & predicted["gw"].eq(gw)
            ][["fantasy_id", "predicted_points", "expected_minutes"]]
            pool = projection.merge(
                engine_predictions, left_on="id", right_on="fantasy_id",
                validate="one_to_one",
            )
            if GW_EXCLUSIONS[gw]:
                pool = pool[~pool["team"].isin(GW_EXCLUSIONS[gw])].copy()
            pool["starter_exp"] = pool["predicted_points"]
            pool["supersub_exp"] = 3.0 * pool["predicted_points"]
            pool["exp_min"] = pool["expected_minutes"]
            squad, _, _ = optimise(pool)
            team_rows.append({
                "gw": gw, "engine": engine,
                "team_points": float(team_points(squad, actuals)),
                "captain": str(squad.loc[squad["is_capt"], "name"].iloc[0]),
                "super_sub": str(squad.loc[squad["is_sub"], "name"].iloc[0]),
            })
    incumbent_path = data / "unified" / "ncr_gw1_3_eval" / "team_metrics.csv"
    incumbent = pd.read_csv(incumbent_path)
    team_rows.extend({
        "gw": int(row.gw), "engine": "ncr_incumbent",
        "team_points": float(row.incumbent_points),
        "captain": str(row.incumbent_captain),
        "super_sub": str(row.incumbent_super_sub),
    } for row in incumbent.itertuples(index=False))
    teams = pd.DataFrame(team_rows)
    totals = teams.groupby("engine", as_index=False)["team_points"].sum()
    totals["gw"] = 0
    totals["captain"] = ""
    totals["super_sub"] = ""
    teams = pd.concat([teams, totals[teams.columns]], ignore_index=True)
    return metrics, teams


def _split_raw_summary(raw: pd.DataFrame) -> list[dict]:
    def bootstrap(group: pd.DataFrame) -> tuple[float, float]:
        fold_delta = group.groupby("fold")["delta"].mean().to_numpy(float)
        rng = np.random.default_rng(17)
        samples = np.empty(2000)
        for index in range(len(samples)):
            chosen = rng.integers(0, len(fold_delta), len(fold_delta))
            samples[index] = float(fold_delta[chosen].mean())
        return float(np.quantile(samples, 0.05)), float(np.quantile(samples, 0.95))

    rows = []
    for split, group in raw.groupby("split", sort=False):
        p05, p95 = bootstrap(group)
        rows.append({
            "split": split,
            "folds": int(group["fold"].nunique()),
            "baseline_score": float(group["baseline_score"].mean()),
            "candidate_score": float(group["candidate_score"].mean()),
            "delta": float(group["candidate_score"].mean() - group["baseline_score"].mean()),
            "delta_p05": p05, "delta_p95": p95,
        })
    p05, p95 = bootstrap(raw)
    rows.append({
        "split": "all", "folds": int(raw["fold"].nunique()),
        "baseline_score": float(raw["baseline_score"].mean()),
        "candidate_score": float(raw["candidate_score"].mean()),
        "delta": float(raw["candidate_score"].mean() - raw["baseline_score"].mean()),
        "delta_p05": p05, "delta_p95": p95,
    })
    return rows


def _markdown(frame: pd.DataFrame, digits: int = 6) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float", "float64"]).columns:
        shown[column] = shown[column].map(lambda value: f"{float(value):.{digits}f}")
    headers = [str(column) for column in shown]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _json_records(frame: pd.DataFrame) -> list[dict]:
    """Return strict JSON records with missing numeric values represented by null."""
    return json.loads(frame.to_json(orient="records", double_precision=15))


ACCEPTED_COLUMNS = (
    "target", "selected_weight_v4", "development_gain", "selection_delta",
)


def accepted_coordinate_table(ledger: Iterable[dict]) -> pd.DataFrame:
    """Return stable report columns, including when all coordinates are rejected."""
    rows = [row for row in ledger if row["accepted"]]
    return pd.DataFrame(rows).reindex(columns=ACCEPTED_COLUMNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_key(path: Path, repository: Path) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_run_manifest(
    benchmark_dir: Path, *, include_official_ncr: bool,
) -> dict:
    """Fingerprint fitted inputs, evaluation data, and implementation code."""
    repository = Path(__file__).resolve().parents[3]
    inputs = [
        benchmark_dir / "player_match.csv",
        benchmark_dir / "event_metrics.csv",
        benchmark_dir / "ranking_metrics.csv",
        *sorted((benchmark_dir / "models" / BASELINE_ENGINE).glob("*.pkl")),
        *sorted((benchmark_dir / "predictions" / BASELINE_ENGINE).glob("*.jsonl")),
        *sorted((benchmark_dir / "manifests" / BASELINE_ENGINE).glob("*.json")),
    ]
    if include_official_ncr:
        data = benchmark_dir.parents[2]
        inputs.extend([
            data / "ncr" / "ncr_gw1_projections.csv",
            data / "ncr" / "ncr_gw2_projections.csv",
            data / "ncr" / "ncr_player_crosswalk.csv",
            data / "ncr" / "ncr_fixtures.csv",
            *(data / "ncr" / "feeds" / f"players_gw{gw}.json" for gw in (1, 2, 3)),
            data / "unified" / "ncr_gw1_3_eval"
            / "corrected_gw3_incumbent_projections.csv",
            data / "unified" / "ncr_gw1_3_eval" / "team_metrics.csv",
        ])
    implementation = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("blend.py"),
        Path(__file__).resolve().with_name("config.py"),
        Path(__file__).resolve().with_name("empirical.py"),
        Path(__file__).resolve().with_name("features.py"),
        Path(__file__).resolve().with_name("folds.py"),
        Path(__file__).resolve().with_name("metrics.py"),
        repository / "model" / "unified" / "contracts.py",
        repository / "model" / "unified" / "schema.py",
        repository / "model" / "unified" / "scoring.py",
        repository / "model" / "rp_rates.py",
    ]
    if include_official_ncr:
        implementation.extend([
            repository / "model" / "ncr_eval.py",
            repository / "model" / "ncr_project.py",
            repository / "model" / "unified" / "benchmark_v2.py",
            repository / "model" / "unified" / "features.py",
            repository / "model" / "unified" / "ncr_gw_eval.py",
            repository / "model" / "unified" / "v3" / "shadow.py",
            repository / "model" / "unified" / "v4" / "features.py",
            repository / "model" / "unified" / "v4" / "gbdt.py",
        ])
    missing = [str(path) for path in (*inputs, *implementation) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"hill-climb manifest inputs are missing: {missing}")
    return {
        "schema_version": 1,
        "include_official_ncr": include_official_ncr,
        "input_sha256": {
            _manifest_key(path, repository): _sha256(path) for path in inputs
        },
        "implementation_sha256": {
            _manifest_key(path, repository): _sha256(path) for path in implementation
        },
        "runtime": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in (
                    "lightgbm", "numpy", "pandas", "scikit-learn", "scipy",
                )
            },
        },
    }


def validate_existing_manifest(output_dir: Path, manifest: Mapping) -> None:
    """Reject an overwrite when an existing run used different sources."""
    path = output_dir / MANIFEST_NAME
    if not path.exists():
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"{output_dir} contains unmanifested results; use a new --output"
            )
        return
    existing = json.loads(path.read_text())
    output_hashes = existing.pop("output_sha256", {})
    if existing != dict(manifest):
        raise FileExistsError(
            f"{output_dir} contains results from different sources; use a new --output"
        )
    if set(output_hashes) != set(RESULT_FILES):
        raise FileExistsError(f"{output_dir} has incomplete manifested results")
    expected_names = {*RESULT_FILES, MANIFEST_NAME}
    if {item.name for item in output_dir.iterdir()} != expected_names:
        raise FileExistsError(f"{output_dir} has unexpected or missing result files")
    for name, expected in output_hashes.items():
        result_path = output_dir / name
        if Path(name).name != name or not result_path.is_file():
            raise FileExistsError(f"{output_dir} has incomplete manifested results")
        if _sha256(result_path) != expected:
            raise FileExistsError(f"{result_path} does not match the result manifest")


def add_output_hashes(manifest: Mapping, output_dir: Path) -> dict:
    """Add hashes for every generated result except the manifest itself."""
    completed = dict(manifest)
    completed["output_sha256"] = {
        name: _sha256(output_dir / name) for name in RESULT_FILES
    }
    return completed


def render_report(
    raw: pd.DataFrame, ledger: list[dict], rankings: list[dict],
    weights: Mapping[str, float], official_metrics: pd.DataFrame,
    official_teams: pd.DataFrame,
) -> str:
    raw_summary = pd.DataFrame(_split_raw_summary(raw))
    accepted = accepted_coordinate_table(ledger)
    ranking_frame = pd.DataFrame(rankings)
    all_rank = ranking_frame[ranking_frame["split"].eq("all")].copy()
    all_rank["model"] = all_rank["engine"].map({
        BASELINE_ENGINE: "P3 50/50", CANDIDATE_ENGINE: "weighted P3",
    })
    all_rank = all_rank[["rubric", "model", "mae", "spearman", "mean_capture", "slates"]]
    folds = raw.groupby(["split", "fold"], as_index=False)[
        ["baseline_score", "candidate_score"]
    ].mean()
    folds["delta"] = folds["candidate_score"] - folds["baseline_score"]
    confirmation = raw_summary[raw_summary["split"].eq("confirmation")].iloc[0]
    all_raw = raw_summary[raw_summary["split"].eq("all")].iloc[0]
    all_improvement = 100.0 * (
        all_raw["baseline_score"] - all_raw["candidate_score"]
    ) / all_raw["baseline_score"]
    official_mean = official_metrics[official_metrics["scope"].eq("GW1-3 mean")][
        ["engine", "mae", "spearman", "mean_capture"]
    ].copy()
    official_mean["model"] = official_mean["engine"].map({
        BASELINE_ENGINE: "P3 50/50", CANDIDATE_ENGINE: "weighted P3",
    })
    official_mean = official_mean[["model", "mae", "spearman", "mean_capture"]]
    team_totals = official_teams[official_teams["gw"].eq(0)][
        ["engine", "team_points"]
    ].copy()
    team_totals["model"] = team_totals["engine"].map({
        BASELINE_ENGINE: "P3 50/50", CANDIDATE_ENGINE: "weighted P3",
        "ncr_incumbent": "NCR incumbent",
    })
    team_totals = team_totals[["model", "team_points"]]
    verdict = (
        "The retrospective 2026 confirmation also improves."
        if confirmation["delta"] < 0
        else "The retrospective 2026 confirmation regresses."
    )
    lines = [
        "# P3 local raw-event hill climb", "",
        "The search optimises raw rugby events. Tournament fantasy metrics are secondary diagnostics.",
        "The search uses 2022-2024 development folds and 2025 selection folds.",
        "The selector does not use 2026. The repository already contained the 2026 results.", "",
        "## Result", "",
        f"The search accepts {len(accepted)} event coordinates. {verdict}",
        f"The all-fold raw score improves by {all_improvement:.2f}% against P3 50/50.",
        "This result defines a research candidate. It does not change the active P3 shadow artifact.", "",
        "## Raw-event objective", "", _markdown(raw_summary), "",
        "Lower scores are better. Each score averages training-naive-relative loss across folds and stable event heads.",
        "The final two columns give the paired 90% fold-bootstrap interval for the score delta.", "",
        "## Accepted coordinates", "", _markdown(accepted), "",
        "All unspecified event weights remain at 0.5. Extended events remain at 0.5 because their support is limited.", "",
        "## Fantasy-adapter diagnostics", "", _markdown(all_rank), "",
        "Fantasy metrics do not select or reject the candidate. They show intended-application behaviour.", "",
        "## Official NCR GW1-3 soft diagnostic", "", _markdown(official_mean), "",
        _markdown(team_totals), "",
        "The official NCR diagnostic runs only after the event weights are frozen.",
        "It is retrospective evidence and does not alter the selected weights.", "",
        "## Interpretation limits", "",
        "The search adds one competition-independent weight for each accepted raw event head.",
        "The 2025 results guide every coordinate and are not independent validation.",
        "The 2026 blocks do not enter selection, but they are existing repository evidence.",
        "Historical fantasy diagnostics reweight expected means and retain baseline dispersion.",
        "The official NCR diagnostic exercises the deployable weighted distribution blend.",
        "Official NCR GW1-3 is retrospective evidence. Future NCR GW4-7 and a new raw tournament remain prospective tests.", "",
        "## Fold evidence", "", _markdown(folds), "",
        "## Reproduce", "", "```bash",
        "/tmp/6n-model-pinned/bin/python -m model.unified.cli raw-benchmark p3-hillclimb",
        "```", "",
        "The command reads frozen cache-derived data and fitted P3 artifacts. It makes no network request.",
    ]
    return "\n".join(lines) + "\n"


def run_hillclimb(
    *, benchmark_dir: Path = OUT, output_dir: Path = DEFAULT_OUTPUT,
    include_official_ncr: bool = True,
) -> dict:
    manifest = build_run_manifest(
        benchmark_dir, include_official_ncr=include_official_ncr,
    )
    validate_existing_manifest(output_dir, manifest)
    evidence, bundles = load_evidence(benchmark_dir)
    baseline_rebuilt = global_score(
        evidence, {}, (*DEVELOPMENT_YEARS, *SELECTION_YEARS, *CONFIRMATION_YEARS),
    )
    frozen_metrics = pd.read_csv(benchmark_dir / "event_metrics.csv")
    expected = frozen_metrics[
        frozen_metrics["engine"].eq(BASELINE_ENGINE)
        & frozen_metrics["cohort"].eq("all")
        & frozen_metrics["tier"].isin(["minutes", "stable"])
    ]["relative_loss"].mean()
    if not np.isclose(baseline_rebuilt, expected, rtol=0.0, atol=1e-10):
        raise AssertionError(
            f"rebuilt P3 score {baseline_rebuilt} differs from frozen score {expected}"
        )
    weights, trials, ledger = run_coordinate_search(evidence)
    raw = raw_metric_table(evidence, weights)
    rankings = ranking_table(bundles, weights, benchmark_dir)
    ranking_rows = ranking_summary(rankings)
    if include_official_ncr:
        official_metrics, official_teams = official_ncr_proof(weights, benchmark_dir)
    else:
        official_metrics = pd.DataFrame(columns=[
            "scope", "engine", "mae", "spearman", "mean_capture",
        ])
        official_teams = pd.DataFrame(columns=["gw", "engine", "team_points"])
    raw_rows = _split_raw_summary(raw)
    summary = {
        "schema_version": 1,
        "candidate": CANDIDATE_ENGINE,
        "baseline": BASELINE_ENGINE,
        "objective": "fold-and-stable-target balanced relative raw-event loss",
        "development_years": list(DEVELOPMENT_YEARS),
        "selection_years": list(SELECTION_YEARS),
        "confirmation_years": list(CONFIRMATION_YEARS),
        "confirmation_used_for_search": False,
        "confirmation_status": "retrospective existing evidence",
        "accepted_coordinates": len(weights),
        "event_weights_v4": dict(sorted(weights.items())),
        "raw": raw_rows,
        "fantasy_diagnostics": ranking_rows,
        "official_ncr_used_for_search": False,
        "official_ncr_metrics": _json_records(official_metrics),
        "official_ncr_team_metrics": _json_records(official_teams),
    }
    config = {
        "schema_version": 1,
        "model": CANDIDATE_ENGINE,
        "base_model": BASELINE_ENGINE,
        "default_weight_v4": BASELINE_WEIGHT,
        "event_weights_v4": dict(sorted(weights.items())),
        "extended_event_policy": "retain default 0.5 weight",
        "selection": {
            "minimum_target_development_gain": MIN_TARGET_DEVELOPMENT_GAIN,
            "maximum_target_selection_regression": MAX_TARGET_SELECTION_REGRESSION,
            "weight_grid": list(WEIGHT_GRID),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output_dir / "trials.csv", index=False)
    raw.to_csv(output_dir / "raw_metrics.csv", index=False)
    rankings.to_csv(output_dir / "ranking_metrics.csv", index=False)
    official_metrics.to_csv(output_dir / "official_ncr_metrics.csv", index=False)
    official_teams.to_csv(output_dir / "official_ncr_team_metrics.csv", index=False)
    (output_dir / "ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger)
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "REPORT.md").write_text(
        render_report(
            raw, ledger, ranking_rows, weights, official_metrics, official_teams,
        )
    )
    completed_manifest = add_output_hashes(manifest, output_dir)
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(completed_manifest, indent=2, sort_keys=True) + "\n"
    )
    return summary
