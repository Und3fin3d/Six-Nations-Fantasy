"""Balanced raw-event and reconstructed-rubric benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contracts import RawPrediction
from ..evaluation import tie_aware_top_n
from ..schema import distribution_family
from ..scoring import NationsChampionshipScorer, SixNationsScorer
from .config import EXTENDED_EVENTS, STABLE_EVENTS, TOP_NS


def _poisson_deviance(actual: np.ndarray, predicted: np.ndarray) -> float:
    predicted = np.clip(predicted, 1e-8, None)
    term = np.where(
        actual > 0,
        actual * np.log(np.clip(actual, 1e-8, None) / predicted),
        0.0,
    )
    return float(np.mean(2.0 * (term - (actual - predicted))))


def target_loss(target: str, actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 0, None)
    if target == "minutes":
        return float(np.mean(np.abs(actual - predicted)))
    family = distribution_family(target)
    if family == "bernoulli":
        probability = np.clip(predicted, 1e-7, 1 - 1e-7)
        binary = (actual > 0).astype(float)
        return float(np.mean(-(binary * np.log(probability) + (1 - binary) * np.log(1 - probability))))
    if family == "lognormal":
        return float(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2))
    return _poisson_deviance(actual, predicted)


@dataclass
class NaiveComparator:
    """Training-only position/status/level target means with safe fallbacks."""

    group_means: dict[str, pd.Series] = field(default_factory=dict)
    position_means: dict[str, pd.Series] = field(default_factory=dict)
    global_means: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, train: pd.DataFrame, targets: tuple[str, ...]) -> "NaiveComparator":
        model = cls()
        for target in targets:
            mask = train.get(f"available__{target}", pd.Series(False, index=train.index))
            valid = mask.fillna(False).astype(bool) & pd.to_numeric(train[target], errors="coerce").notna()
            observed = train[valid].copy()
            observed[target] = pd.to_numeric(observed[target], errors="coerce")
            if observed.empty:
                model.group_means[target] = pd.Series(dtype=float)
                model.position_means[target] = pd.Series(dtype=float)
                model.global_means[target] = 0.0
                continue
            keys = ["position", "started", "competition_level"]
            model.group_means[target] = observed.groupby(keys, dropna=False)[target].mean()
            model.position_means[target] = observed.groupby("position", dropna=False)[target].mean()
            model.global_means[target] = float(observed[target].mean())
        return model

    def predict(self, rows: pd.DataFrame, target: str) -> np.ndarray:
        grouped = self.group_means[target]
        position = self.position_means[target]
        keys = pd.MultiIndex.from_frame(
            rows[["position", "started", "competition_level"]],
            names=["position", "started", "competition_level"],
        )
        values = grouped.reindex(keys).to_numpy(dtype=float)
        missing = ~np.isfinite(values)
        if missing.any():
            fallback = rows["position"].map(position).to_numpy(dtype=float)
            values[missing] = fallback[missing]
        values[~np.isfinite(values)] = self.global_means[target]
        return values


def prediction_means(predictions: list[RawPrediction], target: str) -> np.ndarray:
    if target == "minutes":
        return np.asarray([prediction.minutes.mean for prediction in predictions], dtype=float)
    return np.asarray([
        prediction.events[target].mean if target in prediction.events else np.nan
        for prediction in predictions
    ], dtype=float)


def _calibration(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    if len(actual) < 2 or float(np.var(predicted)) < 1e-12:
        return float(np.mean(actual)) if len(actual) else np.nan, np.nan
    slope, intercept = np.polyfit(predicted, actual, 1)
    return float(intercept), float(slope)


def _cohorts(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    minutes = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0).to_numpy(float)
    started = frame["started"].fillna(False).astype(bool).to_numpy()
    career = pd.to_numeric(frame.get("career_matches", 0), errors="coerce").fillna(0).to_numpy(float)
    cohorts = {
        "all": np.ones(len(frame), dtype=bool),
        "starter": started,
        "bench": ~started,
        "zero_minute": minutes <= 0,
        "low_history": career < 5,
        "north": frame["hemisphere"].eq("north").to_numpy(),
        "south": frame["hemisphere"].eq("south").to_numpy(),
    }
    for position in sorted(frame["position"].dropna().astype(str).unique()):
        cohorts[f"position:{position}"] = frame["position"].astype(str).eq(position).to_numpy()
    return cohorts


def event_metrics(
    frame: pd.DataFrame, predictions: list[RawPrediction], naive: NaiveComparator,
    *, engine: str, fold: str,
) -> pd.DataFrame:
    rows = []
    cohorts = _cohorts(frame)
    for target in ("minutes", *STABLE_EVENTS, *EXTENDED_EVENTS):
        available = frame.get(f"available__{target}", pd.Series(False, index=frame.index))
        actual_all = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
        predicted_all = prediction_means(predictions, target)
        naive_all = naive.predict(frame, target)
        base_valid = available.fillna(False).astype(bool).to_numpy() & np.isfinite(actual_all)
        base_valid &= np.isfinite(predicted_all) & np.isfinite(naive_all)
        for cohort, cohort_mask in cohorts.items():
            valid = base_valid & cohort_mask
            if not valid.any():
                continue
            actual, predicted, naive_pred = actual_all[valid], predicted_all[valid], naive_all[valid]
            loss = target_loss(target, actual, predicted)
            naive_loss = target_loss(target, actual, naive_pred)
            intercept, slope = _calibration(actual, predicted)
            selected = frame.loc[valid, ["fixture_id", "team"]].copy()
            selected["actual"] = actual
            selected["predicted"] = predicted
            totals = selected.groupby(["fixture_id", "team"])[["actual", "predicted"]].sum()
            rows.append({
                "engine": engine, "fold": fold, "target": target,
                "tier": "minutes" if target == "minutes" else (
                    "stable" if target in STABLE_EVENTS else "extended"
                ),
                "cohort": cohort, "n": int(valid.sum()),
                "fixtures": int(frame.loc[valid, "fixture_id"].nunique()),
                "mae": float(np.mean(np.abs(actual - predicted))),
                "loss": loss, "naive_loss": naive_loss,
                "relative_loss": loss / max(naive_loss, 1e-12),
                "calibration_intercept": intercept, "calibration_slope": slope,
                "actual_positive_rate": float(np.mean(actual > 0)),
                "predicted_positive_rate": float(np.mean(predicted > 1e-9)),
                "actual_mean": float(np.mean(actual)), "predicted_mean": float(np.mean(predicted)),
                "team_total_mae": float(np.mean(np.abs(totals["actual"] - totals["predicted"]))),
            })
    return pd.DataFrame(rows)


def _score_rows(
    frame: pd.DataFrame, predictions: list[RawPrediction], scorer,
    allowed_events: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allowed = tuple(event for event in allowed_events if event in frame)
    valid = np.ones(len(frame), dtype=bool)
    for event in allowed:
        valid &= frame[f"available__{event}"].fillna(False).astype(bool).to_numpy()
        valid &= np.isfinite(prediction_means(predictions, event))
    actual, predicted = [], []
    indices = np.flatnonzero(valid)
    for index in indices:
        actual_events = {
            event: np.asarray([float(frame.iloc[index][event])]) for event in allowed
        }
        predicted_events = {
            event: np.asarray([float(predictions[index].events[event].mean)]) for event in allowed
        }
        is_forward = bool(frame.iloc[index]["is_forward"])
        actual.append(float(scorer.score_samples(actual_events, is_forward=is_forward)[0]))
        predicted.append(float(scorer.score_samples(predicted_events, is_forward=is_forward)[0]))
    return indices, np.asarray(actual), np.asarray(predicted)


def _scorer_events(scorer, events: tuple[str, ...]) -> tuple[str, ...]:
    # Tries and forward scrum wins are handled outside the flat weight mapping.
    # Metres are a Six Nations-only scoring term and must not leak into the NCR
    # observable-event label, even though supplying it to the NCR adapter would
    # happen to leave the numerical score unchanged.
    scored = set(scorer.weights) | {"tries", "scrums_won"}
    if isinstance(scorer, SixNationsScorer):
        scored.add("metres")
    return tuple(event for event in events if event in scored)


def ranking_metrics(
    frame: pd.DataFrame, predictions: list[RawPrediction], *, engine: str, fold: str,
) -> pd.DataFrame:
    rows = []
    for rubric, scorer in (
        ("six_nations", SixNationsScorer()), ("ncr", NationsChampionshipScorer()),
    ):
        stable = _scorer_events(scorer, STABLE_EVENTS)
        extended_candidates = _scorer_events(scorer, EXTENDED_EVENTS)
        for slate_id, slate in frame.groupby("slate_id", sort=True):
            local_indices = slate.index.to_numpy()
            local_predictions = [predictions[int(index)] for index in local_indices]
            local = slate.reset_index(drop=True)
            for tier, allowed in (("stable", stable), ("extended_observable", stable + extended_candidates)):
                indices, actual, predicted = _score_rows(local, local_predictions, scorer, allowed)
                if tier == "extended_observable":
                    complete_extended = all(
                        local[f"available__{event}"].fillna(False).astype(bool).all()
                        for event in extended_candidates
                    )
                    if not extended_candidates or not complete_extended:
                        continue
                if len(actual) < min(TOP_NS):
                    continue
                actual_s, predicted_s = pd.Series(actual), pd.Series(predicted)
                if float(predicted_s.std(ddof=0)) < 1e-12 or float(actual_s.std(ddof=0)) < 1e-12:
                    spearman = np.nan
                else:
                    spearman = float(predicted_s.corr(actual_s, method="spearman"))
                row = {
                    "engine": engine, "fold": fold, "slate_id": str(slate_id),
                    "rubric": rubric, "tier": tier, "n": int(len(actual)),
                    "mae": float(np.mean(np.abs(actual - predicted))),
                    "spearman": spearman,
                    "allowed_events": ",".join(allowed),
                }
                for n in TOP_NS:
                    if len(actual) < n:
                        continue
                    overlap, capture = tie_aware_top_n(predicted_s, actual_s, n)
                    row[f"top_{n}_overlap"] = overlap
                    row[f"top_{n}_capture"] = capture
                rows.append(row)
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    rows: pd.DataFrame, candidate: str, baseline: str, metric: str,
    *, cluster: str, n_boot: int = 2000, seed: int = 17,
) -> dict[str, float]:
    pivot = rows.pivot_table(index=cluster, columns="engine", values=metric, aggfunc="mean")
    pivot = pivot.dropna(subset=[candidate, baseline])
    if pivot.empty:
        return {"n_clusters": 0, "mean": np.nan, "p05": np.nan, "p95": np.nan}
    differences = pivot[candidate].to_numpy(float) - pivot[baseline].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        index = rng.integers(0, len(differences), len(differences))
        samples[i] = float(np.mean(differences[index]))
    return {
        "n_clusters": int(len(differences)), "mean": float(np.mean(differences)),
        "p05": float(np.quantile(samples, 0.05)), "p95": float(np.quantile(samples, 0.95)),
    }
