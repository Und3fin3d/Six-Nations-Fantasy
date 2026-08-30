"""All-rugby raw-prediction path for the Six Nations champion components.

Research module for the all-rugby raw-event objective (minimise the
competition-balanced stable raw score of the Historical Raw Benchmark v1).

DEPLOYMENT SAFETY
-----------------
This module is default-off and is never imported by the deployed Six Nations
path.  ``model/research.py``, ``model/assemble.py``, ``model/baselines.py`` and
``research/promoted_config.json`` are untouched.  Everything here is a *raw
prediction path*: it post-processes the champion's already-fitted component
heads, or adds raw-only heads for stable targets the champion never modelled.
The champion's selection behaviour (``selector_pts_hat``, ``target_pts_hat``,
the XV overlays) is not read and not changed.

WHY A SEPARATE RAW PATH
-----------------------
The champion's component layer applies a deliberate volume deflation.  The
project's Experiment 18 established that the deflation is load-bearing implicit
variance regularisation for quota-constrained XV *selection*: it helps pick a
team and hurts point-accurate raw prediction.  Under the all-rugby raw-event
objective that trade flips, so this path can undo the deflation while the
selection path keeps it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from model.baselines import SCORED
from model.data import load as load_six_nations_store
from model.research import PROMOTED_CONFIG, _predict_config, config_from_dict

from ..contracts import EventDistribution, RawPrediction
from ..schema import distribution_family
from .config import EXTENDED_EVENTS, SEED, STABLE_EVENTS
from .folds import build_folds, evaluation_frame, strict_training_frame
from .metrics import NaiveComparator, event_metrics, ranking_metrics

ENGINE = "six_nations_champion_allrugby"
BASELINE_ENGINE = "six_nations_champion_components"

#: Seasons of the Six Nations native store that carry champion-predictable rows.
NATIVE_SEASONS = (2023, 2024, 2025, 2026)
#: Folds the champion can be evaluated on at all (its feature store is 6N-only).
EVAL_SEASONS = (2025, 2026)

#: Raw event heads the champion actually has (13 SCORED components).
RAW_EVENTS = tuple(event for event in SCORED if event in STABLE_EVENTS + EXTENDED_EVENTS)
#: The 12 stable events among them, plus minutes -> the frozen 13-target
#: like-for-like comparison set used by ``champion_partial_comparison.csv``.
SHARED_STABLE_TARGETS = ("minutes", *(e for e in RAW_EVENTS if e in STABLE_EVENTS))
#: Stable events the champion has no head for at all.
MISSING_STABLE_EVENTS = tuple(e for e in STABLE_EVENTS if e not in RAW_EVENTS)


@dataclass(frozen=True)
class RawPathConfig:
    """Knobs for the raw-prediction path.  All default to the frozen champion."""

    name: str = "baseline"
    #: none | global | position -- undo the component-layer volume deflation by
    #: matching the predicted level to the strictly-prior observed level.
    level_calibration: str = "none"
    #: Events to calibrate.  None means every champion head.
    calibrate_events: tuple[str, ...] | None = None
    #: Pseudo-count added to both sides of the ratio so ultra-sparse events
    #: shrink toward 1.0 instead of exploding.
    calibration_shrinkage: float = 1.0
    #: Hard clip on the fitted multiplicative factor.
    calibration_clip: tuple[float, float] = (0.1, 10.0)
    #: none | rate_minutes -- revive champion heads that emit identically zero.
    #: The champion's drop_goals_converted and red_cards heads are degenerate:
    #: predicting a flat zero is optimal for quota-constrained XV selection (it
    #: never changes a pick) but is catastrophic under Poisson deviance and
    #: log-loss, where a zero against a real occurrence costs ~35 and ~16 nats.
    dead_head_fallback: str = "none"
    #: Restrict revival to these events.  None means every detected dead head.
    revive_events: tuple[str, ...] | None = None
    #: none | rate_minutes -- add raw-only heads for MISSING_STABLE_EVENTS.
    extra_heads: str = "none"
    #: Shrinkage pseudo-minutes for the extra-head player rate.
    extra_head_shrinkage: float = 240.0
    #: 0.0 = pure champion minutes_hat, 1.0 = strictly-prior stratum mean minutes.
    minutes_blend: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level_calibration": self.level_calibration,
            "calibrate_events": list(self.calibrate_events) if self.calibrate_events else None,
            "calibration_shrinkage": self.calibration_shrinkage,
            "calibration_clip": list(self.calibration_clip),
            "dead_head_fallback": self.dead_head_fallback,
            "revive_events": list(self.revive_events) if self.revive_events else None,
            "extra_heads": self.extra_heads,
            "extra_head_shrinkage": self.extra_head_shrinkage,
            "minutes_blend": self.minutes_blend,
        }


# ---------------------------------------------------------------------------
# champion component predictions (cached -- each call refits the component stack)
# ---------------------------------------------------------------------------

def champion_component_predictions(
    native: pd.DataFrame, config, season: int, cache_dir: Path,
) -> pd.DataFrame:
    """Champion ``hat_*`` / ``minutes_hat`` for one native season, cached to disk.

    Training is the champion's own forward-chained split (``season < season``),
    so every cached frame is a strictly point-in-time out-of-sample prediction.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"champion_hats_{season}.csv"
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    predicted, _, _ = _predict_config(native, config, season)
    frame = predicted.copy()
    frame["fixture_id"] = frame["fixture_id"].astype(str)
    frame["player_id"] = frame["player_id"].astype(str)
    keep = ["fixture_id", "player_id", "season", "minutes_hat"] + [
        f"hat_{event}" for event in RAW_EVENTS
    ]
    frame = frame[[column for column in keep if column in frame.columns]]
    frame.to_csv(path, index=False)
    return frame


# ---------------------------------------------------------------------------
# level calibration -- the deflation-removal candidate
# ---------------------------------------------------------------------------

def fit_level_calibration(
    benchmark: pd.DataFrame, hats: pd.DataFrame, events: tuple[str, ...],
    *, shrinkage: float, clip: tuple[float, float], scope: str,
) -> dict[str, dict[str, float]]:
    """Multiplicative factors matching predicted level to observed level.

    ``hats`` must contain ONLY strictly-prior seasons.  ``benchmark`` supplies
    the actuals in exactly the units the benchmark metric scores, so the
    calibration targets the measured quantity rather than the native rubric.

    Returns ``{event: {stratum: factor}}`` with ``"__global__"`` always present
    as the fallback stratum.
    """
    joined = benchmark.merge(
        hats, on=["fixture_id", "player_id"], how="inner", validate="one_to_one",
    )
    factors: dict[str, dict[str, float]] = {}
    for event in events:
        # The champion names the minutes head `minutes_hat`, not `hat_minutes`.
        column = "minutes_hat" if event == "minutes" else f"hat_{event}"
        if column not in joined.columns:
            continue
        available = joined.get(f"available__{event}", pd.Series(False, index=joined.index))
        actual = pd.to_numeric(joined[event], errors="coerce")
        predicted = pd.to_numeric(joined[column], errors="coerce").clip(lower=0.0)
        valid = (
            available.fillna(False).astype(bool)
            & actual.notna() & predicted.notna()
        )
        block = joined[valid].assign(
            __actual=actual[valid].to_numpy(float),
            __predicted=predicted[valid].to_numpy(float),
        )
        if block.empty:
            continue
        event_factors: dict[str, float] = {}
        total_actual = float(block["__actual"].sum())
        total_predicted = float(block["__predicted"].sum())
        event_factors["__global__"] = _ratio(
            total_actual, total_predicted, shrinkage, clip,
        )
        if scope == "position":
            for position, group in block.groupby(block["position"].astype(str)):
                event_factors[position] = _ratio(
                    float(group["__actual"].sum()), float(group["__predicted"].sum()),
                    shrinkage, clip,
                )
        factors[event] = event_factors
    return factors


def dead_heads(hats: pd.DataFrame, events: tuple[str, ...]) -> tuple[str, ...]:
    """Champion heads that emit identically zero on strictly-prior seasons.

    Detected from prior-season predictions only, so the decision to revive a
    head is point-in-time honest and never reads the evaluation fold.
    """
    dead = []
    for event in events:
        column = f"hat_{event}"
        if column not in hats.columns:
            continue
        values = pd.to_numeric(hats[column], errors="coerce").abs()
        if float(values.max(skipna=True) or 0.0) < 1e-12:
            dead.append(event)
    return tuple(dead)


def _ratio(
    actual: float, predicted: float, shrinkage: float, clip: tuple[float, float],
) -> float:
    """Shrunk actual/predicted ratio, clipped.  Shrinks to 1.0 when both are tiny."""
    numerator = actual + shrinkage
    denominator = predicted + shrinkage
    if denominator <= 0.0:
        return 1.0
    return float(np.clip(numerator / denominator, clip[0], clip[1]))


def _apply_calibration(
    values: np.ndarray, positions: np.ndarray, event: str,
    factors: dict[str, dict[str, float]], scope: str,
) -> np.ndarray:
    table = factors.get(event)
    if not table:
        return values
    if scope != "position":
        return values * table["__global__"]
    lookup = np.asarray([
        table.get(str(position), table["__global__"]) for position in positions
    ], dtype=float)
    return values * lookup


# ---------------------------------------------------------------------------
# extra heads -- champion architecture (per-minute rate x minutes_hat) for the
# stable events the champion never modelled, fitted on the ALL-RUGBY training
# frame that is strictly before the fold cutoff.
# ---------------------------------------------------------------------------

@dataclass
class RateMinutesHead:
    """Shrunk per-minute event rate, hierarchically pooled player -> stratum -> global."""

    events: tuple[str, ...]
    shrinkage: float = 240.0
    player_rates: dict[str, pd.Series] = field(default_factory=dict)
    stratum_rates: dict[str, pd.Series] = field(default_factory=dict)
    position_rates: dict[str, pd.Series] = field(default_factory=dict)
    global_rates: dict[str, float] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "RateMinutesHead":
        minutes = pd.to_numeric(train["minutes"], errors="coerce")
        for event in self.events:
            mask = train.get(f"available__{event}", pd.Series(False, index=train.index))
            actual = pd.to_numeric(train[event], errors="coerce")
            valid = (
                mask.fillna(False).astype(bool) & actual.notna()
                & minutes.notna() & minutes.gt(0)
            )
            block = train[valid].copy()
            if block.empty:
                self.player_rates[event] = pd.Series(dtype=float)
                self.stratum_rates[event] = pd.Series(dtype=float)
                self.position_rates[event] = pd.Series(dtype=float)
                self.global_rates[event] = 0.0
                continue
            block["__actual"] = actual[valid].to_numpy(float)
            block["__minutes"] = minutes[valid].to_numpy(float)
            block["__player"] = block["player_id"].astype(str)
            block["__position"] = block["position"].astype(str)
            global_rate = float(block["__actual"].sum() / max(block["__minutes"].sum(), 1e-9))
            self.global_rates[event] = global_rate
            position = block.groupby("__position")[["__actual", "__minutes"]].sum()
            self.position_rates[event] = (
                (position["__actual"] + self.shrinkage * global_rate)
                / (position["__minutes"] + self.shrinkage)
            )
            block["__stratum"] = (
                block["__position"] + "|" + block["started"].fillna(False).astype(bool).astype(str)
            )
            stratum = block.groupby("__stratum")[["__actual", "__minutes"]].sum()
            stratum_prior = (
                block.groupby("__stratum")["__position"].first()
                .map(self.position_rates[event]).fillna(global_rate)
            )
            self.stratum_rates[event] = (
                (stratum["__actual"] + self.shrinkage * stratum_prior)
                / (stratum["__minutes"] + self.shrinkage)
            )
            player = block.groupby("__player")[["__actual", "__minutes"]].sum()
            player_prior = (
                block.groupby("__player")["__position"].first()
                .map(self.position_rates[event]).fillna(global_rate)
            )
            self.player_rates[event] = (
                (player["__actual"] + self.shrinkage * player_prior)
                / (player["__minutes"] + self.shrinkage)
            )
        return self

    def rate(self, rows: pd.DataFrame, event: str) -> np.ndarray:
        global_rate = self.global_rates.get(event, 0.0)
        players = rows["player_id"].astype(str)
        positions = rows["position"].astype(str)
        stratum = positions + "|" + rows["started"].fillna(False).astype(bool).astype(str)
        values = players.map(self.player_rates.get(event, pd.Series(dtype=float))).to_numpy(float)
        fallback = stratum.map(self.stratum_rates.get(event, pd.Series(dtype=float))).to_numpy(float)
        values = np.where(np.isfinite(values), values, fallback)
        fallback = positions.map(self.position_rates.get(event, pd.Series(dtype=float))).to_numpy(float)
        values = np.where(np.isfinite(values), values, fallback)
        return np.where(np.isfinite(values), values, global_rate)


# ---------------------------------------------------------------------------
# raw prediction assembly
# ---------------------------------------------------------------------------

def build_raw_predictions(
    frame: pd.DataFrame, hats: pd.DataFrame, cfg: RawPathConfig,
    *, calibration: dict, extra_head: RateMinutesHead | None,
    stratum_minutes: pd.Series | None, revived: tuple[str, ...] = (),
    revive_head: RateMinutesHead | None = None,
) -> list[RawPrediction]:
    """Assemble benchmark ``RawPrediction`` rows for one evaluation fold."""
    table = hats.copy()
    table["fixture_id"] = table["fixture_id"].astype(str)
    table["player_id"] = table["player_id"].astype(str)
    by_key = table.set_index(["fixture_id", "player_id"], drop=False)
    keys = list(zip(frame["fixture_id"].astype(str), frame["player_id"].astype(str)))
    missing = [key for key in keys if key not in by_key.index]
    if missing:
        raise ValueError(f"champion missed {len(missing)} benchmark players: {missing[:5]}")
    aligned = by_key.loc[keys]
    if len(aligned) != len(frame):
        raise ValueError("champion prediction keys are ambiguous")

    positions = frame["position"].astype(str).to_numpy()
    minutes_hat = np.clip(
        pd.to_numeric(aligned["minutes_hat"], errors="coerce").to_numpy(float), 0.0, None,
    )
    if cfg.minutes_blend > 0.0 and stratum_minutes is not None:
        stratum = (
            frame["position"].astype(str) + "|"
            + frame["started"].fillna(False).astype(bool).astype(str)
        )
        prior = stratum.map(stratum_minutes).to_numpy(float)
        prior = np.where(np.isfinite(prior), prior, float(np.nanmean(minutes_hat)))
        minutes_hat = (1.0 - cfg.minutes_blend) * minutes_hat + cfg.minutes_blend * prior
    if cfg.level_calibration != "none":
        minutes_hat = _apply_calibration(
            minutes_hat, positions, "minutes", calibration, cfg.level_calibration,
        )

    event_values: dict[str, np.ndarray] = {}
    for event in RAW_EVENTS:
        values = np.clip(
            pd.to_numeric(aligned[f"hat_{event}"], errors="coerce").to_numpy(float), 0.0, None,
        )
        if cfg.level_calibration != "none":
            wanted = cfg.calibrate_events or RAW_EVENTS
            if event in wanted:
                values = _apply_calibration(
                    values, positions, event, calibration, cfg.level_calibration,
                )
        event_values[event] = values

    # Revive degenerate heads with the champion's own architecture -- a shrunk
    # per-minute rate fitted on strictly-prior all-rugby rows, times minutes_hat.
    if revive_head is not None:
        for event in revived:
            event_values[event] = np.clip(
                revive_head.rate(frame, event) * minutes_hat, 0.0, None,
            )

    if cfg.extra_heads == "rate_minutes" and extra_head is not None:
        for event in MISSING_STABLE_EVENTS:
            event_values[event] = np.clip(
                extra_head.rate(frame, event) * minutes_hat, 0.0, None,
            )

    output: list[RawPrediction] = []
    for index, row in enumerate(frame.itertuples(index=False)):
        events = {}
        for event, values in event_values.items():
            mean = float(values[index])
            if distribution_family(event) == "bernoulli":
                mean = min(mean, 1.0)
            events[event] = EventDistribution(distribution_family(event), max(mean, 0.0), 1.0)
        output.append(RawPrediction(
            fixture_id=str(row.fixture_id), player_id=str(row.player_id),
            player_name=str(row.player_name), team=str(row.team),
            opponent=str(row.opponent), position=str(row.position),
            is_forward=bool(row.is_forward), events=events,
            minutes=EventDistribution("lognormal", float(minutes_hat[index]), 1.0),
            metadata={"model": ENGINE, "raw_path": cfg.name},
        ))
    return output


# ---------------------------------------------------------------------------
# trial runner
# ---------------------------------------------------------------------------

def run_trial(
    cfg: RawPathConfig, *, benchmark: pd.DataFrame, folds: dict, native: pd.DataFrame,
    champion_config, cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one raw-path config across the champion's evaluable folds."""
    event_frames, ranking_frames = [], []
    for season in EVAL_SEASONS:
        fold = folds[f"six_nations_{season}"]
        evaluation = evaluation_frame(benchmark, fold).reset_index(drop=True)
        train = strict_training_frame(benchmark, fold)
        history = train.groupby(train["player_id"].astype(str))["fixture_id"].nunique()
        evaluation["career_matches"] = evaluation["player_id"].astype(str).map(history).fillna(0)

        hats = champion_component_predictions(native, champion_config, season, cache_dir)

        calibration: dict = {}
        if cfg.level_calibration != "none":
            calibration_seasons = [
                other for other in NATIVE_SEASONS
                if other < season and other > min(NATIVE_SEASONS)
            ]
            prior_hats = [
                champion_component_predictions(native, champion_config, other, cache_dir)
                for other in calibration_seasons
            ]
            if prior_hats:
                stacked = pd.concat(prior_hats, ignore_index=True, sort=False)
                stacked["fixture_id"] = stacked["fixture_id"].astype(str)
                stacked["player_id"] = stacked["player_id"].astype(str)
                prior_frame = train.copy()
                prior_frame["fixture_id"] = prior_frame["fixture_id"].astype(str)
                prior_frame["player_id"] = prior_frame["player_id"].astype(str)
                calibration = fit_level_calibration(
                    prior_frame, stacked, ("minutes", *RAW_EVENTS),
                    shrinkage=cfg.calibration_shrinkage, clip=cfg.calibration_clip,
                    scope=cfg.level_calibration,
                )

        revived: tuple[str, ...] = ()
        revive_head = None
        if cfg.dead_head_fallback == "rate_minutes":
            prior_hats = [
                champion_component_predictions(native, champion_config, other, cache_dir)
                for other in NATIVE_SEASONS
                if other < season and other > min(NATIVE_SEASONS)
            ]
            if prior_hats:
                revived = dead_heads(
                    pd.concat(prior_hats, ignore_index=True, sort=False), RAW_EVENTS,
                )
            if cfg.revive_events is not None:
                revived = tuple(e for e in revived if e in cfg.revive_events)
            if revived:
                revive_head = RateMinutesHead(
                    events=revived, shrinkage=cfg.extra_head_shrinkage,
                ).fit(train)

        extra_head = None
        if cfg.extra_heads == "rate_minutes":
            extra_head = RateMinutesHead(
                events=MISSING_STABLE_EVENTS, shrinkage=cfg.extra_head_shrinkage,
            ).fit(train)

        stratum_minutes = None
        if cfg.minutes_blend > 0.0:
            observed = train[pd.to_numeric(train["minutes"], errors="coerce").notna()].copy()
            stratum_minutes = observed.groupby(
                observed["position"].astype(str) + "|"
                + observed["started"].fillna(False).astype(bool).astype(str)
            )["minutes"].mean()

        predictions = build_raw_predictions(
            evaluation, hats, cfg, calibration=calibration,
            extra_head=extra_head, stratum_minutes=stratum_minutes,
            revived=revived, revive_head=revive_head,
        )
        naive = NaiveComparator.fit(train, ("minutes", *STABLE_EVENTS, *EXTENDED_EVENTS))
        events = event_metrics(
            evaluation, predictions, naive, engine=ENGINE, fold=fold.label,
        ).assign(
            tournament=fold.tournament, calendar_year=fold.calendar_year,
            fold_hemisphere=fold.hemisphere, raw_path=cfg.name,
        )
        rankings = ranking_metrics(
            evaluation, predictions, engine=ENGINE, fold=fold.label,
        ).assign(
            tournament=fold.tournament, calendar_year=fold.calendar_year,
            fold_hemisphere=fold.hemisphere, raw_path=cfg.name,
        )
        event_frames.append(events)
        ranking_frames.append(rankings)
    return (
        pd.concat(event_frames, ignore_index=True, sort=False),
        pd.concat(ranking_frames, ignore_index=True, sort=False),
    )


def load_context(output_dir: Path, store_path: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame, object]:
    benchmark = pd.read_csv(
        store_path, low_memory=False, parse_dates=["date", "match_at"],
    )
    folds = {fold.label: fold for fold in build_folds(benchmark)}
    native = load_six_nations_store()
    champion_config = config_from_dict(json.loads(PROMOTED_CONFIG.read_text()))
    return benchmark, folds, native, champion_config
