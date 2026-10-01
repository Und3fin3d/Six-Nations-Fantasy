"""Status-aware per-minute rates for the robust empirical event model.

Replacements produce more per minute than starters: they play the open last
quarter against tired defences. Within the same player, per-minute rates off
the bench are about 1.1-1.5 times his rates as a starter, for tries, metres,
defenders beaten, carries and tackles alike, in club and international rugby
and in every era of the store. The robust empirical model pools a player's
starting and bench minutes into one per-80 rate and scales it by expected
minutes, so it under-forecasts replacements and slightly over-forecasts
starters whose history includes bench cameos. The tree component conditions
on starting status directly and does not share the bias.

``StatusAwareEmpiricalEventModel`` removes it without any tuned parameter:

1. At each lock, a within-player multiplicative model ``E[y] = r_player *
   s[position, started] * minutes`` is fitted to every recency-weighted
   pre-lock row (club and international alike) by alternating ratio updates.
   Bench factors are expressed relative to the same position's starter
   factor and lightly shrunk towards 1.
2. Training events are converted to starter-equivalent counts (divided by
   the row's factor) before the unchanged robust fit, so player profiles and
   positional priors are starter rates.
3. Replacement forecasts multiply the starter rate by the bench factor.

Goal kicking, cards and the sparse official-only events keep factor 1. The
estimate is competition-independent: no competition identifier, fantasy
outcome or evaluation season selects anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from model.history import past_matches, utc_cutoff

from ..contracts import EventDistribution, RawPrediction
from .empirical import HALFLIFE_DAYS
from .robust_empirical import RobustEmpiricalEventModel

UNADJUSTED_EVENTS = frozenset({
    "conversion_goals", "missed_conversion_goals", "penalty_goals",
    "missed_penalty_goals", "drop_goals_converted", "drop_goal_missed",
    "yellow_cards", "red_cards", "fifty_22", "lineout_steals", "scrums_won",
    "kicks_retained", "potm",
})
POOLED = "*"


def estimate_status_factors(
    frame: pd.DataFrame, events, asof, *, halflife_days: float = HALFLIFE_DAYS,
    prior_events: float = 20.0, iterations: int = 4,
) -> dict[tuple[str, bool, str], float]:
    """Bench/starter per-minute rate ratios by (position, started, event).

    Starters are the reference (factor 1). ``(POOLED, False, event)`` is the
    all-position bench factor used for unknown positions. Only rows with
    positive minutes and an available event contribute.
    """
    rows = frame[pd.to_numeric(frame["minutes"], errors="coerce").gt(0)].copy()
    if rows.empty:
        return {}
    dates = pd.to_datetime(rows["date"], errors="coerce", utc=True)
    age = (utc_cutoff(asof) - dates).dt.days.clip(lower=0).fillna(1e9)
    weight = np.power(0.5, age.to_numpy(float) / halflife_days)
    minutes = pd.to_numeric(rows["minutes"], errors="coerce").to_numpy(float)
    started = rows["started"].astype(bool).to_numpy()
    position = rows["position"].astype(str).to_numpy()
    player = pd.factorize(rows["player_id"].astype(str))[0]
    known = position != "Unknown"
    labels = np.where(known, position, POOLED)
    groups = pd.Index(sorted(set(zip(labels, started))))
    group = groups.get_indexer(list(zip(labels, started)))
    start_of = np.array([groups.get_loc((label, True)) if (label, True) in groups else -1
                         for label, _ in groups])
    output: dict[tuple[str, bool, str], float] = {}
    for event in events:
        if event in UNADJUSTED_EVENTS or event not in rows:
            continue
        available = rows[f"available__{event}"] if f"available__{event}" in rows else pd.Series(True, index=rows.index)
        value = pd.to_numeric(rows[event], errors="coerce")
        ok = available.fillna(False).astype(bool).to_numpy() & value.notna().to_numpy() & known
        if ok.sum() < 200:
            continue
        y = value.to_numpy(float)[ok] * weight[ok]
        exposure = minutes[ok] * weight[ok]
        p, g = player[ok], group[ok]
        factor = np.ones(len(groups))
        for _ in range(iterations):
            denominator = np.bincount(p, weights=exposure * factor[g], minlength=player.max() + 1)
            rate = np.bincount(p, weights=y, minlength=player.max() + 1) / np.maximum(denominator, 1e-12)
            observed_g = np.bincount(g, weights=y, minlength=len(groups))
            expected_g = np.bincount(g, weights=exposure * rate[p], minlength=len(groups))
            factor = np.where(expected_g > 0, observed_g / np.maximum(expected_g, 1e-12), 1.0)
        # Expected counts had every row carried its position's starter factor.
        start = start_of[g]
        as_starter = exposure * rate[p] * np.where(start >= 0, factor[np.maximum(start, 0)], np.nan)
        bench = ~started[ok] & np.isfinite(as_starter)
        observed_b = np.bincount(g[bench], weights=y[bench], minlength=len(groups))
        expected_b = np.bincount(g[bench], weights=as_starter[bench], minlength=len(groups))
        if expected_b.sum() <= 0:
            continue
        pooled = float(observed_b.sum() / expected_b.sum())
        output[(POOLED, False, event)] = pooled
        for k, (label, is_start) in enumerate(groups):
            if is_start or expected_b[k] <= 0:
                continue
            # A thin bench group shrinks towards the all-position ratio.
            output[(str(label), False, event)] = float(
                (observed_b[k] + prior_events * pooled) / (expected_b[k] + prior_events))
    return output


def status_factor(factors, position: str, started: bool, event: str) -> float:
    if started:
        return 1.0
    return factors.get((str(position), False, event), factors.get((POOLED, False, event), 1.0))


def starter_equivalent(frame: pd.DataFrame, factors, events) -> pd.DataFrame:
    """Divide each bench row's events by its factor (starter-rate units)."""
    output = frame.copy()
    bench = ~output["started"].astype(bool)
    if not bench.any():
        return output
    position = output.loc[bench, "position"].astype(str)
    for event in events:
        if event in UNADJUSTED_EVENTS or event not in output:
            continue
        scale = position.map(lambda pos: status_factor(factors, pos, False, event)).astype(float)
        output.loc[bench, event] = pd.to_numeric(output.loc[bench, event], errors="coerce") / scale
    return output


def _scale(distribution: EventDistribution, factor: float, variance_floor: float) -> EventDistribution:
    mean = distribution.mean * factor
    if distribution.family == "bernoulli":
        return EventDistribution("bernoulli", min(mean, 1.0), 1.0)
    if distribution.family == "negative_binomial":
        variance = max(variance_floor, mean + 1e-6)
        return EventDistribution("negative_binomial", mean, max(mean * mean / max(variance - mean, 1e-6), 0.05))
    return EventDistribution(distribution.family, mean, distribution.dispersion)


@dataclass
class StatusAwareEmpiricalEventModel(RobustEmpiricalEventModel):
    """Robust empirical model with starter-equivalent rates and bench factors."""

    status_factors: dict[tuple[str, bool, str], float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> "StatusAwareEmpiricalEventModel":
        train = past_matches(frame, self.asof)
        self.status_factors = estimate_status_factors(train, self.events, self.asof)
        super().fit(starter_equivalent(train, self.status_factors, self.events))
        # Dispersions describe raw counts, not starter-equivalent counts.
        intl = train[train["competition_level"].eq("international")]
        minutes = pd.to_numeric(intl["minutes"], errors="coerce")
        for event in self.events:
            if event not in self.dispersion or f"available__{event}" not in intl:
                continue
            valid = intl[f"available__{event}"].fillna(False).astype(bool) & minutes.gt(0)
            actual = pd.to_numeric(intl.loc[valid, event], errors="coerce").to_numpy(float)
            if len(actual):
                self.dispersion[event] = max(float(np.nanvar(actual)), float(np.nanmean(actual)), 1e-4)
        return self

    def predict_frame(self, frame: pd.DataFrame) -> list[RawPrediction]:
        predictions = super().predict_frame(frame)
        output = []
        for prediction, started in zip(predictions, frame["started"].astype(bool)):
            if started:
                output.append(prediction)
                continue
            events = {
                event: _scale(dist, status_factor(self.status_factors, prediction.position, False, event),
                              self.dispersion.get(event, dist.mean + 1.0))
                if event not in UNADJUSTED_EVENTS else dist
                for event, dist in prediction.events.items()
            }
            output.append(RawPrediction(
                fixture_id=prediction.fixture_id, player_id=prediction.player_id,
                player_name=prediction.player_name, team=prediction.team,
                opponent=prediction.opponent, position=prediction.position,
                is_forward=prediction.is_forward, events=events, minutes=prediction.minutes,
                metadata={**prediction.metadata, "status_rates": True},
            ))
        return output
