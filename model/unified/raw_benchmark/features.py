"""Frozen-history features: held-out tournament rows never update one another."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..features import build_pit_features
from ..schema import EVENTS
from ..v4.features import add_v4_base_stats


def _last_by_player(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    return values.groupby(frame["player_id"], sort=False).last()


def build_frozen_feature_frames(
    train: pd.DataFrame, candidates: pd.DataFrame, *, v4: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train features and candidate features using training history only."""
    ordered = train.sort_values(["date", "fixture_id", "team", "player_id"]).copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    training_features = build_pit_features(ordered)
    candidate_features = candidates.copy()
    candidate_features["date"] = pd.to_datetime(candidate_features["date"], errors="coerce")
    players = ordered["player_id"]
    player_group = ordered.groupby("player_id", sort=False)
    last_date = player_group["date"].max()
    candidate_features["days_since_last"] = (
        candidate_features["date"]
        - candidate_features["player_id"].map(last_date)
    ).dt.days.clip(0, 1000)
    candidate_features["career_matches"] = candidate_features["player_id"].map(
        player_group.size()
    ).fillna(0).astype(float)

    minutes = pd.to_numeric(ordered["minutes"], errors="coerce")
    minutes_state = minutes.groupby(players).transform(
        lambda values: values.ewm(span=6, min_periods=1).mean()
    )
    start_state = ordered["started"].astype(float).groupby(players).transform(
        lambda values: values.ewm(span=6, min_periods=1).mean()
    )
    candidate_features["recent_minutes"] = candidate_features["player_id"].map(
        _last_by_player(ordered, minutes_state)
    )
    candidate_features["recent_start_rate"] = candidate_features["player_id"].map(
        _last_by_player(ordered, start_state)
    )

    team_score = pd.to_numeric(
        ordered.get("team_score", pd.Series(np.nan, index=ordered.index)), errors="coerce"
    )
    opponent_score = pd.to_numeric(
        ordered.get("opp_score", pd.Series(np.nan, index=ordered.index)), errors="coerce"
    )
    team_margin = team_score - opponent_score
    margin_state = team_margin.groupby(ordered["team"], sort=False).transform(
        lambda values: values.ewm(span=8, min_periods=1).mean()
    )
    candidate_features["team_recent_margin"] = candidate_features["team"].map(
        margin_state.groupby(ordered["team"], sort=False).last()
    )

    is_intl = ordered["competition_level"].eq("international")
    if v4:
        counts_intl = is_intl.astype(int).groupby(players).sum()
        counts_club = (~is_intl).astype(int).groupby(players).sum()
        candidate_features["intl_prior_matches"] = candidate_features["player_id"].map(
            counts_intl
        ).fillna(0.0)
        candidate_features["club_prior_matches"] = candidate_features["player_id"].map(
            counts_club
        ).fillna(0.0)

    derived_columns: dict[str, pd.Series] = {}
    for event in EVENTS:
        values = pd.to_numeric(ordered[event], errors="coerce")
        valid = ordered[f"available__{event}"].fillna(False).astype(bool)
        per80 = values.where(valid) / minutes.where(minutes > 0) * 80.0
        state = per80.groupby(players).transform(
            lambda series: series.ewm(halflife=4, min_periods=1).mean()
        )
        derived_columns[f"form_per80__{event}"] = candidate_features["player_id"].map(
            _last_by_player(ordered, state)
        )
        count = valid.groupby(players).sum()
        derived_columns[f"history_count__{event}"] = candidate_features["player_id"].map(
            count
        ).fillna(0).astype(float)
        if v4:
            for level, mask in (("intl", is_intl), ("club", ~is_intl)):
                level_state = per80.where(mask).groupby(players).transform(
                    lambda series: series.ewm(halflife=4, min_periods=1).mean()
                )
                derived_columns[f"form_per80_{level}__{event}"] = (
                    candidate_features["player_id"].map(
                        _last_by_player(ordered, level_state)
                    )
                )
    candidate_features = pd.concat(
        [candidate_features, pd.DataFrame(derived_columns, index=candidate_features.index)],
        axis=1,
    )
    if v4:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            training_features = add_v4_base_stats(training_features)
    return training_features.copy(), candidate_features.copy()
