"""v4 point-in-time feature augmentation: EB shrinkage stats and level splits.

Every column here is computed from strictly-prior rows (`shift(1)` before any
cumulative/rolling aggregation), matching the discipline of
``model.unified.features.build_pit_features``. The empirical-Bayes shrinkage
constant K is NOT a feature input: it is fitted per fold from the training
frame only (`fit_shrinkage_k`) and applied to the K-independent cumulative
stats (`add_v4_base_stats`) via `apply_eb_features`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import COUNT_EVENTS, SCORING_EVENTS

# Count events that enter either competition's scoring — the tail that matters.
EB_EVENTS = tuple(e for e in SCORING_EVENTS if e in COUNT_EVENTS and e != "potm")
THIN_INTL_MATCHES = 5
K_DEFAULT = 220.0
K_BOUNDS = (40.0, 4000.0)


def add_v4_base_stats(store: pd.DataFrame) -> pd.DataFrame:
    """K-independent PIT stats: intl history, level-split form, EB cumulatives."""
    df = store.copy()
    player = df.groupby("player_id", sort=False)
    is_intl = df["competition_level"].eq("international").astype(float)
    df["intl_prior_matches"] = (
        is_intl.groupby(df["player_id"]).transform(lambda x: x.shift(1).cumsum())
    ).fillna(0.0)
    df["club_prior_matches"] = (
        (1.0 - is_intl).groupby(df["player_id"]).transform(lambda x: x.shift(1).cumsum())
    ).fillna(0.0)

    minutes = pd.to_numeric(df["minutes"], errors="coerce")
    for event in EB_EVENTS:
        values = pd.to_numeric(df[event], errors="coerce")
        valid = df[f"available__{event}"].astype(bool) & minutes.gt(0)
        count = values.where(valid)
        mins = minutes.where(valid)
        per80 = count / mins * 80.0

        # Level-split recent form (club vs international), same halflife as base.
        for level, mask in (("intl", is_intl.astype(bool)), ("club", ~is_intl.astype(bool))):
            level_per80 = per80.where(mask)
            df[f"form_per80_{level}__{event}"] = level_per80.groupby(
                df["player_id"]
            ).transform(lambda x: x.shift(1).ewm(halflife=4, min_periods=1).mean())

        # EB cumulatives: prior event count and prior observed minutes.
        df[f"cum_count__{event}"] = count.fillna(0.0).groupby(
            df["player_id"]
        ).transform(lambda x: x.shift(1).cumsum()).fillna(0.0)
        df[f"cum_min__{event}"] = mins.fillna(0.0).groupby(
            df["player_id"]
        ).transform(lambda x: x.shift(1).cumsum()).fillna(0.0)

        # Expanding position-by-level prior per-80 rate from strictly-prior rows.
        group = [df["position"].fillna("unknown"), df["competition_level"].fillna("unknown")]
        df[f"prior_per80__{event}"] = per80.groupby(group).transform(
            lambda x: x.shift(1).expanding(min_periods=5).mean()
        )
        overall = per80.groupby(df["competition_level"].fillna("unknown")).transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        df[f"prior_per80__{event}"] = df[f"prior_per80__{event}"].fillna(overall).fillna(0.0)
    return df.copy()  # defragment after the many column inserts


def fit_shrinkage_k(train: pd.DataFrame) -> dict[str, float]:
    """Per-event K (minutes) by variance moment-matching on the training frame.

    Model: player true per-80 rate ~ (mu, tau^2); an M-minute observed rate has
    Poisson sampling variance ~ 80*mu/M. Weighted between-player variance minus
    expected sampling variance estimates tau^2; K = 80*mu/tau^2.
    """
    minutes = pd.to_numeric(train["minutes"], errors="coerce")
    out: dict[str, float] = {}
    for event in EB_EVENTS:
        valid = train[f"available__{event}"].astype(bool) & minutes.gt(0)
        sub = train[valid]
        if len(sub) < 500:
            out[event] = K_DEFAULT
            continue
        mins = pd.to_numeric(sub["minutes"], errors="coerce")
        counts = pd.to_numeric(sub[event], errors="coerce").clip(lower=0)
        agg = pd.DataFrame({
            "count": counts.groupby(sub["player_id"]).sum(),
            "mins": mins.groupby(sub["player_id"]).sum(),
        })
        agg = agg[agg["mins"] >= 80.0]
        if len(agg) < 50:
            out[event] = K_DEFAULT
            continue
        rate = 80.0 * agg["count"] / agg["mins"]
        weight = agg["mins"].to_numpy(float)
        mu = 80.0 * agg["count"].sum() / agg["mins"].sum()
        between = float(np.average((rate - mu) ** 2, weights=weight))
        sampling = float(np.average(80.0 * mu / agg["mins"], weights=weight))
        tau2 = between - sampling
        if mu <= 0 or tau2 <= 1e-9:
            out[event] = K_BOUNDS[1]
        else:
            out[event] = float(np.clip(80.0 * mu / tau2, *K_BOUNDS))
    return out


def apply_eb_features(store: pd.DataFrame, k_by_event: dict[str, float]) -> pd.DataFrame:
    """eb_rate__event = (80*cum_count + K*prior) / (cum_min + K); PIT by inputs."""
    df = store.copy()
    for event in EB_EVENTS:
        k = float(k_by_event.get(event, K_DEFAULT))
        cum_count = df[f"cum_count__{event}"]
        cum_min = df[f"cum_min__{event}"]
        prior = df[f"prior_per80__{event}"]
        df[f"eb_rate__{event}"] = (80.0 * cum_count + k * prior) / (cum_min + k)
        df[f"eb_weight__{event}"] = cum_min / (cum_min + k)
    return df


V4_FEATURE_PREFIXES = ("form_per80_intl__", "form_per80_club__", "eb_rate__", "eb_weight__")
V4_BASE_NUMERIC = ("intl_prior_matches", "club_prior_matches")
