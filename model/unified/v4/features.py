from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import COUNT_EVENTS, SCORING_EVENTS
from ..features import grouped_ewm

EB_EVENTS = tuple(e for e in SCORING_EVENTS if e in COUNT_EVENTS and e != "potm")
K_DEFAULT, K_BOUNDS = 220.0, (40.0, 4000.0)
V4_FEATURE_PREFIXES = ("form_per80_intl__", "form_per80_club__", "eb_rate__", "eb_weight__")
V4_BASE_NUMERIC = ("intl_prior_matches", "club_prior_matches")


def prior_rates(frame: pd.DataFrame, rates: pd.Series, keys: list[str], min_count: int) -> pd.Series:
    groups = frame[keys + ["date"]].copy()
    groups[keys] = groups[keys].fillna("unknown")
    groups["rate"] = rates
    totals = groups.groupby(keys + ["date"])["rate"].agg(["sum", "count"]).sort_index()
    past = totals.groupby(level=keys).cumsum() - totals
    prior = (past["sum"] / past["count"]).where(past["count"] >= min_count)
    index = pd.MultiIndex.from_frame(groups[keys + ["date"]])
    return pd.Series(prior.reindex(index).to_numpy(float), index=frame.index)


def add_v4_base_stats(store: pd.DataFrame) -> pd.DataFrame:
    df = store.copy()
    is_intl = df["competition_level"].eq("international").astype(float)
    df["intl_prior_matches"] = is_intl.groupby(df["player_id"]).cumsum() - is_intl
    df["club_prior_matches"] = (1.0 - is_intl).groupby(df["player_id"]).cumsum() - (1.0 - is_intl)

    minutes = pd.to_numeric(df["minutes"], errors="coerce")
    for event in EB_EVENTS:
        values = pd.to_numeric(df[event], errors="coerce")
        valid = df[f"available__{event}"].astype(bool) & minutes.gt(0)
        count, mins = values.where(valid), minutes.where(valid)
        per80 = count / mins * 80.0
        for level, mask in (("intl", is_intl.astype(bool)), ("club", ~is_intl.astype(bool))):
            df[f"form_per80_{level}__{event}"] = grouped_ewm(per80.where(mask), df["player_id"], shifted=True, halflife=4)
        df[f"cum_count__{event}"] = count.fillna(0).groupby(df["player_id"]).cumsum() - count.fillna(0)
        df[f"cum_min__{event}"] = mins.fillna(0).groupby(df["player_id"]).cumsum() - mins.fillna(0)
        prior = prior_rates(df, per80, ["position", "competition_level"], 5)
        overall = prior_rates(df, per80, ["competition_level"], 1)
        df[f"prior_per80__{event}"] = prior.fillna(overall).fillna(0.0)
    return df.copy()


def fit_shrinkage_k(train: pd.DataFrame) -> dict[str, float]:
    minutes = pd.to_numeric(train["minutes"], errors="coerce")
    out: dict[str, float] = {}
    for event in EB_EVENTS:
        valid = train[f"available__{event}"].astype(bool) & minutes.gt(0)
        sub = train[valid]
        agg = pd.DataFrame(
            {
                "count": pd.to_numeric(sub[event], errors="coerce").clip(lower=0).groupby(sub["player_id"]).sum(),
                "mins": pd.to_numeric(sub["minutes"], errors="coerce").groupby(sub["player_id"]).sum(),
            }
        )
        agg = agg[agg["mins"] >= 80.0]
        if len(sub) < 500 or len(agg) < 50:
            out[event] = K_DEFAULT
            continue
        weight = agg["mins"].to_numpy(float)
        mu = 80.0 * agg["count"].sum() / agg["mins"].sum()
        tau2 = float(np.average((80.0 * agg["count"] / agg["mins"] - mu) ** 2, weights=weight)) - float(np.average(80.0 * mu / agg["mins"], weights=weight))
        out[event] = K_BOUNDS[1] if mu <= 0 or tau2 <= 1e-9 else float(np.clip(80.0 * mu / tau2, *K_BOUNDS))
    return out
