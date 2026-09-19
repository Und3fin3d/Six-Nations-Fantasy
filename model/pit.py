"""Shared information boundaries for retrospective and live predictions."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


def day_cutoff(asof: object) -> pd.Timestamp:
    """UTC midnight: date-only sources cannot prove availability during lock day."""
    value = pd.Timestamp(asof)
    if pd.isna(value):
        raise ValueError("prediction cutoff is missing")
    value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    return value.normalize()


def history_before(frame: pd.DataFrame, asof: object) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
    if dates.isna().any():
        raise ValueError("history contains an undated match")
    return frame.loc[dates.lt(day_cutoff(asof))].copy()


def completed_seasons(seasons: pd.Series, asof: object) -> pd.Series:
    """Conservative availability for full-season aggregates, not match histories.

    Calendar-year totals are usable next January; split-year totals next July.
    In particular, 2026 international totals cannot enter July 2026 predictions.
    Unknown labels are unavailable. This is a retrospective convention, not proof
    of when a historical source snapshot was actually published.
    """
    cutoff = day_cutoff(asof)
    def available(value: object) -> bool:
        text = str(value).strip()
        annual = re.fullmatch(r"(\d{4})(?:\.0)?", text)
        split = re.fullmatch(r"(\d{4})\s*[/\-–]\s*(\d{4})", text)
        if annual:
            when = pd.Timestamp(year=int(annual[1]) + 1, month=1, day=1, tz="UTC")
        elif split and int(split[2]) == int(split[1]) + 1:
            when = pd.Timestamp(year=int(split[2]), month=7, day=1, tz="UTC")
        else:
            return False
        return bool(when <= cutoff)
    return seasons.map(available).astype(bool)


def team_margin_history(frame: pd.DataFrame, *, prior_only: bool) -> pd.Series:
    """Compute form once per fixture, never once per teammate."""
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    keys = ["fixture_id", "team"]
    matches = frame.sort_values(["date", "fixture_id", "team"]).drop_duplicates(keys).copy()
    matches["_margin"] = (
        pd.to_numeric(matches.get("team_score", pd.Series(np.nan, index=matches.index)), errors="coerce")
        - pd.to_numeric(matches.get("opp_score", pd.Series(np.nan, index=matches.index)), errors="coerce")
    )
    matches["_history"] = matches.groupby("team", sort=False)["_margin"].transform(
        lambda values: (values.shift(1) if prior_only else values).ewm(span=8, min_periods=1).mean()
    )
    index = pd.MultiIndex.from_frame(frame[keys])
    values = matches.set_index(keys)["_history"].reindex(index).to_numpy()
    return pd.Series(values, index=frame.index, dtype=float)
