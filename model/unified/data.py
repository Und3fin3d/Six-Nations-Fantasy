from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.sixn.official_labels import match_official

from .schema import EVENTS, FORWARD_POSITIONS, KEY_COLUMNS, POSITION_BY_JERSEY

DEFAULT_SOURCES = (
    ("data/ncr/ncr_player_match.csv", "ncr_international", 30, "international"), ("data/ncr/club_player_match.csv", "ncr_club", 20, "club"),
    ("data/api_player_match.csv", "sixn_api", 10, "international"),
)

ALIASES = {"comp_name": "competition", "comp_id": "competition_id", "canonical_pos": "position"}

OFFICIAL_EVENT_COLUMNS = {"50-22": "fifty_22", "LS": "lineout_steals", "SW": "scrums_won", "KR": "kicks_retained", "POTM": "potm"}


def _attach_official_labels(frame: pd.DataFrame) -> pd.DataFrame:
    official = pd.read_csv("data/official_player_match.csv", low_memory=False)
    out = frame.copy()
    six = out[pd.to_numeric(out.competition_id, errors='coerce').eq(1266)]
    labels = match_official(six, official)
    for source_col, event in OFFICIAL_EVENT_COLUMNS.items():
        out[f'official__{event}'] = labels[source_col]
    joined = pd.Series(False, index=out.index)
    for event in OFFICIAL_EVENT_COLUMNS.values():
        label = pd.to_numeric(out.pop(f"official__{event}"), errors="coerce")
        has_label = label.notna()
        out[event] = out[event].where(out[event].notna(), label)
        out[f"available__{event}"] = out[f"available__{event}"].astype(bool) | has_label
        joined |= has_label
    out.loc[joined, "source"] = out.loc[joined, "source"].astype(str) + "+sixn_official"
    out.loc[joined, "source_count"] = out.loc[joined, "source_count"].astype(int) + 1
    return out


def _stable_id(value: object, name: object, team: object) -> str:
    if pd.notna(value):
        return str(int(value)) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
    return "anon_" + hashlib.sha1(f"{name}|{team}".casefold().encode()).hexdigest()[:14]


def _flag(series: pd.Series) -> pd.Series:
    return series.map(lambda x: x if isinstance(x, bool) else str(x).strip().lower() in {"1", "true", "t", "yes"})


def _normalise_source(path: str, source: str, priority: int, level: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False).rename(columns=ALIASES)
    original = set(frame.columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["source"], frame["source_priority"], frame["competition_level"] = source, priority, level
    for col, default in (("competition", "unknown"), ("competition_id", np.nan)):
        if col not in frame:
            frame[col] = default
    if "season" not in frame:
        frame["season"] = frame["date"].dt.year
    if "round" not in frame:
        frame["round"] = np.nan
    for col in ("jersey", "minutes"):
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    if "started" not in frame:
        frame["started"] = frame["jersey"].between(1, 15)
    frame["started"] = _flag(frame["started"])
    inferred = frame["jersey"].map(POSITION_BY_JERSEY)
    frame["position"] = frame["position"].fillna(inferred) if "position" in frame else inferred
    frame["is_forward"] = frame["position"].isin(FORWARD_POSITIONS)
    frame["player_id"] = [_stable_id(pid, name, team) for pid, name, team in zip(frame.get("player_id"), frame.get("player_name"), frame.get("team"))]
    frame["fixture_id"] = frame["fixture_id"].astype(str)
    for event in EVENTS:
        frame[event] = pd.to_numeric(frame[event], errors="coerce") if event in frame else np.nan
        mask = f"available__{event}"
        frame[mask] = (_flag(frame[mask]).astype(bool) if mask in original else event in original) & frame[event].notna()
    frame["available__minutes"] = (_flag(frame["available__minutes"]).astype(bool) if "available__minutes" in original else True) & frame["minutes"].notna()
    return frame


def build_canonical_store(sources: Iterable[tuple[str, str, int, str]] = DEFAULT_SOURCES) -> pd.DataFrame:
    all_rows = pd.concat([_normalise_source(*source) for source in sources], ignore_index=True, sort=False).dropna(
        subset=["date", "fixture_id", "player_id", "team"]
    )
    all_rows = all_rows.sort_values([*KEY_COLUMNS, "source_priority"], ascending=[True, True, True, False], kind="stable")
    keys = list(KEY_COLUMNS)
    nonkeys = [c for c in all_rows.columns if c not in keys]
    grouped = all_rows.groupby(keys, sort=False)
    filled = all_rows.copy()
    filled[nonkeys] = grouped[nonkeys].bfill()
    availability = [c for c in nonkeys if c.startswith("available__")]
    filled[availability] = grouped[availability].transform("max").astype(bool)
    filled["source_count"] = grouped["source"].transform("size")
    out = _attach_official_labels(filled.drop_duplicates(keys, keep="first").reset_index(drop=True))
    return out.sort_values(["date", "fixture_id", "team", "player_id"]).reset_index(drop=True)
