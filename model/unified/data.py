"""Canonical data module for all club and international player-match sources."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import EVENTS, FORWARD_POSITIONS, KEY_COLUMNS, POSITION_BY_JERSEY

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = (
    (ROOT / "data/ncr/ncr_player_match.csv", "ncr_international", 30, "international"),
    (ROOT / "data/ncr/club_player_match.csv", "ncr_club", 20, "club"),
    (ROOT / "data/api_player_match.csv", "sixn_api", 10, "international"),
)

ALIASES = {
    "comp_name": "competition", "comp_id": "competition_id",
    "canonical_pos": "position",
}

OFFICIAL_EVENT_COLUMNS = {
    "50-22": "fifty_22", "LS": "lineout_steals", "SW": "scrums_won",
    "KR": "kicks_retained", "POTM": "potm",
}


def _name_key(name: object) -> str:
    """Match ``Louis Bielle-Biarrey`` to official ``L. Bielle-Biarrey``."""
    if not isinstance(name, str) or not name.strip():
        return ""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().strip()
    parts = plain.split()
    initial = parts[0][0].lower()
    surname = "".join(parts[1:] if len(parts) > 1 else parts)
    return initial + "|" + re.sub(r"[^a-z]", "", surname.lower())


def _attach_official_labels(frame: pd.DataFrame) -> pd.DataFrame:
    path = ROOT / "data/official_player_match.csv"
    if not path.exists():
        return frame
    official = pd.read_csv(path, low_memory=False)
    official["name_key"] = official["name"].map(_name_key)
    keep = ["season", "round", "team", "name_key", *OFFICIAL_EVENT_COLUMNS]
    official = official[keep].drop_duplicates(["season", "round", "team", "name_key"])
    official = official.rename(columns={k: f"official__{v}" for k, v in OFFICIAL_EVENT_COLUMNS.items()})
    out = frame.copy()
    out["name_key"] = out["player_name"].map(_name_key)
    out = out.merge(official, on=["season", "round", "team", "name_key"], how="left", validate="many_to_one")
    joined = pd.Series(False, index=out.index)
    for event in OFFICIAL_EVENT_COLUMNS.values():
        label = pd.to_numeric(out.pop(f"official__{event}"), errors="coerce")
        has_label = label.notna()
        out[event] = out[event].where(out[event].notna(), label)
        out[f"available__{event}"] = out[f"available__{event}"].astype(bool) | has_label
        joined |= has_label
    out.loc[joined, "source"] = out.loc[joined, "source"].astype(str) + "+sixn_official"
    out.loc[joined, "source_count"] = out.loc[joined, "source_count"].astype(int) + 1
    return out.drop(columns="name_key")


def _stable_id(value: object, name: object, team: object) -> str:
    if pd.notna(value):
        return str(int(value)) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
    raw = f"{name}|{team}".casefold().encode()
    return "anon_" + hashlib.sha1(raw).hexdigest()[:14]


def _normalise_source(
    path: Path, source: str, priority: int, level: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False).rename(columns=ALIASES)
    original = set(frame.columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["source"] = source
    frame["source_priority"] = priority
    frame["competition_level"] = level
    if "competition" not in frame:
        frame["competition"] = "unknown"
    if "competition_id" not in frame:
        frame["competition_id"] = np.nan
    if "season" not in frame:
        frame["season"] = frame["date"].dt.year
    if "round" not in frame:
        frame["round"] = np.nan
    for col in ("jersey", "minutes"):
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    if "started" not in frame:
        frame["started"] = frame["jersey"].between(1, 15)
    frame["started"] = frame["started"].map(
        lambda x: x if isinstance(x, bool) else str(x).strip().lower() in {"1", "true", "t", "yes"}
    )
    inferred = frame["jersey"].map(POSITION_BY_JERSEY)
    if "position" not in frame:
        frame["position"] = inferred
    else:
        frame["position"] = frame["position"].fillna(inferred)
    frame["is_forward"] = frame["position"].isin(FORWARD_POSITIONS)
    frame["player_id"] = [
        _stable_id(pid, name, team)
        for pid, name, team in zip(frame.get("player_id"), frame.get("player_name"), frame.get("team"))
    ]
    frame["fixture_id"] = frame["fixture_id"].astype(str)
    for event in EVENTS:
        available = event in original
        if event not in frame:
            frame[event] = np.nan
        frame[event] = pd.to_numeric(frame[event], errors="coerce")
        mask = f"available__{event}"
        if mask in original:
            explicit = frame[mask].map(
                lambda value: value if isinstance(value, bool)
                else str(value).strip().lower() in {"1", "true", "t", "yes"}
            )
            frame[mask] = explicit.astype(bool) & frame[event].notna()
        else:
            frame[mask] = bool(available) & frame[event].notna()
    if "available__minutes" in original:
        explicit_minutes = frame["available__minutes"].map(
            lambda value: value if isinstance(value, bool)
            else str(value).strip().lower() in {"1", "true", "t", "yes"}
        )
        frame["available__minutes"] = explicit_minutes.astype(bool) & frame["minutes"].notna()
    else:
        frame["available__minutes"] = frame["minutes"].notna()
    return frame


def build_canonical_store(
    sources: Iterable[tuple[Path, str, int, str]] = DEFAULT_SOURCES,
    *, asof: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return one canonical row per ``(fixture_id, player_id, team)``.

    Duplicate rows caused by overlapping historical downloads are resolved by
    source priority, then merged column-by-column so a lower-priority source may
    fill only values absent from the chosen row.  It can never overwrite them.
    """

    frames = [_normalise_source(Path(p), name, priority, level)
              for p, name, priority, level in sources if Path(p).exists()]
    if not frames:
        raise FileNotFoundError("none of the canonical data sources exist")
    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    all_rows = all_rows.dropna(subset=["date", "fixture_id", "player_id", "team"])
    if asof is not None:
        cutoff = pd.Timestamp(asof)
        all_rows = all_rows[all_rows["date"] < cutoff]
    all_rows = all_rows.sort_values(
        [*KEY_COLUMNS, "source_priority"], ascending=[True, True, True, False], kind="stable"
    )

    keys = list(KEY_COLUMNS)
    nonkeys = [c for c in all_rows.columns if c not in keys]
    grouped = all_rows.groupby(keys, sort=False)
    # The sort puts the authoritative source first. bfill supplies only null
    # cells from lower-priority rows, then drop_duplicates selects that filled
    # first row. This is vectorized across ~200k rows.
    filled = all_rows.copy()
    filled[nonkeys] = grouped[nonkeys].bfill()
    availability = [c for c in nonkeys if c.startswith("available__")]
    filled[availability] = grouped[availability].transform("max").astype(bool)
    filled["source_count"] = grouped["source"].transform("size")
    out = filled.drop_duplicates(keys, keep="first").reset_index(drop=True)
    out = _attach_official_labels(out)
    out = out.sort_values(["date", "fixture_id", "team", "player_id"]).reset_index(drop=True)
    if out.duplicated(list(KEY_COLUMNS)).any():
        raise AssertionError("canonical player-match grain is not unique")
    for event in EVENTS:
        missing = ~out[f"available__{event}"].astype(bool)
        if out.loc[missing, event].notna().any():
            raise AssertionError(f"{event}: unavailable labels must remain missing")
    return out


def write_canonical_store(path: Path, *, asof: str | None = None) -> pd.DataFrame:
    frame = build_canonical_store(asof=asof)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
