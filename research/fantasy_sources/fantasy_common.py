"""Shared helpers for the fantasy-source probes: HTTP, name keys, store joins.

Nothing here touches ``data/cache`` or RapidAPI. The store is read only.
"""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "6n-fantasy-research/1.0 (+source probe; low volume)"

STORE_COLS = [
    "date", "competition", "season", "round", "fixture_id", "team", "opponent",
    "player_id", "player_name", "started", "minutes", "tries", "tackles", "metres", "potm",
]


def http_json(url: str, headers: dict[str, str] | None = None, data: bytes | None = None,
              timeout: int = 30):
    """GET (or POST when ``data`` is given) a JSON document."""
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    req_headers.update(headers or {})
    req = urllib.request.Request(url, headers=req_headers, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()


def name_key(name: str) -> str:
    """'Thomas Ramos' / 'THOMAS  RAMOS' / 'Thomás Ramos' -> 'thomas ramos'."""
    s = ascii_fold(name).lower().replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def initial_key(name: str) -> str:
    """'Thomas Ramos' and 'T. Ramos' -> 't ramos'."""
    parts = name_key(name).split()
    if len(parts) < 2:
        return name_key(name)
    return f"{parts[0][0]} {' '.join(parts[1:])}"


def surname_key(name: str) -> str:
    parts = name_key(name).split()
    return parts[-1] if parts else ""


def load_store(path: str | Path, competitions: Iterable[str] | None = None,
               seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Read the player-match store (read only), optionally filtered."""
    usecols = lambda c: c in STORE_COLS  # noqa: E731
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if competitions is not None:
        df = df[df["competition"].isin(list(competitions))]
    if seasons is not None:
        df = df[df["season"].isin(list(seasons))]
    return df.reset_index(drop=True)


def match_players(src: pd.DataFrame, store: pd.DataFrame, src_name: str = "name",
                  src_team: str = "team") -> pd.DataFrame:
    """Map source (name, team) rows to store player_ids within the same team.

    Tiered: exact full name, then initial+surname, then unique surname. Returns
    ``src`` with ``store_player_id`` and ``match_tier`` columns (NaN = no match).
    """
    roster = store[["team", "player_id", "player_name"]].drop_duplicates()
    roster = roster.assign(full=roster.player_name.map(name_key),
                           ini=roster.player_name.map(initial_key),
                           sur=roster.player_name.map(surname_key))
    lookups = {}
    for tier in ("full", "ini", "sur"):
        g = roster.groupby(["team", tier]).player_id.agg(lambda s: sorted(set(s)))
        lookups[tier] = {k: v[0] for k, v in g.items() if len(v) == 1}
    out_id, out_tier = [], []
    for name, team in zip(src[src_name], src[src_team]):
        found = (None, None)
        for tier, fn in (("full", name_key), ("ini", initial_key), ("sur", surname_key)):
            pid = lookups[tier].get((team, fn(name)))
            if pid is not None:
                found = (pid, tier)
                break
        out_id.append(found[0])
        out_tier.append(found[1])
    return src.assign(store_player_id=out_id, match_tier=out_tier)


def assign_round_by_windows(dates: pd.Series, windows: dict[int, tuple[str, str]]) -> pd.Series:
    """Map ISO dates to fantasy rounds using inclusive [start, end] windows."""
    d = pd.to_datetime(dates)
    out = pd.Series(pd.NA, index=dates.index, dtype="Int64")
    for rnd, (start, end) in windows.items():
        mask = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
        out[mask] = rnd
    return out
