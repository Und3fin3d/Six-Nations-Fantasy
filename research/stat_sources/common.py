"""Shared helpers for external stat-source probes.

Everything here is offline: name normalisation, fixture matching by date and
player-name overlap, and agreement metrics against the repo's player-match store.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STORE = os.environ.get(
    "PLAYER_MATCH_STORE",
    "/tmp/claude-0/-home-user-Six-Nations-Fantasy/f11ffa4a-ba0b-5432-952a-6accc17d3a46/"
    "scratchpad/runs/base/inputs/player_match.csv",
)
STORE_COLS = [
    "date", "competition", "fixture_id", "team", "opponent", "season", "round",
    "player_id", "player_name", "jersey", "started", "minutes", "metres", "tackles",
    "missed_tackles", "tackle_turnover", "available__tackle_turnover", "carries", "runs",
    "defenders_beaten", "clean_breaks", "offload", "passes", "lineouts_won",
    "lineout_steals", "conversion_goals", "penalty_goals", "tries", "try_assists",
    "turnovers_conceded", "penalties_conceded", "home_away", "position",
]


def ascii_lower(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return s.lower()


def surname_key(name: str) -> str:
    """'Louis Bielle-Biarrey' / 'L. Bielle-Biarrey' / 'Bielle-Biarrey' -> 'biellebiarrey'."""
    if not isinstance(name, str) or not name.strip():
        return ""
    parts = ascii_lower(name).replace(".", ". ").split()
    if len(parts) > 1:  # drop forename or initial
        parts = parts[1:]
    return re.sub(r"[^a-z]", "", "".join(parts))


def initial_key(name: str) -> str:
    """'Louis Bielle-Biarrey' -> 'l|biellebiarrey' (same contract as compare_api_official.norm_key)."""
    if not isinstance(name, str) or not name.strip():
        return ""
    parts = ascii_lower(name).split()
    sur = re.sub(r"[^a-z]", "", "".join(parts[1:]) if len(parts) > 1 else parts[0])
    return f"{parts[0][0]}|{sur}"


def load_store(path: str | None = None, competitions=None) -> pd.DataFrame:
    path = path or DEFAULT_STORE
    head = pd.read_csv(path, nrows=0).columns
    cols = [c for c in STORE_COLS if c in head]
    d = pd.read_csv(path, usecols=cols, low_memory=False)
    if competitions is not None:
        d = d[d.competition.isin(competitions)]
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d["sur"] = d.player_name.map(surname_key)
    d["ikey"] = d.player_name.map(initial_key)
    return d


def match_fixtures(ext: pd.DataFrame, store: pd.DataFrame, ext_match_col="match_id",
                   ext_date_col="match_date", ext_name_col="player", tol_days=1,
                   min_overlap=8) -> pd.DataFrame:
    """Map each external match to the store fixture (same date +/- tol) with the
    largest surname overlap. Returns ext_match_id -> fixture_id, overlap."""
    store_by_date: dict = {}
    for (dt, fid), g in store.groupby(["date", "fixture_id"]):
        store_by_date.setdefault(dt, []).append((fid, set(g.sur)))
    out = []
    for mid, g in ext.groupby(ext_match_col):
        dt = pd.to_datetime(g[ext_date_col].iloc[0]).date()
        names = set(g[ext_name_col].map(surname_key))
        best = (None, 0)
        for k in range(-tol_days, tol_days + 1):
            for fid, s in store_by_date.get(dt + pd.Timedelta(days=k), []):
                ov = len(names & s)
                if ov > best[1]:
                    best = (fid, ov)
        out.append({ext_match_col: mid, "fixture_id": best[0], "overlap": best[1],
                    "n_ext": len(names)})
    m = pd.DataFrame(out)
    m.loc[m.overlap < min_overlap, "fixture_id"] = None
    return m


def join_players(ext: pd.DataFrame, store: pd.DataFrame, fmap: pd.DataFrame,
                 ext_match_col="match_id", ext_name_col="player") -> pd.DataFrame:
    """Join external player rows to store rows within the matched fixture.
    Key: surname (+ first initial when the surname is ambiguous inside the fixture)."""
    e = ext.merge(fmap[[ext_match_col, "fixture_id"]], on=ext_match_col, how="left")
    e = e[e.fixture_id.notna()].copy()
    e["fixture_id"] = e.fixture_id.astype(store.fixture_id.dtype)
    e["sur"] = e[ext_name_col].map(surname_key)
    e["ikey"] = e[ext_name_col].map(initial_key)
    s = store[store.fixture_id.isin(e.fixture_id.unique())]
    # unique-surname join first
    su = s.groupby(["fixture_id", "sur"]).filter(lambda g: len(g) == 1)
    eu = e.groupby(["fixture_id", "sur"]).filter(lambda g: len(g) == 1)
    j1 = eu.merge(su, on=["fixture_id", "sur"], suffixes=("_ext", ""))
    # ambiguous surnames: use initial|surname
    rest = e[~e.index.isin(eu.index)]
    j2 = rest.merge(s, on=["fixture_id", "ikey"], suffixes=("_ext", ""))
    j2 = j2.groupby(["fixture_id", "ikey"]).filter(lambda g: len(g) == 1)
    j = pd.concat([j1, j2], ignore_index=True, sort=False)
    return j


def agreement(a: pd.Series, b: pd.Series) -> dict:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    if len(a) < 5:
        return {"n": int(len(a))}
    return {
        "n": int(len(a)),
        "corr": round(float(np.corrcoef(a, b)[0, 1]), 3) if a.std() > 0 and b.std() > 0 else None,
        "exact": round(float((a == b).mean()), 3),
        "mean_ext": round(float(a.mean()), 3),
        "mean_ours": round(float(b.mean()), 3),
        "ratio_sum": round(float(a.sum() / b.sum()), 3) if b.sum() else None,
    }
