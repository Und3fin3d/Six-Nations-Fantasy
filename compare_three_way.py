#!/usr/bin/env python3
"""
Three-way comparison: official fantasy  vs  RapidAPI  vs  RugbyPass
==================================================================
RugbyPass only exposes a rich stat line at the *competition-season* level
(competition_stats), and a thin per-match line (mins/tries/conv/cards) in
match_log. So the three-way is done at the Six Nations *season-aggregate*
level, where all three sources can be compared on the stats that matter:
minutes, tries, assists, metres, defenders beaten, tackles, offloads,
penalties conceded, and the breakdown-steal family (official BS vs API
tackle_turnover vs RugbyPass turnovers_won).

Official & API are summed per player per season from their per-match tables;
RugbyPass is read straight from its season aggregate.

Usage:  python compare_three_way.py [seasons...]   (default 2025)
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from compare_api_official import (OFFICIAL, STAT_MAP, load_api, norm_key,
                                  parse_official)
from rugbypass_batch import NAME_TO_SLUG

BASE = Path(__file__).parent
SLUG_TO_NAME = {v: k for k, v in NAME_TO_SLUG.items()}

# stat  ->  (official col, api col, rugbypass col)   None = source lacks it
TRIPLE = {
    "minutes":      ("Min",  "minutes",            "minutes"),
    "tries":        ("T",    "tries",              "tries"),
    "assists":      ("As",   "try_assists",        "try_assists"),
    "metres":       ("MC",   "metres",             "metres"),
    "def_beaten":   ("DB",   "defenders_beaten",   "defenders_beaten"),
    "tackles":      ("Ta",   "tackles",            "tackles"),
    "offloads":     ("OF",   "offload",            "offloads"),
    "pens_conc":    ("CPen", "penalties_conceded", "penalties_conceded"),
    "breakdown":    ("BS",   "tackle_turnover",    "turnovers_won"),
}


def slug_key(slug: str) -> str:
    name = SLUG_TO_NAME.get(slug)
    if name:
        return norm_key(name)
    parts = slug.split("-")                      # fallback: first|rest
    return f"{parts[0][0]}|{''.join(parts[1:])}" if len(parts) > 1 else slug


def load_rugbypass(seasons) -> pd.DataFrame:
    seasons = {str(s) for s in seasons}
    rows = []
    for f in glob.glob(str(BASE / "players" / "rugbypass_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        slug = d.get("player") or Path(f).stem.replace("rugbypass_", "")
        key = slug_key(slug)
        for e in d.get("competition_stats", []):
            if e.get("competition") != "Six Nations":
                continue
            if str(e.get("season")) not in seasons:
                continue
            rows.append({"season": int(e["season"]), "key": key, "slug": slug,
                         "rp_games": e.get("games", 0),
                         **{f"rp_{c}": e.get(c, 0) for c in
                            ("minutes", "tries", "try_assists", "metres",
                             "defenders_beaten", "tackles", "offloads",
                             "penalties_conceded", "turnovers_won", "points")}})
    return pd.DataFrame(rows)


def agg_official(seasons) -> pd.DataFrame:
    off = pd.concat([parse_official(p, s) for s, p in OFFICIAL.items()
                     if s in seasons and p.exists()], ignore_index=True)
    cols = [c for c in (set(STAT_MAP) | {"Pts"}) if c in off.columns]
    g = (off.groupby(["season", "key"])
            .agg(off_games=("round", "nunique"), name=("name", "first"),
                 team=("team", "first"),
                 **{f"off_{c}": (c, "sum") for c in cols})
            .reset_index())
    return g


def agg_api(seasons) -> pd.DataFrame:
    api = load_api()
    api = api[api.season.isin(seasons)].copy()
    api["key"] = api.player_name.map(norm_key)
    fields = ["minutes", "tries", "try_assists", "conversion_goals",
              "penalty_goals", "metres", "defenders_beaten", "tackles",
              "offload", "penalties_conceded", "tackle_turnover",
              "fantasy_pts_recon"]
    g = (api.groupby(["season", "key"])
            .agg(api_games=("fixture_id", "nunique"),
                 **{f"api_{c}": (c, "sum") for c in fields})
            .reset_index())
    return g


def corr(a, b):
    a, b = a.astype(float), b.astype(float)
    return a.corr(b) if a.std() and b.std() else float("nan")


def main():
    seasons = [int(x) for x in sys.argv[1:]] or [2025]
    off = agg_official(seasons)
    api = agg_api(seasons)
    rp = load_rugbypass(seasons)

    j = off.merge(api, on=["season", "key"], how="inner") \
           .merge(rp, on=["season", "key"], how="inner")
    print("=" * 78)
    print(f"THREE-WAY  seasons={seasons}  |  players in all 3 sources: {len(j)}")
    print(f"  official-only players: {len(off)}   api: {len(api)}   rugbypass: {len(rp)}")
    print("=" * 78)

    # ---- per-stat three-way table -------------------------------------------
    hdr = (f"{'stat':<12}{'OFF tot':>9}{'API tot':>9}{'RP tot':>9}"
           f"{'API/OFF':>9}{'RP/OFF':>9}{'r(O,A)':>8}{'r(O,RP)':>9}{'r(A,RP)':>9}")
    print("\nSIX NATIONS SEASON TOTALS & AGREEMENT  (players present in all 3)")
    print("-" * len(hdr)); print(hdr); print("-" * len(hdr))
    for stat, (oc, ac, rc) in TRIPLE.items():
        o = pd.to_numeric(j[f"off_{oc}"], errors="coerce").fillna(0)
        a = pd.to_numeric(j[f"api_{ac}"], errors="coerce").fillna(0)
        r = pd.to_numeric(j[f"rp_{rc}"], errors="coerce").fillna(0)
        print(f"{stat:<12}{o.sum():>9.0f}{a.sum():>9.0f}{r.sum():>9.0f}"
              f"{a.sum()/o.sum() if o.sum() else float('nan'):>8.2f}x"
              f"{r.sum()/o.sum() if o.sum() else float('nan'):>8.2f}x"
              f"{corr(o,a):>8.2f}{corr(o,r):>9.2f}{corr(a,r):>9.2f}")
    print("-" * len(hdr))
    print("ratios vs official total (1.00 = same volume); r = correlation across players")

    # ---- metres adjudication -------------------------------------------------
    o = pd.to_numeric(j["off_MC"], errors="coerce").fillna(0)
    a = pd.to_numeric(j["api_metres"], errors="coerce").fillna(0)
    r = pd.to_numeric(j["rp_metres"], errors="coerce").fillna(0)
    m = (o > 0) & (a > 0) & (r > 0)
    print("\nMETRES — who matches official? (players with all>0, median ratio to OFFICIAL)")
    print(f"  API/official = {(a[m]/o[m]).median():.2f}    "
          f"RugbyPass/official = {(r[m]/o[m]).median():.2f}    "
          f"API/RugbyPass = {(a[m]/r[m]).median():.2f}")

    # ---- breakdown-steal adjudication ---------------------------------------
    o = pd.to_numeric(j["off_BS"], errors="coerce").fillna(0)
    a = pd.to_numeric(j["api_tackle_turnover"], errors="coerce").fillna(0)
    r = pd.to_numeric(j["rp_turnovers_won"], errors="coerce").fillna(0)
    print("\nBREAKDOWN STEAL — which external stat tracks official BS best?")
    print(f"  official total={o.sum():.0f}  API tackle_turnover={a.sum():.0f} "
          f"(corr {corr(o,a):.2f})   RugbyPass turnovers_won={r.sum():.0f} "
          f"(corr {corr(o,r):.2f})")

    # ---- top metre-getters side by side -------------------------------------
    j["_m"] = pd.to_numeric(j["off_MC"], errors="coerce").fillna(0)
    top = j.sort_values("_m", ascending=False).head(10)
    print("\nTOP METRE-MAKERS — official / API / RugbyPass (season metres):")
    for _, x in top.iterrows():
        print(f"  {str(x['name']):<22} {int(x.season)}  "
              f"OFF={int(x.off_MC):>4}  API={int(x.api_metres):>4}  "
              f"RP={int(x.rp_metres):>4}")

    j.to_csv(BASE / "data" / "three_way_6n.csv", index=False)
    print(f"\n💾  data/three_way_6n.csv  ({len(j)} players)")
    return j


if __name__ == "__main__":
    main()
