#!/usr/bin/env python3
"""
Official fantasy labels  →  data/official_player_match.csv  (0 API spend)
========================================================================
Consolidates the manual official xlsx files into one long, model-ready label
table — one row per (season, round, team, player) — carrying the official
fantasy `Pts` label, the fantasy-only categories the API can't see
(50-22, KR, LS, POTM, SW) and the directly comparable components.

Seasons:
  • 2025, 2026 — reuse parse_official() from compare_api_official (dynamic
    Min+Pts header detection over stacked per-team blocks).
  • 2023      — DIFFERENT layout: 'Round N Data' (raw components incl. the
    Game Total = Pts) + 'Round N Points' (per-category points) split sheets,
    Rounds 1-4 ONLY (no R5). Handled by parse_official_2023() below, which
    reads the 'Round N Data' sheet (header on row index 8), assigns each
    player's team via the 'Player List' sheet, and maps the 2023 column labels
    onto the same names parse_official produces.

Output columns (superset; absent ones are simply missing per season):
    season, round, team, name, key,
    Min, T, As, C, Pen, DG, MC, Ta, OF, DB, CPen, BS, YC, RC, Pts,
    50-22, KR, LS, POTM, SW

Usage:  python build_official.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from compare_api_official import (API_BLIND, OFFICIAL, STAT_MAP, norm_key,
                                  parse_official)

BASE = Path(__file__).parent

# 2023 'Round N Data' header label  ->  canonical parse_official column name.
# Anything not listed (Nation, Win/Loss, Dominant Tackles, Line Breaks, MoM,
# Game Total, Total Points) is handled separately or dropped.
COL_MAP_2023 = {
    "Time Played":    "Min",
    "Tackles":        "Ta",
    "Metres Carried": "MC",
    "50-22":          "50-22",
    "Lineout Steal":  "LS",
    "Breakdown Steal": "BS",
    "Try":            "T",
    "Assists":        "As",
    "Conversion":     "C",
    "Penalty":        "Pen",
    "Drop Goal":      "DG",
    "Yellow Card":    "YC",
    "Red Card":       "RC",
    "Game Total":     "Pts",
}
# numeric columns to coerce (blank official cell = 0 events)
NUMERIC_OUT = sorted(set(STAT_MAP) | set(API_BLIND) | {"Pts"})


def _player_nation_map(xl: pd.ExcelFile) -> dict:
    """Build {norm_key(name): nation} from the 2023 'Player List' sheet."""
    pl = pd.read_excel(xl, sheet_name="Player List", header=None)
    pl = pl.dropna(subset=[0])
    out = {}
    for _, row in pl.iterrows():
        name = row[0]
        if not isinstance(name, str) or not name.strip():
            continue
        out[norm_key(name)] = row[1] if isinstance(row[1], str) else ""
    return out


def parse_official_2023(path: Path, season: int = 2023) -> pd.DataFrame:
    """Parse the 2023 split-sheet layout (Rounds 1-4) -> tidy long frame.

    Reads each 'Round N Data' sheet (the component counts + Game Total = Pts),
    finds the header row dynamically (the one whose first cell == 'Name'),
    assigns team via the 'Player List' name->nation map, and renames the 2023
    labels to the parse_official column names.
    """
    xl = pd.ExcelFile(path)
    nation = _player_nation_map(xl)
    out = []
    for r in (1, 2, 3, 4):
        sheet = f"Round {r} Data"
        if sheet not in xl.sheet_names:
            continue
        raw = pd.read_excel(xl, sheet_name=sheet, header=None)
        # locate header row: first cell == 'Name'
        hdr_idx = None
        for i in range(len(raw)):
            if str(raw.iloc[i, 0]).strip() == "Name":
                hdr_idx = i
                break
        if hdr_idx is None:
            continue
        labels = [str(v) for v in raw.iloc[hdr_idx].values]
        for i in range(hdr_idx + 1, len(raw)):
            vals = list(raw.iloc[i].values)
            cell0 = vals[0]
            if not isinstance(cell0, str) or not cell0.strip():
                continue
            name = cell0.strip()
            key = norm_key(name)
            rec = {"season": season, "round": r, "team": nation.get(key, ""),
                   "name": name, "key": key}
            for lab, v in zip(labels, vals):
                col = COL_MAP_2023.get(lab)
                if col:
                    rec[col] = v
            # MoM column (man of the match) -> POTM blind-spot category
            if "MoM" in labels:
                mom = vals[labels.index("MoM")]
                rec["POTM"] = mom
            out.append(rec)
    df = pd.DataFrame(out)
    for lab in NUMERIC_OUT:
        if lab in df.columns:
            df[lab] = pd.to_numeric(df[lab], errors="coerce").fillna(0)
    return df


def build() -> pd.DataFrame:
    frames = []
    for season, path in sorted(OFFICIAL.items()):
        if not path.exists():
            print(f"⚠️  missing {path} (season {season}) — skipped")
            continue
        if season == 2023:
            df = parse_official_2023(path, season)
        else:
            df = parse_official(path, season)
        if df.empty:
            print(f"⚠️  season {season} parsed 0 rows — skipped")
            continue
        print(f"season {season}: {len(df)} rows, "
              f"rounds={sorted(int(x) for x in df['round'].unique())}, "
              f"teams={df['team'].nunique()}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def main():
    out_dir = BASE / "data"
    out_dir.mkdir(exist_ok=True)
    df = build()
    path = out_dir / "official_player_match.csv"
    df.to_csv(path, index=False)
    print(f"\n💾  {path}")
    print(f"   total rows: {len(df)}")
    print(f"   seasons present: {sorted(df['season'].unique())}")
    for s in sorted(df["season"].unique()):
        sub = df[df["season"] == s]
        print(f"     {int(s)}: rows={len(sub)}  "
              f"rounds={sorted(int(x) for x in sub['round'].unique())}")
    miss = 100.0 * df["key"].astype(str).eq("").mean()
    print(f"   blank key: {miss:.2f}%")


if __name__ == "__main__":
    main()
