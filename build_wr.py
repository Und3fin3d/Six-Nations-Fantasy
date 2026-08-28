#!/usr/bin/env python3
"""
World Rugby rankings  —  point-in-time team strength
====================================================
World Rugby publishes its men's ranking through the Pulselive backend, and the
endpoint takes a `date=` parameter that returns the table **as it stood on that
date** (rankings update the Monday after matches, so date == fixture date is the
pre-match table → no leakage). Free, official, exact.

We fetch one snapshot per distinct fixture date in data/api_team_match.csv,
cache each to data/cache/wr_rank_{date}.json, and emit one tidy long table.

    https://api.wr-rims-prod.pulselive.com/rugby/v3/rankings/mru?date=YYYY-MM-DD

Output: data/wr_rankings.csv  (snapshot_date, team, team_abbr, wr_pos, wr_pts,
prev_pos, prev_pts) — join onto the team/player tables on (date, team name).
A team's PIT WR strength for a fixture = the row where snapshot_date == the
fixture date.

Usage:
    python build_wr.py                       # dates from api_team_match.csv
    python build_wr.py --dates 2026-02-09 2026-03-14
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
CACHE = BASE / "data" / "cache"
URL = ("https://api.wr-rims-prod.pulselive.com/rugby/v3/rankings/mru"
       "?date={date}&language=en")
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def fetch_snapshot(date: str) -> dict:
    """Cached GET of the WR ranking as of `date` (YYYY-MM-DD)."""
    cf = CACHE / f"wr_rank_{date}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    req = urllib.request.Request(URL.format(date=date), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    CACHE.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(data))
    time.sleep(0.3)
    print(f"  ↓ WR rankings {date}  ({len(data.get('entries', []))} teams)")
    return data


def snapshot_rows(date: str, data: dict) -> list[dict]:
    rows = []
    for e in data.get("entries", []):
        t = e.get("team", {})
        rows.append({
            "snapshot_date": date,
            "team": t.get("name"),
            "team_abbr": t.get("abbreviation"),
            "wr_pos": e.get("pos"),
            "wr_pts": e.get("pts"),
            "prev_pos": e.get("previousPos"),
            "prev_pts": e.get("previousPts"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=None,
                    help="explicit YYYY-MM-DD dates; default = fixture dates in "
                         "data/api_team_match.csv")
    args = ap.parse_args()

    if args.dates:
        dates = sorted(set(args.dates))
    else:
        tm = pd.read_csv(BASE / "data" / "api_team_match.csv")
        dates = sorted(d for d in tm["date"].dropna().astype(str).unique() if d)

    print(f"fetching WR snapshots for {len(dates)} fixture dates")
    rows = []
    for d in dates:
        rows += snapshot_rows(d, fetch_snapshot(d))

    df = pd.DataFrame(rows)
    out = BASE / "data" / "wr_rankings.csv"
    # MERGE, never clobber: a `--dates` run used to REPLACE the whole table, silently
    # discarding every snapshot it wasn't asked for. Union on (snapshot_date, team).
    if out.exists():
        prev = pd.read_csv(out)
        df = (pd.concat([prev, df], ignore_index=True)
                .drop_duplicates(["snapshot_date", "team"], keep="last")
                .sort_values(["snapshot_date", "wr_pos"]))
    df.to_csv(out, index=False)
    print(f"\n✅  {len(df)} rows, {df.snapshot_date.nunique()} snapshots, "
          f"{df.team.nunique()} teams")
    print(f"💾  {out}")


if __name__ == "__main__":
    main()
