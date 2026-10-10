from __future__ import annotations

import argparse
import json
import time
import os
import urllib.request

import pandas as pd

CACHE = "data/cache"
URL = ("https://api.wr-rims-prod.pulselive.com/rugby/v3/rankings/mru"
       "?date={date}&language=en")
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def fetch_snapshot(date: str) -> dict:
    cf = f"{CACHE}/wr_rank_{date}.json"
    if os.path.exists(cf):
        return json.load(open(cf))
    req = urllib.request.Request(URL.format(date=date), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    os.makedirs(CACHE, exist_ok=True)
    open(cf, "w").write(json.dumps(data))
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
        tm = pd.read_csv("data/api_team_match.csv")
        dates = sorted(d for d in tm["date"].dropna().astype(str).unique() if d)

    print(f"fetching WR snapshots for {len(dates)} fixture dates")
    rows = []
    for d in dates:
        rows += snapshot_rows(d, fetch_snapshot(d))

    df = pd.DataFrame(rows)
    out = "data/wr_rankings.csv"
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
