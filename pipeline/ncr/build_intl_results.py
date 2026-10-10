from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import os
import urllib.request

import pandas as pd

CACHE = "data/cache"
URL = ("https://api.wr-rims-prod.pulselive.com/rugby/v3/match"
       "?startDate={start}&endDate={end}&sort=asc&pageSize=100&page={page}"
       "&states=C&language=en")
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

SIX = {"England", "France", "Ireland", "Italy", "Scotland", "Wales"}
NCR = SIX | {"Argentina", "Australia", "Fiji", "Japan", "New Zealand", "South Africa"}
SENIOR_MENS = "MRU"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def fetch_chunk(start: str, end: str) -> list[dict]:
    cf = f"{CACHE}/wr_match_{start}_{end}.json"
    if os.path.exists(cf):
        return json.load(open(cf))

    first = _get(URL.format(start=start, end=end, page=0))
    time.sleep(0.3)
    num_pages = int(first.get("pageInfo", {}).get("numPages", 1))
    content = list(first.get("content", []))
    for page in range(1, num_pages):
        data = _get(URL.format(start=start, end=end, page=page))
        content += data.get("content", [])
        time.sleep(0.3)

    os.makedirs(CACHE, exist_ok=True)
    open(cf, "w").write(json.dumps(content))
    print(f"  ↓ WR matches {start}…{end}  ({len(content)} items, {num_pages} pages)")
    return content


def month_chunks(start: dt.date, end: dt.date) -> list[tuple[str, str]]:
    chunks = []
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        nxt = dt.date(cur.year + (cur.month // 12), (cur.month % 12) + 1, 1)
        last = nxt - dt.timedelta(days=1)
        chunks.append((max(cur, start).isoformat(), min(last, end).isoformat()))
        cur = nxt
    return chunks


def rows_from_match(
    m: dict,
    target_teams: set[str] = SIX,
    pair_field: str = "is_six_nations_pair",
) -> list[dict]:
    if m.get("sport") != SENIOR_MENS:
        return []
    teams = m.get("teams", [])
    scores = m.get("scores", [])
    if len(teams) != 2 or len(scores) != 2:
        return []
    names = [t.get("name") for t in teams]
    if not any(n in target_teams for n in names):
        return []

    date = (m.get("time") or {}).get("label")
    comp = m.get("competition")
    rw = m.get("rankingsWeight")
    is_pair = all(n in target_teams for n in names)

    out = []
    for i, j in ((0, 1), (1, 0)):
        if names[i] not in target_teams:
            continue
        ts, os_ = scores[i], scores[j]
        margin = ts - os_
        result = "W" if margin > 0 else ("L" if margin < 0 else "D")
        out.append({
            "date": date,
            "comp": comp,
            "team": names[i],
            "team_id": teams[i].get("id"),
            "opponent": names[j],
            "opponent_id": teams[j].get("id"),
            "team_score": ts,
            "opp_score": os_,
            "margin": margin,
            "result": result,
            "rankings_weight": rw,
            pair_field: is_pair,
        })
    return out


def _season(date_iso: str) -> int:
    d = dt.date.fromisoformat(date_iso)
    return d.year + 1 if d.month >= 7 else d.year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-07-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    chunks = month_chunks(start, end)
    print(f"fetching Pulselive matches {start}…{end} in {len(chunks)} monthly chunks")

    rows: list[dict] = []
    ncr_rows: list[dict] = []
    for s, e in chunks:
        for m in fetch_chunk(s, e):
            rows += rows_from_match(m)
            ncr_rows += rows_from_match(m, NCR, "is_ncr_pair")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["date", "team", "opponent"])
        df = df.sort_values(["date", "team"]).reset_index(drop=True)

    out = "data/intl_results.csv"
    df.to_csv(out, index=False)

    ncr_df = pd.DataFrame(ncr_rows)
    if not ncr_df.empty:
        ncr_df = (ncr_df.drop_duplicates(subset=["date", "team", "opponent"])
                         .sort_values(["date", "team"])
                         .reset_index(drop=True))
    ncr_out = "data/ncr/ncr_intl_results.csv"
    ncr_df.to_csv(ncr_out, index=False)

    n_matches = df["is_six_nations_pair"].notna().sum()
    distinct_matches = df.drop_duplicates(
        subset=["date", "team", "opponent"]).shape[0] if not df.empty else 0
    n_unique_fixtures = 0
    seen = set()
    for r in df.itertuples(index=False):
        key = tuple(sorted([r.team, r.opponent])) + (r.date,)
        if key not in seen:
            seen.add(key)
            n_unique_fixtures += 1

    df["_season"] = df["date"].map(_season)
    per_season = df.drop_duplicates(
        subset=["date", "team", "opponent"]).groupby("_season").apply(
        lambda g: len({tuple(sorted([t, o])) + (d,)
                       for t, o, d in zip(g.team, g.opponent, g.date)}))
    seen_nations = sorted(SIX & set(df["team"].unique()))
    n_pair = df[df.is_six_nations_pair].drop_duplicates(
        subset=["date", "team", "opponent"]).shape[0] // 2

    print(f"\n✅  {len(df)} team-side rows  ({n_unique_fixtures} distinct intl fixtures kept)")
    print(f"    seasons covered: {sorted(df['_season'].unique().tolist())}")
    print("    distinct fixtures / season:")
    for ssn, cnt in per_season.items():
        print(f"      {ssn}: {cnt}")
    print(f"    six-nations-pair fixtures (autumn/summer 6N vs 6N): {n_pair}")
    print(f"    6 nations present as `team`: {seen_nations} "
          f"({'ALL 6' if len(seen_nations) == 6 else 'MISSING ' + str(sorted(SIX - set(seen_nations)))})")
    print(f"💾  {out}")
    print(f"💾  {ncr_out}  ({len(ncr_df)} team-side rows, all 12 NCR nations)")


if __name__ == "__main__":
    main()
