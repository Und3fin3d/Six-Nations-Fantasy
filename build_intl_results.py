#!/usr/bin/env python3
"""
International results  —  free, widened H2H / FORM pool (Pulselive)
==================================================================
World Rugby's public Pulselive backend exposes every completed senior men's
international through the same `/match` endpoint we already use for rankings.
This pulls 2020-07 → today, keeps matches involving a Six Nations side, and
emits ONE ROW PER TEAM-SIDE (mirror, like data/api_team_match.csv) so the
international result pool can deepen the H2H feature without touching the paid
rugby-live-data quota.

    GET https://api.wr-rims-prod.pulselive.com/rugby/v3/match
        ?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&sort=asc
        &pageSize=100&page=0..numPages-1&states=C&language=en

Each content item:
    {matchId, sport, teams:[{id,name}], scores:[a,b], status:"C",
     competition, time:{label:"YYYY-MM-DD"}, rankingsWeight, venue}

CRITICAL FILTERS
  • sport == "MRU"  — senior men's XV only. Women's (WRU), U20 (JMU) and
    sevens (MRS/WRS) reuse the SAME team names ("England", "France", …) so
    filtering on name alone would silently merge women's/U20 results.
  • the Six Nations output keeps sides from the six northern teams.
  • the NCR output keeps sides from all 12 Nations Championship teams.

Pulselive team ids (England=34, …) are a DIFFERENT id-space from
rugby-live-data (England=1667, …). We store the Pulselive id for traceability,
but any join back onto api_team_match.csv must be by team NAME.

Output: data/intl_results.csv
    date, comp, team, team_id, opponent, opponent_id, team_score, opp_score,
    margin, result(W/L/D), rankings_weight, is_six_nations_pair

Additional output: data/ncr/ncr_intl_results.csv
    The same team-side grain for all 12 Nations Championship teams, with
    `is_ncr_pair` in place of `is_six_nations_pair`.

Usage:
    python build_intl_results.py
    python build_intl_results.py --start 2020-07-01 --end 2026-06-15
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
CACHE = BASE / "data" / "cache"
URL = ("https://api.wr-rims-prod.pulselive.com/rugby/v3/match"
       "?startDate={start}&endDate={end}&sort=asc&pageSize=100&page={page}"
       "&states=C&language=en")
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

SIX = {"England", "France", "Ireland", "Italy", "Scotland", "Wales"}
NCR = SIX | {"Argentina", "Australia", "Fiji", "Japan", "New Zealand", "South Africa"}
SENIOR_MENS = "MRU"   # senior men's XV; exclude WRU/JMU/MRS/WRS


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def fetch_chunk(start: str, end: str) -> list[dict]:
    """Cached, paginated GET of all completed matches in [start, end].

    Caches the FULL combined content list to data/cache/wr_match_{start}_{end}.json
    so re-runs cost zero network. Returns the raw content items."""
    cf = CACHE / f"wr_match_{start}_{end}.json"
    if cf.exists():
        return json.loads(cf.read_text())

    first = _get(URL.format(start=start, end=end, page=0))
    time.sleep(0.3)
    num_pages = int(first.get("pageInfo", {}).get("numPages", 1))
    content = list(first.get("content", []))
    for page in range(1, num_pages):
        data = _get(URL.format(start=start, end=end, page=page))
        content += data.get("content", [])
        time.sleep(0.3)

    CACHE.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(content))
    print(f"  ↓ WR matches {start}…{end}  ({len(content)} items, {num_pages} pages)")
    return content


def month_chunks(start: dt.date, end: dt.date) -> list[tuple[str, str]]:
    """Calendar-month [first, last] chunks covering [start, end]. Month-sized
    keeps each page set tiny and the cache files stable across re-runs."""
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
    """Two mirrored rows (one per side) for a senior men's match where at least
    one team is a 6N side. Only sides whose `team` is in SIX are emitted."""
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
    """6N-style season label: a match in Jul..Dec belongs to the NEXT calendar
    season's autumn block; Jan..Jun is that calendar year. We use a simple
    rugby-year heuristic so 'matches/season' is meaningful."""
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
        # dedup any overlap (a match can only fall in one month, but be safe)
        df = df.drop_duplicates(subset=["date", "team", "opponent"])
        df = df.sort_values(["date", "team"]).reset_index(drop=True)

    out = BASE / "data" / "intl_results.csv"
    df.to_csv(out, index=False)

    ncr_df = pd.DataFrame(ncr_rows)
    if not ncr_df.empty:
        ncr_df = (ncr_df.drop_duplicates(subset=["date", "team", "opponent"])
                         .sort_values(["date", "team"])
                         .reset_index(drop=True))
    ncr_out = BASE / "data" / "ncr" / "ncr_intl_results.csv"
    ncr_out.parent.mkdir(parents=True, exist_ok=True)
    ncr_df.to_csv(ncr_out, index=False)

    # ---- report ----
    n_matches = df["is_six_nations_pair"].notna().sum()        # = total sides emitted
    distinct_matches = df.drop_duplicates(
        subset=["date", "team", "opponent"]).shape[0] if not df.empty else 0
    # a "match" counted once: a 6N-pair match yields 2 sides; vs-non-6N yields 1
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
        subset=["date", "team", "opponent"]).shape[0] // 2  # each pair = 2 sides

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
