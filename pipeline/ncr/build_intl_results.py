from __future__ import annotations

import datetime as dt
import json
import time
import os
import urllib.request

import pandas as pd

CACHE = "data/cache"
URL = "https://api.wr-rims-prod.pulselive.com/rugby/v3/match?startDate={start}&endDate={end}&sort=asc&pageSize=100&page={page}&states=C&language=en"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

SIX = {"England", "France", "Ireland", "Italy", "Scotland", "Wales"}
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


def rows_from_match(m: dict) -> list[dict]:
    teams, scores = m.get("teams", []), m.get("scores", [])
    if m.get("sport") != SENIOR_MENS or len(teams) != 2 or len(scores) != 2:
        return []
    names = [t.get("name") for t in teams]
    is_pair = all(n in SIX for n in names)
    return [
        {
            "date": (m.get("time") or {}).get("label"), "comp": m.get("competition"), "team": names[i], "team_id": teams[i].get("id"), "opponent": names[j],
            "opponent_id": teams[j].get("id"), "team_score": scores[i], "opp_score": scores[j], "margin": scores[i] - scores[j],
            "result": "W" if scores[i] > scores[j] else ("L" if scores[i] < scores[j] else "D"), "rankings_weight": m.get("rankingsWeight"),
            "is_six_nations_pair": is_pair,
        }
        for i, j in ((0, 1), (1, 0))
        if names[i] in SIX
    ]


def _season(date_iso: str) -> int:
    d = dt.date.fromisoformat(date_iso)
    return d.year + 1 if d.month >= 7 else d.year


if __name__ == "__main__":
    start, end = dt.date(2020, 7, 1), dt.date.today()
    chunks = month_chunks(start, end)
    print(f"fetching Pulselive matches {start}…{end} in {len(chunks)} monthly chunks")
    df = pd.DataFrame([row for s, e in chunks for m in fetch_chunk(s, e) for row in rows_from_match(m)])
    df = df.drop_duplicates(subset=["date", "team", "opponent"]).sort_values(["date", "team"]).reset_index(drop=True)
    df.to_csv("data/intl_results.csv", index=False)
    fixtures = lambda g: len({tuple(sorted([t, o])) + (d,) for t, o, d in zip(g.team, g.opponent, g.date)})
    df["_season"] = df["date"].map(_season)
    seen_nations = sorted(SIX & set(df["team"].unique()))
    print(f"\n✅  {len(df)} team-side rows  ({fixtures(df)} distinct intl fixtures kept)")
    print(f"    seasons covered: {sorted(df['_season'].unique().tolist())}")
    print("    distinct fixtures / season:")
    for ssn, cnt in df.groupby("_season").apply(fixtures).items():
        print(f"      {ssn}: {cnt}")
    print(f"    six-nations-pair fixtures (autumn/summer 6N vs 6N): {df[df.is_six_nations_pair].shape[0] // 2}")
    print(f"    6 nations present as `team`: {seen_nations} ({'ALL 6' if len(seen_nations) == 6 else 'MISSING ' + str(sorted(SIX - set(seen_nations)))})")
    print("💾  data/intl_results.csv")
