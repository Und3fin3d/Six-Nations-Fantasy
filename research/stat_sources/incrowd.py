"""InCrowd Sports "rugby-union-feeds" (RugbyViz / Opta-fed) probe + downloader.

This is the public JSON feed behind premiershiprugby.com, unitedrugby.com, the
LNR/EPCR/Japan League One match centres. No key, no login.

    fixtures: https://rugby-union-feeds.incrowdsports.com/v1/matches?provider=rugbyviz&season=202501&compId=1068&sort=date
    match:    https://rugby-union-feeds.incrowdsports.com/v1/matches/{id}?provider=rugbyviz&seasonId=202501[&clientId=PRL]

Competition ids (from public open-source clients): Premiership 1011 (clientId PRL),
Prem Cup 1297 (PRL), Top 14 1002, Pro D2 1013, Champions Cup 1008, Challenge Cup 1026,
URC 1068, Japan League One 2074, RFU Championship 1051.

Modes
-----
  python research/stat_sources/incrowd.py fetch --comp 1068 --season 202501 --limit 1 --out DIR
      live download (needs egress to rugby-union-feeds.incrowdsports.com; blocked in
      the 2026-10 cloud session). Writes raw JSON per match + tidy CSV.
  python research/stat_sources/incrowd.py compare --mirror DIR [--store player_match.csv]
      offline: join a tidy per-player table (our own fetch output, or the CSV mirror
      published by github.com/bpcsaund/rugby-analytics data/raw/*_player_stats.csv,
      which is the same feed flattened) to the repo store and report join rate and
      stat agreement.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import agreement, join_players, load_store, match_fixtures  # noqa: E402

BASE = "https://rugby-union-feeds.incrowdsports.com/v1/matches"
COMPS = {
    "prem": (1011, "PRL"), "premcup": (1297, "PRL"), "top14": (1002, ""), "prod2": (1013, ""),
    "champs": (1008, ""), "challenge": (1026, ""), "urc": (1068, ""), "japan": (2074, ""),
    "championship": (1051, ""),
}
UA = {"User-Agent": "six-nations-fantasy research probe (low volume)", "Accept": "application/json"}

# feed key -> repo store column (player level)
FEED_TO_STORE = {
    "metres": "metres", "tackles": "tackles", "missedTackles": "missed_tackles",
    "turnoverWon": "tackle_turnover", "carries": "runs", "defendersBeaten": "defenders_beaten",
    "cleanBreaks": "clean_breaks", "offload": "offload", "passes": "passes",
    "lineoutsWon": "lineouts_won", "conversionGoals": "conversion_goals",
    "penaltyGoals": "penalty_goals", "tries": "tries", "turnoversConceded": "turnovers_conceded",
    "penaltiesConceded": "penalties_conceded", "minutesPlayedTotal": "minutes",
}
# bpcsaund CSV mirror column -> feed key
MIRROR_TO_FEED = {
    "Metres_Gained": "metres", "Tackles_Attempted": "tackles", "Missed_Tackles": "missedTackles",
    "Turnovers_Won": "turnoverWon", "Carries": "carries", "Defenders_Beaten": "defendersBeaten",
    "Clean_Breaks": "cleanBreaks", "Offloads": "offload", "Passes": "passes",
    "Lineouts_Won": "lineoutsWon", "Conversions": "conversionGoals", "Penalty_Goals": "penaltyGoals",
    "Tries": "tries", "Turnovers_Conceded": "turnoversConceded",
    "Penalties_Conceded": "penaltiesConceded", "minutes_played": "minutesPlayedTotal",
    "Kicks_In_Play": "kicksFromHand", "Lineout_Steals": "lineoutSteals",
}


def parse_match(payload: dict) -> pd.DataFrame:
    """Raw match JSON -> one row per player with every scalar stat the feed exposes,
    plus on/off minutes reconstructed from the event log (bench timing)."""
    m = payload.get("data", payload)
    rows = []
    sub_on, sub_off = {}, {}
    for ev in m.get("events") or []:
        pid = ev.get("playerId")
        if pid is None:
            continue
        if ev.get("type") == "Sub On":
            sub_on.setdefault(pid, []).append(ev.get("minute"))
        elif ev.get("type") == "Sub Off":
            sub_off.setdefault(pid, []).append(ev.get("minute"))
    for side in ("homeTeam", "awayTeam"):
        t = m.get(side) or {}
        for p in t.get("players") or []:
            pos = int(p.get("positionId") or 0)
            r = {
                "match_id": m.get("id"), "comp_id": m.get("compId"), "season": m.get("season"),
                "match_date": (m.get("date") or "")[:10], "status": m.get("status"),
                "team": t.get("name"), "home_away": "home" if side == "homeTeam" else "away",
                "player_id": p.get("id"), "player": p.get("known") or p.get("name"),
                "jersey": pos, "position": p.get("position"), "starter": pos <= 15,
                "on_minutes": json.dumps(([0] if pos <= 15 else []) + sub_on.get(p.get("id"), [])),
                "off_minutes": json.dumps(sub_off.get(p.get("id"), [])),
            }
            for k, v in (p.get("stats") or {}).items():
                if isinstance(v, (int, float, str)) or v is None:
                    r[k] = v
            rows.append(r)
    return pd.DataFrame(rows)


def fetch(comp: str, season: str, limit: int, out: str, sleep: float = 1.0) -> None:
    import requests

    comp_id, client = COMPS[comp]
    os.makedirs(out, exist_ok=True)
    fx = requests.get(BASE, params={"provider": "rugbyviz", "season": season, "compId": comp_id,
                                    "sort": "date"}, headers=UA, timeout=30)
    fx.raise_for_status()
    matches = [x for x in fx.json().get("data", []) if (x.get("status") or "").lower() == "result"]
    print(f"{len(matches)} completed matches in comp {comp_id} season {season}")
    frames = []
    for x in matches[:limit]:
        params = {"provider": "rugbyviz", "seasonId": season}
        if client:
            params["clientId"] = client
        r = requests.get(f"{BASE}/{x['id']}", params=params, headers=UA, timeout=30)
        r.raise_for_status()
        Path(out, f"incrowd_{x['id']}.json").write_text(r.text)
        frames.append(parse_match(r.json()))
        time.sleep(sleep)
    if frames:
        tidy = pd.concat(frames, ignore_index=True)
        tidy.to_csv(Path(out, f"incrowd_{comp}_{season}_players.csv"), index=False)
        print(f"wrote {len(tidy)} player rows; stat keys: {sorted(set(tidy.columns) - {'match_id'})}")


def load_mirror(path_glob: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(path_glob)):
        d = pd.read_csv(f, low_memory=False)
        d = d.rename(columns=MIRROR_TO_FEED)
        d["source_file"] = os.path.basename(f)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def compare(mirror_glob: str, store_path: str | None, out_json: str | None) -> dict:
    ext = load_mirror(mirror_glob)
    store = load_store(store_path)
    store = store[store.date >= pd.to_datetime(ext.match_date).min().date() - pd.Timedelta(days=2)]
    fmap = match_fixtures(ext, store)
    j = join_players(ext, store, fmap)
    res = {"ext_rows": len(ext), "ext_matches": int(ext.match_id.nunique()),
           "matched_matches": int(fmap.fixture_id.notna().sum()),
           "joined_rows": len(j), "by_file": {}, "agreement": {}}
    for f, g in ext.groupby("source_file"):
        mm = fmap[fmap.match_id.isin(g.match_id)]
        res["by_file"][f] = {"rows": len(g), "matches": int(g.match_id.nunique()),
                             "matched_matches": int(mm.fixture_id.notna().sum()),
                             "joined_rows": int(j.source_file.eq(f).sum())}
    for feed, ours in FEED_TO_STORE.items():
        if feed in j.columns and ours in j.columns:
            a = j[feed]
            b = j[ours]
            if ours == "tackle_turnover" and "available__tackle_turnover" in j:
                keep = j["available__tackle_turnover"].astype(str).eq("True")
                a, b = a[keep], b[keep]
            res["agreement"][f"{feed}~{ours}"] = agreement(a, b)
    # what the feed has that our store does not
    res["feed_only_cols"] = [c for c in ("kicksFromHand", "lineoutSteals") if c in ext.columns]
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=2, default=str))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--comp", default="urc", choices=sorted(COMPS))
    f.add_argument("--season", default="202501")
    f.add_argument("--limit", type=int, default=1)
    f.add_argument("--out", default="incrowd_out")
    c = sub.add_parser("compare")
    c.add_argument("--mirror", required=True, help="glob of *_player_stats.csv")
    c.add_argument("--store")
    c.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.comp, a.season, a.limit, a.out)
    else:
        print(json.dumps(compare(a.mirror, a.store, a.json), indent=2, default=str))


if __name__ == "__main__":
    main()
