#!/usr/bin/env python3
"""Probe for the Six Nations official fantasy platform family.

The Guinness Six Nations game (fantasy.sixnationsrugby.com, identity 600) and
La Grande Mêlée, the LNR-listed Top 14 game run by Midi Olympique
(lagrandemelee.midi-olympique.fr, identity 740), are the same vendor stack:
``X-Access-Key: <identity>@<version>@...`` headers, French field names
(``journeecalendrier``, ``statsjoueur``, ``valeur``) and the same stat taxonomy.

Two access levels:

* ``/v1/public/sportifs`` -- current catalogue (price, ownership, availability).
  No login. Already collected for the 6N game by ``snapshot_fantasy_market.py``.
* ``/v1/private/statsjoueur`` (POST ``{"credentials": {"idj": round, "idf": player,
  "detail": true}}``) -- per-player, per-round detail: minutes, starter/sub flag,
  official fantasy points, price before/after the round and 17 official stats
  (tries, assists, conversions, penalties, drop goals, metres carried, defenders
  beaten, tackles, conceded penalties, 50-22 kicks, kicks recovered, offloads,
  lineout steals, breakdown steals, player of the match, yellow and red cards).
  Needs a logged-in account token (``Authorization: Token ...``). Check the game's
  terms before automating this with a personal account; this module never logs in.

The detail format is parsed by :func:`tidy_detail`; it was validated against a
public mirror of the complete 2026 Six Nations pull (238 players x 5 rounds).

Usage:
  python research/fantasy_sources/sixn_platform.py tidy --players players.json --out rounds.csv
  python research/fantasy_sources/sixn_platform.py compare --tidy rounds.csv \
      --official data/official_player_match.csv --season 2026
  # with your own token, one request per player (about 700 for a Top 14 season):
  FANTASY_TOKEN=... python research/fantasy_sources/sixn_platform.py fetch-detail \
      --base https://lagrandemelee.midi-olympique.fr --access-key '740@<ver>@<uuid>' \
      --round 26 --ids ids.txt --out detail.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fantasy_common import http_json, initial_key  # noqa: E402

STAT_COLUMNS = {
    "Try": "tries", "Assists": "try_assists", "Conversion": "conversions", "Penalty": "penalties",
    "Drop goal": "drop_goals", "Metres carried": "metres_carried", "Defenders beaten": "defenders_beaten",
    "Tackles": "tackles", "Conceded penalty": "penalties_conceded", "Kick 50-22": "kick_50_22",
    "Kicks recovered": "kicks_recovered", "Offloads": "offloads", "Lineout steal": "lineout_steals",
    "Breakdown steal": "breakdown_steals", "Man of the match": "potm", "Yellow cards": "yellow_cards",
    "Red cards": "red_cards",
}


def tidy_detail(players: list[dict]) -> pd.DataFrame:
    """One row per (player, round) from ``searchjoueurs`` rows carrying ``stats.detail``."""
    rows = []
    for p in players:
        stats = p.get("stats") or {}
        # Price path for every round, played or not: valeur_footballeur[k-1] is the
        # price after round k and points_marques[k-1].delta that round's change.
        after = list(stats.get("valeur_footballeur") or [])
        deltas = {int(m["numero"]): float(m.get("delta") or 0) for m in stats.get("points_marques") or []}
        for d in stats.get("detail") or []:
            k = int(d.get("numero"))
            series_after = after[k - 1] if 0 < k <= len(after) else None
            series_before = (round(series_after - deltas[k], 2)
                             if series_after is not None and k in deltas else None)
            row = {
                "platform_player_id": p.get("id"), "feed_id": p.get("idws"),
                "short_name": p.get("nom"), "name": p.get("nomcomplet"),
                "team": p.get("club"), "position_id": p.get("id_position"),
                "position": p.get("position"), "round": d.get("numero"),
                "played": bool(d.get("joue")), "minutes": d.get("minutes"),
                "starter": bool(d.get("titulaire")), "replacement": bool(d.get("remplacant")),
                "points": pd.to_numeric(d.get("points"), errors="coerce"),
                "price_before": d.get("valeuravant") if d.get("valeuravant") is not None else series_before,
                "price_after": d.get("valeurapres") if d.get("valeurapres") is not None else series_after,
                "opponent": (d.get("adversaire") or {}).get("nom"),
                "home": (d.get("club") or {}).get("domicile"), "score": d.get("score"),
            }
            for s in d.get("stats") or []:
                col = STAT_COLUMNS.get(str(s.get("libelle", "")).strip())
                if col:
                    row[col] = s.get("total")
            rows.append(row)
    return pd.DataFrame(rows)


def compare_official(tidy: pd.DataFrame, official: pd.DataFrame, season: int) -> dict:
    """Check the platform detail against the repo's official per-stat workbook rows."""
    off = official[official["season"] == season].copy()
    off["k"] = off["name"].map(initial_key)
    t = tidy[tidy["played"]].copy()
    t["k"] = t["short_name"].map(initial_key)
    m = t.merge(off, left_on=["team", "round", "k"], right_on=["team", "round", "k"], suffixes=("", "_off"))
    out = {"platform_played_rows": int(len(t)), "official_rows": int(len(off)), "joined": int(len(m)),
           "join_rate_platform": round(len(m) / max(1, len(t)), 4)}
    pairs = {"points": "Pts", "minutes": "Min", "tackles": "Ta", "metres_carried": "MC",
             "breakdown_steals": "BS", "potm": "POTM", "tries": "T"}
    for a, b in pairs.items():
        if a in m and b in m:
            x = pd.to_numeric(m[a], errors="coerce")
            y = pd.to_numeric(m[b], errors="coerce")
            ok = x.notna() & y.notna()
            out[f"exact_{a}"] = round(float((abs(x[ok] - y[ok]) < 0.051).mean()), 4)
    return out


def fetch_detail(base: str, access_key: str, round_no: int, ids: list[int], out: Path,
                 pause: float = 0.5) -> int:
    """Per-player detail with the caller's own token (FANTASY_TOKEN). Not run in CI."""
    token = os.environ.get("FANTASY_TOKEN")
    if not token:
        raise SystemExit("FANTASY_TOKEN is not set (use your own account; check the game's terms)")
    headers = {"Authorization": f"Token {token}", "X-Access-Key": access_key,
               "Content-Type": "application/json", "Origin": base, "Referer": base + "/"}
    result = []
    for pid in ids:
        body = json.dumps({"credentials": {"idj": round_no, "idf": pid, "detail": True}}).encode()
        stats = http_json(f"{base}/v1/private/statsjoueur?lg=en", headers=headers, data=body)
        result.append({"id": pid, "stats": stats})
        time.sleep(pause)
    out.write_text(json.dumps(result), encoding="utf-8")
    return len(result)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tidy")
    t.add_argument("--players", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--tidy", type=Path, required=True)
    c.add_argument("--official", type=Path, required=True)
    c.add_argument("--season", type=int, required=True)
    f = sub.add_parser("fetch-detail")
    f.add_argument("--base", required=True)
    f.add_argument("--access-key", required=True)
    f.add_argument("--round", type=int, required=True)
    f.add_argument("--ids", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "tidy":
        df = tidy_detail(json.loads(args.players.read_text(encoding="utf-8")))
        df.to_csv(args.out, index=False)
        print(f"{len(df)} player-round rows ({int(df.played.sum())} played), {df.platform_player_id.nunique()} players")
    elif args.cmd == "compare":
        res = compare_official(pd.read_csv(args.tidy), pd.read_csv(args.official), args.season)
        print(json.dumps(res, indent=2))
    else:
        ids = [int(x) for x in args.ids.read_text().split()]
        print(fetch_detail(args.base, args.access_key, args.round, ids, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
