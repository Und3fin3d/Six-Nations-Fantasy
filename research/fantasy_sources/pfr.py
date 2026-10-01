#!/usr/bin/env python3
"""PlayFantasyRugby-platform probe (NZ Rugby / SANZAAR / World Rugby games).

The same white-label platform serves several official games with an identical
public, unauthenticated JSON layout under ``<site>/json/fantasy/``:

  * https://www.playfantasyrugby.com          Super Rugby Pacific (2025-), Rugby Championship (2025-)
  * https://fantasy.rugbyworldcup.com         Rugby World Cup 2023 (and later RWC games)

Files: ``players.json`` (one row per player with ``stats.scores`` = {round: points}
for every completed round, ``selected`` = {round: ownership %}, ``cost`` and, from
2026, ``priceHistory`` = {round: price}), ``rounds.json`` (rounds -> fixtures with
dates and squads), ``squads.json`` and ``player_stats.json`` (season totals only;
there is no per-round stat breakdown).

Because the season-end ``players.json`` keeps every round, one download after the
final round recovers the whole season (points, ownership and price path). Past
seasons are only recoverable from archives (Wayback) or third-party mirrors.

Usage:
  python research/fantasy_sources/pfr.py fetch --site https://www.playfantasyrugby.com --out DIR
  python research/fantasy_sources/pfr.py tidy --players DIR/players.json [--rounds DIR/rounds.json]
        [--squads DIR/squads.json] --out player_rounds.csv
  python research/fantasy_sources/pfr.py join --tidy player_rounds.csv --store player_match.csv
        --competition "Rugby World Cup" --season 2023 [--windows rwc2023]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fantasy_common import assign_round_by_windows, http_json, load_store, match_players  # noqa: E402

FILES = ("players", "rounds", "squads", "player_stats")

# Fantasy round windows when rounds.json is not available (RWC 2023 mirror).
WINDOWS = {
    "rwc2023": {
        1: ("2023-09-08", "2023-09-11"), 2: ("2023-09-14", "2023-09-18"),
        3: ("2023-09-20", "2023-09-25"), 4: ("2023-09-27", "2023-10-02"),
        5: ("2023-10-05", "2023-10-09"), 6: ("2023-10-14", "2023-10-16"),
        7: ("2023-10-20", "2023-10-22"), 8: ("2023-10-27", "2023-10-29"),
    },
}


def fetch(site: str, out: Path) -> dict[str, int]:
    """Download the four public JSON files (4 requests)."""
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name in FILES:
        payload = http_json(f"{site.rstrip('/')}/json/fantasy/{name}.json")
        text = json.dumps(payload, ensure_ascii=False)
        (out / f"{name}.json").write_text(text, encoding="utf-8")
        sizes[name] = len(payload)
    return sizes


def _position(value) -> str:
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return str(value or "")


def tidy_players(players: list[dict], squads: list[dict] | None = None) -> pd.DataFrame:
    """One row per (player, round) with points, ownership and price where known."""
    team_of = {s["id"]: s.get("name") for s in (squads or [])}
    rows = []
    for p in players:
        stats = p.get("stats") or {}
        scores = stats.get("scores") or {}
        selected = p.get("selected") or {}
        prices = p.get("priceHistory") or {}
        rounds = sorted({int(r) for r in list(scores) + list(selected) + list(prices)})
        base = {
            "pfr_player_id": p["id"], "feed_id": p.get("feedId"),
            "name": f"{p.get('firstName') or ''} {p.get('lastName') or ''}".strip(),
            "position": _position(p.get("position")), "squad_id": p.get("squadId"),
            "team": team_of.get(p.get("squadId")), "cost_now": p.get("cost"),
            "status_now": p.get("status"),
        }
        for rnd in rounds:
            key = str(rnd)
            rows.append({**base, "round": rnd,
                         "points": scores.get(key, scores.get(rnd)),
                         "selected_pct": selected.get(key, selected.get(rnd)),
                         "price": prices.get(key, prices.get(rnd))})
    return pd.DataFrame(rows)


def fixtures_from_rounds(rounds: list[dict]) -> pd.DataFrame:
    rows = []
    for r in rounds:
        for t in r.get("tournaments", []):
            for side, opp in (("home", "away"), ("away", "home")):
                rows.append({"round": r["number"], "date": str(t.get("date", ""))[:10],
                             "squad_id": t.get(f"{side}SquadId"),
                             "team": t.get(f"{side}SquadName"),
                             "opponent": t.get(f"{opp}SquadName"), "status": t.get("status")})
    return pd.DataFrame(rows)


def join_store(tidy: pd.DataFrame, store: pd.DataFrame, windows: dict | None = None,
               team_map: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Attach store rows to fantasy (player, round) score rows; return rates."""
    tidy = tidy.copy()
    if team_map:
        tidy["team"] = tidy["team"].map(lambda t: team_map.get(t, t))
    store = store.copy()
    if windows:
        store["fantasy_round"] = assign_round_by_windows(store["date"], windows)
    elif "fantasy_round" not in store:
        raise ValueError("need round windows or a store 'fantasy_round' column")
    players = tidy[["pfr_player_id", "name", "team"]].drop_duplicates()
    players = match_players(players, store)
    scored = tidy[tidy["points"].notna()].merge(
        players[["pfr_player_id", "store_player_id", "match_tier"]], on="pfr_player_id")
    sto = store.dropna(subset=["fantasy_round"]).astype({"fantasy_round": int})
    merged = scored.merge(sto, how="left", left_on=["store_player_id", "round", "team"],
                          right_on=["player_id", "fantasy_round", "team"], suffixes=("", "_store"))
    in_store_players = players["team"].isin(store["team"].unique())
    stats = {
        "fantasy_players": int(len(players)),
        "fantasy_players_on_store_teams": int(in_store_players.sum()),
        "players_matched": int(players["store_player_id"].notna().sum()),
        "player_match_rate": round(float(players.loc[in_store_players, "store_player_id"].notna().mean()), 4),
        "match_tiers": players["match_tier"].value_counts().to_dict(),
        "scored_player_rounds": int(len(scored)),
        "scored_rounds_joined_to_store_row": int(merged["fixture_id"].notna().sum()),
        "player_round_join_rate": round(float(merged["fixture_id"].notna().mean()), 4),
        "store_rows_in_windows": int(len(sto)),
        "store_rows_with_fantasy_points": int(sto.merge(
            merged.dropna(subset=["fixture_id"])[["player_id", "fixture_id"]].drop_duplicates(),
            on=["player_id", "fixture_id"]).shape[0]),
    }
    stats["store_row_coverage"] = round(stats["store_rows_with_fantasy_points"] / max(1, stats["store_rows_in_windows"]), 4)
    j = merged.dropna(subset=["fixture_id"])
    if len(j) > 10:
        stats["corr_points_minutes"] = round(float(j["points"].corr(j["minutes"])), 3)
        stats["corr_points_tries"] = round(float(j["points"].corr(j["tries"])), 3)
        stats["corr_points_tackles"] = round(float(j["points"].corr(j["tackles"])), 3)
    return merged, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--site", default="https://www.playfantasyrugby.com")
    f.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("tidy")
    t.add_argument("--players", type=Path, required=True)
    t.add_argument("--squads", type=Path)
    t.add_argument("--out", type=Path, required=True)
    j = sub.add_parser("join")
    j.add_argument("--tidy", type=Path, required=True)
    j.add_argument("--store", type=Path, required=True)
    j.add_argument("--competition", required=True)
    j.add_argument("--season", type=int, required=True)
    j.add_argument("--windows", choices=sorted(WINDOWS), required=True)
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        print(json.dumps(fetch(args.site, args.out)))
    elif args.cmd == "tidy":
        players = json.loads(args.players.read_text(encoding="utf-8"))
        squads = json.loads(args.squads.read_text(encoding="utf-8")) if args.squads else None
        df = tidy_players(players, squads)
        df.to_csv(args.out, index=False)
        print(f"{len(df)} player-round rows, {df.pfr_player_id.nunique()} players, rounds {sorted(df['round'].unique())}")
    else:
        tidy = pd.read_csv(args.tidy)
        store = load_store(args.store, [args.competition], [args.season])
        _, stats = join_store(tidy, store, WINDOWS[args.windows])
        print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
