from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import urllib.request
from datetime import datetime, timezone

BASE = "https://fantasy.nationschampionshiprugby.com/fantasy/feeds"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Cache-Control": "no-cache", "Pragma": "no-cache", "Entity": "ed0t4n$3!"}
FEEDS = "data/ncr/feeds"
PLAYER_COLS = [
    "id", "display_name", "full_name", "team_name", "team_id", "team_short_code", "hemisphere", "skill", "skill_desc", "value", "sel_percentage",
    "sel_cap_percentage", "is_active", "available_status", "injury_status", "player_status", "cur_gd_points", "min_in_game",
]


def get(path: str) -> bytes:
    req = urllib.request.Request(BASE + path, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def current_gameday(fixtures: list[dict]) -> int:
    current = {int(float(f["gameday"])) for f in fixtures if int(float(f.get("iscurrent") or 0)) == 1}
    if len(current) != 1:
        raise ValueError(f"fixtures must identify exactly one current gameday; got {sorted(current)}")
    return current.pop()


def player_feed_path(gameday: int) -> str:
    return f"/players/players_1_en_{int(gameday)}.json"


def latest_completed_gameday(fixtures: list[dict], now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).date()
    completed = set()
    for fixture in fixtures:
        raw_date = str(fixture["game_date"]).replace("Z", "+00:00")
        if datetime.fromisoformat(raw_date).date() < today:
            completed.add(int(float(fixture["gameday"])))
    if not completed:
        raise ValueError("fixtures do not contain a completed gameweek")
    return max(completed)


def validate_player_gameday(players: list[dict], requested_gw: int) -> None:
    payload_gws = {int(float(p["gameday_id"])) for p in players if p.get("gameday_id") not in (None, "")}
    if payload_gws != {int(requested_gw)}:
        raise ValueError(f"requested GW{requested_gw} player feed but payload contains gameday_id {sorted(payload_gws)}; refusing to promote stale data")


def carryover_gameday(players: list[dict], requested_gw: int, feeds_dir: str = FEEDS) -> int | None:
    signature = {int(float(player["id"])): str(player.get("player_status")) for player in players if player.get("player_status") in ("P", "B")}
    if len(signature) < 250:
        return None
    for path in sorted(glob.glob(f"{feeds_dir}/players_gw*.json")):
        match = re.fullmatch(r"players_gw(\d+)\.json", os.path.basename(path))
        if not match or int(match.group(1)) == int(requested_gw):
            continue
        prior = json.load(open(path))["Data"]["Value"]["Players"]
        prior_signature = {int(float(player["id"])): str(player.get("player_status")) for player in prior if player.get("player_status") in ("P", "B")}
        if signature == prior_signature:
            return int(match.group(1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="archive suffix (default: UTC timestamp), e.g. 'post_gw2'")
    ap.add_argument("--gameday", type=int, default=None, help="override the current gameweek derived from fixtures")
    args = ap.parse_args()

    raw_fixtures = get("/fixtures/fixtures_1_en.json")
    fixtures = json.loads(raw_fixtures)["Data"]["Value"]
    requested_gw = args.gameday or current_gameday(fixtures)
    results_gw = args.gameday or latest_completed_gameday(fixtures)
    raw_players = get(player_feed_path(requested_gw))
    players = json.loads(raw_players)["Data"]["Value"]["Players"]
    validate_player_gameday(players, requested_gw)
    carryover = carryover_gameday(players, requested_gw)
    if carryover is not None:
        raise ValueError(f"GW{requested_gw} exposes the exact GW{carryover} P/B lineup; refusing to promote carryover team sheets")
    if results_gw == requested_gw:
        raw_results, result_players = raw_players, players
    else:
        raw_results = get(player_feed_path(results_gw))
        result_players = json.loads(raw_results)["Data"]["Value"]["Players"]
        validate_player_gameday(result_players, results_gw)

    stamp = args.label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = f"{FEEDS}/players_{stamp}.json"
    for name, raw in (
        (f"players_{stamp}", raw_players), (f"players_gw{requested_gw}", raw_players), (f"players_gw{results_gw}", raw_results),
        ("players_latest", raw_results), ("players_latest_results", raw_results), ("players_cur", raw_players),
    ):
        open(f"{FEEDS}/{name}.json", "wb").write(raw)

    with open("data/ncr/ncr_players.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PLAYER_COLS)
        w.writeheader()
        for p in players:
            w.writerow({c: p.get(c) for c in PLAYER_COLS})

    rows = []
    for f in fixtures:
        a, b = f["participants"]
        rows.append(
            {
                "match_id": f["match_id"], "gameday": f["gameday"], "game_date": f["game_date"], "lock_date": f["start_date"], "home": a["name"],
                "home_id": a["id"], "away": b["name"], "away_id": b["id"], "iscurrent": f["iscurrent"],
            }
        )
    with open("data/ncr/ncr_fixtures.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cur = sorted({r["gameday"] for r in rows if int(float(r["iscurrent"])) == 1})
    named = sum(1 for p in players if p.get("player_status") in ("P", "B"))
    starters = sum(1 for p in players if p.get("player_status") == "P")
    gd_id = {str(int(float(p["gameday_id"]))) for p in players}
    print(f"players: {len(players)}   fixtures: {len(rows)}   current gameday: {cur}")
    print(f"team sheets: {starters} starters + {named - starters} bench = {named} named")
    sheet_state = "NOT POSTED" if named == 0 else "PARTIAL" if named < 250 else "POSTED"
    print(f"current player catalogue: GW{requested_gw}   gameday_id: {sorted(gd_id)}")
    print(f"GW{requested_gw} team sheets: {sheet_state}")
    print(f"latest completed fantasy points: GW{results_gw} → {FEEDS}/players_latest.json")
    print(f"archived raw feed → {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
