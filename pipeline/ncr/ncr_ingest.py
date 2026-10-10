from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from pipeline.sixn.ingest_6n import (
    STAT_FIELDS,
    TEAM_ID_COLS,
    _flatten_one_team_stats,
    _num,
    derive_minutes,
    player_minutes,
)
from pipeline.sources.rugby_api import RugbyAPI

NCR_DIR = "data/ncr"
DONE = ("Result", "Full Time")

NATIONS = {"New Zealand", "Australia", "South Africa", "Argentina", "Fiji", "Japan",
           "England", "France", "Ireland", "Italy", "Scotland", "Wales"}
TEAM_IDS = {2567, 317, 2717, 2417, 2879, 9017, 1667, 1967, 1817, 2873, 2117, 1517}

INTL_PLAN = ([(1266, s) for s in range(2021, 2027)]
             + [(1296, s) for s in range(2021, 2027)]
             + [(1326, s) for s in (2020, 2022, 2025, 2026)]
             + [(30, s) for s in range(2021, 2027)]
             + [(1272, 2024), (2208, 2021), (2202, 2021), (696, 2026)])
OPEN_PLAN = [(1338, 2025), (18, 2023), (18, 2025), (18, 2026)]
CLUB_PLAN = ([(c, s) for c in (1236, 1230, 1218, 1464, 1470) for s in (2023, 2024, 2025, 2026, 2027)]
             + [(1242, s) for s in range(2022, 2028)]
             + [(2538, 2024), (2538, 2025), (2538, 2026)]
             + [(1260, 2024), (1260, 2025), (1260, 2026), (1260, 2027)]
             + [(1254, 2023), (1254, 2024), (1254, 2025)])
LIVE_COMP = 696


def parse_match(m: dict, keep) -> list[dict]:
    match = m.get("match", {})
    home, away = m.get("home", {}), m.get("away", {})
    if not (home.get("teamsheet") and away.get("teamsheet")):
        return []
    off, on, cards = derive_minutes(m.get("events") or [])
    date = (match.get("date") or "")[:10]
    rows = []
    for side in ("home", "away"):
        opp = "away" if side == "home" else "home"
        team = match.get(f"{side}_team")
        if not keep(team):
            continue
        for p in m[side]["teamsheet"]:
            pid = p["player_id"]
            stats = p.get("match_stats") or {}
            rec = {
                "date": date, "comp_id": match.get("comp_id"), "comp_name": match.get("comp_name"),
                "fixture_id": match.get("id"), "team": team, "team_id": match.get(f"{side}_id"),
                "opponent": match.get(f"{opp}_team"), "opponent_id": match.get(f"{opp}_id"),
                "team_score": match.get(f"{side}_score"), "opp_score": match.get(f"{opp}_score"),
                "player_id": pid, "player_name": p.get("name"), "jersey": p.get("position"),
                "started": not p.get("substitute", False),
                "minutes": player_minutes(pid, not p.get("substitute", False), off, on),
                "yellow_cards": cards[pid]["yellow_cards"], "red_cards": cards[pid]["red_cards"],
                "available__minutes": True, "available__yellow_cards": True,
                "available__red_cards": True,
            }
            for f in STAT_FIELDS:
                present = f in stats
                rec[f] = _num(stats[f]) if present else None
                rec[f"available__{f}"] = present
            rows.append(rec)
    return rows


def parse_team_match(m: dict, keep) -> list[dict]:
    match = m.get("match", {})
    rows = []
    for side in ("home", "away"):
        opp = "away" if side == "home" else "home"
        team = match.get(f"{side}_team")
        if not keep(team):
            continue
        stats = (m.get(side) or {}).get("team_stats")
        if not stats:
            continue
        team_score = _num(match.get(f"{side}_score"))
        opp_score = _num(match.get(f"{opp}_score"))
        result = "W" if team_score > opp_score else "L" if team_score < opp_score else "D"
        rec = {
            "season": match.get("season"),
            "comp_id": match.get("comp_id"),
            "comp_name": match.get("comp_name"),
            "round": match.get("game_week") or match.get("round_id"),
            "fixture_id": match.get("id"),
            "date": (match.get("date") or "")[:10],
            "team": team,
            "team_id": match.get(f"{side}_id"),
            "opponent": match.get(f"{opp}_team"),
            "opponent_id": match.get(f"{opp}_id"),
            "home_away": side,
            "team_score": team_score,
            "opp_score": opp_score,
            "result": result,
            "margin": team_score - opp_score,
            "tries_for": _num(match.get(f"{side}_tries")),
            "tries_against": _num(match.get(f"{opp}_tries")),
            "conv_for": _num(match.get(f"{side}_conversions")),
            "conv_against": _num(match.get(f"{opp}_conversions")),
            "pens_for": _num(match.get(f"{side}_penalties")),
            "pens_against": _num(match.get(f"{opp}_penalties")),
            "dg_for": _num(match.get(f"{side}_drop_goals")),
            "dg_against": _num(match.get(f"{opp}_drop_goals")),
        }
        rec.update(_flatten_one_team_stats(stats))
        rows.append(rec)
    return rows


def sweep(api: RugbyAPI, plan, keep, id_filter=None) -> list[dict]:
    rows = []
    for comp, season in plan:
        try:
            fixtures = api.fixtures(comp, season)
        except Exception:
            continue
        for f in fixtures:
            if f.get("status") not in DONE:
                continue
            if id_filter and not id_filter(f):
                continue
            if not os.path.exists(api._cache_path(f"/match/{f['id']}")):
                continue
            rows += parse_match(api.match(f["id"]), keep)
    return rows


def sweep_team(api: RugbyAPI, plan, keep) -> list[dict]:
    rows = []
    for comp, season in plan:
        try:
            fixtures = api.fixtures(comp, season)
        except Exception:
            continue
        for fixture in fixtures:
            if fixture.get("status") not in DONE:
                continue
            match_path = api._cache_path(f"/match/{fixture['id']}")
            if not os.path.exists(match_path):
                continue
            rows += parse_team_match(api.match(fixture["id"]), keep)
    return rows


def fetch_new_results(api: RugbyAPI, floor: int) -> int:
    fixtures = api.get(f"/fixtures/{LIVE_COMP}/2026", refresh=True).get("results", [])
    todo = [f for f in fixtures
            if f.get("status") in DONE and not os.path.exists(api._cache_path(f"/match/{f['id']}"))]
    print(f"new completed NCR matches to fetch: {len(todo)}  (quota {api.remaining})")
    got = 0
    for f in todo:
        if api.remaining is not None and api.remaining <= floor:
            print(f"quota floor {floor} reached; {len(todo) - got} left")
            break
        try:
            api.match(f["id"]); got += 1
        except RuntimeError as exc:
            if "429" in str(exc):
                time.sleep(8)
            else:
                print(f"  ERR {f['id']}: {str(exc)[:80]}")
    return got


def rebuild(api: RugbyAPI) -> None:
    ncr_teams = NATIONS.__contains__
    rows = sweep(api, INTL_PLAN, ncr_teams)
    rows += sweep(api, OPEN_PLAN, lambda t: True)
    intl = (pd.DataFrame(rows).drop_duplicates(["fixture_id", "player_id", "team"]))
    intl["date"] = pd.to_datetime(intl["date"])
    intl = intl.sort_values("date")
    intl.to_csv(f"{NCR_DIR}/ncr_player_match.csv", index=False)
    print(f"ncr_player_match.csv : {len(intl):>6} rows  {intl.fixture_id.nunique():>4} matches  "
          f"{intl.player_id.nunique()} players  → {intl.date.max().date()}")

    team_rows = sweep_team(api, INTL_PLAN, ncr_teams)
    team_rows += sweep_team(api, OPEN_PLAN, lambda t: True)
    team = pd.DataFrame(team_rows).drop_duplicates(["fixture_id", "team"])
    team["date"] = pd.to_datetime(team["date"])
    stat_cols = sorted(column for column in team.columns if column not in TEAM_ID_COLS)
    team = team.reindex(columns=[column for column in TEAM_ID_COLS if column in team] + stat_cols)
    team = team.sort_values("date")
    team.to_csv(f"{NCR_DIR}/ncr_team_match.csv", index=False)
    print(f"ncr_team_match.csv   : {len(team):>6} rows  {team.fixture_id.nunique():>4} matches  "
          f"→ {team.date.max().date()}")

    club = pd.DataFrame(sweep(api, CLUB_PLAN, lambda t: True))
    club = club.drop_duplicates(["fixture_id", "player_id", "team"])
    club["date"] = pd.to_datetime(club["date"])
    club = club.sort_values("date")
    club.to_csv(f"{NCR_DIR}/club_player_match.csv", index=False)
    print(f"club_player_match.csv: {len(club):>6} rows  {club.fixture_id.nunique():>4} matches  "
          f"{club.player_id.nunique()} players  → {club.date.max().date()}")
    per_comp = club.groupby("comp_name").fixture_id.nunique().sort_values(ascending=False)
    print("  " + "  ".join(f"{k}:{v}" for k, v in per_comp.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="pull new NCR results (spends quota)")
    ap.add_argument("--rebuild", action="store_true", help="re-assemble tables from cache (free)")
    ap.add_argument("--floor", type=int, default=50, help="stop fetching at this quota")
    args = ap.parse_args()
    if not args.fetch and not args.rebuild:
        args.fetch = args.rebuild = True

    api = RugbyAPI(verbose=False, min_interval=0.7)
    if args.fetch:
        n = fetch_new_results(api, args.floor)
        print(f"fetched {n} new matches; quota left {api.remaining}")
    if args.rebuild:
        rebuild(api)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
