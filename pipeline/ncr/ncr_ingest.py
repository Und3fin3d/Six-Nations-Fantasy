from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from pipeline.sixn.ingest_6n import STAT_FIELDS, TEAM_ID_COLS, _flatten_one_team_stats, _num, derive_minutes, player_minutes
from pipeline.sources.rugby_api import RugbyAPI

NCR_DIR = "data/ncr"
DONE = ("Result", "Full Time")

NATIONS = {"New Zealand", "Australia", "South Africa", "Argentina", "Fiji", "Japan", "England", "France", "Ireland", "Italy", "Scotland", "Wales"}

INTL_PLAN = (
    [(1266, s) for s in range(2021, 2027)]
    + [(1296, s) for s in range(2021, 2027)]
    + [(1326, s) for s in (2020, 2022, 2025, 2026)]
    + [(30, s) for s in range(2021, 2027)]
    + [(1272, 2024), (2208, 2021), (2202, 2021), (696, 2026)]
)
OPEN_PLAN = [(1338, 2025), (18, 2023), (18, 2025), (18, 2026)]
CLUB_PLAN = (
    [(c, s) for c in (1236, 1230, 1218, 1464, 1470) for s in (2023, 2024, 2025, 2026, 2027)]
    + [(1242, s) for s in range(2022, 2028)]
    + [(2538, 2024), (2538, 2025), (2538, 2026)]
    + [(1260, 2024), (1260, 2025), (1260, 2026), (1260, 2027)]
    + [(1254, 2023), (1254, 2024), (1254, 2025)]
)
LIVE_COMP = 696


def parse_match(m: dict, keep) -> list[dict]:
    match = m.get("match", {})
    home, away = m.get("home", {}), m.get("away", {})
    if not (home.get("teamsheet") and away.get("teamsheet")):
        return []
    off, on, cards = derive_minutes(m.get("events") or [])
    date = (match.get("date") or "")[:10]
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        team = match.get(f"{side}_team")
        if not keep(team):
            continue
        for p in m[side]["teamsheet"]:
            pid, stats = p["player_id"], p.get("match_stats") or {}
            rows.append(
                {
                    "date": date,
                    "comp_id": match.get("comp_id"),
                    "comp_name": match.get("comp_name"),
                    "fixture_id": match.get("id"),
                    "team": team,
                    "team_id": match.get(f"{side}_id"),
                    "opponent": match.get(f"{opp}_team"),
                    "opponent_id": match.get(f"{opp}_id"),
                    "team_score": match.get(f"{side}_score"),
                    "opp_score": match.get(f"{opp}_score"),
                    "player_id": pid,
                    "player_name": p.get("name"),
                    "jersey": p.get("position"),
                    "started": not p.get("substitute", False),
                    "minutes": player_minutes(pid, not p.get("substitute", False), off, on),
                    "yellow_cards": cards[pid]["yellow_cards"],
                    "red_cards": cards[pid]["red_cards"],
                    "available__minutes": True,
                    "available__yellow_cards": True,
                    "available__red_cards": True,
                    **{k: v for f in STAT_FIELDS for k, v in ((f, _num(stats[f]) if f in stats else None), (f"available__{f}", f in stats))},
                }
            )
    return rows


def parse_team_match(m: dict, keep) -> list[dict]:
    match = m.get("match", {})
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        team = match.get(f"{side}_team")
        if not keep(team):
            continue
        stats = (m.get(side) or {}).get("team_stats")
        if not stats:
            continue
        team_score, opp_score = _num(match.get(f"{side}_score")), _num(match.get(f"{opp}_score"))
        result = "W" if team_score > opp_score else "L" if team_score < opp_score else "D"
        rows.append(
            {
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
                **_flatten_one_team_stats(stats),
            }
        )
    return rows


def sweep(api: RugbyAPI, plan, keep, parse=parse_match) -> list[dict]:
    rows = []
    for comp, season in plan:
        try:
            fixtures = api.fixtures(comp, season)
        except Exception:
            continue
        for f in fixtures:
            if f.get("status") in DONE and os.path.exists(api._cache_path(f"/match/{f['id']}")):
                rows += parse(api.match(f["id"]), keep)
    return rows


def fetch_new_results(api: RugbyAPI, floor: int) -> int:
    fixtures = api.get(f"/fixtures/{LIVE_COMP}/2026", refresh=True).get("results", [])
    todo = [f for f in fixtures if f.get("status") in DONE and not os.path.exists(api._cache_path(f"/match/{f['id']}"))]
    print(f"new completed NCR matches to fetch: {len(todo)}  (quota {api.remaining})")
    got = 0
    for f in todo:
        if api.remaining is not None and api.remaining <= floor:
            print(f"quota floor {floor} reached; {len(todo) - got} left")
            break
        try:
            api.match(f["id"])
            got += 1
        except RuntimeError as exc:
            if "429" in str(exc):
                time.sleep(8)
            else:
                print(f"  ERR {f['id']}: {str(exc)[:80]}")
    return got


def _save(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    frame.to_csv(f"{NCR_DIR}/{name}", index=False)
    return frame


def rebuild(api: RugbyAPI) -> None:
    every = lambda t: True
    intl = _save(
        pd.DataFrame(sweep(api, INTL_PLAN, NATIONS.__contains__) + sweep(api, OPEN_PLAN, every)).drop_duplicates(["fixture_id", "player_id", "team"]),
        "ncr_player_match.csv",
    )
    print(
        f"ncr_player_match.csv : {len(intl):>6} rows  {intl.fixture_id.nunique():>4} matches  {intl.player_id.nunique()} players  "
        f"→ {intl.date.max().date()}"
    )
    team = pd.DataFrame(sweep(api, INTL_PLAN, NATIONS.__contains__, parse_team_match) + sweep(api, OPEN_PLAN, every, parse_team_match)).drop_duplicates(
        ["fixture_id", "team"]
    )
    team = _save(team.reindex(columns=[c for c in TEAM_ID_COLS if c in team] + sorted(c for c in team.columns if c not in TEAM_ID_COLS)), "ncr_team_match.csv")
    print(f"ncr_team_match.csv   : {len(team):>6} rows  {team.fixture_id.nunique():>4} matches  → {team.date.max().date()}")
    club = _save(pd.DataFrame(sweep(api, CLUB_PLAN, every)).drop_duplicates(["fixture_id", "player_id", "team"]), "club_player_match.csv")
    print(
        f"club_player_match.csv: {len(club):>6} rows  {club.fixture_id.nunique():>4} matches  {club.player_id.nunique()} players  "
        f"→ {club.date.max().date()}"
    )
    print("  " + "  ".join(f"{k}:{v}" for k, v in club.groupby("comp_name").fixture_id.nunique().sort_values(ascending=False).items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--floor", type=int, default=50)
    args = ap.parse_args()
    api = RugbyAPI(verbose=False, min_interval=0.7)
    if not args.rebuild:
        print(f"fetched {fetch_new_results(api, args.floor)} new matches; quota left {api.remaining}")
    rebuild(api)
