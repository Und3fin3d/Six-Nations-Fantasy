from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

import model.ncr.project as NP

DATA = Path("data/ncr")

SCORE_WEIGHTS = {**NP.ATT, **NP.DEF, **NP.DISC}
RESULT_START = "<!-- GW_RESULT_START -->"
RESULT_END = "<!-- GW_RESULT_END -->"
TEAM_FILES = {"Bayes / NCR empirical": "ncr_gw{gw}_squad.csv", "6N champion": "ncr_gw{gw}_squad_champion.csv", "Blend": "ncr_gw{gw}_squad_blend.csv"}


def load_official_actuals(gw: int) -> dict[int, dict]:
    path = f"{DATA}/feeds/players_gw{gw}.json"
    players = json.load(open(path))["Data"]["Value"]["Players"]
    payload_gws = {int(float(p["gameday_id"])) for p in players if p.get("gameday_id") not in (None, "")}
    if payload_gws != {int(gw)}:
        raise ValueError(f"players_gw{gw}.json contains gameday_id {sorted(payload_gws)}, expected GW{gw}")
    return {
        int(float(p["id"])): {
            "points": float(p.get("cur_gd_points") or 0), "status": p.get("player_status") or "",
            "name": p.get("full_name") or p.get("display_name") or str(p["id"]), "team": p.get("team_name") or "",
        }
        for p in players
    }


def _result(rows: list[dict]) -> tuple[pd.DataFrame, float]:
    result = pd.DataFrame(rows)
    return result, float(result["Contribution"].sum())


def _row(player, points: float, multiplier: float, role: str, potm: bool) -> dict:
    return {
        "Player": player.name, "Team": player.team, "Base": points, "Multiplier": multiplier, "Contribution": points * multiplier, "Role": role, "POTM": potm,
    }


def score_squad_official(path: str, actuals: dict[int, dict]) -> tuple[pd.DataFrame, float]:
    roles = {"B": (3.0, "Super sub (bench)"), "P": (0.5, "Super sub (started)")}
    rows = []
    for player in pd.read_csv(path).itertuples(index=False):
        if int(player.id) not in actuals:
            raise ValueError(f"No official GW points row for {player.name} ({int(player.id)})")
        actual = actuals[int(player.id)]
        multiplier, role = (
            (2.0, "Captain") if bool(player.is_capt) else roles.get(actual["status"], (0.0, "Super sub (did not play)")) if bool(player.is_sub) else (1.0, "")
        )
        rows.append(_row(player, float(actual["points"]), multiplier, role, False))
    return _result(rows)


def gameweek_date(gw: int) -> str:
    fixtures = pd.read_csv(f"{DATA}/ncr_fixtures.csv")
    rows = fixtures[pd.to_numeric(fixtures["gameday"], errors="coerce").eq(gw)]
    if rows.empty:
        raise ValueError(f"GW{gw} is absent from ncr_fixtures.csv")
    dates = pd.to_datetime(rows["game_date"]).dt.strftime("%Y-%m-%d").unique()
    if len(dates) != 1:
        raise ValueError(f"GW{gw} has ambiguous fixture dates: {dates.tolist()}")
    return str(dates[0])


def actual_player_points(gw: int) -> pd.DataFrame:
    date = gameweek_date(gw)
    hist = pd.read_csv(f"{DATA}/ncr_player_match.csv")
    played = hist[hist["date"].astype(str).eq(date)].copy()
    if played["fixture_id"].nunique() != 6 or len(played) != 276:
        raise ValueError(f"GW{gw} history is incomplete: {len(played)} rows, {played['fixture_id'].nunique()} fixtures (expected 276 and 6)")
    for column in SCORE_WEIGHTS:
        played[column] = pd.to_numeric(played[column], errors="coerce").fillna(0)
    played["actual_points"] = sum(played[column] * weight for column, weight in SCORE_WEIGHTS.items())
    potm = pd.read_csv(f"{DATA}/ncr_potm.csv")
    potm = potm[potm["date"].astype(str).eq(date)]
    if potm["fixture_id"].nunique() != 6:
        raise ValueError(f"GW{gw} POTM data is incomplete: {potm['fixture_id'].nunique()} fixtures (expected 6)")
    played["potm"] = played["player_id"].astype(int).isin(set(pd.to_numeric(potm["player_id"], errors="coerce").dropna().astype(int)))
    played["actual_points"] += 15 * played["potm"].astype(int)
    return played


def score_squad(path: str, played: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    squad = pd.read_csv(path)
    pool = pd.read_csv(f"{DATA}/ncr_players.csv")
    crosswalk = NP.match_players(pool[pool["id"].astype(int).isin(squad["id"].astype(int))], played)
    by_api_id = played.set_index(played["player_id"].astype(int))
    rows = []
    for player in squad.itertuples(index=False):
        api_id = crosswalk.get(int(player.id))
        if api_id is None or int(api_id) not in by_api_id.index:
            raise ValueError(f"No GW match row for {player.name} ({player.team})")
        actual = by_api_id.loc[int(api_id)]
        if isinstance(actual, pd.DataFrame):
            raise ValueError(f"Multiple GW match rows for {player.name} ({player.team})")
        bench = not bool(actual["started"])
        multiplier, role = (
            (2.0, "Captain")
            if bool(player.is_capt)
            else ((3.0, "Super sub (bench)") if bench else (0.5, "Super sub (started)"))
            if bool(player.is_sub)
            else (1.0, "")
        )
        rows.append(_row(player, float(actual["actual_points"]), multiplier, role, bool(actual["potm"])))
    return _result(rows)


def number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def result_section(gw: int, date: str, selected_label: str, selected: pd.DataFrame, totals: dict[str, float], source: str) -> str:
    lines = [
        RESULT_START, f"## GW{gw} scored result", "", f"**{selected_label}: {number(totals[selected_label])} points**", "", "| Model team | Points |",
        "|---|---:|", *(f"| {label} | **{number(total)}** |" for label, total in totals.items()), "", f"### {selected_label} player contributions", "",
        "| Player | Nation | Base points | Multiplier | Contribution | Role |", "|---|---|---:|---:|---:|---|",
    ]
    lines += [
        f"| {r.Player} | {r.Team} | {number(r.Base)} | {number(r.Multiplier)}x | **{number(r.Contribution)}** | "
        f"{(r.Role + (' + POTM' if r.POTM else '')).strip()} |"
        for r in selected.itertuples(index=False)
    ]
    lines.append("")
    lines.append(
        f"Platform-confirmed result from the gameweek-versioned official GW{gw} fantasy feed, covering the fixtures on {date}."
        if source == "official"
        else f"Scored from the six ingested fixtures on {date}. "
        "This is an **API-observable reconstruction** because the official fantasy feed was unavailable. "
        "Exact scrum-won, own-lineout, lineout-steal and interception allocations are unavailable per player."
    )
    return "\n".join(lines + [RESULT_END])


def update_markdown(path: str, section: str) -> None:
    text = re.sub(rf"\n*{re.escape(RESULT_START)}.*?{re.escape(RESULT_END)}\n?", "\n", open(path).read(), flags=re.DOTALL)
    open(path, "w").write(text.rstrip() + "\n\n" + section + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--update-markdown", action="store_true")
    args = ap.parse_args()
    date = gameweek_date(args.gw)
    squads = {
        label: f"{DATA}/{template.format(gw=args.gw)}" for label, template in TEAM_FILES.items() if os.path.exists(f"{DATA}/{template.format(gw=args.gw)}")
    }
    if os.path.exists(f"{DATA}/feeds/players_gw{args.gw}.json"):
        source, actuals = "official", load_official_actuals(args.gw)
        scored = {label: score_squad_official(path, actuals) for label, path in squads.items()}
    else:
        source, played = "reconstructed", actual_player_points(args.gw)
        scored = {label: score_squad(path, played) for label, path in squads.items()}
    totals = {label: total for label, (_, total) in scored.items()}
    print(f"GW{args.gw} ({date}) [{source}]")
    for label, total in totals.items():
        print(f"  {label:<23} {number(total)}")
    if args.update_markdown:
        for label, path in squads.items():
            update_markdown(path.removesuffix(".csv") + ".md", result_section(args.gw, date, label, scored[label][0], totals, source))
            print(f"  updated {path.removesuffix('.csv')}.md")
