#!/usr/bin/env python3
"""Score saved NCR squads from the ingested match-stat history.

The official fantasy player feed has not reliably rolled from one completed
gameday to the next. This scorer therefore reconstructs the categories exposed
by the match API and joins the separately verified player-of-the-match awards.
Categories that cannot be assigned to an individual from the API are called out
in the generated Markdown rather than estimated silently.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

import model.ncr_project as NP

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ncr"

SCORE_WEIGHTS = {**NP.ATT, **NP.DEF, **NP.DISC}
RESULT_START = "<!-- GW_RESULT_START -->"
RESULT_END = "<!-- GW_RESULT_END -->"
TEAM_FILES = {
    "Bayes / NCR empirical": "ncr_gw{gw}_squad.csv",
    "6N champion": "ncr_gw{gw}_squad_champion.csv",
    "Blend": "ncr_gw{gw}_squad_blend.csv",
}


def load_official_actuals(gw: int) -> dict[int, dict]:
    """Load platform points/status from the feed explicitly versioned for ``gw``."""
    path = DATA / "feeds" / f"players_gw{gw}.json"
    players = json.loads(path.read_text())["Data"]["Value"]["Players"]
    payload_gws = {
        int(float(p["gameday_id"]))
        for p in players
        if p.get("gameday_id") not in (None, "")
    }
    if payload_gws != {int(gw)}:
        raise ValueError(
            f"{path.name} contains gameday_id {sorted(payload_gws)}, expected GW{gw}"
        )
    return {
        int(float(p["id"])): {
            "points": float(p.get("cur_gd_points") or 0),
            "status": p.get("player_status") or "",
            "name": p.get("full_name") or p.get("display_name") or str(p["id"]),
            "team": p.get("team_name") or "",
        }
        for p in players
    }


def score_squad_official(path: Path, actuals: dict[int, dict]) -> tuple[pd.DataFrame, float]:
    """Score a saved squad using exact platform points and that GW's P/B status."""
    squad = pd.read_csv(path)
    rows = []
    for player in squad.itertuples(index=False):
        fantasy_id = int(player.id)
        if fantasy_id not in actuals:
            raise ValueError(f"No official GW points row for {player.name} ({fantasy_id})")
        actual = actuals[fantasy_id]
        points = float(actual["points"])
        if bool(player.is_capt):
            multiplier, role = 2.0, "Captain"
        elif bool(player.is_sub):
            if actual["status"] == "B":
                multiplier, role = 3.0, "Super sub (bench)"
            elif actual["status"] == "P":
                multiplier, role = 0.5, "Super sub (started)"
            else:
                multiplier, role = 0.0, "Super sub (did not play)"
        else:
            multiplier, role = 1.0, ""
        rows.append({
            "Player": player.name,
            "Team": player.team,
            "Base": points,
            "Multiplier": multiplier,
            "Contribution": points * multiplier,
            "Role": role,
            "POTM": False,
        })
    result = pd.DataFrame(rows)
    return result, float(result["Contribution"].sum())


def gameweek_date(gw: int) -> str:
    fixtures = pd.read_csv(DATA / "ncr_fixtures.csv")
    rows = fixtures[pd.to_numeric(fixtures["gameday"], errors="coerce").eq(gw)]
    if rows.empty:
        raise ValueError(f"GW{gw} is absent from ncr_fixtures.csv")
    dates = pd.to_datetime(rows["game_date"]).dt.strftime("%Y-%m-%d").unique()
    if len(dates) != 1:
        raise ValueError(f"GW{gw} has ambiguous fixture dates: {dates.tolist()}")
    return str(dates[0])


def actual_player_points(gw: int) -> tuple[pd.DataFrame, str]:
    date = gameweek_date(gw)
    hist = pd.read_csv(DATA / "ncr_player_match.csv")
    played = hist[hist["date"].astype(str).eq(date)].copy()
    if played["fixture_id"].nunique() != 6 or len(played) != 276:
        raise ValueError(
            f"GW{gw} history is incomplete: {len(played)} rows, "
            f"{played['fixture_id'].nunique()} fixtures (expected 276 and 6)"
        )
    for column in SCORE_WEIGHTS:
        played[column] = pd.to_numeric(played[column], errors="coerce").fillna(0)
    played["actual_points"] = sum(
        played[column] * weight for column, weight in SCORE_WEIGHTS.items()
    )

    potm_path = DATA / "ncr_potm.csv"
    potm = pd.read_csv(potm_path)
    potm_for_date = potm[potm["date"].astype(str).eq(date)]
    if potm_for_date["fixture_id"].nunique() != 6:
        raise ValueError(
            f"GW{gw} POTM data is incomplete: "
            f"{potm_for_date['fixture_id'].nunique()} fixtures (expected 6)"
        )
    potm_ids = set(
        pd.to_numeric(potm_for_date["player_id"], errors="coerce")
        .dropna().astype(int)
    )
    played["potm"] = played["player_id"].astype(int).isin(potm_ids)
    played["actual_points"] += 15 * played["potm"].astype(int)
    return played, date


def score_squad(path: Path, played: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    squad = pd.read_csv(path)
    pool = pd.read_csv(DATA / "ncr_players.csv")
    wanted = pool[pool["id"].astype(int).isin(squad["id"].astype(int))]
    crosswalk = NP.match_players(wanted, played)
    by_api_id = played.set_index(played["player_id"].astype(int))

    rows = []
    for player in squad.itertuples(index=False):
        fantasy_id = int(player.id)
        api_id = crosswalk.get(fantasy_id)
        if api_id is None or int(api_id) not in by_api_id.index:
            raise ValueError(f"No GW match row for {player.name} ({player.team})")
        actual = by_api_id.loc[int(api_id)]
        if isinstance(actual, pd.DataFrame):
            raise ValueError(f"Multiple GW match rows for {player.name} ({player.team})")
        points = float(actual["actual_points"])
        if bool(player.is_capt):
            multiplier, role = 2.0, "Captain"
        elif bool(player.is_sub):
            multiplier = 3.0 if not bool(actual["started"]) else 0.5
            role = "Super sub (bench)" if multiplier == 3 else "Super sub (started)"
        else:
            multiplier, role = 1.0, ""
        rows.append({
            "Player": player.name,
            "Team": player.team,
            "Base": points,
            "Multiplier": multiplier,
            "Contribution": points * multiplier,
            "Role": role,
            "POTM": bool(actual["potm"]),
        })
    result = pd.DataFrame(rows)
    return result, float(result["Contribution"].sum())


def number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def result_section(
    gw: int,
    date: str,
    selected_label: str,
    selected: pd.DataFrame,
    totals: dict[str, float],
    source: str,
) -> str:
    lines = [
        RESULT_START,
        f"## GW{gw} scored result",
        "",
        f"**{selected_label}: {number(totals[selected_label])} points**",
        "",
        "| Model team | Points |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | **{number(total)}** |" for label, total in totals.items())
    lines.extend([
        "",
        f"### {selected_label} player contributions",
        "",
        "| Player | Nation | Base points | Multiplier | Contribution | Role |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in selected.itertuples(index=False):
        potm = " + POTM" if row.POTM else ""
        role = f"{row.Role}{potm}".strip()
        lines.append(
            f"| {row.Player} | {row.Team} | {number(row.Base)} | "
            f"{number(row.Multiplier)}x | **{number(row.Contribution)}** | {role} |"
        )
    lines.append("")
    if source == "official":
        lines.append(
            f"Platform-confirmed result from the gameweek-versioned official GW{gw} "
            f"fantasy feed, covering the fixtures on {date}."
        )
    else:
        lines.append(
            f"Scored from the six ingested fixtures on {date}. This is an "
            "**API-observable reconstruction** because the official fantasy feed was "
            "unavailable. Exact scrum-won, own-lineout, lineout-steal and interception "
            "allocations are unavailable per player."
        )
    lines.append(RESULT_END)
    return "\n".join(lines)


def update_markdown(path: Path, section: str) -> None:
    text = path.read_text()
    pattern = re.compile(
        rf"\n*{re.escape(RESULT_START)}.*?{re.escape(RESULT_END)}\n?", re.DOTALL
    )
    text = pattern.sub("\n", text).rstrip() + "\n\n" + section + "\n"
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, required=True)
    parser.add_argument("--update-markdown", action="store_true")
    args = parser.parse_args()

    date = gameweek_date(args.gw)
    scored: dict[str, pd.DataFrame] = {}
    totals: dict[str, float] = {}
    try:
        official_actuals = load_official_actuals(args.gw)
        source = "official"
        for label, template in TEAM_FILES.items():
            csv_path = DATA / template.format(gw=args.gw)
            scored[label], totals[label] = score_squad_official(csv_path, official_actuals)
    except FileNotFoundError:
        played, _ = actual_player_points(args.gw)
        source = "reconstructed"
        for label, template in TEAM_FILES.items():
            csv_path = DATA / template.format(gw=args.gw)
            scored[label], totals[label] = score_squad(csv_path, played)

    print(f"GW{args.gw} ({date}) [{source}]")
    for label, total in totals.items():
        print(f"  {label:<23} {number(total)}")

    if args.update_markdown:
        markdowns = {
            "Bayes / NCR empirical": DATA / f"ncr_gw{args.gw}_squad.md",
            "6N champion": DATA / f"ncr_gw{args.gw}_squad_champion.md",
            "Blend": DATA / f"ncr_gw{args.gw}_squad_blend.md",
        }
        for label, path in markdowns.items():
            update_markdown(
                path, result_section(args.gw, date, label, scored[label], totals, source)
            )
            print(f"  updated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
