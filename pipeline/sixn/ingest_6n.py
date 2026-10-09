from __future__ import annotations

import argparse
import collections
from pathlib import Path

import pandas as pd

from pipeline.sources.rugby_api import RugbyAPI

SIX_NATIONS = 1266

SIXN_START = {
    2023: "2023-02-04", 2024: "2024-02-02",
    2025: "2025-01-31", 2026: "2026-02-05",
}
QUOTA_FLOOR = 50


class QuotaExhausted(RuntimeError):
    pass


def _in_form_window(date_str: str, comp: int, season: int) -> bool:
    if comp == SIX_NATIONS:
        return True
    if season not in SIXN_START:
        return False
    return bool(date_str) and f"{season - 1}-10-01" <= date_str < SIXN_START[season]


def _skip_comp_season(comp: int, season: int) -> bool:
    return comp != SIX_NATIONS and season not in SIXN_START


def _quota_guard(api) -> None:
    if api.remaining is not None and api.remaining < QUOTA_FLOOR:
        raise QuotaExhausted(
            f"live quota {api.remaining} < floor {QUOTA_FLOOR}; stopping before "
            f"the next /match to avoid overage")


JERSEY_GROUP = {
    1: "Prop", 2: "Hooker", 3: "Prop",
    4: "Second-row", 5: "Second-row",
    6: "Back-row", 7: "Back-row", 8: "Back-row",
    9: "Scrum-half", 10: "Fly-half",
    11: "Back-three", 12: "Centre", 13: "Centre", 14: "Back-three", 15: "Back-three",
}
FORWARD_GROUPS = {"Prop", "Hooker", "Second-row", "Back-row"}

STAT_FIELDS = [
    "tries", "try_assists", "conversion_goals", "penalty_goals", "drop_goals_converted",
    "metres", "defenders_beaten", "clean_breaks", "offload", "runs",
    "tackles", "missed_tackles", "dominant_tackles", "tackle_success",
    "tackle_turnover", "tackle_try_saver",
    "rucks_won", "rucks_lost", "passes", "bad_passes", "lineouts_won",
    "turnovers_conceded", "penalties_conceded",
    "missed_conversion_goals", "missed_penalty_goals", "drop_goal_missed",
    "points",
]


def _num(v):
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return 0


def derive_minutes(events: list[dict]) -> tuple[dict, dict, dict]:
    off, on = {}, {}
    cards = collections.defaultdict(lambda: {"yellow_cards": 0, "red_cards": 0})
    for e in events:
        t = (e.get("type") or "").lower()
        pid1 = e.get("player_1_id")
        if e.get("type") == "Substitution":
            if pid1:
                off[pid1] = min(off.get(pid1, 99), _num(e.get("time")))
            pid2 = e.get("player_2_id")
            if pid2:
                on[pid2] = min(on.get(pid2, 99), _num(e.get("time")))
        elif "yellow" in t:
            cards[pid1]["yellow_cards"] += 1
        elif "red" in t:
            cards[pid1]["red_cards"] += 1
    return off, on, cards


def player_minutes(pid, started, off, on) -> int:
    if started:
        return int(off[pid]) if pid in off else 80
    if pid in on:
        end = int(off[pid]) if pid in off else 80
        return max(0, end - int(on[pid]))
    return 0


def ingest(seasons: list[int], comps: list[int] | None = None) -> pd.DataFrame:
    if comps is None:
        comps = [SIX_NATIONS]
    api = RugbyAPI()
    rows: list[dict] = []
    skipped = 0

    try:
      for comp in comps:
        for season in seasons:
            if _skip_comp_season(comp, season):
                continue
            fixtures = api.fixtures(comp, season)
            done = [f for f in fixtures if f.get("status") in ("Result", "Full Time")]
            done = [f for f in done
                    if _in_form_window((f.get("date") or "")[:10], comp, season)]
            tag = "windowed " if comp != SIX_NATIONS else ""
            print(f"\ncomp {comp} season {season}: {len(done)} {tag}completed fixtures")
            for fx in done:
                _quota_guard(api)
                m = api.match(fx["id"])
                match = m.get("match", {})
                events = m.get("events") or []
                off, on, cards = derive_minutes(events)
                if not (m.get("home", {}).get("teamsheet") and m["home"]["teamsheet"][0].get("match_stats")):
                    skipped += 1
                    continue

                for side in ("home", "away"):
                    opp = "away" if side == "home" else "home"
                    team = match.get(f"{side}_team")
                    team_id = match.get(f"{side}_id")
                    opp_team = match.get(f"{opp}_team")
                    opp_id = match.get(f"{opp}_id")
                    ts = match.get(f"{side}_score")
                    os_ = match.get(f"{opp}_score")
                    result = "W" if (ts or 0) > (os_ or 0) else "L" if (ts or 0) < (os_ or 0) else "D"

                    for p in m[side]["teamsheet"]:
                        pid = p["player_id"]
                        ms = p.get("match_stats") or {}
                        started = not p.get("substitute", False)
                        jersey = p.get("position")
                        rec = {
                            "season": season, "comp_id": match.get("comp_id"),
                            "comp_name": match.get("comp_name"),
                            "round": match.get("game_week") or match.get("round_id"),
                            "fixture_id": match.get("id"), "date": (match.get("date") or "")[:10],
                            "team": team, "team_id": team_id,
                            "opponent": opp_team, "opponent_id": opp_id,
                            "home_away": side, "team_score": ts, "opp_score": os_,
                            "result": result, "margin": (ts or 0) - (os_ or 0),
                            "player_id": pid, "player_name": p.get("name"),
                            "jersey": jersey, "started": started,
                            "minutes": player_minutes(pid, started, off, on),
                            "yellow_cards": cards[pid]["yellow_cards"],
                            "red_cards": cards[pid]["red_cards"],
                            "available__minutes": True,
                            "available__yellow_cards": True,
                            "available__red_cards": True,
                        }
                        for f in STAT_FIELDS:
                            present = f in ms
                            rec[f] = _num(ms[f]) if present else None
                            rec[f"available__{f}"] = present
                        rows.append(rec)

    except QuotaExhausted as e:
        print(f"\n🛑 {e}\n   returning the {len(rows)} player-rows fetched before the floor")

    df = pd.DataFrame(rows)
    if skipped:
        print(f"\n⚠️  skipped {skipped} matches without per-player stats")
    return df


TEAM_ID_COLS = [
    "season", "comp_id", "comp_name", "round", "fixture_id", "date",
    "team", "team_id", "opponent", "opponent_id", "home_away",
    "team_score", "opp_score", "result", "margin",
    "tries_for", "tries_against", "conv_for", "conv_against",
    "pens_for", "pens_against", "dg_for", "dg_against",
]


def _flatten_one_team_stats(ts) -> dict:
    out: dict = {}
    seen: dict = {}
    groups = ts.items() if isinstance(ts, dict) else []
    for grp, arr in groups:
        for item in (arr or []):
            if not isinstance(item, dict):
                continue
            stat = item.get("stat")
            if not stat:
                continue
            col = stat if stat not in seen else f"{grp}_{stat}"
            seen[stat] = grp
            out[col] = _num(item.get("value"))
    return out


def flatten_team_stats(comp_season_pairs: list[tuple[int, int]]) -> pd.DataFrame:
    api = RugbyAPI()
    rows: list[dict] = []
    skipped = 0
    try:
      for comp, season in comp_season_pairs:
        if _skip_comp_season(comp, season):
            continue
        fixtures = api.fixtures(comp, season)
        done = [f for f in fixtures if f.get("status") in ("Result", "Full Time")]
        done = [f for f in done
                if _in_form_window((f.get("date") or "")[:10], comp, season)]
        tag = "windowed " if comp != SIX_NATIONS else ""
        print(f"\ncomp {comp} season {season}: {len(done)} {tag}completed fixtures")
        for fx in done:
            _quota_guard(api)
            m = api.match(fx["id"])
            match = m.get("match", {})
            for side in ("home", "away"):
                opp = "away" if side == "home" else "home"
                ts = (m.get(side) or {}).get("team_stats")
                if not ts:
                    skipped += 1
                    continue
                tscore = match.get(f"{side}_score")
                oscore = match.get(f"{opp}_score")
                result = ("W" if (tscore or 0) > (oscore or 0)
                          else "L" if (tscore or 0) < (oscore or 0) else "D")
                rec = {
                    "season": match.get("season") or season,
                    "comp_id": match.get("comp_id") or comp,
                    "comp_name": match.get("comp_name"),
                    "round": match.get("game_week") or match.get("round_id"),
                    "fixture_id": match.get("id"), "date": (match.get("date") or "")[:10],
                    "team": match.get(f"{side}_team"), "team_id": match.get(f"{side}_id"),
                    "opponent": match.get(f"{opp}_team"), "opponent_id": match.get(f"{opp}_id"),
                    "home_away": side, "team_score": tscore, "opp_score": oscore,
                    "result": result, "margin": (tscore or 0) - (oscore or 0),
                    "tries_for": _num(match.get(f"{side}_tries")),
                    "tries_against": _num(match.get(f"{opp}_tries")),
                    "conv_for": _num(match.get(f"{side}_conversions")),
                    "conv_against": _num(match.get(f"{opp}_conversions")),
                    "pens_for": _num(match.get(f"{side}_penalties")),
                    "pens_against": _num(match.get(f"{opp}_penalties")),
                    "dg_for": _num(match.get(f"{side}_drop_goals")),
                    "dg_against": _num(match.get(f"{opp}_drop_goals")),
                }
                rec.update(_flatten_one_team_stats(ts))
                rows.append(rec)
    except QuotaExhausted as e:
        print(f"\n🛑 {e}\n   returning the {len(rows)} team-rows fetched before the floor")

    df = pd.DataFrame(rows)
    if skipped:
        print(f"\n⚠️  {skipped} team-sides had no team_stats")
    if not df.empty:
        stat_cols = sorted(c for c in df.columns if c not in TEAM_ID_COLS)
        df = df.reindex(columns=[c for c in TEAM_ID_COLS if c in df.columns] + stat_cols)
    return df


def add_position_and_recon(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = df[df.started & df.jersey.between(1, 15)]
    canon = {}
    for pid, grp in starts.groupby("player_id"):
        jersey_mode = grp.jersey.mode()
        j = int(jersey_mode.iloc[0]) if len(jersey_mode) else None
        canon[pid] = JERSEY_GROUP.get(j)
    df["canonical_pos"] = df.player_id.map(canon)
    df.loc[df.canonical_pos.isna() & df.jersey.between(1, 15), "canonical_pos"] = \
        df.jersey.map(JERSEY_GROUP)
    df["is_forward"] = df.canonical_pos.isin(FORWARD_GROUPS)

    try_pts = df.is_forward.map({True: 15, False: 10}).fillna(12) * df.tries
    df["fantasy_pts_recon"] = (
        try_pts
        + 4 * df.try_assists
        + 2 * df.conversion_goals
        + 3 * df.penalty_goals
        + 5 * df.drop_goals_converted
        + (df.metres // 10)
        + 2 * df.defenders_beaten
        + 1 * df.tackles
        + 2 * df.offload
        + 5 * df.tackle_turnover
        - 1 * df.penalties_conceded
        - 5 * df.yellow_cards
        - 10 * df.red_cards
    ).astype(int)

    players = (df.groupby(["player_id", "player_name"])
                 .agg(canonical_pos=("canonical_pos", "first"),
                      is_forward=("is_forward", "first"),
                      matches=("fixture_id", "nunique"),
                      starts=("started", "sum"),
                      teams=("team", lambda s: ",".join(sorted(set(s)))),
                      seasons=("season", lambda s: ",".join(map(str, sorted(set(s))))))
                 .reset_index())
    return df, players


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=[2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--comps", nargs="+", type=int, default=None,
                    help="competition ids (default: Six Nations only). When "
                         "passed for the player table, writes "
                         "data/api_player_match.csv instead of the 6N files.")
    ap.add_argument("--team", action="store_true",
                    help="build the team-level table (api_team_match.csv) instead "
                         "of the player table")
    args = ap.parse_args()

    comps_given = args.comps is not None
    comps = args.comps if comps_given else [SIX_NATIONS]

    out_dir = Path(__file__).resolve().parents[2] / "data"
    out_dir.mkdir(exist_ok=True)

    if args.team:
        pairs = [(c, s) for c in comps for s in args.seasons]
        tm = flatten_team_stats(pairs)
        tm_path = out_dir / "api_team_match.csv"
        tm.to_csv(tm_path, index=False)
        print(f"\n✅  {len(tm)} team-match rows, {tm.fixture_id.nunique()} matches, "
              f"comps {sorted(tm.comp_id.unique())}, seasons {sorted(tm.season.unique())}")
        print(f"    {len(tm.columns)} cols; stat fields: "
              f"{[c for c in tm.columns if c not in TEAM_ID_COLS]}")
        print(f"💾  {tm_path}")
        return

    df = ingest(args.seasons, comps)
    df, players = add_position_and_recon(df)

    if comps_given:
        pm_path = out_dir / "api_player_match.csv"
        df.to_csv(pm_path, index=False)
        print(f"\n✅  {len(df)} player-match rows, {len(players)} players, "
              f"{df.fixture_id.nunique()} matches, "
              f"comps {sorted(df.comp_id.dropna().unique())}, "
              f"seasons {sorted(df.season.unique())}")
        print(f"💾  {pm_path}")
        return

    pm_path = out_dir / "6n_player_match.csv"
    pl_path = out_dir / "6n_players.csv"
    df.to_csv(pm_path, index=False)
    players.to_csv(pl_path, index=False)

    print(f"\n✅  {len(df)} player-match rows, {len(players)} players, "
          f"{df.fixture_id.nunique()} matches, seasons {sorted(df.season.unique())}")
    print(f"💾  {pm_path}")
    print(f"💾  {pl_path}")


if __name__ == "__main__":
    main()
