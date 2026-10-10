from __future__ import annotations

import argparse
import collections

import pandas as pd

from pipeline.sources.rugby_api import RugbyAPI

SIX_NATIONS = 1266

SIXN_START = {2023: "2023-02-04", 2024: "2024-02-02", 2025: "2025-01-31", 2026: "2026-02-05"}
QUOTA_FLOOR = 50


class QuotaExhausted(RuntimeError):
    pass


def _in_form_window(date_str: str, comp: int, season: int) -> bool:
    if comp == SIX_NATIONS:
        return True
    if season not in SIXN_START:
        return False
    return bool(date_str) and f"{season - 1}-10-01" <= date_str < SIXN_START[season]


def _quota_guard(api) -> None:
    if api.remaining is not None and api.remaining < QUOTA_FLOOR:
        raise QuotaExhausted(f"live quota {api.remaining} < floor {QUOTA_FLOOR}; stopping before the next /match to avoid overage")


JERSEY_GROUP = {
    1: "Prop", 2: "Hooker", 3: "Prop", 4: "Second-row", 5: "Second-row", 6: "Back-row", 7: "Back-row", 8: "Back-row", 9: "Scrum-half", 10: "Fly-half",
    11: "Back-three", 12: "Centre", 13: "Centre", 14: "Back-three", 15: "Back-three",
}
FORWARD_GROUPS = {"Prop", "Hooker", "Second-row", "Back-row"}

STAT_FIELDS = [
    "tries", "try_assists", "conversion_goals", "penalty_goals", "drop_goals_converted", "metres", "defenders_beaten", "clean_breaks", "offload", "runs",
    "tackles", "missed_tackles", "dominant_tackles", "tackle_success", "tackle_turnover", "tackle_try_saver", "rucks_won", "rucks_lost", "passes", "bad_passes",
    "lineouts_won", "turnovers_conceded", "penalties_conceded", "missed_conversion_goals", "missed_penalty_goals", "drop_goal_missed", "points",
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


def _matches(api: RugbyAPI, pairs: list[tuple[int, int]]):
    for comp, season in pairs:
        if comp != SIX_NATIONS and season not in SIXN_START:
            continue
        done = [
            f for f in api.fixtures(comp, season) if f.get("status") in ("Result", "Full Time") and _in_form_window((f.get("date") or "")[:10], comp, season)
        ]
        print(f"\ncomp {comp} season {season}: {len(done)} {'windowed ' if comp != SIX_NATIONS else ''}completed fixtures")
        for fx in done:
            _quota_guard(api)
            yield comp, season, api.match(fx["id"])


def _score(match: dict, side: str, opp: str) -> tuple:
    ts, os_ = match.get(f"{side}_score"), match.get(f"{opp}_score")
    return ts, os_, "W" if (ts or 0) > (os_ or 0) else "L" if (ts or 0) < (os_ or 0) else "D", (ts or 0) - (os_ or 0)


def ingest(seasons: list[int], comps: list[int]) -> pd.DataFrame:
    api = RugbyAPI()
    rows, skipped = [], 0
    try:
        for _, season, m in _matches(api, [(c, s) for c in comps for s in seasons]):
            match = m.get("match", {})
            off, on, cards = derive_minutes(m.get("events") or [])
            if not (m.get("home", {}).get("teamsheet") and m["home"]["teamsheet"][0].get("match_stats")):
                skipped += 1
                continue
            for side, opp in (("home", "away"), ("away", "home")):
                ts, os_, result, margin = _score(match, side, opp)
                for p in m[side]["teamsheet"]:
                    pid, ms, started = p["player_id"], p.get("match_stats") or {}, not p.get("substitute", False)
                    rows.append(
                        {
                            "season": season,
                            "comp_id": match.get("comp_id"),
                            "comp_name": match.get("comp_name"),
                            "round": match.get("game_week") or match.get("round_id"),
                            "fixture_id": match.get("id"),
                            "date": (match.get("date") or "")[:10],
                            "team": match.get(f"{side}_team"),
                            "team_id": match.get(f"{side}_id"),
                            "opponent": match.get(f"{opp}_team"),
                            "opponent_id": match.get(f"{opp}_id"),
                            "home_away": side,
                            "team_score": ts,
                            "opp_score": os_,
                            "result": result,
                            "margin": margin,
                            "player_id": pid,
                            "player_name": p.get("name"),
                            "jersey": p.get("position"),
                            "started": started,
                            "minutes": player_minutes(pid, started, off, on),
                            "yellow_cards": cards[pid]["yellow_cards"],
                            "red_cards": cards[pid]["red_cards"],
                            "available__minutes": True,
                            "available__yellow_cards": True,
                            "available__red_cards": True,
                            **{k: v for f in STAT_FIELDS for k, v in ((f, _num(ms[f]) if f in ms else None), (f"available__{f}", f in ms))},
                        }
                    )
    except QuotaExhausted as e:
        print(f"\n🛑 {e}\n   returning the {len(rows)} player-rows fetched before the floor")
    if skipped:
        print(f"\n⚠️  skipped {skipped} matches without per-player stats")
    return pd.DataFrame(rows)


TEAM_ID_COLS = [
    "season", "comp_id", "comp_name", "round", "fixture_id", "date", "team", "team_id", "opponent", "opponent_id", "home_away", "team_score", "opp_score",
    "result", "margin", "tries_for", "tries_against", "conv_for", "conv_against", "pens_for", "pens_against", "dg_for", "dg_against",
]


def _flatten_one_team_stats(ts) -> dict:
    out: dict = {}
    seen: dict = {}
    groups = ts.items() if isinstance(ts, dict) else []
    for grp, arr in groups:
        for item in arr or []:
            if not isinstance(item, dict):
                continue
            stat = item.get("stat")
            if not stat:
                continue
            col = stat if stat not in seen else f"{grp}_{stat}"
            seen[stat] = grp
            out[col] = _num(item.get("value"))
    return out


def flatten_team_stats(pairs: list[tuple[int, int]]) -> pd.DataFrame:
    api = RugbyAPI()
    rows, skipped = [], 0
    try:
        for comp, season, m in _matches(api, pairs):
            match = m.get("match", {})
            for side, opp in (("home", "away"), ("away", "home")):
                ts = (m.get(side) or {}).get("team_stats")
                if not ts:
                    skipped += 1
                    continue
                tscore, oscore, result, margin = _score(match, side, opp)
                rows.append(
                    {
                        "season": match.get("season") or season,
                        "comp_id": match.get("comp_id") or comp,
                        "comp_name": match.get("comp_name"),
                        "round": match.get("game_week") or match.get("round_id"),
                        "fixture_id": match.get("id"),
                        "date": (match.get("date") or "")[:10],
                        "team": match.get(f"{side}_team"),
                        "team_id": match.get(f"{side}_id"),
                        "opponent": match.get(f"{opp}_team"),
                        "opponent_id": match.get(f"{opp}_id"),
                        "home_away": side,
                        "team_score": tscore,
                        "opp_score": oscore,
                        "result": result,
                        "margin": margin,
                        **{
                            f"{k}_{who}": _num(match.get(f"{s}_{field}"))
                            for k, field in (("tries", "tries"), ("conv", "conversions"), ("pens", "penalties"), ("dg", "drop_goals"))
                            for who, s in (("for", side), ("against", opp))
                        },
                        **_flatten_one_team_stats(ts),
                    }
                )
    except QuotaExhausted as e:
        print(f"\n🛑 {e}\n   returning the {len(rows)} team-rows fetched before the floor")
    df = pd.DataFrame(rows)
    if skipped:
        print(f"\n⚠️  {skipped} team-sides had no team_stats")
    return df.reindex(columns=[c for c in TEAM_ID_COLS if c in df.columns] + sorted(c for c in df.columns if c not in TEAM_ID_COLS))


def add_position_and_recon(df: pd.DataFrame) -> pd.DataFrame:
    canon = {
        pid: JERSEY_GROUP.get(int(mode.iloc[0]) if len(mode := grp.jersey.mode()) else None)
        for pid, grp in df[df.started & df.jersey.between(1, 15)].groupby("player_id")
    }
    df["canonical_pos"] = df.player_id.map(canon)
    df.loc[df.canonical_pos.isna() & df.jersey.between(1, 15), "canonical_pos"] = df.jersey.map(JERSEY_GROUP)
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
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", nargs="+", type=int, required=True)
    ap.add_argument("--team", action="store_true")
    args = ap.parse_args()
    seasons = [2021, 2022, 2023, 2024, 2025, 2026]
    if args.team:
        tm = flatten_team_stats([(c, s) for c in args.comps for s in seasons])
        tm.to_csv("data/api_team_match.csv", index=False)
        print(f"\n✅  {len(tm)} team-match rows, {tm.fixture_id.nunique()} matches, comps {sorted(tm.comp_id.unique())}, seasons {sorted(tm.season.unique())}")
        print(f"    {len(tm.columns)} cols; stat fields: {[c for c in tm.columns if c not in TEAM_ID_COLS]}")
        print("💾  data/api_team_match.csv")
    else:
        df = add_position_and_recon(ingest(seasons, args.comps))
        df.to_csv("data/api_player_match.csv", index=False)
        print(
            f"\n✅  {len(df)} player-match rows, {df.groupby(['player_id', 'player_name']).ngroups} players, {df.fixture_id.nunique()} matches, "
            f"comps {sorted(df.comp_id.dropna().unique())}, seasons {sorted(df.season.unique())}"
        )
        print("💾  data/api_player_match.csv")
