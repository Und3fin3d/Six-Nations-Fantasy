"""Offline tests for research/stat_sources parsers (no network)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "stat_sources"))

from common import agreement, initial_key, join_players, match_fixtures, surname_key  # noqa: E402
from incrowd import parse_match  # noqa: E402
from pulselive import minutes_from_timeline  # noqa: E402


def test_name_keys():
    assert surname_key("Louis Bielle-Biarrey") == "biellebiarrey"
    assert surname_key("L. Bielle-Biarrey") == "biellebiarrey"
    assert surname_key("Bielle-Biarrey") == "biellebiarrey"
    assert surname_key("Thomas du Toit") == "dutoit"
    assert initial_key("Antoine Dupont") == "a|dupont"


def _incrowd_payload():
    # Shape as consumed by public clients of rugby-union-feeds.incrowdsports.com/v1/matches/{id}
    return {"data": {
        "id": 1, "compId": 1068, "season": 202501, "date": "2025-10-04T14:00:00.000Z", "status": "result",
        "homeTeam": {"id": 10, "name": "Home", "players": [
            {"id": 101, "name": "A Starter", "positionId": 7, "position": "flanker",
             "stats": {"turnoverWon": 2, "metres": 12, "minutesPlayedTotal": 58, "kicksFromHand": 0}},
            {"id": 102, "name": "B Bench", "positionId": 20, "position": "sub",
             "stats": {"turnoverWon": 1, "metres": 4, "minutesPlayedTotal": 22}},
        ]},
        "awayTeam": {"id": 20, "name": "Away", "players": [
            {"id": 201, "name": "C Unused", "positionId": 23, "stats": {}}]},
        "events": [
            {"type": "Sub Off", "teamId": 10, "playerId": 101, "minute": 58},
            {"type": "Sub On", "teamId": 10, "playerId": 102, "minute": 58},
            {"type": "Try", "teamId": 10, "playerId": 102, "minute": 70},
        ]}}


def test_incrowd_parse_match():
    df = parse_match(_incrowd_payload())
    assert len(df) == 3
    a = df.set_index("player_id")
    assert a.loc[101, "turnoverWon"] == 2 and bool(a.loc[101, "starter"])
    assert a.loc[101, "off_minutes"] == "[58]"
    assert a.loc[102, "on_minutes"] == "[58]" and not bool(a.loc[102, "starter"])
    assert a.loc[201, "on_minutes"] == "[]"


def test_pulselive_minutes_from_timeline():
    summary = {"teams": [{"teamList": {"list": [
        {"number": 2, "player": {"id": 1, "name": {"display": "Hooker"}}},
        {"number": 16, "player": {"id": 2, "name": {"display": "Bench Hooker"}}},
        {"number": 10, "player": {"id": 3, "name": {"display": "Fly Half"}}},
        {"number": 22, "player": {"id": 4, "name": {"display": "Bench Ten"}}},
    ]}}]}
    timeline = {"timeline": [
        {"type": "Sub Off", "playerId": 1, "time": {"secs": 50 * 60}},
        {"type": "Sub On", "playerId": 2, "time": {"secs": 50 * 60}},
        # HIA: fly-half off at 20', back on at 32'
        {"type": "HIAO", "playerId": 3, "time": {"secs": 20 * 60}},
        {"type": "Sub On", "playerId": 4, "time": {"secs": 20 * 60}},
        {"type": "IR", "playerId": 3, "time": {"secs": 32 * 60}},
        {"type": "Sub Off", "playerId": 4, "time": {"secs": 32 * 60}},
    ]}
    m = minutes_from_timeline(summary, timeline, 0).set_index("player_id")
    assert m.loc["1", "minutes"] == 50
    assert m.loc["2", "minutes"] == 30 and m.loc["2", "first_on_min"] == 50
    assert m.loc["3", "minutes"] == 68
    assert m.loc["4", "minutes"] == 12


def test_fixture_match_and_join():
    store = pd.DataFrame({
        "date": [pd.Timestamp("2025-10-04").date()] * 3,
        "fixture_id": [9, 9, 9], "player_name": ["Ann Starter", "Bob Bench", "Cy Other"],
        "metres": [12, 4, 0]})
    store["sur"] = store.player_name.map(surname_key)
    store["ikey"] = store.player_name.map(initial_key)
    ext = pd.DataFrame({"match_id": [1, 1], "match_date": ["2025-10-04"] * 2,
                        "player": ["A Starter", "B Bench"], "metres_ext": [12, 4]})
    fmap = match_fixtures(ext, store, min_overlap=2)
    assert fmap.fixture_id.iloc[0] == 9
    j = join_players(ext, store, fmap)
    assert len(j) == 2
    assert agreement(pd.Series([1, 2, 3, 4, 5]), pd.Series([1, 2, 3, 4, 5]))["exact"] == 1.0
