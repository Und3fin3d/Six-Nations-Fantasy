import json

import pytest
from datetime import datetime, timezone

import ncr_snapshot as snapshot


def test_current_fixture_gameday_selects_matching_player_feed():
    fixtures = [
        {"gameday": 1, "iscurrent": 2},
        {"gameday": 2, "iscurrent": 2},
        {"gameday": 3, "iscurrent": 1},
    ]

    assert snapshot.current_gameday(fixtures) == 3
    assert snapshot.player_feed_path(3) == "/players/players_1_en_3.json"


def test_stale_player_payload_is_rejected_before_promotion():
    players = [
        {"id": 1, "gameday_id": 1, "cur_gd_points": "10.00"},
        {"id": 2, "gameday_id": 1, "cur_gd_points": "20.00"},
    ]

    with pytest.raises(ValueError, match="requested GW3.*contains gameday_id.*1"):
        snapshot.validate_player_gameday(players, requested_gw=3)


def test_latest_completed_results_are_separate_from_current_catalogue():
    fixtures = [
        {"gameday": 1, "game_date": "2026-07-04T05:10:00.000"},
        {"gameday": 2, "game_date": "2026-07-11T05:10:00.000"},
        {"gameday": 3, "game_date": "2026-07-18T05:10:00.000"},
    ]
    now = datetime(2026, 7, 15, 18, tzinfo=timezone.utc)

    assert snapshot.latest_completed_gameday(fixtures, now=now) == 2


def test_exact_prior_lineup_is_rejected_as_carryover(tmp_path):
    players = [
        {"id": i, "gameday_id": "3", "player_status": "P" if i < 179 else "B"}
        for i in range(270)
    ]
    prior = {
        "Data": {"Value": {"Players": [
            {**player, "gameday_id": "1"} for player in players
        ]}}
    }
    (tmp_path / "players_gw1.json").write_text(json.dumps(prior))
    assert snapshot.carryover_gameday(players, 3, tmp_path) == 1


def test_changed_lineup_is_not_rejected_as_carryover(tmp_path):
    players = [
        {"id": i, "gameday_id": "3", "player_status": "P" if i < 179 else "B"}
        for i in range(270)
    ]
    prior_players = [{**player, "gameday_id": "2"} for player in players]
    prior_players[0]["player_status"] = "B"
    prior = {"Data": {"Value": {"Players": prior_players}}}
    (tmp_path / "players_gw2.json").write_text(json.dumps(prior))
    assert snapshot.carryover_gameday(players, 3, tmp_path) is None
