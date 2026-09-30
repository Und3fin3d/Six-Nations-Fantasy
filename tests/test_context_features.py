import numpy as np
import pandas as pd
import pytest

from model.unified.v4.context import (COLUMNS, INITIAL, add_candidate_context,
                                      add_training_context, pre_match_context)
from model.unified.v4.features import V4_FEATURE_PREFIXES
from model.unified.v4.gbdt import V4FeatureEncoder


def _rows(fixture, when, team, opponent, scored, conceded, home, players=2):
    return [{
        "fixture_id": fixture, "match_at": pd.Timestamp(when, tz="UTC"), "team": team,
        "opponent": opponent, "team_score": scored, "opp_score": conceded,
        "home_away": home, "player_id": f"{team}-{i}",
    } for i in range(players)]


def _store():
    rows = []
    rows += _rows("1", "2025-01-01 15:00", "A", "B", 30, 10, "home")
    rows += _rows("1", "2025-01-01 15:00", "B", "A", 10, 30, "away")
    rows += _rows("2", "2025-01-08 15:00", "A", "C", 20, 20, "away")
    rows += _rows("3", "2025-01-15 15:00", "B", "C", 5, 40, "home")
    return pd.DataFrame(rows)


def test_each_fixture_updates_both_teams_once_and_conserves_rating():
    table, state = pre_match_context(_store())
    assert len(table) == 6
    assert table.groupby("fixture_id").size().eq(2).all()
    assert sum(state.elo.values()) == pytest.approx(3 * INITIAL)
    assert state.elo["A"] > INITIAL > state.elo["B"]


def test_training_rows_only_see_earlier_fixtures():
    store = _store()
    table, _ = pre_match_context(store)
    features = add_training_context(store, table)
    first = features[features.fixture_id.eq("1")]
    assert first["ctx__team_elo"].eq(INITIAL).all()
    assert first["ctx__team_pf"].isna().all()
    later = features[features.fixture_id.eq("2") & features.team.eq("A")].iloc[0]
    assert later["ctx__team_elo"] > INITIAL
    assert later["ctx__team_pf"] == 30
    assert later["ctx__home"] == 0.0
    assert later["ctx__elo_edge"] == pytest.approx(later["ctx__team_elo"] - later["ctx__opp_elo"] - 45.0)


def test_mirrored_player_rows_do_not_double_update():
    one_side = _store()
    one_side = one_side[~(one_side.fixture_id.eq("1") & one_side.team.eq("B"))]
    assert pre_match_context(one_side)[1].elo == pre_match_context(_store())[1].elo


def test_candidates_use_final_training_state_not_their_own_results():
    store = _store()
    _, state = pre_match_context(store)
    candidates = pd.DataFrame([{"fixture_id": "9", "team": "C", "opponent": "A", "home_away": "home",
                                "team_score": 0, "opp_score": 99}])
    context = add_candidate_context(candidates, state).iloc[0]
    assert context["ctx__team_elo"] == pytest.approx(state.elo["C"])
    assert context["ctx__opp_elo"] == pytest.approx(state.elo["A"])
    assert context["ctx__home"] == 1.0
    assert context["ctx__opp_pf"] == pytest.approx(state.pf["A"])
    assert state.elo == pre_match_context(store)[1].elo


def test_unknown_venue_is_neutral_and_new_teams_start_level():
    _, state = pre_match_context(_store())
    candidates = pd.DataFrame([{"fixture_id": "9", "team": "X", "opponent": "Y", "home_away": ""}])
    context = add_candidate_context(candidates, state).iloc[0]
    assert context["ctx__home"] == 0.5
    assert context["ctx__elo_edge"] == 0.0
    assert np.isnan(context["ctx__team_pf"])


def test_v4_encoder_admits_context_only_when_present():
    assert "ctx__" in V4_FEATURE_PREFIXES
    frame = pd.DataFrame({
        "player_id": ["1", "2"], "position": ["Prop", "Hooker"], "team": ["A", "B"],
        "opponent": ["B", "A"], "competition_level": ["club", "club"],
        "is_forward": [1.0, 1.0], "started": [1.0, 0.0],
    })
    plain = V4FeatureEncoder().fit(frame)
    assert not any(c.startswith("ctx__") for c in plain.numeric_columns)
    with_context = frame.assign(**{c: [1.0, 2.0] for c in COLUMNS})
    encoder = V4FeatureEncoder().fit(with_context)
    assert set(COLUMNS) <= set(encoder.numeric_columns)
