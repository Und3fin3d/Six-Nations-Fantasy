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


from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.v4.context import (TeamStrengthCalibration, fit_team_strength_beta,
                                      team_strength_scale)


def _prediction():
    return RawPrediction("9", "p", "P", "A", "B", "Centre", False, {
        "tries": EventDistribution("negative_binomial", 0.4, 2.0),
        "metres": EventDistribution("lognormal", 50.0, 900.0),
        "yellow_cards": EventDistribution("bernoulli", 0.9, 1.0),
    }, EventDistribution("lognormal", 70.0, 100.0))


def test_team_strength_scale_preserves_shape_and_bounds():
    scaled = team_strength_scale(_prediction(), 400.0, {"tries": 0.5, "metres": -0.2, "yellow_cards": 1.0})
    assert scaled.events["tries"].mean == pytest.approx(0.4 * np.exp(0.5))
    assert scaled.events["tries"].dispersion == 2.0
    factor = np.exp(-0.2)
    assert scaled.events["metres"].mean == pytest.approx(50.0 * factor)
    assert scaled.events["metres"].dispersion == pytest.approx(900.0 * factor ** 2)
    assert scaled.events["yellow_cards"].mean == 1.0
    assert scaled.minutes == _prediction().minutes


def test_zero_edge_or_missing_beta_is_identity():
    original = _prediction()
    assert team_strength_scale(original, 0.0, {"tries": 0.5}).events == original.events
    assert team_strength_scale(original, 300.0, {}).events == original.events


def test_beta_fit_recovers_known_coefficient_and_skips_sparse_events():
    rng = np.random.default_rng(3)
    edge = rng.uniform(-400, 400, 40000)
    forecast = rng.uniform(1, 3, 40000)
    observed = rng.poisson(forecast * np.exp(0.3 * edge / 400))
    frame = pd.DataFrame({"edge": edge, "p_tries": forecast, "y_tries": observed,
                          "p_red_cards": 0.001, "y_red_cards": 0.0})
    beta = fit_team_strength_beta(frame, ("tries", "red_cards"))
    assert beta["tries"] == pytest.approx(0.3, abs=0.03)
    assert beta["red_cards"] == 0.0


def test_calibration_uses_lock_state_and_candidate_venue():
    _, state = pre_match_context(_store())

    class Fixed:
        def predict_frame(self, frame):
            return [_prediction() for _ in range(len(frame))]

    frame = pd.DataFrame([{"fixture_id": "9", "team": "A", "opponent": "B", "home_away": "home"}])
    out = TeamStrengthCalibration(Fixed(), {"tries": 0.4}, state).predict_frame(frame)[0]
    edge = state.elo["A"] - state.elo["B"] + 45.0
    assert out.events["tries"].mean == pytest.approx(0.4 * np.exp(0.4 * edge / 400))
