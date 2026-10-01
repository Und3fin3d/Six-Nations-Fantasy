import numpy as np
import pandas as pd
import pytest

from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.hillclimb import MODEL_WEIGHT, adjust_raw, blend_points
from model.unified.kicking import KICKING_EVENTS, KickingConcentration, concentrate_kicking

MINUTES = EventDistribution("lognormal", 70.0, 100.0)


def _player(player, team, conversions, penalties, fixture="1", tries=0.2):
    events = {
        "conversion_goals": EventDistribution("negative_binomial", conversions, 2.0),
        "missed_conversion_goals": EventDistribution("negative_binomial", conversions / 3, 2.0),
        "penalty_goals": EventDistribution("negative_binomial", penalties, 2.0),
        "missed_penalty_goals": EventDistribution("negative_binomial", penalties / 4, 2.0),
        "tries": EventDistribution("negative_binomial", tries, 2.0),
    }
    return RawPrediction(fixture, player, player, team, "B" if team == "A" else "A", "Fly-half",
                         False, events, MINUTES)


def _attempts(prediction):
    return sum(prediction.events[e].mean for e in KICKING_EVENTS)


def test_gamma_one_is_identity():
    raw = [_player("k", "A", 2.0, 1.5), _player("x", "A", 0.5, 0.3)]
    assert concentrate_kicking(raw, 1.0) == raw


def test_concentration_preserves_team_attempts_and_success_split():
    raw = [_player("k", "A", 2.0, 1.5), _player("x", "A", 0.5, 0.3), _player("y", "A", 0.1, 0.0)]
    out = concentrate_kicking(raw, 2.0)
    assert sum(map(_attempts, out)) == pytest.approx(sum(map(_attempts, raw)))
    assert _attempts(out[0]) > _attempts(raw[0])
    assert _attempts(out[1]) < _attempts(raw[1])
    before, after = raw[0].events, out[0].events
    assert after["conversion_goals"].mean / after["penalty_goals"].mean == pytest.approx(
        before["conversion_goals"].mean / before["penalty_goals"].mean)
    assert after["tries"] == before["tries"]
    assert after["conversion_goals"].dispersion == before["conversion_goals"].dispersion


def test_teams_and_fixtures_are_concentrated_separately():
    raw = [_player("k", "A", 2.0, 1.0), _player("x", "A", 1.0, 0.0),
           _player("j", "B", 0.2, 0.1), _player("k2", "A", 1.0, 1.0, fixture="2")]
    out = concentrate_kicking(raw, 3.0)
    assert _attempts(out[2]) == pytest.approx(_attempts(raw[2]))
    assert _attempts(out[3]) == pytest.approx(_attempts(raw[3]))


def test_invalid_gamma_and_wrapper():
    with pytest.raises(ValueError):
        concentrate_kicking([], 0.0)

    class Fixed:
        def predict_frame(self, frame):
            return [_player("k", "A", 2.0, 1.5), _player("x", "A", 0.5, 0.3)]

    out = KickingConcentration(Fixed(), 1.5).predict_frame(None)
    assert _attempts(out[0]) > 2.0 + 2.0 / 3 + 1.5 + 1.5 / 4 - 1e-9


def test_candidate_adjustment_uses_history_state_and_fixed_blend():
    history = pd.DataFrame([
        {"fixture_id": "0", "team": "A", "opponent": "B", "team_score": 40, "opp_score": 3,
         "home_away": "home", "match_at": pd.Timestamp("2025-01-01", tz="UTC")},
    ])
    candidates = pd.DataFrame([{"fixture_id": "1", "team": "A", "opponent": "B", "home_away": "away"},
                               {"fixture_id": "1", "team": "B", "opponent": "A", "home_away": "home"}])
    raw = [_player("a", "A", 1.0, 1.0), _player("b", "B", 1.0, 1.0)]
    out = adjust_raw(raw, candidates, history, beta={"tries": 0.5})
    assert out[0].events["tries"].mean > raw[0].events["tries"].mean > out[1].events["tries"].mean
    blended = blend_points(np.array([10.0]), np.array([20.0]))
    assert blended[0] == pytest.approx(MODEL_WEIGHT * 10 + (1 - MODEL_WEIGHT) * 20)
