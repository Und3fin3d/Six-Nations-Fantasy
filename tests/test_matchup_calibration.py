import numpy as np
import pandas as pd
import pytest

from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.matchup import (calibrate_matchups, conceded_profiles, fit_matchup_coefficients,
                                   matchup_scale, team_event_totals)

MINUTES = EventDistribution("lognormal", 70.0, 100.0)


def _history():
    rows = []
    fixtures = [("1", "A", "B", "2024-01-01", 30, 10), ("2", "C", "B", "2024-02-01", 50, 0),
                ("3", "A", "C", "2024-03-01", 20, 25), ("9", "A", "B", "2024-12-01", 99, 99)]
    for fixture, home, away, date, home_tackles, away_tackles in fixtures:
        for team, opponent, tackles, venue in ((home, away, home_tackles, "home"), (away, home, away_tackles, "away")):
            for player in range(2):
                rows.append({"fixture_id": fixture, "team": team, "opponent": opponent, "player_id": f"{team}{player}",
                             "match_at": pd.Timestamp(date, tz="UTC"), "competition_level": "international",
                             "tackles": tackles / 2, "available__tackles": True, "home_away": venue,
                             "team_score": 10, "opp_score": 5})
    return pd.DataFrame(rows)


def test_team_totals_pair_each_side_with_its_opponent():
    totals = team_event_totals(_history(), ("tackles",)).set_index(["fixture_id", "team"])
    assert totals.loc[("1", "A"), "tackles"] == 30
    assert totals.loc[("1", "A"), "a_tackles"] == 10
    assert totals.loc[("2", "B"), "a_tackles"] == 50


def test_profiles_use_only_pre_cutoff_fixtures_and_shrink_to_one():
    totals = team_event_totals(_history(), ("tackles",))
    profile = conceded_profiles(totals, pd.Timestamp("2024-06-01", tz="UTC"), ("tackles",))["tackles"]
    # B conceded 30 and 50 tackles (opponents tackled a lot), above the pooled mean.
    assert profile["B"] > 1.0 > profile["A"]
    later = conceded_profiles(totals, pd.Timestamp("2025-01-01", tz="UTC"), ("tackles",))["tackles"]
    assert later["B"] != pytest.approx(profile["B"])
    flat = conceded_profiles(totals, pd.Timestamp("2024-06-01", tz="UTC"), ("tackles",), prior=1e9)["tackles"]
    assert np.allclose(flat.to_numpy(), 1.0)


def _prediction(team="A", opponent="B", player="A0"):
    events = {"tackles": EventDistribution("negative_binomial", 5.0, 3.0),
              "metres": EventDistribution("lognormal", 40.0, 400.0),
              "potm": EventDistribution("bernoulli", 0.9, 1.0)}
    return RawPrediction("10", player, player, team, opponent, "Prop", True, events, MINUTES)


def test_scale_preserves_shapes_and_unlisted_events():
    coef = {"tackles": {"edge": 0.0, "opp": 1.0}, "metres": {"edge": 0.4, "opp": 0.0},
            "potm": {"edge": 4.0, "opp": 0.0}}
    out = matchup_scale(_prediction(), {"edge": 400.0, "opp__tackles": np.log(1.2)}, coef)
    assert out.events["tackles"].mean == pytest.approx(6.0)
    assert out.events["tackles"].dispersion == 3.0
    factor = np.exp(0.4)
    assert out.events["metres"].mean == pytest.approx(40 * factor)
    assert out.events["metres"].dispersion == pytest.approx(400 * factor ** 2)
    assert out.events["potm"].mean == 1.0
    assert out.minutes == MINUTES
    same = matchup_scale(_prediction(), {"edge": 0.0}, {})
    assert same.events == _prediction().events


def test_calibration_checks_alignment():
    history = _history()
    candidates = pd.DataFrame({"fixture_id": ["10"], "player_id": ["A0"], "team": ["A"], "opponent": ["B"],
                               "home_away": ["home"]})
    coef = {"tackles": {"edge": 0.0, "opp": 1.0}}
    out = calibrate_matchups([_prediction()], candidates, history, pd.Timestamp("2024-06-01", tz="UTC"), coef)
    assert out[0].events["tackles"].mean > 5.0
    with pytest.raises(ValueError):
        calibrate_matchups([_prediction(player="A1")], candidates, history, pd.Timestamp("2024-06-01", tz="UTC"), coef)


def test_fit_recovers_known_opponent_effect():
    rng = np.random.default_rng(0)
    n = 4000
    edge = rng.normal(0, 150, n)
    opp = rng.normal(0, 0.2, n)
    p = rng.uniform(80, 120, n)
    y = rng.poisson(p * np.exp(0.1 * edge / 400 + 0.8 * opp))
    teams = pd.DataFrame({"edge": edge, "opp__tackles": opp, "p_tackles": p, "y_tackles": y,
                          "opp__tries": opp, "p_tries": 0.001, "y_tries": 0})
    coef = fit_matchup_coefficients(teams, ("tackles", "tries"), ridge=0.0)
    assert coef["tackles"]["opp"] == pytest.approx(0.8, abs=0.05)
    assert coef["tackles"]["edge"] == pytest.approx(0.1, abs=0.05)
    assert coef["tries"] == {"edge": 0.0, "opp": 0.0}
