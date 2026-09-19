import numpy as np
import pandas as pd
import pytest

from model.pit import completed_seasons, day_cutoff, history_before, team_margin_history
from model.unified.raw_benchmark.folds import HistoricalFold, strict_training_frame


def test_prior_round_is_available_but_current_and_future_are_not():
    store = pd.DataFrame({
        "fixture_id": ["before", "round1", "round2", "round3"],
        "match_at": pd.to_datetime(["2026-06-20", "2026-07-04", "2026-07-11", "2026-07-18"], utc=True),
    })
    fold = HistoricalFold("round2", "ncr", 2026, "2026-07-11T00:00:00Z", ("round2",), ("round2",), "mixed")
    assert set(strict_training_frame(store, fold).fixture_id) == {"before", "round1"}
    frozen = HistoricalFold("tournament", "ncr", 2026, "2026-07-04T00:00:00Z", ("round1", "round2", "round3"), ("r1", "r2", "r3"), "mixed")
    assert set(strict_training_frame(store, frozen).fixture_id) == {"before"}


def test_date_only_history_excludes_the_entire_lock_day():
    frame = pd.DataFrame({"date": ["2026-07-03", "2026-07-04", "2026-07-05"], "value": [1, 2, 3]})
    assert history_before(frame, "2026-07-04T18:00:00Z").value.tolist() == [1]
    assert day_cutoff("2026-07-04T01:00:00+01:00") == pd.Timestamp("2026-07-04", tz="UTC")
    with pytest.raises(ValueError):
        history_before(pd.DataFrame({"date": [None]}), "2026-07-04")


def test_calendar_year_totals_cannot_see_later_rounds():
    seasons = pd.Series(["2026", "2025", "2025/2026", "2026/2027", "unknown"])
    assert completed_seasons(seasons, "2026-07-04").tolist() == [False, True, True, False, False]
    assert not completed_seasons(pd.Series(["2025/2026"]), "2026-06-30").iloc[0]
    assert completed_seasons(pd.Series(["2025/2026"]), "2026-07-01").iloc[0]
    assert completed_seasons(pd.Series(["2026"]), "2027-01-01").iloc[0]


def test_team_margin_is_once_per_match_and_shared_by_teammates():
    frame = pd.DataFrame({
        "fixture_id": ["old", "old", "current", "current", "next"],
        "team": ["A"] * 5,
        "date": pd.to_datetime(["2026-01-01"] * 2 + ["2026-01-08"] * 2 + ["2026-01-15"]),
        "team_score": [20, 20, 90, 90, 30], "opp_score": [10, 10, 0, 0, 0],
    })
    before = team_margin_history(frame, prior_only=True)
    assert before.iloc[:2].isna().all()
    assert before.iloc[2:4].tolist() == [10, 10]
    changed = frame.copy()
    changed.loc[changed.fixture_id.eq("current"), "team_score"] = 900
    after = team_margin_history(changed, prior_only=True)
    pd.testing.assert_series_equal(before.iloc[:4], after.iloc[:4])
    extra = pd.concat([frame.iloc[:2], frame.iloc[[0]], frame.iloc[2:]], ignore_index=True)
    assert team_margin_history(extra, prior_only=True).iloc[-1] == pytest.approx(before.iloc[-1])


def test_empirical_profiles_do_not_depend_on_future_results():
    from model.ncr_project import player_profiles, ATT, DEF, DISC
    rows = []
    for fixture, date, count in [("past", "2026-07-01", 1), ("future", "2026-07-20", 100)]:
        rows.append({"fixture_id": fixture, "date": date, "player_id": "p", "minutes": 80.0,
                     "started": True, **{event: float(count) for event in set(ATT) | set(DEF) | set(DISC)}})
    history = pd.DataFrame(rows)
    expected = player_profiles(history.iloc[:1], asof="2026-07-04")
    actual = player_profiles(history, asof="2026-07-04")
    assert actual["p"]["att"] == expected["p"]["att"]
    assert actual["p"]["n"] == 1
    assert player_profiles(history, asof="2026-07-21")["p"]["n"] == 2


def test_expected_metres_score_does_not_floor_the_mean():
    from model.unified.contracts import EventDistribution, RawPrediction
    from model.unified.rolling_evaluation import expected_points
    p = RawPrediction("f", "p", "player", "A", "B", "Back-three", False,
                      {"metres": EventDistribution("lognormal", 10.0, 1e-8)},
                      EventDistribution("lognormal", 80.0, 1.0))
    assert expected_points([p], "six_nations")[0] == pytest.approx(0.5, abs=0.001)
    assert expected_points([p], "ncr")[0] == 0.0


def test_promotion_gate_rejects_missing_competitions(tmp_path):
    import json
    from model.unified.rolling_evaluation import summarize, MODELS
    directory = tmp_path / "ncr-2026-gw1"
    directory.mkdir()
    rows = [{"round": "ncr-2026-gw1", "competition": "ncr", "season": 2026,
             "model": model, "n": 10, "mae": 2 if model == "empirical_baseline" else 1,
             "team_points": 100 if model == "empirical_baseline" else 200} for model in MODELS]
    (directory / "metrics.json").write_text(json.dumps(rows))
    result = summarize(tmp_path)
    assert not result["pr_eligible"]
    assert result["missing_rounds"]
    assert not result["candidates"]["p3_weighted"]["all_available_metric_checks_passed"]
