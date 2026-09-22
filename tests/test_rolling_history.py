"""Regression tests for equal, strictly historical model information."""
import numpy as np
import pandas as pd
import pytest

from model.ncr_project import ATT, DEF, DISC, player_profiles
from model.unified.features import build_pit_features
from model.unified.schema import EVENTS


def _history():
    rows = []
    for fixture, date, tries in [('past', '2026-07-01', 0), ('current', '2026-07-11', 9), ('future', '2026-07-18', 20)]:
        row = dict(fixture_id=fixture, date=date, player_id='p', player_name='Player', team='A', opponent='B', position='Prop', started=True, minutes=80.0, jersey=1, is_forward=True, competition_level='international', team_score=10.0, opp_score=0.0)
        row.update({e: 0.0 for e in EVENTS})
        row.update({f'available__{e}': True for e in EVENTS})
        row['available__minutes'] = True
        row['tries'] = tries
        rows.append(row)
    return pd.DataFrame(rows)


def test_empirical_profiles_exclude_current_and_future_matches():
    profile = player_profiles(_history(), asof=pd.Timestamp('2026-07-11'))['p']
    assert profile['att'] == 0.0
    assert profile['n'] == 1


def test_empirical_club_profiles_exclude_future_matches():
    history = _history().iloc[:1]
    club = _history().iloc[1:]
    profile = player_profiles(history, club, asof=pd.Timestamp('2026-07-11'))['p']
    assert profile['att'] == 0.0


def test_team_form_is_prior_fixture_not_prior_player_row():
    history = _history().iloc[:2].copy()
    second_player = history.copy()
    second_player['player_id'] = 'q'
    frame = pd.concat([history, second_player], ignore_index=True)
    frame['date'] = pd.to_datetime(frame['date'])
    before = build_pit_features(frame)
    frame.loc[frame.fixture_id.eq('current'), 'team_score'] = 9999
    after = build_pit_features(frame)
    key = ['fixture_id', 'player_id', 'team']
    a = before.set_index(key)['team_recent_margin'].sort_index()
    b = after.set_index(key)['team_recent_margin'].sort_index()
    pd.testing.assert_series_equal(a, b)
    assert before.loc[before.fixture_id.eq('past'), 'team_recent_margin'].isna().all()
    assert before.loc[before.fixture_id.eq('current'), 'team_recent_margin'].eq(10).all()


def test_date_only_histories_fail_closed_on_cutoff_day_and_missing_dates():
    from model.history import past_matches
    frame = pd.DataFrame({'date': ['2026-07-10', '2026-07-11', None]})
    result = past_matches(frame, '2026-07-11T18:00:00Z')
    assert result.index.tolist() == [0]
    assert len(frame) == 3


def test_exact_kickoff_cutoff_is_exclusive_and_timezone_aware():
    from model.history import past_matches
    frame = pd.DataFrame({'match_at': ['2026-07-10T20:00Z', '2026-07-11T17:00Z', '2026-07-11T18:00Z', None]})
    assert past_matches(frame, '2026-07-11T19:00+02:00').index.tolist() == [0]


def test_season_aggregates_cannot_use_an_unfinished_calendar_year():
    from model.history import past_seasons
    frame = pd.DataFrame({'season': ['2024', '2025', '2024/2025', '2025/2026', 'unknown']})
    assert past_seasons(frame, '2025-07-02').index.tolist() == [0, 2]
    assert past_seasons(frame, '2025-02-01').index.tolist() == [0]


def test_round_folds_admit_earlier_rounds_but_not_current_or_future():
    from model.unified.raw_benchmark.folds import HistoricalFold, rolling_folds, strict_training_frame
    frame = _history()
    frame['match_at'] = pd.to_datetime(frame.date, utc=True)
    tournament = HistoricalFold('ncr_2026', 'nations_championship', 2026,
                                '2026-07-01T00:00:00Z',
                                ('past', 'current', 'future'), ('r1', 'r2', 'r3'), 'mixed')
    rounds = rolling_folds(frame, [tournament])
    assert [fold.label for fold in rounds] == ['r1', 'r2', 'r3']
    assert strict_training_frame(frame, tournament).empty
    assert set(strict_training_frame(frame, rounds[1]).fixture_id) == {'past'}
    assert set(strict_training_frame(frame, rounds[2]).fixture_id) == {'past', 'current'}


def test_current_round_players_share_one_history_cutoff():
    from model.unified.raw_benchmark.folds import HistoricalFold, rolling_folds, strict_training_frame
    frame = _history()
    frame['match_at'] = pd.to_datetime(frame.date, utc=True)
    tournament = HistoricalFold('ncr_2026', 'nations_championship', 2026,
                                '2026-07-01T00:00:00Z',
                                ('past', 'current', 'future'), ('r1', 'r2', 'r2'), 'mixed')
    rounds = rolling_folds(frame, [tournament])
    assert set(rounds[1].fixture_ids) == {'current', 'future'}
    assert set(strict_training_frame(frame, rounds[1]).fixture_id) == {'past'}


def test_precomputed_training_features_equal_cutoff_rebuild():
    from model.unified.raw_benchmark.features import build_frozen_feature_frames
    from model.unified.raw_benchmark.folds import masked_candidates
    from model.unified.v4.features import add_v4_base_stats
    from model.unified.v4.gbdt import V4FeatureEncoder
    frame = _history()
    frame['date'] = pd.to_datetime(frame.date)
    all_features = add_v4_base_stats(build_pit_features(frame))
    train = frame.iloc[:2]
    candidate = masked_candidates(frame.iloc[2:])
    rebuilt, prediction = build_frozen_feature_frames(train, candidate, v4=True)
    cached, cached_prediction = build_frozen_feature_frames(
        train, candidate, v4=True, prepared_train=all_features.iloc[:2])
    columns = V4FeatureEncoder().fit(rebuilt).numeric_columns
    pd.testing.assert_frame_equal(rebuilt[columns], cached[columns])
    pd.testing.assert_frame_equal(prediction, cached_prediction)


def test_candidate_outcomes_cannot_change_features():
    from model.unified.raw_benchmark.features import build_frozen_feature_frames
    from model.unified.raw_benchmark.folds import masked_candidates
    from model.unified.v4.gbdt import V4FeatureEncoder
    frame = _history()
    frame['date'] = pd.to_datetime(frame.date)
    candidate = frame.iloc[2:].copy()
    a, before = build_frozen_feature_frames(frame.iloc[:2], masked_candidates(candidate), v4=True)
    candidate.loc[:, list(EVENTS)] = 99999.0
    candidate['minutes'] = 99999.0
    _, after = build_frozen_feature_frames(frame.iloc[:2], masked_candidates(candidate), v4=True)
    columns = V4FeatureEncoder().fit(a).numeric_columns
    pd.testing.assert_frame_equal(before.reindex(columns=columns), after.reindex(columns=columns))


def test_robust_minutes_use_international_role_not_club_minutes(monkeypatch):
    from model.unified.raw_benchmark.empirical import EmpiricalEventModel
    from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
    monkeypatch.setattr(EmpiricalEventModel, '_fit_rp_priors', lambda self: None)
    frame = _history().iloc[:1].copy()
    frame['minutes'] = 20.0
    frame['started'] = False
    club = pd.concat([frame]*20, ignore_index=True)
    club['competition_level'] = 'club'
    club['minutes'] = 80.0
    model = RobustEmpiricalEventModel(asof='2026-07-03').fit(pd.concat([frame,club],ignore_index=True))
    assert model.minutes_by_player[('p',False)] == pytest.approx(20.0)


def test_robust_position_rate_uses_observed_exposure(monkeypatch):
    from model.unified.raw_benchmark.empirical import EmpiricalEventModel
    from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
    monkeypatch.setattr(EmpiricalEventModel, '_fit_rp_priors', lambda self: None)
    frame = pd.concat([_history().iloc[:1]]*2,ignore_index=True)
    frame.loc[0,['player_id','minutes','tries']] = ['cameo',1.0,1.0]
    frame.loc[1,['player_id','minutes','tries']] = ['starter',80.0,0.0]
    model = RobustEmpiricalEventModel(asof='2026-07-03').fit(frame)
    assert model.position_priors[('Prop','tries')] == pytest.approx(80.0/81.0)


def test_expected_six_nations_points_accounts_for_metres_distribution():
    from model.unified.contracts import EventDistribution, RawPrediction
    from model.unified.rolling_eval import expected_points
    prediction = RawPrediction('f','p','Player','A','B','Prop',True,
        {'metres': EventDistribution('lognormal',15.0,100.0)},
        EventDistribution('lognormal',70.0,20.0))
    points = expected_points([prediction],'six_nations')[0]
    draws = prediction.events['metres'].sample(np.random.default_rng(17),200000)
    assert points == pytest.approx(np.floor(draws/10).mean(),abs=0.015)
    assert abs(points - np.floor(15/10)) > 0.01


def test_kickoff_before_lock_does_not_make_ongoing_match_results_available():
    from model.history import past_matches
    frame = pd.DataFrame({'match_at': ['2026-07-11T09:00Z','2026-07-11T11:00Z',
                                      '2026-07-11T11:30Z','2026-07-11T13:30Z']})
    assert past_matches(frame,'2026-07-11T14:15Z').index.tolist() == [0,1]


@pytest.mark.parametrize('shifted', [False,True])
def test_grouped_ewm_is_numerically_identical_with_missing_values_and_duplicate_indices(shifted):
    from model.unified.features import grouped_ewm
    values = pd.Series([1.0,np.nan,4.0,3.0,8.0,np.nan,2.0],index=[5,5,1,3,3,8,1])
    groups = pd.Series(['b','a','b','a','c','b','a'],index=values.index)
    expected = values.groupby(groups,sort=False).transform(
        lambda x: (x.shift(1) if shifted else x).ewm(halflife=4,min_periods=1).mean())
    pd.testing.assert_series_equal(grouped_ewm(values,groups,shifted=shifted,halflife=4),expected)
