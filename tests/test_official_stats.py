import numpy as np
import pandas as pd
import pytest

from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.official_stats import (
    OfficialStatConfig, SixNationsOfficialStats, _metres_mapping, potm_probabilities, rugbypass_turnovers,
    win_probability,
)

CUTOFF = pd.Timestamp('2026-02-07T14:00:00Z')


def official_rows():
    rows = []
    for season, day, scale in ((2025, '2025-02-01', 3.0), (2026, '2026-02-01', 0.0), (2026, '2026-03-01', 0.0)):
        for i, position in enumerate(['Back-row', 'Back-row', 'Back-three', 'Back-three']):
            metres, runs = 20.0 + 10*i, 5.0 + i
            rows.append({'fixture_id': f'f{season}{day[5:7]}', 'player_id': f'p{i}', 'team': 'A',
                         'player_name': f'P. Player{"abcd"[i]}', 'key': f'p|player{"abcd"[i]}', 'match_at': pd.Timestamp(day, tz='UTC'),
                         'season': season, 'round': 1, 'position': position, 'minutes': 80.0, 'metres': metres,
                         'runs': runs, 'off_Min': 80.0, 'off_MC': metres + scale*runs,
                         'off_BS': 1.0 if position == 'Back-row' else 0.0, 'off_POTM': 0.0, 'off_Pts': 10.0})
    return pd.DataFrame(rows)


def store_rows():
    return pd.DataFrame({'fixture_id': ['h1'], 'team': ['A'], 'opponent': ['B'], 'team_score': [30], 'opp_score': [10],
                         'home_away': ['home'], 'match_at': [pd.Timestamp('2025-11-01', tz='UTC')]})


def compstats():
    return pd.DataFrame({
        'key': ['p|playera', 'p|playerb', 'p|playera', 'x|dup', 'x|dup'],
        'slug': ['playera', 'playerb', 'playera', 'dup-a', 'dup-b'],
        'competition': ['URC', 'URC', 'URC', 'URC', 'URC'],
        'season': ['2024/2025', '2024/2025', '2025/2026', '2024/2025', '2024/2025'],
        'minutes': [800, 800, 800, 800, 700], 'turnovers_won': [20, 2, 50, 5, 5]})


def prediction(i, team='A', opponent='B', position='Back-row', minutes=80.0, xp_tries=0.1):
    events = {'tackle_turnover': EventDistribution('negative_binomial', 0.2, 1.0),
              'metres': EventDistribution('lognormal', 30.0, 400.0),
              'runs': EventDistribution('negative_binomial', 6.0, 5.0),
              'tries': EventDistribution('poisson', xp_tries),
              'potm': EventDistribution('bernoulli', 0.02)}
    return RawPrediction('g1', f'p{i}', f'P. Player{"abcd"[i]}', team, opponent, position, True, events,
                         EventDistribution('lognormal', minutes, 25.0))


def candidates(raw, started):
    return pd.DataFrame({'fixture_id': [p.fixture_id for p in raw], 'player_id': [p.player_id for p in raw],
                         'team': [p.team for p in raw], 'opponent': [p.opponent for p in raw],
                         'home_away': ['home' if p.team == 'A' else 'away' for p in raw], 'started': started})


def test_rugbypass_uses_completed_seasons_and_drops_ambiguous_keys():
    table = rugbypass_turnovers(compstats(), CUTOFF)
    assert table.loc['p|playera', 'tw'] == 20  # 2025/2026 is not complete before February 2026
    assert 'x|dup' not in table.index


def test_metres_mapping_prefers_current_season_rounds():
    history = official_rows()
    a, b = _metres_mapping(history, 2026)
    assert a == pytest.approx(1.0) and b == pytest.approx(0.0, abs=1e-9)
    a, b = _metres_mapping(history[history.season.eq(2025)], 2026)  # no 2026 rounds yet: latest season
    assert a == pytest.approx(1.0) and b == pytest.approx(3.0)


def test_fit_ignores_rows_after_the_lock():
    history = official_rows()
    early = SixNationsOfficialStats().fit(history, compstats(), store_rows(), '2026-02-15', 2026)
    later = history.assign(off_BS=np.where(history.match_at > pd.Timestamp('2026-02-15', tz='UTC'), 9.0, history.off_BS))
    same = SixNationsOfficialStats().fit(later, compstats(), store_rows(), '2026-02-15', 2026)
    assert early.position_rate == same.position_rate
    assert early.metres_map == same.metres_map


def test_breakdown_steals_follow_official_rates_and_minutes():
    raw = [prediction(0), prediction(2, position='Back-three', minutes=40.0)]
    config = OfficialStatConfig(metres=False, potm=False)
    adapter = SixNationsOfficialStats(config).fit(official_rows(), compstats(), store_rows(), CUTOFF, 2026)
    adjusted = adapter.adjust(raw, candidates(raw, [True, True]))
    rates = adapter.breakdown_rates(['p|playera', 'p|playerc'], ['Back-row', 'Back-three'])
    assert adjusted[0].events['tackle_turnover'].mean == pytest.approx(rates[0])
    assert adjusted[1].events['tackle_turnover'].mean == pytest.approx(rates[1]*40/80)
    assert rates[0] > 2*rates[1]
    assert adjusted[0].events['metres'] == raw[0].events['metres']


def test_metres_keep_coefficient_of_variation():
    raw = [prediction(0)]
    adapter = SixNationsOfficialStats(OfficialStatConfig(breakdown_steals=False, potm=False)).fit(
        official_rows()[lambda f: f.season.eq(2025)], compstats(), store_rows(), CUTOFF, 2026)
    new = adapter.adjust(raw, candidates(raw, [True]))[0].events['metres']
    old = raw[0].events['metres']
    assert new.mean == pytest.approx(30.0 + 3.0*6.0)
    assert new.dispersion/new.mean**2 == pytest.approx(old.dispersion/old.mean**2)


def test_potm_probabilities_one_per_match_and_favour_winners():
    frame = pd.DataFrame({'fixture_id': ['g'] * 4, 'team': ['A', 'A', 'B', 'B'], 'started': [True, False, True, True],
                          'xp': [20.0, 20.0, 30.0, 10.0], 'pwin': [0.8, 0.8, 0.2, 0.2]})
    p = potm_probabilities(frame, 0.1, 0.05)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(0.8/1.05) and p[1] == pytest.approx(0.8*0.05/1.05)
    assert p[2] > p[3]
    assert win_probability(np.array([0.0]))[0] == pytest.approx(0.5)


def test_adjust_rejects_misaligned_candidates():
    raw = [prediction(0), prediction(1)]
    adapter = SixNationsOfficialStats().fit(official_rows(), compstats(), store_rows(), CUTOFF, 2026)
    with pytest.raises(ValueError):
        adapter.adjust(raw, candidates(raw[::-1], [True, True]))


def test_full_adapter_sets_potm_per_match():
    raw = [prediction(0, xp_tries=0.5), prediction(1), prediction(2, team='B', opponent='A'),
           prediction(3, team='B', opponent='A')]
    adapter = SixNationsOfficialStats().fit(official_rows(), compstats(), store_rows(), CUTOFF, 2026)
    adjusted = adapter.adjust(raw, candidates(raw, [True, True, True, True]))
    total = sum(p.events['potm'].mean for p in adjusted)
    assert total == pytest.approx(1.0)
    assert adjusted[0].events['potm'].mean > adjusted[2].events['potm'].mean  # A won its only Elo match at home
