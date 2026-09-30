import json

import numpy as np
import pandas as pd
import pytest

from model.unified.friendly_cohort import complete_fixture_cohort
from model.unified.raw_benchmark.folds import masked_candidates
from model.unified.schema import EVENTS


def setup_cohort(tmp_path):
    match = dict(id=10, home_team='A', away_team='B', date='2024-11-01T12:00:00Z', comp_id=30)
    payload = {'results': {'match': match, 'events': [],
        'home': {'teamsheet': [dict(player_id=1, name='Known', position=1, substitute=False,
            match_stats={'tries': 2, 'tackles': 4})]},
        'away': {'teamsheet': [dict(player_id=2, name='New', position=16, substitute=True,
            match_stats={'tries': 0})]}}}
    (tmp_path/'match_10.json').write_text(json.dumps(payload))
    store = pd.DataFrame([dict(fixture_id='10', player_id='1', team='A', position='Prop',
        tries=99, tackles=99, minutes=80, available__tackles=True)])
    return store, payload


def test_full_cached_opponent_is_evaluated_and_cold_start_is_recorded(tmp_path):
    store, _ = setup_cohort(tmp_path)
    result = complete_fixture_cohort(store, '10', tmp_path).set_index('player_id')
    assert set(result.team) == {'A', 'B'}
    assert result.loc['1', 'canonical_player']
    assert not result.loc['2', 'canonical_player']
    assert result.loc['2', 'position'] == 'Hooker'
    assert result.loc['2', 'is_forward']
    assert not result.loc['2', 'started']
    assert result.loc['2', 'minutes'] == 0
    assert result.loc['2', 'tries'] == 0
    assert result.loc['1', 'tries'] == 2
    assert not result.loc['2', 'available__tackles']
    assert np.isnan(result.loc['2', 'tackles'])


def test_incomplete_cached_teamsheet_fails(tmp_path):
    store, payload = setup_cohort(tmp_path)
    payload['results']['away']['teamsheet'] = []
    (tmp_path/'match_10.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='both complete'):
        complete_fixture_cohort(store, '10', tmp_path)


def test_duplicate_canonical_key_fails(tmp_path):
    store, _ = setup_cohort(tmp_path)
    with pytest.raises(ValueError, match='unique canonical'):
        complete_fixture_cohort(pd.concat([store, store]), '10', tmp_path)


def test_full_cohort_targets_are_masked_before_feature_building(tmp_path):
    store, _ = setup_cohort(tmp_path)
    candidates = masked_candidates(complete_fixture_cohort(store, '10', tmp_path))
    for target in ('minutes', *EVENTS):
        assert candidates[target].isna().all()
        assert not candidates[f'available__{target}'].any()
