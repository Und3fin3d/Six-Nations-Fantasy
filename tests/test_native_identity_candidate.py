"""The experimental player representation must not change the default control."""
from dataclasses import asdict
import json

import pandas as pd
import pytest

from model.unified.domain_experiment import Candidate, CONTROL, DATA, STUDIES, candidate_tree


def test_control_wiring_is_unchanged():
    tree = candidate_tree(STUDIES['identity'][CONTROL], 17)
    assert tree.pool_player_id is True
    assert tree.player_effects is True
    assert tree.native_categories is True
    assert tree.random_state == 17
    assert tree.weighting == 'natural'
    assert tree.time_half_life_days is None
    assert tree.n_estimators == 180
    assert tree.num_leaves == 23


def test_identity_wiring_preserves_actual_player_ids():
    tree = candidate_tree(STUDIES['identity']['p3_identity_native'], 17)
    frame = pd.DataFrame({'player_id':['a','b']})
    assert tree.pool_player_id is False
    assert tree.player_effects is False
    assert tree.native_categories is True
    assert tree._frame(frame).player_id.tolist() == ['a','b']
    assert tree.random_state == 17
    control = candidate_tree(STUDIES['identity'][CONTROL], 17)
    assert control._frame(frame).player_id.tolist() == ['pooled','pooled']
    assert frame.player_id.tolist() == ['a','b']


def test_default_candidate_has_no_identity_switch():
    assert Candidate('natural').player_identity is False
    assert STUDIES['identity'][CONTROL] == STUDIES['domain'][CONTROL]
    assert STUDIES['identity'][CONTROL] == STUDIES['seedbag'][CONTROL]


@pytest.mark.parametrize('value', ['yes',None,1])
def test_identity_switch_requires_boolean(value):
    with pytest.raises(ValueError, match='identity'):
        Candidate('natural',player_identity=value)


def test_frozen_identity_configuration_matches_the_factory():
    path = DATA/'unified/identity_search/selection.json'
    declared = json.loads(path.read_text())
    actual = json.loads(json.dumps({name:asdict(spec) for name,spec in STUDIES['identity'].items()}))
    assert declared['candidate_configs'] == actual
    assert declared['selected'] == 'p3_identity_native'
    assert declared['selection_used_fantasy_outcomes'] is False
    assert declared['selection_used_friendly15'] is False
