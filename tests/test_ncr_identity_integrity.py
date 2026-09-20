"""Preserve distinct people before computing features or ensemble predictions."""
import json
from pathlib import Path

import pandas as pd
import pytest

from model.unified.data import ROOT
from model.unified.v3 import shadow


def inputs(tmp_path, monkeypatch, *, duplicate=False):
    directory = tmp_path / 'ncr'
    directory.mkdir()
    pd.DataFrame([
        {'fantasy_id':1,'api_player_id':232158,'full_name':'Eddie James'},
        {'fantasy_id':2,'api_player_id':232158 if duplicate else 366690,'full_name':'Kane James'},
    ]).to_csv(directory/'ncr_player_crosswalk.csv',index=False)
    pd.DataFrame([{'gameday':2,'home':'Wales','away':'Argentina',
                   'match_id':291583,'game_date':'2026-07-11'}]).to_csv(directory/'ncr_fixtures.csv',index=False)
    monkeypatch.setattr(shadow,'DATA',tmp_path)
    return pd.DataFrame([
        {'id':1,'name':'Eddie James','team':'Wales','pos':'Centre','status':'P'},
        {'id':2,'name':'Kane James','team':'Wales','pos':'Loose Forward','status':'B'},
    ])


def test_duplicate_canonical_id_is_rejected_before_prediction(tmp_path,monkeypatch):
    projection = inputs(tmp_path,monkeypatch,duplicate=True)
    with pytest.raises(ValueError,match='fixture/player/team'):
        shadow.ncr_candidates(2,projection)


def test_distinct_players_keep_both_fantasy_entries_and_masked_labels(tmp_path,monkeypatch):
    projection = inputs(tmp_path,monkeypatch)
    candidates = shadow.ncr_candidates(2,projection)
    assert candidates.fantasy_id.tolist() == [1,2]
    assert candidates.player_id.tolist() == ['232158','366690']
    assert candidates.player_name.tolist() == ['Eddie James','Kane James']
    assert candidates.position.tolist() == ['Centre','Back-row']
    assert candidates.minutes.isna().all()
    assert not candidates['available__tries'].any()
    assert not candidates.duplicated(['fixture_id','player_id','team']).any()


def test_corrected_crosswalk_matches_a_pre_tournament_cached_identity():
    path = ROOT/'data/cache/match_8492398.json'
    fixture = json.loads(path.read_text())['results']
    assert pd.Timestamp(fixture['match']['date']) < pd.Timestamp('2026-07-04',tz='UTC')
    people = [p for side in ('home','away') for p in fixture[side]['teamsheet'] if p['name'] == 'Kane James']
    assert len(people) == 1
    assert int(people[0]['player_id']) == 366690
    crosswalk = pd.read_csv(ROOT/'data/ncr/ncr_player_crosswalk.csv')
    kane = crosswalk.loc[crosswalk.fantasy_id.eq(243167)].iloc[0]
    eddie = crosswalk.loc[crosswalk.fantasy_id.eq(237271)].iloc[0]
    assert kane.full_name == 'Kane James'
    assert int(kane.api_player_id) == 366690
    assert eddie.full_name == 'Eddie James'
    assert int(eddie.api_player_id) == 232158
