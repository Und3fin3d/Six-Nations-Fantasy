import copy
import json

import numpy as np
import pandas as pd
import pytest

from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.friendly_eval import (
    TARGETS, COUNT_TARGETS, score_stats, summarise_stats,
    select_fixtures, validate_manifest, write_manifest,
)


def truth_and_predictions():
    truth, predictions = [], []
    for fixture, count, error in [('a', 1, 2.0), ('b', 3, 4.0)]:
        for i in range(count):
            row = dict(fixture_id=fixture, player_id=str(i), team='A')
            row.update({event: 0.0 for event in TARGETS})
            row.update({f'available__{event}': True for event in TARGETS})
            truth.append(row)
            events = {event: EventDistribution('poisson', error) for event in COUNT_TARGETS}
            events['metres'] = EventDistribution('lognormal', 200, 1)
            predictions.append(RawPrediction(fixture, str(i), 'P', 'A', 'B', 'Prop', True,
                events, EventDistribution('lognormal', 80, 1)))
    return pd.DataFrame(truth), predictions


def test_mean_weights_games_equally_and_keeps_units_separate():
    truth, predictions = truth_and_predictions()
    metrics = score_stats(truth, predictions, engine='m')
    summary, per_stat = summarise_stats(metrics, ['a', 'b'])
    assert summary.iloc[0].count_stat_mae == pytest.approx(3.0)  # not 3.5 by player
    assert summary.iloc[0].minutes_mae == 80
    assert summary.iloc[0].metres_mae == 200
    assert len(per_stat) == len(TARGETS)


def test_reordering_predictions_is_safe_because_alignment_is_keyed():
    truth, predictions = truth_and_predictions()
    pd.testing.assert_frame_equal(score_stats(truth, predictions, engine='m'),
        score_stats(truth, predictions[::-1], engine='m'))


def test_missing_label_is_not_zero_but_observed_zero_counts():
    truth, predictions = truth_and_predictions()
    truth.loc[1, 'tries'] = 999
    truth.loc[1, 'available__tries'] = False
    truth.loc[2, 'minutes'] = 0  # unused substitute still included
    metrics = score_stats(truth, predictions, engine='m')
    row = metrics[metrics.fixture_id.eq('b') & metrics.target.eq('tries')].iloc[0]
    assert row.n == 2 and row.mae == 4


def test_missing_observed_prediction_fails_not_silently_dropped():
    truth, predictions = truth_and_predictions()
    del predictions[0].events['tries']
    with pytest.raises(ValueError, match='missing predictions'):
        score_stats(truth, predictions, engine='m')


@pytest.mark.parametrize('mode', ['missing', 'duplicate', 'wrong_team'])
def test_prediction_key_parity(mode):
    truth, predictions = truth_and_predictions()
    if mode == 'missing':
        predictions.pop()
    elif mode == 'duplicate':
        predictions[-1] = predictions[0]
    else:
        from dataclasses import replace
        predictions[-1] = replace(predictions[-1], team='wrong')
    with pytest.raises(ValueError, match='cohort'):
        score_stats(truth, predictions, engine='m')


def test_incomplete_or_unequal_comparison_support_fails():
    truth, predictions = truth_and_predictions()
    a = score_stats(truth, predictions, engine='a')
    with pytest.raises(ValueError, match='incomplete'):
        summarise_stats(a.iloc[1:], ['a', 'b'])
    b = a.assign(engine='b')
    b.loc[0, 'n'] += 1
    with pytest.raises(ValueError, match='supports differ'):
        summarise_stats(pd.concat([a, b]), ['a', 'b'])


def test_wholly_unobserved_stat_is_flagged_not_an_artificial_improvement():
    truth, predictions = truth_and_predictions()
    truth['available__tries'] = False
    summary, per_stat = summarise_stats(score_stats(truth, predictions, engine='m'), ['a', 'b'])
    assert not summary.iloc[0].count_support_complete
    assert np.isnan(summary.iloc[0].count_stat_mae)
    assert per_stat[per_stat.target.eq('tries')].iloc[0].observed_fixtures == 0


def test_metadata_only_selection_excludes_fantasy_and_future_and_is_write_once(tmp_path):
    rows = []
    for i in range(20):
        date = f'2025-11-{i+1:02d}T12:00:00Z'
        match = dict(id=i, comp_id=30, status='Result', date=date, home_team='A', away_team='B')
        (tmp_path / f'match_{i}.json').write_text(json.dumps({'results': {'match': match}}))
        rows.append(dict(fixture_id=str(i), player_id='p', team='A', competition_level='international',
            competition_id_cache=30, match_at=date))
    store = pd.DataFrame(rows)
    manifest = select_fixtures(store, tmp_path, forbidden=['17'], asof='2025-11-19', count=15)
    ids = [item['fixture_id'] for item in manifest['fixtures']]
    assert ids == [str(i) for i in range(2, 17)]
    validate_manifest(manifest, store, tmp_path, forbidden=['17'])
    path = tmp_path/'manifest.json'
    write_manifest(path, manifest)
    write_manifest(path, manifest)
    changed = copy.deepcopy(manifest)
    changed['asof'] = 'changed'
    with pytest.raises(FileExistsError):
        write_manifest(path, changed)
    with pytest.raises(ValueError, match='fantasy-labelled'):
        validate_manifest(manifest, store, tmp_path, forbidden=[ids[0]])
    fixture_path = tmp_path/f'match_{ids[0]}.json'
    fixture_path.write_text(fixture_path.read_text() + ' ')
    with pytest.raises(ValueError, match='cache changed'):
        validate_manifest(manifest, store, tmp_path, forbidden=[])
