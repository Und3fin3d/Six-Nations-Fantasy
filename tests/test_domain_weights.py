import numpy as np
import pandas as pd
import pytest

from model.unified.gbdt import training_weights


def test_natural_weights_leave_existing_fit_unchanged():
    frame = pd.DataFrame({'competition_level': ['club']*9+['international'], 'date': ['2024-01-01']*10})
    np.testing.assert_array_equal(training_weights(frame), np.ones(10))


def test_domain_weighting_equalises_total_weight_not_available_information():
    frame = pd.DataFrame({'competition_level': ['club']*9+['international'], 'date': ['2024-01-01']*10})
    weight = training_weights(frame, 'level_balanced')
    assert weight[:9].sum() == pytest.approx(weight[9])
    assert weight.mean() == pytest.approx(1)
    assert np.all(weight > 0)


def test_half_life_and_domain_weights_are_shared_by_the_two_stages():
    frame = pd.DataFrame({'competition_level': ['international']*2, 'date': ['2022-01-01', '2024-01-01']})
    weight = training_weights(frame, 'level_balanced', 730)
    assert weight[1] / weight[0] == pytest.approx(2)
    assert weight.mean() == pytest.approx(1)


@pytest.mark.parametrize('half_life', [0,-1,np.nan,np.inf])
def test_half_life_must_be_positive_and_finite(half_life):
    with pytest.raises(ValueError, match='half-life'):
        training_weights(pd.DataFrame({'competition_level': ['club']}), time_half_life_days=half_life)


def test_player_effects_use_the_same_domain_weights(monkeypatch):
    from model.unified.v4.gbdt import V4GBDT
    from model.unified.v4 import features as vf
    monkeypatch.setattr(vf, 'fit_shrinkage_k', lambda frame: {'tries': 220.0})
    class Encoder:
        def transform(self, frame):
            return np.zeros((len(frame),1)), np.zeros((len(frame),1))
    class Model:
        def predict(self, X):
            return np.ones(len(X))
    frame = pd.DataFrame({'player_id': ['p']*10, 'minutes': 80.0,
        'competition_level': ['club']*9+['international'], 'tries': [0.0]*9+[10.0],
        'available__tries': True, 'date': '2024-01-01'})
    model = V4GBDT(events=('tries',), pool_player_id=True, player_effects=True,
                   effect_min_rows=1, weighting='level_balanced')
    model.encoder = Encoder()
    model.models = {'tries': Model()}
    model._fit_effects(frame)
    weight = training_weights(frame, 'level_balanced')
    actual = np.dot(frame.tries, weight)
    predicted = weight.sum()
    alpha = 220 * actual / np.dot(frame.minutes, weight)
    assert model.effects[('tries','p')] == pytest.approx((alpha+actual)/(alpha+predicted))


def test_selection_uses_only_frozen_older_fixtures_and_minutes_guard():
    import json
    from model.unified.domain_experiment import CONFIG_DIR, CANDIDATES, CONTROL, select_development
    from model.unified.friendly_eval import TARGETS
    manifest = json.loads((CONFIG_DIR/'development_fixtures.json').read_text())
    fixtures = [row['fixture_id'] for row in manifest['fixtures']]
    rows = []
    for name in CANDIDATES:
        for fixture in fixtures:
            for target in TARGETS:
                value = 1.0 if name == CONTROL else (0.9 if name == 'p3_balanced_native' else 0.8)
                if name == 'p3_balanced_recent_native' and target == 'minutes':
                    value = 1.03  # more than the frozen 2% guard
                rows.append(dict(engine=name, fixture_id=fixture, target=target, n=46, mae=value))
    result = select_development(pd.DataFrame(rows), fixtures)
    assert result['selected'] == 'p3_balanced_native'
    assert result['selection_used_friendly15'] is False
    with pytest.raises(ValueError, match='frozen 12'):
        select_development(pd.DataFrame(rows), fixtures[:-1]+['8518300'])
