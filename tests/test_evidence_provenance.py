import pytest

from model.unified.evidence_provenance import (
    validate_source_reuse, normalized_spec, corrected_candidate_keys,
)


def test_only_orchestration_changes_allow_forecast_reuse():
    old = {'model/unified/gbdt.py':'a','model/unified/domain_experiment.py':'b'}
    new = dict(old, **{'model/unified/domain_experiment.py':'c',
                      'model/unified/evidence_provenance.py':'d'})
    assert set(validate_source_reuse(old,new)) == {
        'model/unified/domain_experiment.py','model/unified/evidence_provenance.py'}
    new['model/unified/gbdt.py'] = 'changed'
    with pytest.raises(ValueError,match='refit'):
        validate_source_reuse(old,new)


@pytest.mark.parametrize('change',['add','remove'])
def test_added_or_deleted_numerical_sources_require_refitting(change):
    old = {'model/unified/gbdt.py':'a'}
    new = {'model/unified/gbdt.py':'a','model/unified/unknown_model.py':'b'}
    with pytest.raises(ValueError,match='refit'):
        validate_source_reuse(*( (old,new) if change == 'add' else (new,old)))


def test_old_default_config_normalizes_without_changing_supplied_values():
    value = {'weighting':'natural','half_life_days':None,'seeds':[17,29,43]}
    assert normalized_spec(value) == dict(value,player_identity=False)
    assert 'player_identity' not in value
    assert normalized_spec({'seeds':[43],'player_identity':True})['seeds'] == [43]
    assert normalized_spec({'player_identity':True})['player_identity'] is True


def ncr2_keys():
    keys = [['291583',str(i),'Wales'] for i in range(250)]
    keys[219] = ['291583','232158','Wales']
    keys[233] = ['291583','232158','Wales']
    return keys


def test_exact_kane_identity_correction_preserves_every_other_row():
    old = ncr2_keys()
    new = [row.copy() for row in old]
    new[233][1] = '366690'
    assert corrected_candidate_keys(old,new,'ncr_2026_r2') == 1
    assert old[233][1] == '232158'
    new[3][1] = 'unexpected'
    with pytest.raises(ValueError,match='unexpected'):
        corrected_candidate_keys(old,new,'ncr_2026_r2')


def test_wrong_person_or_dropped_player_cannot_pass_correction():
    old = ncr2_keys()
    new = [row.copy() for row in old]
    new[219][1] = '366690'
    with pytest.raises(ValueError):
        corrected_candidate_keys(old,new,'ncr_2026_r2')
    with pytest.raises(ValueError):
        corrected_candidate_keys(old,old[:233]+old[234:],'ncr_2026_r2')


def test_unaffected_cohorts_must_be_exact_and_unique():
    keys = [['a','1','team'],['a','2','team']]
    assert corrected_candidate_keys(keys,keys,'six_nations_2025_r1') == 0
    with pytest.raises(ValueError):
        corrected_candidate_keys(keys,keys[::-1],'ncr_2026_r1')
    with pytest.raises(ValueError):
        corrected_candidate_keys([keys[0],keys[0]],[keys[0],keys[0]],'ncr_2026_r3')
