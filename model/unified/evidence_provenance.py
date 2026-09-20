"""Explicit provenance for immutable reused forecasts after a crosswalk repair.

This does not rewrite old manifests. Only evaluation/orchestration code may
change when reusing forecasts; all fitted-model, feature and scoring code must
match. NCR is always refit and rescored rather than exempted from the audit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .data import ROOT
from .domain_experiment import CONTROL, STUDIES

# These exact orchestration/validation modules are not fitted model mechanisms.
INFRASTRUCTURE = frozenset({
    'model/unified/domain_experiment.py',
    'model/unified/comparison_report.py',
    'model/unified/evidence_provenance.py',
    'model/unified/v3/shadow.py',
})


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_spec(spec: dict) -> dict:
    result = dict(spec)
    result.setdefault('player_identity',False)
    result.setdefault('seeds',[17])
    return result


def validate_source_reuse(recorded: dict[str,str], current: dict[str,str]) -> list[str]:
    """Any added/deleted/modified numerical model module forbids forecast reuse."""
    changed = [name for name in sorted(set(recorded) | set(current))
               if recorded.get(name) != current.get(name)]
    unauthorized = set(changed) - INFRASTRUCTURE
    if unauthorized:
        raise ValueError(f'numerical source changed; refit required: {sorted(unauthorized)}')
    return changed


def validate_configs(recorded: dict, study: str) -> None:
    if study not in STUDIES:
        raise ValueError('unknown source study')
    expected = json.loads(json.dumps({name:asdict(spec) for name,spec in STUDIES[study].items()}))
    actual = {name:normalized_spec(spec) for name,spec in recorded.items()}
    if actual != expected:
        raise ValueError('recorded candidate configuration differs from the prespecified study')


def corrected_candidate_keys(previous: list[list[str]], current: list[list[str]], name: str) -> int:
    """Only Kane's unique, identified NCR2 record may change canonical identity."""
    expected = [list(key) for key in previous]
    changes = 0
    if name == 'ncr_2026_r2':
        duplicates = [i for i,key in enumerate(previous) if key == ['291583','232158','Wales']]
        if len(duplicates) != 2:
            raise ValueError('the archived NCR2 collision differs from the diagnosed record')
        # Index and fantasy ID are verified independently against archived predictions.
        if duplicates != [219,233]:
            raise ValueError('NCR2 candidate order differs from the pinned crosswalk correction')
        expected[233][1] = '366690'
        changes = 1
    if expected != current or len({tuple(key) for key in current}) != len(current):
        raise ValueError('unexpected candidate-cohort change after identity correction')
    return changes


def validate_correction_files() -> dict:
    """Require the exact one-cell repair and a named historical cache record."""
    directory = ROOT/'data/unified/candidate_search'
    provenance = json.loads((directory/'identity_correction.json').read_text())
    path = ROOT/'data/ncr/ncr_player_crosswalk.csv'
    if file_hash(path) != provenance['crosswalk_after_sha256']:
        raise ValueError('corrected crosswalk changed')
    old = directory/'original_crosswalk.csv'
    if file_hash(old) != provenance['crosswalk_before_sha256']:
        raise ValueError('archived original crosswalk changed')
    before,after = pd.read_csv(old),pd.read_csv(path)
    expected = before.copy()
    selected = expected.fantasy_id.eq(243167)
    if selected.sum() != 1 or expected.loc[selected,'full_name'].iloc[0] != 'Kane James':
        raise ValueError('wrong fantasy identity selected for repair')
    expected.loc[selected,'api_player_id'] = 366690.0
    pd.testing.assert_frame_equal(expected,after,check_exact=True)
    cache = ROOT/provenance['identity_cache']
    if file_hash(cache) != provenance['identity_cache_sha256']:
        raise ValueError('historical identity cache changed')
    fixture = json.loads(cache.read_text())['results']
    if pd.Timestamp(fixture['match']['date']) >= pd.Timestamp('2026-07-04',tz='UTC'):
        raise ValueError('identity evidence must precede NCR')
    people = [p for side in ('home','away') for p in fixture[side]['teamsheet']
              if p['name'] == 'Kane James']
    if len(people) != 1 or int(people[0]['player_id']) != 366690:
        raise ValueError('historical cache does not support repaired identity')
    return provenance


def inspect_manifest(manifest: dict, previous: dict, *, name: str, selection_hashes: dict) -> dict:
    current_source = {str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'model').rglob('*.py'))}
    numerical_changes = validate_source_reuse(manifest['source_sha256'],current_source)
    study = manifest['study']
    validate_configs(manifest['candidate_configs'],study)
    if manifest['selection_sha256'] != selection_hashes[study]:
        raise ValueError('source prediction does not match its pinned selection hash')
    for key in ('store_sha256','features_sha256','weight_config_sha256','packages',
                'cutoff','history_keys_sha256'):
        if manifest[key] != previous[key]:
            raise ValueError(f'{name}: historical input changed: {key}')
    changed_keys = corrected_candidate_keys(previous['candidate_keys'],manifest['candidate_keys'],name)
    if name.startswith('ncr_'):
        if study != 'search' or manifest['source_sha256'] != current_source:
            raise ValueError('all NCR forecasts must be refitted with the tested corrected source')
        crosswalk = ROOT/'data/ncr/ncr_player_crosswalk.csv'
        if manifest.get('ncr_crosswalk_sha256') != file_hash(crosswalk):
            raise ValueError('corrected NCR crosswalk is missing from job provenance')
    elif study not in {'seedbag','identity'}:
        raise ValueError('unexpected source for reused non-NCR forecasts')
    return dict(job=name,study=study,original_source_sha256=manifest['source_sha256'],
                original_selection_sha256=manifest['selection_sha256'],
                orchestration_changes_since_fit=numerical_changes,
                corrected_player_keys=changed_keys,refitted=name.startswith('ncr_'))
