import argparse
import hashlib
import json
import os

import pandas as pd

from model.unified.raw_benchmark import coverage
from model.unified.raw_benchmark.config import DIRECT_EVENTS


KEYS = ['fixture_id', 'player_id', 'team']
EXPECTED_INPUT = '2b70331c0cf215def67831cd27c987362e2ec8330b2afb41c8d204f48ef17313'


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def cache_hashes(fixtures):
    return {fixture: digest(coverage.CACHE / f'match_{fixture}.json') for fixture in sorted(fixtures)}


def raw_differences(original, parsed):
    differences = {}
    for event in ('yellow_cards', 'red_cards', *DIRECT_EVENTS):
        left = original[event]
        right = parsed[f'{event}_cache']
        values = ~(left.eq(right) | (left.isna() & right.isna()))
        masks = original[f'available__{event}'].ne(parsed[f'available__{event}_cache'])
        if values.any() or masks.any():
            differences[event] = {'values': int(values.sum()), 'availability': int(masks.sum())}
    return differences


def build_overlay(input_path, output_dir):
    if os.path.exists(output_dir):
        raise FileExistsError(output_dir)
    input_hash = digest(input_path)
    if input_hash != EXPECTED_INPUT:
        raise ValueError('Canonical input hash differs from the diagnosed store')
    original = pd.read_csv(input_path, low_memory=False, dtype={'fixture_id': str, 'player_id': str})
    fixtures = set(original.fixture_id)
    before_hashes = cache_hashes(fixtures)
    _, players, ledger = coverage.build_cache_tables(fixtures)
    parsed = players.set_index(KEYS).reindex(pd.MultiIndex.from_frame(original[KEYS])).reset_index()
    if parsed['available__minutes_cache'].isna().any():
        raise ValueError('Cache keys do not cover the canonical population')
    differences = raw_differences(original, parsed)
    if differences:
        raise ValueError(f'Raw event differences: {differences}')
    changed = original.available__minutes & ~parsed.available__minutes_cache
    overlay = original.loc[changed, KEYS + ['player_name', 'date', 'competition_level', 'minutes', 'available__minutes', 'provenance__minutes']].copy()
    overlay = overlay.rename(columns={'minutes': 'previous_minutes', 'available__minutes': 'previous_available__minutes', 'provenance__minutes': 'previous_provenance__minutes'})
    overlay['minutes'] = float('nan')
    overlay['available__minutes'] = False
    overlay['provenance__minutes'] = 'unobserved_entry_time'
    overlay['reason'] = 'Positive observed on-field activity; no recorded substitute entry time'
    corrected = original.copy()
    corrected.loc[changed, 'minutes'] = float('nan')
    corrected.loc[changed, 'available__minutes'] = False
    corrected.loc[changed, 'provenance__minutes'] = 'unobserved_entry_time'
    after_hashes = cache_hashes(fixtures)
    if before_hashes != after_hashes or digest(input_path) != input_hash:
        raise ValueError('Immutable inputs changed during overlay construction')
    os.makedirs(output_dir)
    overlay.to_csv(os.path.join(output_dir, 'minutes_overlay.csv'), index=False)
    corrected.to_csv(os.path.join(output_dir, 'player_match.csv'), index=False)
    ledger.to_csv(os.path.join(output_dir, 'fixture_event_coverage.csv'), index=False)
    with open(os.path.join(output_dir, 'cache_hashes.json'), 'w') as handle:
        json.dump(before_hashes, handle, sort_keys=True, indent=2)
        handle.write('\n')
    manifest = {'rule_version': 'unlogged_activity_minutes_v1', 'input_path': input_path, 'input_sha256': input_hash, 'source_path': coverage.__file__, 'source_sha256': digest(coverage.__file__), 'rows': len(original), 'fixtures': len(fixtures), 'changed_rows': int(changed.sum()), 'changed_international_rows': int((changed & original.competition_level.eq('international')).sum()), 'raw_event_differences': differences, 'cache_hashes_unchanged': before_hashes == after_hashes, 'model_fits': 0, 'model_gains': 'untested', 'outputs': {name: digest(os.path.join(output_dir, name)) for name in sorted(os.listdir(output_dir))}}
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build_overlay(args.input, args.output)
