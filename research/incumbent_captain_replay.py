import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from model.data import load
from model.research import _apply_upside_head, config_from_dict
from research.incumbent_replay import ROOT, KEY, record_policy, write_json
from model.unified.raw_benchmark.coverage import sha256


def prefix_frame(forecasts, native, season, round_no):
    visible = native[(native.season < season) | (native.season.eq(season) & native['round'].le(round_no))].reset_index(drop=True)
    current = forecasts.season.eq(season) & forecasts['round'].eq(round_no) & forecasts['mode'].eq('lock_rebuilt')
    previous = forecasts.season.eq(season) & forecasts['round'].lt(round_no) & forecasts['mode'].eq('native_store')
    prefix = pd.concat([forecasts[previous], forecasts[current]], ignore_index=True)
    ordered = visible.loc[visible.season.eq(season), KEY].copy()
    for col in KEY[:2]:
        ordered[col] = ordered[col].astype(str)
    prefix = ordered.merge(prefix, on=KEY, how='left', validate='one_to_one')
    if len(prefix) != 138*round_no or prefix.upside_score.isna().any():
        raise ValueError('Cannot reconstruct complete incumbent prefix')
    return visible, prefix


def corrected_captain(visible, prefix, cfg, season, round_no):
    train_idx = np.where(visible.season.lt(season))[0]
    test_idx = np.where(visible.season.eq(season))[0]
    legacy = visible.copy()
    known = legacy.has_label.astype(bool) & legacy.is_modern.astype(bool) & np.isfinite(legacy.official_pts)
    legacy.loc[~known, 'recon_pts'] = legacy.loc[~known, 'target_pts'].fillna(legacy.loc[~known, 'recon_pts'])
    original = _apply_upside_head(legacy, train_idx, test_idx, prefix, cfg)
    current = prefix['round'].eq(round_no)
    error = float(np.max(np.abs(original.loc[current, 'upside_score']-prefix.loc[current, 'upside_score'])))
    if error > 1e-10:
        raise ValueError(f'Original upside could not be reproduced: {error}')
    updated = _apply_upside_head(visible, train_idx, test_idx, prefix, cfg)
    out = prefix.loc[current].copy()
    out['captain_score'] += cfg.captain_upside_weight*(updated.loc[current, 'upside_score']-out.upside_score)
    out['upside_score'] = updated.loc[current, 'upside_score']
    unchanged = [c for c in out if c.endswith('_hat') or c.startswith('hat_') or c in ('selection_score', 'sel_score', 'supersub_score')]
    if not out[unchanged].equals(prefix.loc[current, unchanged]):
        raise ValueError('Captain repair changed another forecast head')
    return out, {'maximum_legacy_upside_error': error, 'unchanged_heads': unchanged,
                 'training_rows_using_reconstruction': int((~known.iloc[train_idx]).sum())}


def replay_input_hashes(original):
    recorded = json.loads((original/'manifest.json').read_text())
    expected = {f'data/{name}': digest for name, digest in recorded['data_sha256'].items()}
    expected['research/promoted_config.json'] = recorded['promoted_config_sha256']
    actual = {name: sha256(ROOT/name) for name in expected}
    if actual != expected:
        changed = [name for name in expected if actual[name] != expected[name]]
        raise ValueError(f'Captain replay inputs differ from the original incumbent: {changed}')
    return actual


def run(original, output):
    if output.exists():
        raise ValueError('Use a new output directory for captain replay')
    os.makedirs(output)
    input_path = original/'forecasts.csv'
    input_hash = sha256(input_path)
    forecasts = pd.read_csv(input_path, dtype={c: str for c in KEY})
    inputs = replay_input_hashes(original)
    native = load()
    cfg = config_from_dict(json.loads((ROOT/'research/promoted_config.json').read_text()))
    if cfg.selector_upside_weight != 0 or cfg.supersub_upside_weight != 0 or cfg.post_target_refresh_captain:
        raise ValueError('Saved-head repair requires the registered promoted configuration')
    manifest = {'input_forecasts_sha256': input_hash, 'original_manifest_sha256': sha256(original/'manifest.json'),
                'input_sha256': inputs,
                'source_sha256': {p: sha256(ROOT/p) for p in ('model/data.py', 'model/evaluate.py', 'model/research.py', 'research/incumbent_replay.py', 'research/incumbent_captain_replay.py')},
                'repair': 'Known modern official outcomes, otherwise deterministic reconstruction, for prior player variance and positional ceiling.',
                'outcome_status': 'Already consulted historical outcomes; no independent holdout.',
                'scope': 'Captain and upside scores only; point, selector, supersub and raw component forecasts must be identical.'}
    write_json(output/'manifest.json', manifest)
    metrics, corrected, audits = [], [], []
    for season, round_no in forecasts[['season', 'round']].drop_duplicates().sort_values(['season', 'round']).itertuples(index=False, name=None):
        slate = SimpleNamespace(name=f'six_nations_{season}_r{round_no}', season=int(season), round=int(round_no))
        os.makedirs(output/slate.name)
        visible, prefix = prefix_frame(forecasts, native, season, round_no)
        frame, audit = corrected_captain(visible, prefix, cfg, season, round_no)
        frame.to_csv(output/slate.name/'incumbent_causal_captain_full_forecasts.csv', index=False)
        for roles, suffix in [('native_position', 'native_roles'), ('position', 'common_roles')]:
            metrics.append(record_policy(slate, f'incumbent_causal_captain_{suffix}', frame, output, roles))
        audits.append({'slate': slate.name, **audit})
        corrected.append(frame.assign(mode='causal_captain'))
    if sha256(input_path) != input_hash or replay_input_hashes(original) != inputs:
        raise ValueError('Original forecast evidence or replay inputs changed')
    pd.DataFrame(metrics).to_csv(output/'metrics.csv', index=False)
    pd.concat(corrected, ignore_index=True).to_csv(output/'forecasts.csv', index=False)
    write_json(output/'verification.json', {'slates': audits, 'original_forecasts_unchanged': True})
    summary = pd.DataFrame(metrics).groupby(['season', 'engine'], as_index=False).agg(
        team_points=('team_points', lambda x: x.sum(min_count=len(x))),
        xv_points=('xv_points', 'sum'), captain_extra=('captain_extra', 'sum'), supersub_points=('supersub_points', 'sum'),
        illegal_rounds=('country_limit_exceeded', lambda x: int(x.gt(0).sum())))
    summary.to_csv(output/'summary.csv', index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(ROOT/args.original, ROOT/args.output)


if __name__ == '__main__':
    main()
