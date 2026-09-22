import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from model.unified.raw_benchmark.config import STABLE_EVENTS
from model.unified.raw_benchmark.coverage import sha256
from model.unified.rolling_eval import ROOT, season_summary


def paired_summary(frame, keys, metric, reference, grouping):
    results = []
    for group, rows in frame.groupby(grouping, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        pivot = rows.pivot(index=keys, columns='engine', values=metric)
        for opponent in pivot.columns:
            if opponent == reference:
                continue
            pairs = pivot[[reference, opponent]].dropna()
            if pairs.empty:
                continue
            differences = (pairs[reference]-pairs[opponent]).to_numpy(float)
            rng = np.random.default_rng(20260922)
            means = differences[rng.integers(0, len(differences), (10000, len(differences)))].mean(axis=1)
            leave_one = (differences.sum()-differences)/(len(differences)-1) if len(differences)>1 else differences
            results.append(dict(zip(grouping, group), reference=reference, opponent=opponent,
                metric=metric, n_pairs=len(pairs), mean_difference=differences.mean(),
                p025=np.quantile(means, .025), p975=np.quantile(means, .975),
                negative_blocks=int((differences<0).sum()), positive_blocks=int((differences>0).sum()),
                leave_one_min=leave_one.min(), leave_one_max=leave_one.max()))
    return pd.DataFrame(results)


def load_evidence(directory, study):
    manifests = glob.glob(os.path.join(directory, '**', 'run_manifest.json'), recursive=True)
    expected_jobs = len(study['raw_folds']) + len(study['official_slates'])
    if len(manifests) != expected_jobs:
        raise ValueError(f'Expected {expected_jobs} completed jobs, found {len(manifests)} manifests')
    official, events, support, provenance = [], [], [], []
    fixtures = set()
    folds = set()
    stores = set()
    sources = set()
    weights = set()
    environments = set()
    label_versions = set()
    raw_drivers = set()
    for path in sorted(manifests):
        root = os.path.dirname(path)
        with open(path) as handle:
            manifest = json.load(handle)
        stores.add(manifest['store_sha256'])
        weights.add(manifest['weight_config_sha256'])
        environments.add(json.dumps([manifest['python'], manifest['packages']], sort_keys=True))
        sources.add(json.dumps({k:v for k,v in manifest['source_sha256'].items()
                               if k != 'research/raw_comparison.py'}, sort_keys=True))
        provenance.append(dict(job=os.path.basename(os.path.dirname(root)), manifest=manifest))
        if 'fold' in manifest:
            if manifest['evaluation_fixtures'] != study['raw_folds'].get(manifest['fold']):
                raise ValueError('Raw fixture population differs from the frozen study')
            raw_drivers.add(manifest['source_sha256']['research/raw_comparison.py'])
            if manifest['fold'] in folds or fixtures.intersection(manifest['evaluation_fixtures']):
                raise ValueError('Duplicate raw fold or fixture')
            folds.add(manifest['fold'])
            fixtures.update(manifest['evaluation_fixtures'])
            events.append(pd.read_csv(os.path.join(root, 'events.csv')))
            support.append(pd.read_csv(os.path.join(root, 'target_support.csv')))
        else:
            label_versions.add(json.dumps(manifest['evaluation_inputs_sha256'], sort_keys=True))
            official.append(pd.read_csv(os.path.join(root, 'metrics.csv')))
    fingerprints = (stores, sources, weights, environments, label_versions, raw_drivers)
    if (any(len(values)!=1 for values in fingerprints) or folds != set(study['raw_folds'])
            or stores != {study['store_sha256']} or len(official)!=len(study['official_slates'])):
        raise ValueError('Input provenance or evaluation coverage differs from the frozen protocol')
    return pd.concat(official), pd.concat(events), pd.concat(support), provenance


def summarise_raw(events, output, study):
    if events.duplicated(['engine','fold','target','cohort']).any():
        raise ValueError('Duplicate raw metrics')
    all_rows = events[events.cohort.eq('all')].copy()
    stable = all_rows[all_rows.target.isin(STABLE_EVENTS)]
    counts = stable.groupby(['fold','engine']).target.nunique()
    expected = {(fold,engine) for fold in study['raw_folds'] for engine in study['raw_engines']}
    if not counts.eq(len(STABLE_EVENTS)).all() or set(counts.index)!=expected:
        raise ValueError('Stable target coverage differs between candidates or blocks')
    blocks = stable.groupby(['fold','engine'], as_index=False).relative_loss.mean()
    blocks['family'] = blocks.fold.str.replace(r'_\d{4}$', '', regex=True)
    blocks['population'] = 'stable_events'
    blocks.to_csv(os.path.join(output, 'raw_blocks.csv'), index=False)
    paired_summary(blocks, ['fold'], 'relative_loss', 'p3_robust_native', ['population']).to_csv(
        os.path.join(output, 'raw_paired_intervals.csv'), index=False)
    blocks.groupby(['family','engine'], as_index=False).relative_loss.mean().to_csv(
        os.path.join(output, 'raw_families.csv'), index=False)
    paired_summary(all_rows, ['fold'], 'relative_loss', 'p3_robust_native', ['target']).to_csv(
        os.path.join(output, 'raw_target_intervals.csv'), index=False)
    scales = stable[['fold','engine','target','naive_loss']].rename(columns={'naive_loss':'population_naive_loss'})
    cohort_stable = events[events.target.isin(STABLE_EVENTS)].merge(
        scales, on=['fold','engine','target'], validate='many_to_one')
    cohort_stable['main_normalised_loss'] = cohort_stable.loss / cohort_stable.population_naive_loss
    cohort_blocks = cohort_stable.groupby(['fold','engine','cohort'], as_index=False).main_normalised_loss.mean()
    paired_summary(cohort_blocks, ['fold'], 'main_normalised_loss', 'p3_robust_native', ['cohort']).to_csv(
        os.path.join(output, 'raw_cohort_intervals.csv'), index=False)
    return blocks.groupby('engine').relative_loss.mean().sort_values().to_dict()


def summarise_official(official, output, study):
    if (official.duplicated(['slate','engine']).any() or len(official)!=5*len(study['official_slates'])
            or set(official.slate)!=set(study['official_slates'])):
        raise ValueError('Expected five candidates on each of 13 official slates')
    if official.groupby('slate').engine.nunique().ne(5).any():
        raise ValueError('Official model coverage differs between slates')
    official.to_csv(os.path.join(output, 'official_metrics.csv'), index=False)
    summary = season_summary(official)
    summary.to_csv(os.path.join(output, 'official_seasons.csv'), index=False)
    intervals = []
    for metric in ('mae','team_points'):
        intervals.append(paired_summary(official, ['slate'], metric, 'p3_robust_native', ['competition']))
    pd.concat(intervals).to_csv(os.path.join(output, 'official_paired_intervals.csv'), index=False)
    return summary.astype(object).where(pd.notna(summary), None).to_dict('records')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with open(ROOT/'data/unified/complete_comparison/study_manifest.json') as handle:
        study = json.load(handle)
    official, events, support, provenance = load_evidence(args.evidence, study)
    os.makedirs(args.output, exist_ok=True)
    events.to_csv(os.path.join(args.output, 'raw_events.csv'), index=False)
    support.to_csv(os.path.join(args.output, 'raw_target_support.csv'), index=False)
    result = {'raw_equal_block_relative_loss': summarise_raw(events, args.output, study),
              'official_seasons': summarise_official(official, args.output, study),
              'aggregation_source_sha256': sha256(ROOT/'research/summarise_complete_comparison.py'),
              'study_sha256': sha256(ROOT/'data/unified/complete_comparison/study_manifest.json'),
              'bootstrap_draws':10000, 'seed':20260922,
              'intervals':'Paired exploratory 95%; negative differences favour reference for losses only',
              'provenance':provenance}
    with open(os.path.join(args.output, 'summary.json'), 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
