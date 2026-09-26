import argparse
import glob
import json
import os
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.raw_benchmark.empirical import EmpiricalEventModel
from model.unified.raw_benchmark.positions import prior_supported_positions
from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from model.unified.rolling_eval import KEY, expected_points, official_slates
from research.decision_replay import archived_forecasts, current_frame, digest, score_selection, summarise
from research.decision_summary import admissions, paired_rows, paired_summary

REGISTRATION = 'research/TIMING_REMEDY_PROTOCOL_2026-09-26.md'
REGISTRATION_HASH = '3021e3876b0c11d6113a6c0ea9d6b1fb6a67bbf124a8600ed280135972f2e49e'
INPUT_HASHES = ('2b70331c0cf215def67831cd27c987362e2ec8330b2afb41c8d204f48ef17313',
                'e4b20637e2b514d622bb35cc7c43762e0eddb4f3b4e50f1ca2e39155ae96e6be')
CANDIDATE = 'empirical_timing_only'
COMPARATORS = ('p3_robust_native', 'incumbent', 'equal_incumbent')


def input_hashes(args):
    paths = glob.glob('model/**/*.py', recursive=True)
    paths += ['research/timing_ablation.py', 'research/decision_replay.py', 'research/decision_summary.py',
              REGISTRATION, args.old_store, args.new_store, 'data/rp_compstats.csv', 'data/wr_rankings.csv',
              'data/model_targets.csv', 'data/unified/p3_hillclimb/config.json', f'{args.comparison}/forecasts.csv',
              f'{args.comparison}/metrics.csv']
    paths += glob.glob('data/ncr/*.csv') + glob.glob('data/ncr/feeds/*.json')
    return {path: digest(path) for path in sorted(set(paths))}


def read_stores(args):
    stores = []
    for path, expected in zip((args.old_store, args.new_store), INPUT_HASHES):
        if digest(path) != expected:
            raise ValueError(f'Unexpected canonical input hash: {path}')
        stores.append(pd.read_csv(path, dtype={key: str for key in KEY}, low_memory=False))
    unchanged = stores[0].columns.difference(['minutes', 'available__minutes', 'provenance__minutes'])
    if not stores[0][unchanged].equals(stores[1][unchanged]):
        raise ValueError('Timing inputs differ outside the registered columns')
    return stores


def component_predictions(old, new, slate):
    frames = [past_matches(store, slate.cutoff).sort_values(['date', 'fixture_id', 'team', 'player_id']) for store in (old, new)]
    candidates = slate.candidates
    if 'position_source' in candidates:
        candidates = prior_supported_positions(candidates, frames[0])
    models = (EmpiricalEventModel(asof=slate.cutoff).fit(frames[0]),
              RobustEmpiricalEventModel(asof=slate.cutoff).fit(frames[0]),
              RobustEmpiricalEventModel(asof=slate.cutoff).fit(frames[1]))
    forecasts = [model.predict_frame(candidates) for model in models]
    audit = dict(slate=slate.name, cutoff=slate.cutoff.isoformat(), training_rows=len(frames[0]),
                 training_latest_kickoff=frames[0].match_at.max(),
                 newly_unavailable_minutes=int((frames[0].available__minutes & ~frames[1].available__minutes).sum()),
                 fits=3)
    return forecasts, audit


def empirical_weight(prediction, event, config):
    metadata = prediction.metadata
    expected = config['event_weights_v4'].get(event, config['default_weight_v4'])
    recorded = metadata.get('event_weights_v4', {}).get(event, metadata['weight_v4'])
    if recorded != expected:
        raise ValueError(f'Archived event weight differs: {event}')
    return 1.0-recorded


def replace_component(blend, weighted, legacy, old, new, fraction, event):
    recovered = blend.mean-weighted.mean-fraction*(old.mean-legacy.mean)
    if abs(recovered) > 1e-9:
        raise ValueError(f'Empirical component recovery failed for {event}: {recovered}')
    mean = blend.mean + fraction*(new.mean-old.mean)
    if mean < 0 or not np.isfinite(mean):
        raise ValueError(f'Invalid updated mean for {event}: {mean}')
    distribution = replace(blend, mean=float(mean))
    moment_error = 0.0
    if event in ('metres', 'minutes'):
        old_second = old.dispersion+old.mean**2
        legacy_second = legacy.dispersion+legacy.mean**2
        moment_error = blend.dispersion+blend.mean**2-weighted.dispersion-weighted.mean**2-fraction*(old_second-legacy_second)
        if abs(moment_error) > 1e-9:
            raise ValueError(f'Empirical second-moment recovery failed for {event}: {moment_error}')
        second = blend.dispersion+blend.mean**2+fraction*(new.dispersion+new.mean**2-old_second)
        distribution = replace(distribution, dispersion=float(second-mean**2))
    return distribution, recovered, moment_error


def adjusted_prediction(saved, weighted, components, config):
    legacy, old, new = components
    keys = [(item.fixture_id, item.player_id, item.team, item.position, item.is_forward)
            for item in (saved, weighted, legacy, old, new)]
    if len(set(keys)) != 1 or len({tuple(sorted(item.events)) for item in (saved, weighted, legacy, old, new)}) != 1:
        raise ValueError('Component keys or event support differ')
    events, audit = {}, []
    for event in (*sorted(saved.events), 'minutes'):
        values = [item.minutes if event == 'minutes' else item.events[event] for item in (saved, weighted, legacy, old, new)]
        fraction = empirical_weight(saved, event, config)
        result, error, moment_error = replace_component(*values, fraction, event)
        events[event] = result
        audit.append(dict(fixture_id=saved.fixture_id, player_id=saved.player_id, team=saved.team,
                          event=event, empirical_weight=fraction, legacy_empirical=values[2].mean,
                          old_empirical=values[3].mean, new_empirical=values[4].mean, old_blend=values[0].mean,
                          new_blend=result.mean, recovery_error=error, moment_recovery_error=moment_error,
                          scoring_variance=result.dispersion if event in ('metres', 'minutes') else np.nan))
    minutes = events.pop('minutes')
    return replace(saved, events=events, minutes=minutes), audit


def adjusted_forecasts(raw, components, config):
    if len({len(values) for values in (raw['p3_robust_native'], raw['p3_weighted_native'], *components)}) != 1:
        raise ValueError('Component prediction lengths differ')
    forecasts, audit = [], []
    for saved, weighted, legacy, old, new in zip(raw['p3_robust_native'], raw['p3_weighted_native'], *components):
        prediction, rows = adjusted_prediction(saved, weighted, (legacy, old, new), config)
        forecasts.append(prediction)
        audit.extend(rows)
    return forecasts, pd.DataFrame(audit)


def comparator_points(slate, forecasts, archived):
    points = {}
    for engine in COMPARATORS:
        selected = forecasts[forecasts.slate.eq(slate.name) & forecasts.engine.eq(engine)].reset_index(drop=True)
        if not selected[KEY].equals(slate.candidates[KEY].astype(str).reset_index(drop=True)):
            raise ValueError(f'Comparator candidate keys differ: {slate.name}/{engine}')
        if not np.allclose(selected.actual, slate.actual, equal_nan=True):
            raise ValueError(f'Comparator outcomes differ: {slate.name}/{engine}')
        points[engine] = selected.predicted.to_numpy()
    if not np.allclose(points['p3_robust_native'], archived['p3_robust_native'], atol=1e-9, rtol=0):
        raise ValueError(f'Corrected robust comparator failed replay: {slate.name}')
    return points


def score_slate(slate, points, expected_metrics):
    metrics, squads, forecasts = [], [], []
    for engine, prediction in points.items():
        row, squad = score_selection(slate, engine, prediction)
        if engine in COMPARATORS:
            expected = expected_metrics[expected_metrics.slate.eq(slate.name) & expected_metrics.engine.eq(engine)].iloc[0]
            columns = ['total', 'xv', 'captain_extra', 'super_sub', 'mae', 'bias', 'rmse']
            if not np.allclose([row[key] for key in columns], expected[columns].to_numpy(float), atol=1e-9, rtol=0, equal_nan=True):
                raise ValueError(f'Comparator selection failed replay: {slate.name}/{engine}')
        metrics.append(row)
        squads.append(squad)
        forecasts.append(current_frame(slate).assign(engine=engine, predicted=prediction))
    return metrics, pd.concat(squads, ignore_index=True), pd.concat(forecasts, ignore_index=True)


def save_summary(output, metrics):
    summarise(metrics).to_csv(f'{output}/summary.csv', index=False)
    pairs = paired_rows(metrics, comparisons=[(CANDIDATE, comparator) for comparator in COMPARATORS])
    pairs.to_csv(f'{output}/paired_rounds.csv', index=False)
    paired_summary(pairs, ['competition', 'season', 'candidate', 'comparator']).to_csv(f'{output}/paired_seasons.csv', index=False)
    paired_summary(pairs, ['competition', 'candidate', 'comparator']).to_csv(f'{output}/paired_competitions.csv', index=False)
    decisions = admissions(pairs)
    decisions.to_csv(f'{output}/admission.csv', index=False)
    print(summarise(metrics).to_string(index=False), flush=True)
    print(decisions.to_string(index=False), flush=True)


def run(args):
    if digest(REGISTRATION) != REGISTRATION_HASH:
        raise ValueError('Timing remedy registration changed')
    os.makedirs(args.output, exist_ok=False)
    hashes = input_hashes(args)
    old, new = read_stores(args)
    slates = official_slates(old, ('six_nations', 'ncr'))
    if len(slates) != 13:
        raise ValueError('The registered ablation requires thirteen slates')
    with open('data/unified/p3_hillclimb/config.json') as stream:
        config = json.load(stream)
    comparators = pd.read_csv(f'{args.comparison}/forecasts.csv', dtype={key: str for key in KEY})
    comparator_metrics = pd.read_csv(f'{args.comparison}/metrics.csv')
    metrics, squads, forecasts, audits, training = [], [], [], [], []
    archives = {}
    for slate in slates:
        print(f'Fitting empirical components: {slate.name}', flush=True)
        archived, raw, files, _ = archived_forecasts(slate, args.evidence)
        archives.update({path: digest(path) for path in files})
        components, history = component_predictions(old, new, slate)
        training.append(history)
        predictions, audit = adjusted_forecasts(raw, components, config)
        audit['slate'] = slate.name
        audits.append(audit)
        points = comparator_points(slate, comparators, archived)
        points[CANDIDATE] = expected_points(predictions, slate.competition, season=slate.season)
        rows, squad, frame = score_slate(slate, points, comparator_metrics)
        metrics.extend(rows)
        squads.append(squad)
        forecasts.append(frame)
        pd.DataFrame(metrics).to_csv(f'{args.output}/metrics.csv', index=False)
        audit.to_csv(f'{args.output}/{slate.name}_component_audit.csv', index=False)
        print(f'Recovered {slate.name}: {len(audit)} component cells', flush=True)
    pd.concat(squads, ignore_index=True).to_csv(f'{args.output}/squads.csv', index=False)
    pd.concat(forecasts, ignore_index=True).to_csv(f'{args.output}/forecasts.csv', index=False)
    pd.concat(audits, ignore_index=True).to_csv(f'{args.output}/component_audit.csv', index=False)
    pd.DataFrame(training).to_csv(f'{args.output}/training.csv', index=False)
    if input_hashes(args) != hashes or any(digest(path) != value for path, value in archives.items()):
        raise ValueError('Ablation inputs or source changed during execution')
    save_summary(args.output, pd.DataFrame(metrics))
    os.makedirs(f'{args.output}/source', exist_ok=False)
    for path in ('research/timing_ablation.py', REGISTRATION):
        shutil.copy2(path, f'{args.output}/source/{os.path.basename(path)}')
    manifest = dict(registration_sha256=REGISTRATION_HASH, input_sha256=hashes, archive_sha256=archives,
                    empirical_fits=len(slates)*3, tree_fits=0, candidate=CANDIDATE,
                    count_tails='Not recalibrated; updated means used for deterministic scoring only',
                    interpretation='One empirical-component-only ablation on already-consulted historical data; no promotion',
                    outputs={os.path.basename(path): digest(path) for path in glob.glob(f'{args.output}/*.csv')})
    with open(f'{args.output}/manifest.json', 'w') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-store', required=True)
    parser.add_argument('--new-store', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
