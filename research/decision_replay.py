import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

from model.empirical_unified import project_candidates
from model.ncr_project import optimise
from model.unified.contracts import RawPrediction
from model.unified.rolling_eval import KEY, official_slates, expected_points
from model.unified.scoring import NCR_LEGACY_SCORING_VERSION, SIX_NATIONS_LEGACY_SCORING_VERSION
from research.decision_adjustments import apply_kicking_gate, bench_history_diagnostic, calibrate_roles, combination_weight, kicking_history

ENGINES = ('p3_rolling_native', 'p3_weighted_native', 'p3_robust_native')
REGISTRATION = 'research/DECISION_REMEDIES_2026-09-26.json'
REGISTRATION_HASH = 'e946d6f29a1331c567185304b09a7fea11cad258b031904383dfc498dbd9f8cd'


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_raw(path, slate):
    with open(path) as stream:
        rows = [json.loads(line) for line in stream]
    keys = pd.DataFrame([{key: row[key] for key in KEY} for row in rows])
    if not keys.equals(slate.candidates[KEY].astype(str).reset_index(drop=True)):
        raise ValueError(f'{slate.name}: raw forecast alignment changed')
    return [RawPrediction.from_dict(row) for row in rows]


def archived_forecasts(slate, evidence):
    directory = f'{evidence}/comparison-slate-{slate.name}/results'
    with open(REGISTRATION) as stream:
        required_store = json.load(stream)['history_store_sha256']
    with open(f'{directory}/run_manifest.json') as stream:
        if json.load(stream)['store_sha256'] != required_store:
            raise ValueError(f'{slate.name}: archived training store differs from registered history')
    with open(f'{directory}/round_manifests.json') as stream:
        recorded = next(row for row in json.load(stream) if row['slate'] == slate.name)
    if pd.Timestamp(recorded['cutoff']) != slate.cutoff:
        raise ValueError(f'{slate.name}: archive cutoff changed')
    points, raw, checks = {}, {}, []
    files = [f'{directory}/{name}.json' for name in ('run_manifest', 'round_manifests')]
    for engine in ('empirical_baseline', *ENGINES):
        path = f'{directory}/{slate.name}/{engine}_predictions.csv'
        frame = pd.read_csv(path)
        if not np.array_equal(frame.id, slate.pool.id) or not np.allclose(frame.actual, slate.actual, equal_nan=True):
            raise ValueError(f'{slate.name}: archived identity or outcome changed')
        points[engine] = frame.predicted.to_numpy()
        files.append(path)
        if engine == 'empirical_baseline':
            continue
        path = f'{directory}/models/{slate.name}/{engine}.jsonl'
        raw[engine] = load_raw(path, slate)
        legacy = NCR_LEGACY_SCORING_VERSION if slate.competition == 'ncr' else SIX_NATIONS_LEGACY_SCORING_VERSION
        reproduced = expected_points(raw[engine], slate.competition, scoring_version=legacy)
        error = float(np.max(np.abs(reproduced - points[engine])))
        if error > 1e-9:
            raise ValueError(f'{slate.name}/{engine}: archived scoring changed by {error}')
        checks.append(dict(slate=slate.name, engine=engine, rows=len(frame), max_error=error))
        points[engine] = expected_points(raw[engine], slate.competition, season=slate.season)
        files.append(path)
    return points, raw, files, checks


def incumbent_points(slate, directory):
    path = f'{directory}/{slate.name}/incumbent_lock_rebuilt_full_forecasts.csv'
    if not os.path.exists(path):
        raise ValueError(f'Missing complete incumbent replay: {path}')
    frame = pd.read_csv(path, dtype={key: str for key in KEY})
    if not frame[KEY].equals(slate.candidates[KEY].astype(str).reset_index(drop=True)):
        raise ValueError(f'{slate.name}: incumbent identity changed')
    if not np.allclose(frame.actual, slate.actual, equal_nan=True):
        raise ValueError(f'{slate.name}: incumbent outcomes changed')
    return frame.target_pts_hat.to_numpy(), path


def score_selection(slate, engine, points):
    pool = slate.pool.copy()
    pool['actual'] = slate.actual
    pool['starter_exp'] = points
    pool['supersub_exp'] = points * np.where(pool.status.eq('B'), 3.0, 0.5)
    options = {} if slate.competition == 'ncr' else dict(budget=np.inf, max_nation=4, max_hemi=None)
    squad, _, _ = optimise(pool, **options)
    known = np.isfinite(slate.actual)
    if not np.isfinite(points).all():
        raise ValueError(f'{slate.name}/{engine}: non-finite forecast')
    xv = squad.loc[~squad.is_sub, 'actual'].sum(min_count=15)
    captain = squad.loc[squad.is_capt, 'actual'].sum(min_count=1)
    sub = 3 * squad.loc[squad.is_sub, 'actual'].sum(min_count=1)
    errors = points[known] - slate.actual[known]
    metrics = dict(slate=slate.name, competition=slate.competition, season=slate.season, engine=engine,
                   cutoff=slate.cutoff.isoformat(), n=len(points), n_labelled=int(known.sum()),
                   unknown_selected=int(squad.actual.isna().sum()), xv=xv, captain_extra=captain,
                   super_sub=sub, total=xv+captain+sub, mae=np.abs(errors).mean(), bias=errors.mean(),
                   rmse=np.sqrt(np.square(errors).mean()), captain=squad.loc[squad.is_capt, 'name'].iloc[0],
                   sub=squad.loc[squad.is_sub, 'name'].iloc[0])
    return metrics, squad.assign(slate=slate.name, engine=engine)


def add_remedy_forecasts(slate, base, earlier, points, raw, store):
    robust = points['p3_robust_native']
    known = earlier.dropna(subset=['actual']).copy() if not earlier.empty else earlier
    if not known.empty:
        known['residual'] = known.actual - known.robust
    points['role_calibration'], calibration = calibrate_roles(base, known, robust)
    adjustments, gate = apply_kicking_gate(raw['p3_robust_native'], kicking_history(store, slate.cutoff))
    points['kicking_role_gate'] = expected_points(adjustments, slate.competition, season=slate.season)
    weights = []
    for comparator in ('empirical_baseline', 'incumbent'):
        if comparator not in points:
            continue
        points[f'equal_{comparator}'] = 0.5 * (robust + points[comparator])
        weight = combination_weight(known, comparator)
        points[f'regularised_{comparator}'] = points[comparator] + weight * (robust - points[comparator])
        weights.append(dict(slate=slate.name, comparator=comparator, robust_weight=weight,
                            prior_known_rows=len(known), prior_latest_cutoff=str(known.cutoff.max()) if len(known) else None))
    return calibration.assign(slate=slate.name), pd.DataFrame(gate).assign(slate=slate.name), weights


def current_frame(slate):
    columns = [*KEY, 'position']
    frame = slate.candidates[columns].copy().reset_index(drop=True)
    frame['status'], frame['actual'] = slate.pool.status.to_numpy(), slate.actual
    frame['slate'], frame['cutoff'] = slate.name, slate.cutoff.isoformat()
    return frame


def corrected_empirical(slate, store, wr):
    if slate.competition == 'ncr':
        return slate.baseline
    result = project_candidates(slate.candidates, store, slate.competition, wr, asof=slate.cutoff)
    return result.set_index('label_row_id').reindex(slate.candidates.label_row_id).predicted_points.to_numpy()


def summarise(metrics):
    return metrics.groupby(['competition', 'season', 'engine'], as_index=False).agg(
        total=('total', lambda x: x.sum(min_count=len(x))), rounds=('slate', 'nunique'),
        xv=('xv', lambda x: x.sum(min_count=len(x))), captain_extra=('captain_extra', lambda x: x.sum(min_count=len(x))),
        super_sub=('super_sub', lambda x: x.sum(min_count=len(x))), unknown_selected=('unknown_selected', 'sum'),
        mae=('mae', 'mean'), bias=('bias', 'mean'), rmse=('rmse', 'mean'))


def save_tables(output, values):
    for name, frames in values.items():
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        table.to_csv(f'{output}/{name}.csv', index=False)


def replay_inputs(args):
    sources = glob.glob('model/**/*.py', recursive=True)
    sources.extend(['research/decision_replay.py', 'research/decision_adjustments.py', 'model/history.py'])
    data = glob.glob('data/ncr/*.csv') + glob.glob('data/ncr/feeds/*.json')
    data.extend([args.store, REGISTRATION, 'data/wr_rankings.csv', 'data/rp_compstats.csv', 'data/model_targets.csv'])
    for season in (2025, 2026):
        for round_no in range(1, 6):
            path = f'{args.incumbent}/six_nations_{season}_r{round_no}/incumbent_lock_rebuilt_full_forecasts.csv'
            if not os.path.exists(path):
                raise ValueError(f'Complete incumbent replay is required before remedies: {path}')
            data.append(path)
    return {path: digest(path) for path in sorted(set(sources + data))}


def run(args):
    if digest(REGISTRATION) != REGISTRATION_HASH:
        raise ValueError('Remedy registration changed')
    with open(REGISTRATION) as stream:
        registration = json.load(stream)
    if digest(args.store) != registration['history_store_sha256']:
        raise ValueError('History store changed')
    initial_hashes = replay_inputs(args)
    os.makedirs(args.output, exist_ok=False)
    store = pd.read_csv(args.store, dtype={key: str for key in KEY}, low_memory=False)
    wr = pd.read_csv('data/wr_rankings.csv', parse_dates=['snapshot_date'])
    slates = official_slates(store, ('six_nations', 'ncr'))
    history, metrics, weights, checks, files = {}, [], [], [], [args.store, REGISTRATION, 'data/wr_rankings.csv']
    tables = {name: [] for name in ('forecasts', 'squads', 'calibration', 'kicking_gate', 'bench_history')}
    for slate in slates:
        print(slate.name, flush=True)
        base = current_frame(slate)
        points, raw, sources, replay = archived_forecasts(slate, args.evidence)
        files.extend(sources)
        checks.extend(replay)
        points['empirical_archived'] = points['empirical_baseline'].copy()
        points['empirical_baseline'] = corrected_empirical(slate, store, wr)
        if slate.competition == 'ncr':
            points['incumbent'] = points['empirical_baseline'].copy()
        else:
            points['incumbent'], path = incumbent_points(slate, args.incumbent)
            files.append(path)
        earlier = history.get(slate.competition, pd.DataFrame())
        calibration, gate, fitted_weights = add_remedy_forecasts(slate, base, earlier, points, raw, store)
        tables['calibration'].append(calibration)
        tables['kicking_gate'].append(gate)
        tables['bench_history'].append(bench_history_diagnostic(store, slate.cutoff).assign(slate=slate.name))
        weights.extend(fitted_weights)
        for engine, prediction in points.items():
            row, squad = score_selection(slate, engine, prediction)
            metrics.append(row)
            tables['squads'].append(squad)
            tables['forecasts'].append(base.assign(engine=engine, predicted=prediction))
        completed = base.assign(robust=points['p3_robust_native'], empirical_baseline=points['empirical_baseline'])
        if 'incumbent' in points:
            completed['incumbent'] = points['incumbent']
        history[slate.competition] = pd.concat([earlier, completed], ignore_index=True)
        pd.DataFrame(metrics).to_csv(f'{args.output}/metrics.csv', index=False)
    table = pd.DataFrame(metrics)
    summarise(table).to_csv(f'{args.output}/summary.csv', index=False)
    pd.DataFrame(weights).to_csv(f'{args.output}/weights.csv', index=False)
    pd.DataFrame(checks).to_csv(f'{args.output}/archive_replay.csv', index=False)
    save_tables(args.output, tables)
    if initial_hashes != replay_inputs(args):
        raise ValueError('Replay source or inputs changed during computation')
    manifest = dict(registration_sha256=REGISTRATION_HASH, source_base='93f9670f38ed1699c0029192ac38227392b329fb',
                    input_sha256={path: digest(path) for path in sorted(set(files))},
                    runtime_input_sha256=initial_hashes,
                    interpretation='Already consulted historical development/audit; no prospective performance claim',
                    chronology='Strictly previous slates only; no current/future outcome used to fit corrections',
                    missing_outcomes='Remain unknown; full pools optimised before evaluating outcomes')
    with open(f'{args.output}/manifest.json', 'w') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print(summarise(table).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--incumbent', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
