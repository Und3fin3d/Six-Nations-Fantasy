import argparse
import glob
import hashlib
import json
import os
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd

from model.role_selection import SelectionRules, optimise_roles, standardise, verify_selection
from model.unified.rolling_eval import KEY, official_slates
from research.decision_replay import score_selection, summarise
from research.decision_summary import admissions, paired_rows, paired_summary


REGISTRATION = 'research/DECISION_ROLE_POLICY_REGISTRATION_2026-09-26.json'
REGISTERED_COMMIT = 'f0d1d25580f79b42922b7a0a164559201e6f70b7'
CONTROLS = ('incumbent', 'empirical_baseline', 'p3_robust_native', 'equal_incumbent')
POLICIES = ('incumbent_legal_sequential', 'incumbent_legal_joint_roles', 'p3_legal_joint_role_transfer')
EXPECTED = {
    'remedies/forecasts.csv': '46d47d855ef1584898b7a2c88c03f0f991d55c01eefd83d91cde061f19f6610e',
    'remedies/metrics.csv': '41489c6c4a92f840683e07a4b9adfaef4649c2f6fd6c697cae91c728620db6ba',
    'captain_repair/forecasts.csv': 'c26ece257e03ed64be2f0d56f02af96634e5bbb39a0da109523ddc02c651bc61',
}


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, content):
    with open(path, 'w') as handle:
        json.dump(content, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def signature(squad):
    columns = ['id', 'is_sub', 'is_capt']
    return squad[columns].sort_values('id').reset_index(drop=True)


def aligned(frame, slate):
    if frame.duplicated(KEY).any() or len(frame) != len(slate.pool):
        raise ValueError(f'{slate.name}: incomplete or duplicate candidate keys')
    index = pd.MultiIndex.from_frame(slate.candidates[KEY].astype(str))
    result = frame.set_index(KEY).reindex(index).reset_index()
    if not result[KEY].equals(slate.candidates[KEY].astype(str).reset_index(drop=True)):
        raise ValueError(f'{slate.name}: candidate alignment changed')
    if not np.allclose(result.actual, slate.actual, equal_nan=True):
        raise ValueError(f'{slate.name}: observed or unknown labels changed')
    return result


def measure(slate, policy, points, selected):
    selected = selected.copy()
    actual = dict(zip(slate.pool.id, slate.actual))
    selected['actual'] = selected.id.map(actual)
    p = dict(zip(slate.pool.id, points))
    selected['point_forecast'] = selected.id.map(p)
    parts = {}
    for name, mask, count, multiplier in (
        ('xv', ~selected.is_sub, 15, 1), ('captain_extra', selected.is_capt, 1, 1),
        ('super_sub', selected.is_sub, 1, 3),
    ):
        parts[name] = multiplier * selected.loc[mask, 'actual'].sum(min_count=count)
    known = np.isfinite(slate.actual)
    errors = points[known] - slate.actual[known]
    result = dict(slate=slate.name, competition=slate.competition, season=slate.season,
                  cutoff=slate.cutoff.isoformat(), engine=policy, n=len(points), n_labelled=int(known.sum()),
                  unknown_selected=int(selected.actual.isna().sum()), **parts,
                  total=sum(parts.values()), mae=float(np.abs(errors).mean()), bias=float(errors.mean()),
                  rmse=float(np.sqrt(np.square(errors).mean())),
                  captain=selected.loc[selected.is_capt, 'name'].iloc[0], sub=selected.loc[selected.is_sub, 'name'].iloc[0])
    return result, selected.assign(slate=slate.name, engine=policy)


def sources(args):
    files = [args.store, REGISTRATION, 'research/decision_role_policy.py',
             'research/decision_replay.py', 'research/decision_summary.py', 'research/decision_adjustments.py',
             'data/wr_rankings.csv', 'data/rp_compstats.csv', 'data/model_targets.csv',
             f'{args.workflow}/remedies/squads.csv']
    files.extend(glob.glob('model/**/*.py', recursive=True))
    files.extend(glob.glob('data/ncr/*.csv'))
    files.extend(glob.glob('data/ncr/feeds/*.json'))
    files.extend(f'{args.workflow}/{name}' for name in EXPECTED)
    return {p: digest(p) for p in sorted(set(files))}


def verify_control(slate, engine, points, old_metrics, old_squads, rules):
    measured, selected = score_selection(slate, engine, points)
    old = old_metrics[old_metrics.slate.eq(slate.name) & old_metrics.engine.eq(engine)]
    if len(old) != 1:
        raise ValueError('Missing unique archived control result')
    columns = ['total', 'xv', 'captain_extra', 'super_sub', 'mae', 'bias', 'rmse']
    if not np.allclose([measured[c] for c in columns], old[columns].iloc[0].to_numpy(float), atol=1e-8, rtol=0, equal_nan=True):
        raise ValueError(f'{slate.name}/{engine}: archived control metrics did not reproduce')
    archived = old_squads[old_squads.slate.eq(slate.name) & old_squads.engine.eq(engine)]
    if not signature(selected).equals(signature(archived)):
        raise ValueError(f'{slate.name}/{engine}: archived control selected keys changed')
    exact, diagnostic = optimise_roles(slate.pool, points, points,
                                       points * np.where(slate.pool.status.eq('B'), 3.0, 0.5), rules=rules)
    if not signature(exact).equals(signature(selected)):
        raise ValueError(f'{slate.name}/{engine}: exact solver endpoint changed; review comparator before policies')
    verify_selection(selected, rules)
    return measured, selected, diagnostic


def run(args):
    from model_env_preflight import check
    errors = check()
    if errors:
        raise RuntimeError('; '.join(errors))
    if digest(args.store) != '2b70331c0cf215def67831cd27c987362e2ec8330b2afb41c8d204f48ef17313':
        raise ValueError('Original canonical comparison input changed')
    for name, expected in EXPECTED.items():
        if digest(f'{args.workflow}/{name}') != expected:
            raise ValueError(f'Frozen input changed: {name}')
    source_hashes = sources(args)
    os.makedirs(args.output, exist_ok=False)
    manifest = dict(registration=REGISTRATION, registered_commit=REGISTERED_COMMIT, input_and_source_hashes=source_hashes,
                    python=platform.python_version(), packages={n: version(n) for n in ('numpy', 'pandas', 'scipy', 'scikit-learn')},
                    scope='Reused historical, complete-pool, price-free Six Nations; unchanged incumbent NCR route',
                    friendly15='Raw forecasts unchanged by these selection policies; exact Friendly-15 reference not re-evaluated in this run',
                    scoring=dict(six_nations_2025='six_nations_legacy_v1', six_nations_2026='six_nations_2026_v2', ncr='ncr_front_row_v2'),
                    objective_status='active_unfinished', outcome_selection='Outcomes attached only after solver selection')
    write_json(f'{args.output}/manifest.json', manifest)
    store = pd.read_csv(args.store, dtype={k: str for k in KEY}, low_memory=False)
    forecast = pd.read_csv(f'{args.workflow}/remedies/forecasts.csv', dtype={k: str for k in KEY})
    native = pd.read_csv(f'{args.workflow}/captain_repair/forecasts.csv', dtype={k: str for k in KEY})
    old_metrics = pd.read_csv(f'{args.workflow}/remedies/metrics.csv')
    old_squads = pd.read_csv(f'{args.workflow}/remedies/squads.csv')
    metrics, squads, pools, diagnostics = [], [], [], []
    slates = official_slates(store, ('six_nations', 'ncr'))
    point_bank = {}
    for slate in slates:
        print(f'{slate.name}: verify frozen controls', flush=True)
        rules = SelectionRules(budget=np.inf, max_nation=4, max_hemi=None) if slate.competition == 'six_nations' else SelectionRules()
        points = {}
        for engine in CONTROLS:
            frame = aligned(forecast[forecast.slate.eq(slate.name) & forecast.engine.eq(engine)], slate)
            points[engine] = frame.predicted.to_numpy(float)
            if not np.isfinite(points[engine]).all():
                raise ValueError(f'{engine}: missing point forecast')
            result, selected, diagnostic = verify_control(slate, engine, points[engine], old_metrics, old_squads, rules)
            metrics.append(result)
            squads.append(selected.assign(slate=slate.name, engine=engine))
            diagnostics.append(dict(slate=slate.name, engine=engine, **diagnostic))
        point_bank[slate.name] = points
    pd.DataFrame(metrics).to_csv(f'{args.output}/control_metrics.csv', index=False)
    pd.concat(squads, ignore_index=True).to_csv(f'{args.output}/control_squads.csv', index=False)
    write_json(f'{args.output}/control_verification.json', diagnostics)
    for slate in slates:
        print(f'{slate.name}: registered policies', flush=True)
        points = point_bank[slate.name]
        rules = SelectionRules(budget=np.inf, max_nation=4, max_hemi=None) if slate.competition == 'six_nations' else SelectionRules()
        pool_record = slate.pool.copy().assign(slate=slate.name, cutoff=slate.cutoff.isoformat(), actual=slate.actual)
        for key in KEY:
            pool_record[f'key_{key}'] = slate.candidates[key].to_numpy()
        for engine, values in points.items():
            pool_record[engine] = values
        if slate.competition == 'six_nations':
            heads = aligned(native[native.slate.eq(slate.name)], slate)
            if not np.allclose(heads.target_pts_hat, points['incumbent'], atol=1e-10, rtol=0):
                raise ValueError('Causal captain repair changed the point comparator')
            if not heads.cutoff.eq(slate.cutoff.isoformat()).all():
                raise ValueError('Native role-head cutoff changed')
            for column in ('selection_score', 'captain_score', 'supersub_score'):
                pool_record[column] = heads[column].to_numpy()
        for policy in POLICIES:
            if slate.competition == 'ncr':
                result, selected = score_selection(slate, policy, points['incumbent'])
                diagnostic = dict(mode='unchanged_ncr_incumbent', **verify_selection(selected, rules))
            else:
                base = 'p3_robust_native' if policy.startswith('p3_') else 'incumbent'
                xv = standardise(points[base]) if policy.startswith('p3_') else standardise(heads.selection_score)
                captain = standardise(heads.captain_score)
                sub = 3 * standardise(heads.supersub_score)
                mode = 'sequential' if policy.endswith('sequential') else 'joint'
                if mode == 'sequential':
                    xv, captain, sub = (heads[c].to_numpy(float) for c in ('selection_score', 'captain_score', 'supersub_score'))
                selected, diagnostic = optimise_roles(slate.pool, xv, captain, sub, rules=rules, mode=mode)
                result, selected = measure(slate, policy, points[base], selected)
            metrics.append(result)
            squads.append(selected.assign(slate=slate.name, engine=policy))
            diagnostics.append(dict(slate=slate.name, engine=policy, **diagnostic))
        pools.append(pool_record)
        pd.DataFrame(metrics).to_csv(f'{args.output}/metrics.csv', index=False)
        pd.concat(squads, ignore_index=True).to_csv(f'{args.output}/squads.csv', index=False)
        pd.concat(pools, ignore_index=True).to_csv(f'{args.output}/pools.csv', index=False)
        write_json(f'{args.output}/solver_diagnostics.json', diagnostics)
    table = pd.DataFrame(metrics)
    if table.groupby('engine').slate.nunique().ne(13).any():
        raise ValueError('Incomplete three-tournament comparison')
    summarise(table).to_csv(f'{args.output}/summary.csv', index=False)
    comparisons = [(p, c) for p in POLICIES for c in (*CONTROLS, 'incumbent_legal_sequential') if p != c]
    pairs = paired_rows(table, comparisons)
    pairs.to_csv(f'{args.output}/paired_rounds.csv', index=False)
    paired_summary(pairs, ['competition', 'season', 'candidate', 'comparator']).to_csv(f'{args.output}/paired_seasons.csv', index=False)
    paired_summary(pairs, ['competition', 'candidate', 'comparator']).to_csv(f'{args.output}/paired_competitions.csv', index=False)
    decisions = admissions(pairs)
    decisions.to_csv(f'{args.output}/admission.csv', index=False)
    if source_hashes != sources(args):
        raise ValueError('Immutable inputs or executing sources changed during the run')
    write_json(f'{args.output}/completion.json', dict(analysis_completed=True, objective_completed=False,
                                                     output_hashes={n: digest(f'{args.output}/{n}') for n in sorted(os.listdir(args.output))}))
    print(summarise(table).to_string(index=False), flush=True)
    print(decisions.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True)
    parser.add_argument('--workflow', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
