import argparse
import dataclasses
import json
import os
import platform
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd

from model.data import load
from model.evaluate import _pick_xv, _supersub_pool
from model.research import _predict_config, config_from_dict
from model.unified.raw_benchmark.coverage import sha256
from model.unified.rolling_eval import ROOT, KEY, POS, official_slates, evaluate
from research.incumbent_features import IncumbentFeatureHistory


def write_json(path, value):
    with open(path, 'w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def source_manifest(store_path, evidence, cfg):
    source = [*sorted((ROOT/'model').rglob('*.py')), ROOT/'build_features.py', ROOT/'build_team_play.py',
              ROOT/'build_targets.py', ROOT/'build_team_form.py', ROOT/'official_labels.py',
              ROOT/'research/incumbent_replay.py', ROOT/'research/incumbent_features.py',
              ROOT/'report_2026_prediction_xv.py', ROOT/'gw_update.sh']
    data = ['model_player_match.csv', 'model_targets.csv', 'api_player_match.csv', 'api_team_match.csv',
            'official_player_match.csv', 'wr_rankings.csv', 'rp_compstats.csv', 'rp_bio.csv',
            'fixture_difficulty.csv', 'team_play_predictions.csv', 'player_crosswalk.csv', '6n_players.csv']
    return {'source_base': '93f9670f38ed1699c0029192ac38227392b329fb',
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source},
            'data_sha256': {name: sha256(ROOT/'data'/name) for name in data},
            'store_sha256': sha256(store_path), 'evidence_directory': str(evidence),
            'archived_prediction_sha256': {str(p.relative_to(evidence)): sha256(p) for p in sorted(evidence.glob('comparison-slate-six_nations_*/results/six_nations_*/*_predictions.csv'))},
            'promoted_config_sha256': sha256(ROOT/'research/promoted_config.json'),
            'effective_config': dataclasses.asdict(cfg), 'python': platform.python_version(),
            'packages': {name: version(name) for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'xgboost')},
            'future_round_rows': 'excluded before every fit and every season-wide standardisation',
            'current_slate_labels': 'masked before forecasting; restored only for evaluation',
            'fallback': 'none; all 1380 triple keys must exist and receive finite forecasts',
            'interpretation': 'current deployed configuration replay on already consulted historical seasons; no independently sealed outcomes',
            'limitations': ['No archived Six Nations prices or historically issued forecasts.',
                            'Historical bio, aggregate publication times and native role mapping remain unversioned.',
                            'Native-store variant retains fixture-relative features and is compatibility evidence.',
                            'Lock-rebuilt variant freezes match-derived form, role, own-team and team-play features at slate lock.',
                            'Lock reconstruction uses current source histories and preserves fixed native class, bio, fixture context and role definitions; feature refresh and cutoff effects are not separately identified.',
                            'Native component training uses prior Six Nations seasons; PR24 candidate models use broader history.',
                            'Native selector has positional quotas but no country or price constraint; common optimiser uses PR24 roles and four-per-country constraint.']}


def aligned_predictions(slate, predicted, selection_col):
    pred = predicted.copy()
    for col in KEY[:2]:
        pred[col] = pred[col].astype(str)
    candidates = slate.candidates[KEY+['position', 'position_source', 'started', 'label_row_id']].copy()
    pred = pred.rename(columns={'canonical_pos': 'native_position', 'started': 'native_started'})
    joined = candidates.merge(pred, on=KEY, how='left', validate='one_to_one', indicator=True)
    cols = ['target_pts_hat', selection_col, 'captain_score', 'supersub_score']
    if not joined._merge.eq('both').all() or not np.isfinite(joined[cols].to_numpy(float)).all():
        raise ValueError(f'{slate.name}: incomplete incumbent forecasts')
    if not np.array_equal(joined.label_row_id, slate.pool.id):
        raise ValueError('candidate IDs moved during alignment')
    joined['actual'] = slate.actual
    joined['selection_score'] = joined[selection_col]
    joined['cutoff'] = slate.cutoff.isoformat()
    return joined.drop(columns='_merge')


def deployed_squad(frame, positions):
    pool = frame.copy()
    pool['canonical_pos'] = pool[positions]
    xv = _pick_xv(pool, 'selection_score').copy()
    if len(xv) != 15:
        raise ValueError('deployed selector did not produce a complete XV')
    xv['is_capt'] = xv.index == xv.captain_score.idxmax()
    xv['is_sub'] = False
    bench = _supersub_pool(pool, set(xv.player_id))
    if bench.empty:
        raise ValueError('deployed selector has no supersub candidate')
    sub = bench.loc[[bench.supersub_score.idxmax()]].copy()
    sub['is_capt'], sub['is_sub'] = False, True
    return pd.concat([xv, sub], ignore_index=True)


def squad_components(squad, actual):
    points = squad.label_row_id.map(actual).to_numpy(float)
    cap = squad.is_capt.to_numpy(bool)
    sub = squad.is_sub.to_numpy(bool)
    observed = np.isfinite(points)
    known = float(points[observed & ~sub].sum()+points[observed & cap].sum()+3*points[observed & sub].sum())
    def total(mask, multiplier=1.0):
        return float(multiplier*points[mask].sum()) if observed[mask].all() else np.nan
    return {'xv_points': total(~sub), 'captain_extra': total(cap), 'supersub_points': total(sub, 3.0),
            'team_points': known if observed.all() else np.nan, 'known_contribution': known,
            'team_unlabelled': int((~observed).sum()), 'country_limit_exceeded': int(squad.groupby('team').size().gt(4).sum())}


def record_policy(slate, engine, frame, output, positions):
    squad = deployed_squad(frame, positions)
    squad.to_csv(output/slate.name/f'{engine}_squad.csv', index=False)
    actual = dict(zip(frame.label_row_id, frame.actual))
    known = np.isfinite(frame.actual)
    return {'slate': slate.name, 'season': slate.season, 'round': slate.round, 'engine': engine,
            'n': len(frame), 'n_labelled': int(known.sum()), 'n_unlabelled': int((~known).sum()),
            'mae': float(np.abs(frame.loc[known, 'target_pts_hat']-frame.loc[known, 'actual']).mean()),
            **squad_components(squad, actual)}


def add_common_metrics(slate, engine, points, output):
    result = evaluate(slate, engine, points, output)
    squad = pd.read_csv(output/slate.name/f'{engine}_squad.csv').rename(columns={'id': 'label_row_id'})
    actual = dict(zip(slate.pool.id, slate.actual))
    return {**result, **squad_components(squad, actual)}


def archived_comparators(slate, evidence, output):
    job = evidence/f'comparison-slate-{slate.name}'/'results'
    manifest = json.loads((job/'run_manifest.json').read_text())
    output_manifest = json.loads((output/'manifest.json').read_text())
    if manifest['store_sha256'] != output_manifest['store_sha256']:
        raise ValueError('PR24 comparison store changed')
    rows = []
    paths = list((job/slate.name).glob('*_predictions.csv'))
    frozen = evidence/f'comparison-slate-six_nations_{slate.season}_r1'/'results'/slate.name/'p3_tournament_frozen_native_predictions.csv'
    if frozen.exists() and frozen not in paths:
        paths.append(frozen)
    for path in sorted(paths):
        engine = path.name.removesuffix('_predictions.csv')
        frame = pd.read_csv(path)
        if not np.array_equal(frame.id, slate.pool.id) or not np.allclose(frame.actual, slate.actual, equal_nan=True):
            raise ValueError(f'{slate.name}/{engine}: archived candidate alignment failed')
        rows.append(add_common_metrics(slate, engine, frame.predicted.to_numpy(), output))
    return rows


def forecast_slate(native, cfg, slate, output, feature_history, mode):
    visible = native[(native.season < slate.season) | (native.season.eq(slate.season) & native['round'].le(slate.round))].copy()
    audit = {}
    if mode == 'lock_rebuilt':
        visible, audit = feature_history.lock_features(visible, slate)
    current = visible.season.eq(slate.season) & visible['round'].eq(slate.round)
    visible.loc[current, ['has_label', 'is_modern']] = False
    visible.loc[current, ['official_pts', 'target_pts']] = np.nan
    predicted, selection_col, _ = _predict_config(visible.reset_index(drop=True), cfg, slate.season)
    predicted = predicted[predicted['round'].eq(slate.round)]
    frame = aligned_predictions(slate, predicted, selection_col)
    frame.to_csv(output/slate.name/f'incumbent_{mode}_full_forecasts.csv', index=False)
    audit.update({'slate': slate.name, 'mode': mode, 'cutoff': slate.cutoff.isoformat(),
                  'native_rows_visible': len(visible), 'training_rows': int(visible.season.lt(slate.season).sum()),
                  'forecast_rows': len(frame), 'position_differences': int(frame.position.ne(frame.native_position).sum()),
                  'latest_native_training_date': str(visible.loc[visible.season.lt(slate.season), 'date'].max())})
    write_json(output/slate.name/f'incumbent_{mode}_audit.json', audit)
    rows = [add_common_metrics(slate, f'incumbent_{mode}_common', frame.target_pts_hat.to_numpy(), output)]
    for positions, suffix in [('native_position', 'deployed_policy'), ('position', 'common_roles_policy')]:
        rows.append(record_policy(slate, f'incumbent_{mode}_{suffix}', frame, output, positions))
    return rows, frame


def run(store_path, evidence, output, selected_slate=None):
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    os.makedirs(output, exist_ok=True)
    store = pd.read_csv(store_path, dtype={k: str for k in KEY}, parse_dates=['date', 'match_at'], low_memory=False)
    cfg = config_from_dict(json.loads((ROOT/'research/promoted_config.json').read_text()))
    manifest = source_manifest(store_path, evidence, cfg)
    manifest_path = output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('incumbent replay inputs changed; use a new output directory')
    write_json(manifest_path, manifest)
    native = load()
    slates = official_slates(store, ('six_nations',))
    if selected_slate:
        slates = [s for s in slates if s.name == selected_slate]
    history = IncumbentFeatureHistory(ROOT, store, native)
    metrics, predictions = [], []
    for slate in slates:
        os.makedirs(output/slate.name, exist_ok=True)
        print(f'{slate.name}: aligned {len(slate.pool)} candidates; {np.isfinite(slate.actual).sum()} outcomes', flush=True)
        metrics.extend(archived_comparators(slate, evidence, output))
        for mode in ('native_store', 'lock_rebuilt'):
            print(f'{slate.name}: fitting {mode}', flush=True)
            rows, frame = forecast_slate(native, cfg, slate, output, history, mode)
            metrics.extend(rows)
            predictions.append(frame.assign(slate=slate.name, mode=mode))
        pd.DataFrame(metrics).to_csv(output/'metrics.csv', index=False)
        pd.concat(predictions, ignore_index=True).to_csv(output/'forecasts.csv', index=False)
    table = pd.DataFrame(metrics)
    summary = table.groupby(['season', 'engine'], as_index=False).agg(
        team_points=('team_points', lambda x: x.sum(min_count=len(x))), mae=('mae', 'mean'),
        known_contribution=('known_contribution', 'sum'), n=('n', 'sum'), n_labelled=('n_labelled', 'sum'),
        team_unlabelled=('team_unlabelled', 'sum'), rounds=('slate', 'nunique'))
    summary.to_csv(output/'summary.csv', index=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--slate')
    args = parser.parse_args()
    run(ROOT/args.store, ROOT/args.evidence, ROOT/args.output, args.slate)


if __name__ == '__main__':
    main()
