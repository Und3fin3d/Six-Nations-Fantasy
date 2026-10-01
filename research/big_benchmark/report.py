"""Pooled metrics, paired bootstraps and power analysis for the large benchmark.

Accuracy metrics are computed per slate (MAE, MSE, bias, Pearson and Spearman
within the slate) and averaged with every slate weighted equally, as the
official ledger averages round MAEs. Pooled player-weighted MAE/RMSE and
decile calibration are also written.

Differences against the reference are bootstrapped three ways: resampling
slates (i.i.d.), whole blocks (competition seasons) and whole locks (one
model fit per month). The block/lock design effects feed the power analysis,
which converts the measured standard deviation of paired per-slate
differences into the number of slates needed to detect a given difference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .rubrics import RUBRICS
from .score import ENGINES, REFERENCE
from .stats import (calibration_deciles, calibration_line, detectable, paired_bootstrap, power_slates)

LOWER_IS_BETTER = {'mae': True, 'mse': True, 'abs_bias': True, 'pearson': False, 'spearman': False,
                   'realised': False, 'smoothed': False, 'flat': False, 'smoothed_flat': False,
                   'regret': True, 'smoothed_regret': True}
DELTAS = {'mae': 0.02, 'mse': 0.5, 'spearman': 0.005, 'pearson': 0.005, 'realised': 5.0, 'smoothed': 5.0,
          'flat': 5.0, 'smoothed_flat': 5.0}


def load_all_players(output: Path, manifest: dict) -> pd.DataFrame:
    frames = []
    for lock in manifest['locks']:
        path = output/'players'/f"{pd.Timestamp(lock).strftime('%Y-%m')}.pkl"
        if path.exists():
            frames.append(pd.read_pickle(path))
    if not frames:
        raise FileNotFoundError('no scored locks')
    return pd.concat(frames, ignore_index=True)


def slate_metrics(players: pd.DataFrame, engines, rubrics) -> pd.DataFrame:
    rows = []
    meta = ['slate', 'lock', 'slate_competition', 'family', 'block']
    for keys, group in players.groupby(meta, sort=False):
        for rubric in rubrics:
            actual = group[f'actual__{rubric}'].to_numpy(float)
            valid = np.isfinite(actual)
            if valid.sum() < 10:
                continue
            a = actual[valid]
            for engine in engines:
                p = group[f'pred__{engine}__{rubric}'].to_numpy(float)[valid]
                err = p - a
                ps, as_ = pd.Series(p), pd.Series(a)
                rows.append({**dict(zip(meta, keys)), 'rubric': rubric, 'engine': engine, 'n': int(valid.sum()),
                             'mae': float(np.abs(err).mean()), 'mse': float((err**2).mean()),
                             'bias': float(err.mean()), 'abs_bias': float(abs(err.mean())),
                             'pearson': float(ps.corr(as_)), 'spearman': float(ps.corr(as_, method='spearman')),
                             'sum_abs': float(np.abs(err).sum()), 'sum_sq': float((err**2).sum())})
    return pd.DataFrame(rows)


def pooled_summary(players: pd.DataFrame, slates: pd.DataFrame, engines, rubrics) -> pd.DataFrame:
    rows = []
    scopes = {'all': players.index}
    for family, group in players.groupby('family'):
        scopes[family] = group.index
    for comp, group in players.groupby('slate_competition'):
        scopes[comp] = group.index
    for scope, index in scopes.items():
        subset = players.loc[index]
        for rubric in rubrics:
            actual = subset[f'actual__{rubric}'].to_numpy(float)
            valid = np.isfinite(actual)
            for engine in engines:
                pred = subset[f'pred__{engine}__{rubric}'].to_numpy(float)[valid]
                err = pred - actual[valid]
                intercept, slope = calibration_line(pred, actual[valid])
                srows = slates[(slates.rubric == rubric) & (slates.engine == engine)]
                if scope != 'all':
                    srows = srows[(srows.family == scope) | (srows.slate_competition == scope)]
                rows.append({'scope': scope, 'rubric': rubric, 'engine': engine, 'players': int(valid.sum()),
                             'slates': int(len(srows)), 'slate_mae': float(srows.mae.mean()),
                             'slate_rmse': float(np.sqrt(srows.mse.mean())),
                             'pooled_mae': float(np.abs(err).mean()), 'pooled_rmse': float(np.sqrt((err**2).mean())),
                             'bias': float(err.mean()), 'pearson': float(srows.pearson.mean()),
                             'spearman': float(srows.spearman.mean()), 'calibration_intercept': intercept,
                             'calibration_slope': slope, 'mean_actual': float(actual[valid].mean()),
                             'mean_predicted': float(pred.mean())})
    return pd.DataFrame(rows)


def differences(frame: pd.DataFrame, metrics, engines, rubrics, scope_columns=('family', 'slate_competition'),
                reference: str = REFERENCE) -> pd.DataFrame:
    rows = []
    scopes = [('all', frame)]
    for column in scope_columns:
        for value, group in frame.groupby(column):
            scopes.append((value, group))
    for scope, subset in scopes:
        for rubric in rubrics:
            block = subset[subset.rubric == rubric]
            if block.empty:
                continue
            for metric in metrics:
                for engine in engines:
                    if engine == reference or engine not in set(block.engine):
                        continue
                    slate = paired_bootstrap(block, engine, reference, metric, cluster='slate')
                    by_block = paired_bootstrap(block, engine, reference, metric, cluster='block', unit='slate')
                    by_lock = paired_bootstrap(block, engine, reference, metric, cluster='lock', unit='slate')
                    better = slate['share_better'] if LOWER_IS_BETTER[metric] else slate['share_positive']
                    rows.append({'scope': scope, 'rubric': rubric, 'metric': metric, 'engine': engine,
                                 'reference': reference, 'slates': slate['n'], 'blocks': by_block['clusters'],
                                 'locks': by_lock['clusters'], 'diff': slate['mean'], 'sd_slate_diff': slate['sd'],
                                 'slate_p05': slate['p05'], 'slate_p95': slate['p95'],
                                 'block_p05': by_block['p05'], 'block_p95': by_block['p95'],
                                 'lock_p05': by_lock['p05'], 'lock_p95': by_lock['p95'],
                                 'share_slates_better': better})
    return pd.DataFrame(rows)


def design_effect(row: pd.Series) -> float:
    iid = (row.slate_p95 - row.slate_p05)**2
    clustered = max((row.block_p95 - row.block_p05)**2, (row.lock_p95 - row.lock_p05)**2)
    return float(max(clustered/iid, 1.0)) if iid > 0 else np.nan


def official_round_sds(path: Path | None) -> pd.DataFrame:
    """Per-round paired SDs on the 13 official rounds (``hillclimb2_eval`` rounds.csv)."""
    if path is None or not path.exists():
        return pd.DataFrame()
    rounds = pd.read_csv(path)
    rows = []
    names = {'ref_recomputed': 'p3_robust', 'p3_robust_native': 'p3_robust', 'H2': 'h2', 'MK': 'mk',
             'H1dev': 'h1_oct1', 'empirical_baseline': 'empirical_baseline'}
    rounds['engine'] = rounds['engine'].map(names)
    rounds = rounds.dropna(subset=['engine'])
    for metric, column in (('mae', 'mae'), ('realised', 'team_points'), ('smoothed', 'smoothed_points')):
        pivot = rounds.pivot_table(index='slate', columns='engine', values=column)
        for engine in pivot.columns:
            if engine == 'p3_robust':
                continue
            diff = (pivot[engine] - pivot['p3_robust']).dropna()
            rows.append({'metric': metric, 'engine': engine, 'rounds': len(diff), 'diff': float(diff.mean()),
                         'sd_round_diff': float(diff.std(ddof=1))})
    return pd.DataFrame(rows)


def power_table(diffs: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scope = diffs[diffs.scope.isin(['all', 'international', 'club', 'six_nations'])]
    for row in scope.itertuples(index=False):
        if row.metric not in DELTAS:
            continue
        delta = DELTAS[row.metric]
        deff = design_effect(pd.Series(row._asdict()))
        n_iid = power_slates(row.sd_slate_diff, delta)
        record = {'scope': row.scope, 'rubric': row.rubric, 'metric': row.metric, 'engine': row.engine,
                  'delta': delta, 'sd_slate_diff': row.sd_slate_diff, 'design_effect': deff,
                  'slates_needed': n_iid*deff if np.isfinite(deff) else np.nan, 'slates_available': row.slates,
                  'mde_available': detectable(row.sd_slate_diff, row.slates/deff) if np.isfinite(deff) else np.nan,
                  'observed_diff': row.diff}
        if len(official) and row.rubric == 'six_nations':
            match = official[(official.metric == row.metric) & (official.engine == row.engine)]
            if len(match):
                sd = float(match.sd_round_diff.iloc[0])
                record.update({'official_sd_round_diff': sd, 'official_rounds_needed': power_slates(sd, delta),
                               'official_mde_13': detectable(sd, 13)})
        rows.append(record)
    return pd.DataFrame(rows)


def decision_frame(output: Path, manifest: dict) -> pd.DataFrame:
    path = output/'decisions.csv'
    if not path.exists():
        return pd.DataFrame()
    decisions = pd.read_csv(path)
    info = pd.DataFrame(manifest['slates'])[['slate', 'block', 'family', 'competition']].rename(
        columns={'competition': 'slate_competition'})
    return decisions.merge(info, on='slate', how='left', validate='many_to_one')


def events_summary(output: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((output/'events').glob('*.csv'))]
    if not frames:
        return pd.DataFrame()
    events = pd.concat(frames, ignore_index=True)
    events = events[events.cohort.eq('all')]
    events['loss_n'] = events['loss']*events['n']
    events['mae_n'] = events['mae']*events['n']
    pooled = events.groupby(['target', 'tier', 'engine']).agg(n=('n', 'sum'), loss_n=('loss_n', 'sum'),
                                                              mae_n=('mae_n', 'sum')).reset_index()
    pooled['loss'] = pooled.loss_n/pooled.n
    pooled['mae'] = pooled.mae_n/pooled.n
    reference = pooled[pooled.engine.eq(REFERENCE)].set_index('target')
    pooled['loss_vs_reference'] = pooled.loss - pooled.target.map(reference.loss)
    pooled['relative_loss_vs_reference'] = pooled.loss/pooled.target.map(reference.loss)
    return pooled.drop(columns=['loss_n', 'mae_n'])


def report_stage(manifest: dict, output: Path, engines, rubrics, official_rounds: Path | None = None) -> dict:
    engines = list(engines or ENGINES)
    rubrics = list(rubrics or RUBRICS)
    directory = output/'report'
    directory.mkdir(parents=True, exist_ok=True)
    players = load_all_players(output, manifest)
    slates = slate_metrics(players, engines, rubrics)
    slates.to_csv(directory/'slate_metrics.csv', index=False)
    summary = pooled_summary(players, slates, engines, rubrics)
    summary.to_csv(directory/'accuracy_summary.csv', index=False)
    deciles = []
    for rubric in rubrics:
        valid = np.isfinite(players[f'actual__{rubric}'])
        for engine in engines:
            for family, group in [('all', players[valid]), *players[valid].groupby('family')]:
                table = calibration_deciles(group[f'pred__{engine}__{rubric}'].to_numpy(float),
                                            group[f'actual__{rubric}'].to_numpy(float))
                deciles.append(table.assign(rubric=rubric, engine=engine, family=family))
    pd.concat(deciles, ignore_index=True).to_csv(directory/'calibration_deciles.csv', index=False)
    accuracy_diffs = differences(slates, ('mae', 'mse', 'abs_bias', 'pearson', 'spearman'), engines, rubrics)
    decisions = decision_frame(output, manifest)
    decision_diffs = pd.DataFrame()
    if len(decisions):
        decisions.groupby(['rubric', 'engine', 'family']).agg(
            slates=('slate', 'nunique'), realised=('realised', 'mean'), smoothed=('smoothed', 'mean'),
            flat=('flat', 'mean'), smoothed_flat=('smoothed_flat', 'mean'), oracle=('oracle', 'mean'),
            regret=('regret', 'mean'), smoothed_regret=('smoothed_regret', 'mean'),
            captain_points=('captain_points', 'mean'), sub_points=('sub_points', 'mean'),
            smoothed_sd=('smoothed_sd', 'mean')).reset_index().to_csv(directory/'decision_summary.csv', index=False)
        decision_diffs = differences(decisions, ('realised', 'smoothed', 'flat', 'smoothed_flat'),
                                     [e for e in engines if e in set(decisions.engine)], rubrics)
    diffs = pd.concat([accuracy_diffs, decision_diffs], ignore_index=True)
    diffs.to_csv(directory/'differences.csv', index=False)
    official = official_round_sds(official_rounds)
    official.to_csv(directory/'official_round_sds.csv', index=False)
    power = power_table(diffs, official)
    power.to_csv(directory/'power.csv', index=False)
    events = events_summary(output)
    events.to_csv(directory/'events_summary.csv', index=False)
    result = {'players': int(len(players)), 'slates': int(players.slate.nunique()),
              'locks_scored': int(players['lock'].nunique()),
              'decision_slates': int(decisions.slate.nunique()) if len(decisions) else 0}
    (directory/'summary.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result), flush=True)
    return result
