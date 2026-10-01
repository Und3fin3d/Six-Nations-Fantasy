"""Summaries for ``research.official_stats_eval`` output: bootstraps, components, positions, decisions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

B, SEED = 2000, 7


def paired_bootstrap(players: pd.DataFrame, engine: str, reference: str) -> pd.DataFrame:
    """90% intervals for the change in round-mean MAE and MSE, resampling fixtures within rounds."""
    rows = []
    rng = np.random.default_rng(SEED)
    for season, frame in players.groupby('season'):
        a = frame[frame.engine.eq(engine)].set_index(['slate', 'fixture_id', 'player_id', 'team'])
        b = frame[frame.engine.eq(reference)].set_index(['slate', 'fixture_id', 'player_id', 'team'])
        joined = a[['predicted', 'actual']].join(b[['predicted']], rsuffix='_ref').reset_index()
        joined = joined[np.isfinite(joined.actual)]
        joined['dae'] = (joined.predicted - joined.actual).abs() - (joined.predicted_ref - joined.actual).abs()
        joined['dse'] = (joined.predicted - joined.actual)**2 - (joined.predicted_ref - joined.actual)**2
        fixtures = joined.groupby(['slate', 'fixture_id']).agg(dae=('dae', 'sum'), dse=('dse', 'sum'), n=('dae', 'size'))
        point = joined.groupby('slate')[['dae', 'dse']].mean().mean()
        draws = []
        by_round = [g.to_numpy() for _, g in fixtures.groupby(level='slate')[['dae', 'dse', 'n']]]
        for _ in range(B):
            means = []
            for values in by_round:
                pick = values[rng.integers(0, len(values), len(values))]
                means.append(pick[:, :2].sum(axis=0)/pick[:, 2].sum())
            draws.append(np.mean(means, axis=0))
        draws = np.array(draws)
        rounds_better = int((joined.groupby('slate').dae.mean() < 0).sum())
        rows.append({'season': season, 'engine': engine, 'reference': reference,
                     'd_mae': point.dae, 'mae_lo': np.quantile(draws[:, 0], .05), 'mae_hi': np.quantile(draws[:, 0], .95),
                     'd_mse': point.dse, 'mse_lo': np.quantile(draws[:, 1], .05), 'mse_hi': np.quantile(draws[:, 1], .95),
                     'rounds_mae_better': rounds_better, 'rounds': joined.slate.nunique()})
    return pd.DataFrame(rows)


def components(players: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    frame = players[np.isfinite(players.actual) & players.off_min.notna()].copy()
    frame['bs_act'] = 5*frame.off_BS.fillna(0)
    frame['metres_act'] = np.floor(frame.off_MC.fillna(0)/10)
    frame['potm_act'] = 15*frame.off_POTM.fillna(0)
    out = []
    for keys, g in frame.groupby(by):
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row['n'] = len(g)
        for comp in ('bs', 'metres', 'potm'):
            p, a = g[f'{comp}_pts'], g[f'{comp}_act']
            row[f'{comp}_pred'] = p.mean()
            row[f'{comp}_act'] = a.mean()
            row[f'{comp}_mse'] = ((p - a)**2).mean()
            row[f'{comp}_corr'] = np.corrcoef(p, a)[0, 1] if p.std() > 0 and a.std() > 0 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    players = pd.read_pickle(args.input/'players.pkl')
    rounds = pd.read_csv(args.input/'rounds.csv')
    boots = []
    for engine in sorted(players.engine.unique()):
        base, variant = engine.split(':')
        if variant != 'none':
            boots.append(paired_bootstrap(players, engine, f'{base}:none'))
    if {'robust:none', 'mk:none'} <= set(players.engine):
        boots.append(paired_bootstrap(players, 'mk:BMP', 'robust:none'))
    boots = pd.concat(boots, ignore_index=True)
    boots.to_csv(args.output/'bootstrap.csv', index=False)
    print(boots.round(4).to_string(index=False))
    comp = components(players, ['season', 'engine'])
    comp.to_csv(args.output/'components_by_season.csv', index=False)
    print(comp.round(3).to_string(index=False))
    pos = components(players[players.engine.str.startswith('robust:')], ['engine', 'position', 'started'])
    pos.to_csv(args.output/'components_by_position.csv', index=False)
    # Total predicted-minus-actual by position x started, per engine
    labelled = players[np.isfinite(players.actual)]
    bias = labelled.assign(err=labelled.predicted - labelled.actual).groupby(['engine', 'position', 'started']).agg(
        n=('err', 'size'), pred=('predicted', 'mean'), actual=('actual', 'mean'), bias=('err', 'mean'),
        mae=('err', lambda e: e.abs().mean())).reset_index()
    bias.to_csv(args.output/'total_bias_by_position.csv', index=False)
    decisions = rounds[['slate', 'engine', 'captain', 'supersub', 'team_points'] +
                       (['smoothed_points'] if 'smoothed_points' in rounds else [])]
    decisions.to_csv(args.output/'decisions.csv', index=False)


if __name__ == '__main__':
    main()
