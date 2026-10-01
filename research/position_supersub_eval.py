"""Score status-aware forecasts: fantasy MAE, positional bias, discrimination and super-subs.

Reads the reference robust P3 forecasts and the status-aware forecasts written
by ``research.position_supersub``. Optional matchup/kicking steps reuse
``research.hillclimb2_eval.transform`` with its cached pre-lock features.

Per slate and engine it reports fantasy MAE and MSE, within-round Pearson and
Spearman correlation, realised squad points and smoothed squad points (mean
over the same 40 identical 1% jitters as the hill-climb ledger), and the
super-sub contribution (tripled realised points of the chosen replacement)
both realised and smoothed. Player-level rows are kept for bias tables and
fixture bootstraps.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model.ncr_eval import team_points
from model.unified.rolling_eval import expected_points
from research.hillclimb2_eval import Context, squad, transform
from research.context_experiment import load_store
from research.position_supersub import load_raw
from research.reweight_eval import cached_slates

JITTER, DRAWS, SEED = 0.01, 40, 12345
MK = {'matchup': 'research/hillclimb_2026-10-02/matchup_coefficients.json', 'gamma': 1.5}


def supersub_points(selected: pd.DataFrame, actuals: dict) -> tuple[float, float, str]:
    """Super-sub contribution with the official multiplier rule, and captain points."""
    sub, capt, name = 0.0, 0.0, ''
    for r in selected.itertuples():
        pts, played, status = actuals.get(str(int(float(r.id))), (0.0, 0.0, None))
        if r.is_sub:
            sub = (3.0 if (status == 'B' and played > 0) else (0.5 if played > 0 else 0.0))*pts
            name = r.name
        if r.is_capt:
            capt = 2.0*pts
    return sub, capt, name


def decisions(slate, points: np.ndarray, smooth: bool = True) -> dict:
    chosen = squad(slate, points)
    actuals = {k: (0.0 if not np.isfinite(v[0]) else v[0], v[1], v[2]) for k, v in slate.team_actuals.items()}
    sub, capt, name = supersub_points(chosen, actuals)
    out = {'team_points': team_points(chosen, slate.team_actuals), 'supersub_points': sub, 'captain_points': capt,
           'supersub': name, 'supersub_pred': float(points[slate.pool.id.to_numpy() == chosen[chosen.is_sub].id.iloc[0]][0])}
    if smooth:
        rng = np.random.default_rng(SEED)
        totals, subs, caps, names = [], [], [], []
        for _ in range(DRAWS):
            jittered = squad(slate, points*np.exp(rng.normal(0, JITTER, len(points))))
            totals.append(team_points(jittered, slate.team_actuals))
            s, c, n = supersub_points(jittered, actuals)
            subs.append(s); caps.append(c); names.append(n)
        out.update(smoothed_points=float(np.mean(totals)), smoothed_supersub=float(np.mean(subs)),
                   smoothed_captain=float(np.mean(caps)), supersub_modal_share=pd.Series(names).value_counts(normalize=True).iloc[0])
    return out


def engines_for(slate, reference, status, context: Context | None, mk: bool) -> dict[str, np.ndarray]:
    values = {'p3_robust_native': expected_points(reference, slate.competition),
              'status_rates': expected_points(status, slate.competition),
              'empirical_baseline': np.asarray(slate.baseline, float)}
    if mk and context is not None:
        features = context.features(slate.name, slate.candidates, slate.cutoff)
        values['MK'] = expected_points(transform(reference, features, MK), slate.competition)
        values['status_MK'] = expected_points(transform(status, features, MK), slate.competition)
    return values


def evaluate_set(slates, reference_dir: Path, status_dir: Path, context: Context | None, mk: bool,
                 smooth: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    rounds, players = [], []
    for slate in slates:
        keys = list(zip(slate.candidates.fixture_id.astype(str), slate.candidates.player_id.astype(str),
                        slate.candidates.team.astype(str)))
        reference = load_raw(reference_dir/slate.name/'p3_robust_native.jsonl')
        status = load_raw(status_dir/slate.name/'status.jsonl')
        for raw in (reference, status):
            if [(p.fixture_id, p.player_id, p.team) for p in raw] != keys:
                raise ValueError(f'{slate.name}: forecasts misaligned')
        labelled = np.isfinite(slate.actual)
        minutes = slate.candidates.minutes if 'minutes' in slate.candidates and slate.candidates.minutes.notna().any() else None
        for engine, points in engines_for(slate, reference, status, context, mk).items():
            y, p = slate.actual[labelled], points[labelled]
            row = {'slate': slate.name, 'competition': slate.competition, 'season': slate.season, 'round': slate.round,
                   'engine': engine, 'n': int(labelled.sum()), 'mae': float(np.abs(p-y).mean()),
                   'mse': float(((p-y)**2).mean()), 'bias': float((p-y).mean()),
                   'pearson': float(np.corrcoef(p, y)[0, 1]), 'spearman': float(spearmanr(p, y).correlation)}
            bench = (slate.pool.status.to_numpy() == 'B') & labelled
            row['bench_pearson'] = float(np.corrcoef(points[bench], slate.actual[bench])[0, 1])
            row.update(decisions(slate, points, smooth))
            rounds.append(row)
            players.append(pd.DataFrame({'slate': slate.name, 'engine': engine, 'pool_id': slate.pool.id.to_numpy(),
                                         'fixture_id': slate.candidates.fixture_id.astype(str).to_numpy(),
                                         'player_name': slate.pool.name.to_numpy(), 'team': slate.pool.team.to_numpy(),
                                         'position': slate.candidates.position.to_numpy(), 'status': slate.pool.status.to_numpy(),
                                         'predicted': points, 'actual': slate.actual}))
        print(f'{slate.name} done', flush=True)
    return pd.DataFrame(rounds), pd.concat(players, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True, help='research.position_supersub output')
    parser.add_argument('--set', choices=['dev', 'official'], required=True)
    parser.add_argument('--mk', action='store_true', help='also score matchup + kicking (MK) on both forecasts')
    parser.add_argument('--no-smooth', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    context = None
    if args.mk:
        store, _ = load_store(args.runs/'base')
        context = Context(store, args.runs/'feature_cache')
    if args.set == 'dev':
        slates = pickle.loads((args.runs/'devcr'/'slates.pkl').read_bytes())
        reference_dir, status_dir = args.runs/'devcr'/'models', args.status/'devcr'
    else:
        slates = cached_slates(args.runs/'base')
        reference_dir, status_dir = args.runs/'base'/'models', args.status/'official'
    rounds, players = evaluate_set(slates, reference_dir, status_dir, context, args.mk, not args.no_smooth)
    rounds.to_csv(args.output/'rounds.csv', index=False)
    players.to_pickle(args.output/'players.pkl')
    seasons = rounds.groupby(['competition', 'season', 'engine']).agg(
        mae=('mae', 'mean'), mse=('mse', 'mean'), pearson=('pearson', 'mean'), spearman=('spearman', 'mean'),
        bench_pearson=('bench_pearson', 'mean'), team_points=('team_points', 'sum'),
        smoothed_points=('smoothed_points', 'sum'), supersub=('supersub_points', 'sum'),
        smoothed_supersub=('smoothed_supersub', 'sum'), rounds=('round', 'nunique')).reset_index()
    seasons.to_csv(args.output/'seasons.csv', index=False)
    print(seasons.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
