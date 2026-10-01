"""Evaluate the Six Nations official-statistic adapter on saved raw forecasts (no refit).

For every Six Nations slate the adapter (``model.unified.official_stats``) is
fitted on pre-lock data only, then applied to the saved robust P3 forecasts and,
optionally, to the round-2 MK raw candidate (matchup calibration + kicking).
Variants switch the three components (B = breakdown steals, M = metres,
P = player of the match) on and off.

Outputs per player (predictions, official per-component actuals) and per round
(fantasy MAE, MSE, Pearson/Spearman, realised and smoothed squad points, captain
and super-sub identity). NCR slates are never adjusted.
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
from model.unified.official_stats import OfficialStatConfig, SixNationsOfficialStats, load_compstats, official_history
from model.unified.rolling_eval import ROOT, expected_points
from research.context_experiment import load_store
from research.friendly_experiment import load_raw
from research.hillclimb2_eval import Context, squad, transform

JITTER, DRAWS, SEED = 0.01, 40, 12345
MK = {'matchup': str(ROOT/'research'/'hillclimb_2026-10-02'/'matchup_coefficients.json'), 'gamma': 1.5}
VARIANTS = {
    'none': dict(breakdown_steals=False, metres=False, potm=False),
    'B': dict(breakdown_steals=True, metres=False, potm=False),
    'M': dict(breakdown_steals=False, metres=True, potm=False),
    'P': dict(breakdown_steals=False, metres=False, potm=True),
    'BP': dict(breakdown_steals=True, metres=False, potm=True),
    'BMP': dict(breakdown_steals=True, metres=True, potm=True),
}
COMPONENTS = ('tackle_turnover', 'metres', 'potm')


def squad_detail(slate, points: np.ndarray, smooth: bool) -> dict:
    chosen = squad(slate, points)
    result = {'team_points': team_points(chosen, slate.team_actuals),
              'captain': chosen.loc[chosen.is_capt, 'name'].iloc[0], 'supersub': chosen.loc[chosen.is_sub, 'name'].iloc[0],
              'squad_ids': ','.join(map(str, sorted(chosen.id)))}
    if smooth:
        rng = np.random.default_rng(SEED)
        result['smoothed_points'] = float(np.mean([
            team_points(squad(slate, points*np.exp(rng.normal(0, JITTER, len(points)))), slate.team_actuals)
            for _ in range(DRAWS)]))
    return result


def component_points(raw) -> pd.DataFrame:
    """Expected Six Nations points from each adjusted component."""
    from model.unified.rolling_eval import expected_points as points
    from dataclasses import replace
    rows = {'bs_pts': [5*p.events['tackle_turnover'].mean if 'tackle_turnover' in p.events else 0.0 for p in raw],
            'potm_pts': [15*p.events['potm'].mean if 'potm' in p.events else 0.0 for p in raw]}
    only_metres = [replace(p, events={k: v for k, v in p.events.items() if k == 'metres'}) for p in raw]
    rows['metres_pts'] = points(only_metres, 'six_nations')
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='scratch runs directory (base/, devcr/)')
    parser.add_argument('--set', choices=['dev', 'official'], required=True)
    parser.add_argument('--cache', type=Path, required=True, help='matchup feature cache directory (written if missing)')
    parser.add_argument('--bases', nargs='+', default=['robust', 'mk'])
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS))
    parser.add_argument('--rounds', nargs='*', help='only these slate names')
    parser.add_argument('--no-smooth', action='store_true')
    parser.add_argument('--exclude-official', nargs='*', default=[],
                        help='SEASON:ROUND official rounds withheld from the adapter history (sensitivity checks)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store, _ = load_store(args.runs/'base')
    context = Context(store, args.cache)
    official = official_history(store)
    for item in args.exclude_official:
        season, round_no = map(int, item.split(':'))
        official = official[~(official.season.astype(int).eq(season) & official['round'].astype(int).eq(round_no))]
    compstats = load_compstats()
    if args.set == 'dev':
        slates = [s for s in pickle.loads((args.runs/'devcr'/'slates.pkl').read_bytes())
                  if s.season == 2023 and s.round <= 4]
        raw_dir = args.runs/'devcr'/'models'
    else:
        from research.reweight_eval import cached_slates
        slates = [s for s in cached_slates(args.runs/'base') if s.competition == 'six_nations']
        raw_dir = args.runs/'base'/'models'
    if args.rounds:
        slates = [s for s in slates if s.name in args.rounds]
    rounds, players = [], []
    for slate in slates:
        reference = load_raw(raw_dir/slate.name/'p3_robust_native.jsonl')
        frame = slate.candidates.reset_index(drop=True)
        bases = {'robust': reference}
        if 'mk' in args.bases:
            bases['mk'] = transform(reference, context.features(slate.name, slate.candidates, slate.cutoff), MK)
        labelled = np.isfinite(slate.actual)
        truth = frame[['fixture_id', 'player_id', 'team']].astype(str).merge(
            official.astype({'fixture_id': str, 'player_id': str, 'team': str}),
            on=['fixture_id', 'player_id', 'team'], how='left')
        for variant in args.variants:
            adapter = SixNationsOfficialStats(OfficialStatConfig(**VARIANTS[variant])).fit(
                official, compstats, store, slate.cutoff, slate.season)
            for base_name, raw in bases.items():
                if base_name not in args.bases:
                    continue
                adjusted = adapter.adjust(raw, frame)
                points = expected_points(adjusted, 'six_nations')
                engine = f'{base_name}:{variant}'
                detail = squad_detail(slate, points, not args.no_smooth)
                y, p = slate.actual[labelled], points[labelled]
                rounds.append({'slate': slate.name, 'season': slate.season, 'round': slate.round, 'engine': engine,
                               'mae': float(np.abs(p - y).mean()), 'mse': float(((p - y)**2).mean()),
                               'bias': float((p - y).mean()), 'pearson': float(np.corrcoef(p, y)[0, 1]),
                               'spearman': float(spearmanr(p, y)[0]), **detail})
                comp = component_points(adjusted)
                players.append(pd.DataFrame({
                    'slate': slate.name, 'season': slate.season, 'round': slate.round, 'engine': engine,
                    'fixture_id': frame.fixture_id.astype(str), 'player_id': frame.player_id.astype(str),
                    'team': frame.team.astype(str), 'name': frame.player_name, 'position': [q.position for q in raw],
                    'started': frame.started.astype(bool).to_numpy(), 'predicted': points, 'actual': slate.actual,
                    'off_BS': truth.off_BS.to_numpy(), 'off_MC': truth.off_MC.to_numpy(),
                    'off_POTM': truth.off_POTM.to_numpy(), 'off_min': truth.off_Min.to_numpy(),
                    **{c: comp[c].to_numpy() for c in comp}}))
            print(f'{slate.name} {variant} done', flush=True)
    rounds = pd.DataFrame(rounds)
    rounds.to_csv(args.output/'rounds.csv', index=False)
    pd.concat(players, ignore_index=True).to_pickle(args.output/'players.pkl')
    agg = dict(mae=('mae', 'mean'), mse=('mse', 'mean'), bias=('bias', 'mean'), pearson=('pearson', 'mean'),
               spearman=('spearman', 'mean'), team_points=('team_points', 'sum'))
    if 'smoothed_points' in rounds:
        agg['smoothed_points'] = ('smoothed_points', 'sum')
    seasons = rounds.groupby(['season', 'engine']).agg(**agg).reset_index()
    seasons.to_csv(args.output/'seasons.csv', index=False)
    print(seasons.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
