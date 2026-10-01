"""Combine the October 2026 parallel-agent fixes on top of robust P3 (no refits).

Bases: ``robust`` (reference p3_robust_native raw forecasts) and ``SK`` (the
status-aware, within-position-shrunk empirical component from
``research.position_supersub``, same trees and blend weights). Optional layers:
``MK`` (matchup calibration + kicking concentration, ``research.hillclimb2_eval``),
``BP`` (Six Nations official-stat adapter: breakdown steals + player of the
match, ``model.unified.official_stats``; never applied to NCR), and ``H2`` (MK
followed by the 0.7/0.3 empirical fantasy blend). Each layer's settings were
frozen by its own protocol; nothing here is tuned.

Sets: ``official`` (13 slates), ``dev`` (Six Nations 2023 rounds 2-4 under the
current rubric; round 1 is excluded for its shifted official rows) and
``friendly`` (Friendly-25 raw count statistics; BP and blends do not apply).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model.unified.friendly_eval import score_stats, summarise_stats
from model.unified.official_stats import OfficialStatConfig, SixNationsOfficialStats, load_compstats, official_history
from model.unified.rolling_eval import ROOT, expected_points
from research.context_experiment import load_store
from research.friendly_experiment import load_raw
from research.hillclimb2_eval import Context, squad_points, transform
from research.official_stats_eval import MK

H2_WEIGHT = 0.7
ENGINES = {  # name: (base, mk, bp, h2)
    'robust': ('robust', False, False, False),
    'robust+BP': ('robust', False, True, False),
    'SK': ('SK', False, False, False),
    'SK+BP': ('SK', False, True, False),
    'MK': ('robust', True, False, False),
    'MK+BP': ('robust', True, True, False),
    'SK+MK': ('SK', True, False, False),
    'SK+MK+BP': ('SK', True, True, False),
    'H2': ('robust', True, False, True),
    'SK+H2+BP': ('SK', True, True, True),
}


def aligned(raw, frame):
    by_key = {(p.fixture_id, p.player_id, p.team): p for p in raw}
    return [by_key[k] for k in zip(frame.fixture_id.astype(str), frame.player_id.astype(str), frame.team.astype(str))]


def official_set(args, store, context, slates, raw_dir, sk_dir):
    official = official_history(store)
    official = official[~(official.season.astype(int).eq(2023) & official['round'].astype(int).eq(1))]
    compstats = load_compstats()
    rows, players = [], []
    for slate in slates:
        frame = slate.candidates.reset_index(drop=True)
        bases = {'robust': aligned(load_raw(raw_dir/slate.name/'p3_robust_native.jsonl'), frame),
                 'SK': aligned(load_raw(sk_dir/slate.name/'shrunk.jsonl'), frame)}
        features = context.features(slate.name, slate.candidates, slate.cutoff)
        adapter = None
        if slate.competition == 'six_nations':
            adapter = SixNationsOfficialStats(OfficialStatConfig(breakdown_steals=True, metres=False, potm=True)).fit(
                official, compstats, store, slate.cutoff, slate.season)
        labelled = np.isfinite(slate.actual)
        values = {'empirical_baseline': np.asarray(slate.baseline, float)}
        for name, (base, mk, bp, h2) in ENGINES.items():
            raw = bases[base]
            if mk:
                raw = transform(raw, features, MK)
            if bp and adapter is not None:
                raw = adapter.adjust(raw, frame)
            points = expected_points(raw, slate.competition)
            values[name] = H2_WEIGHT*points + (1 - H2_WEIGHT)*values['empirical_baseline'] if h2 else points
        for name, points in values.items():
            realised, smoothed = squad_points(slate, points) if not args.no_smooth else (np.nan, np.nan)
            y, p = slate.actual[labelled], points[labelled]
            rows.append({'slate': slate.name, 'competition': slate.competition, 'season': slate.season,
                         'round': slate.round, 'engine': name, 'mae': float(np.abs(p - y).mean()),
                         'mse': float(((p - y)**2).mean()), 'pearson': float(np.corrcoef(p, y)[0, 1]),
                         'spearman': float(spearmanr(p, y)[0]), 'team_points': realised, 'smoothed_points': smoothed})
            players.append(pd.DataFrame({'slate': slate.name, 'engine': name,
                                         'fixture_id': frame.fixture_id.astype(str).to_numpy(),
                                         'predicted': points, 'actual': slate.actual}))
        print(f'{slate.name} done', flush=True)
    return pd.DataFrame(rows), pd.concat(players, ignore_index=True)


def friendly_set(runs, context, sk_root):
    import json
    manifest = json.loads((ROOT/'data'/'unified'/'friendly25'/'fixtures.json').read_text())
    fixtures = pd.DataFrame(manifest['fixtures'])
    fixtures['kickoff'] = pd.to_datetime(fixtures.kickoff, utc=True)
    stats = []
    for day, group in fixtures.groupby(fixtures.kickoff.dt.date):
        truth = pd.read_pickle(runs/'f25_ctx'/'days'/str(day)/'truth.pkl')
        frame = truth.reset_index(drop=True)
        robust = aligned(load_raw(runs/'f25_ctx'/'days'/str(day)/'p3_robust_native.jsonl'), frame)
        sk = aligned(load_raw(sk_root/str(day)/'shrunk.jsonl'), frame)
        features = context.features(f'friendly_{day}', frame, group.kickoff.min())
        for name, raw in (('robust', robust), ('SK', sk), ('MK', transform(robust, features, MK)),
                          ('SK+MK', transform(sk, features, MK))):
            stats.append(score_stats(truth, raw, engine=name))
    stats = pd.concat(stats, ignore_index=True)
    return stats, summarise_stats(stats, fixtures.fixture_id.astype(str))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='scratch runs directory (base/, devcr/, f25_ctx/)')
    parser.add_argument('--status', type=Path, required=True, help='research.position_supersub output (official/, devcr/, friendly/)')
    parser.add_argument('--cache', type=Path, required=True, help='matchup feature cache')
    parser.add_argument('--set', choices=['official', 'dev', 'friendly'], required=True)
    parser.add_argument('--no-smooth', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store, _ = load_store(args.runs/'base')
    context = Context(store, args.cache)
    if args.set == 'friendly':
        stats, summary = friendly_set(args.runs, context, args.status/'friendly')
        stats.to_csv(args.output/'friendly_stats.csv', index=False)
        summary.to_csv(args.output/'friendly_summary.csv', index=False)
        print(summary.to_string(index=False))
        return
    if args.set == 'dev':
        slates = [s for s in pickle.loads((args.runs/'devcr'/'slates.pkl').read_bytes())
                  if s.season == 2023 and 2 <= s.round <= 4]
        raw_dir, sk_dir = args.runs/'devcr'/'models', args.status/'devcr'
    else:
        from research.reweight_eval import cached_slates
        slates, raw_dir, sk_dir = cached_slates(args.runs/'base'), args.runs/'base'/'models', args.status/'official'
    rows, players = official_set(args, store, context, slates, raw_dir, sk_dir)
    rows.to_csv(args.output/'rounds.csv', index=False)
    players.to_pickle(args.output/'players.pkl')
    summary = rows.groupby(['competition', 'season', 'engine']).agg(
        mae=('mae', 'mean'), mse=('mse', 'mean'), pearson=('pearson', 'mean'),
        team_points=('team_points', 'sum'), smoothed_points=('smoothed_points', 'sum')).reset_index()
    summary.to_csv(args.output/'seasons.csv', index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
