"""Four-benchmark ledger for every hill-climb candidate, under one scoring contract.

All fantasy points are recomputed from saved raw forecasts with the current
adapters (NCR scrum points restricted to the front row). Squad points are
reported twice: as realised, and smoothed as the mean over 40 identical 1%
multiplicative forecast jitters, which removes knife-edge tie-break luck while
still using only realised outcomes. Fantasy MAE is never jittered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.unified.friendly_eval import score_stats, summarise_stats
from model.unified.rolling_eval import ROOT, evaluate, expected_points
from research.context_experiment import load_store
from research.friendly_experiment import load_raw
from research.raw_transform_eval import transform
from research.reweight_eval import cached_slates

JITTER, DRAWS, SEED = 0.01, 40, 12345


def squad_points(slate, values, directory):
    realised = evaluate(slate, 'ledger', values, directory)
    rng = np.random.default_rng(SEED)
    smoothed = [evaluate(slate, 'ledger', values*np.exp(rng.normal(0, JITTER, len(values))), directory)['team_points']
                for _ in range(DRAWS)]
    return realised, float(np.mean(smoothed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--transforms', type=Path, required=True, help='JSON {name: {gamma?, beta?}}')
    parser.add_argument('--blends', type=Path, required=True, help='JSON {name: {"weight": w, "transform": name|null}}')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = args.runs
    transforms = json.loads(args.transforms.read_text())
    blends = json.loads(args.blends.read_text())
    store, _ = load_store(runs/'base')
    rows = []
    for slate in cached_slates(runs/'base'):
        reference = load_raw(runs/'base'/'models'/slate.name/'p3_robust_native.jsonl')
        raws = {'p3_robust_native': reference}
        for name, spec in transforms.items():
            raws[name] = transform(reference, slate.candidates, store, slate.cutoff, spec)
        for directory in ('ctx', 'ctx_s5', 'ctx_ncr'):
            path = runs/directory/'models'/slate.name/'candidate.jsonl'
            if path.exists():
                raws['context_features'] = load_raw(path)
        values = {name: expected_points(raw, slate.competition) for name, raw in raws.items()}
        values['empirical_baseline'] = slate.baseline
        for name, spec in blends.items():
            source = values[spec['transform'] or 'p3_robust_native']
            values[name] = spec['weight']*source + (1 - spec['weight'])*slate.baseline
        for name, points in values.items():
            realised, smoothed = squad_points(slate, points, args.output/'squads')
            rows.append({**{k: realised[k] for k in ('slate', 'competition', 'season', 'round', 'mae',
                                                     'team_points', 'n_labelled')},
                         'engine': name, 'smoothed_points': smoothed})
        print(f'{slate.name}: {len(values)} candidates', flush=True)
    official = pd.DataFrame(rows)
    official.to_csv(args.output/'official_rounds.csv', index=False)
    season = official.groupby(['competition', 'season', 'engine']).agg(
        mae=('mae', 'mean'), team_points=('team_points', lambda v: v.sum(min_count=len(v))),
        smoothed_points=('smoothed_points', 'sum'), rounds=('round', 'nunique')).reset_index()
    season.to_csv(args.output/'official_seasons.csv', index=False)

    manifest = json.loads((ROOT/'data'/'unified'/'friendly25'/'fixtures.json').read_text())
    fixtures = pd.DataFrame(manifest['fixtures'])
    fixtures['kickoff'] = pd.to_datetime(fixtures.kickoff, utc=True)
    stats = []
    for day, group in fixtures.groupby(fixtures.kickoff.dt.date):
        directory = runs/'f25_ctx'/'days'/str(day)
        truth = pd.read_pickle(directory/'truth.pkl')
        reference = load_raw(directory/'p3_robust_native.jsonl')
        stats.append(score_stats(truth, reference, engine='p3_robust_native'))
        stats.append(score_stats(truth, load_raw(directory/'empirical_raw.jsonl'), engine='empirical_raw'))
        stats.append(score_stats(truth, load_raw(directory/'candidate.jsonl'), engine='context_features'))
        for name, spec in transforms.items():
            stats.append(score_stats(truth, transform(reference, truth, store, group.kickoff.min(), spec), engine=name))
    stats = pd.concat(stats, ignore_index=True)
    stats.to_csv(args.output/'friendly_stats.csv', index=False)
    summary, per_stat = summarise_stats(stats, fixtures.fixture_id.astype(str))
    summary.to_csv(args.output/'friendly_summary.csv', index=False)
    per_stat.to_csv(args.output/'friendly_per_stat.csv', index=False)
    print(season.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
