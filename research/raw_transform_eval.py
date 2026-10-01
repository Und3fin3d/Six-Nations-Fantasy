"""Score raw-forecast post-processors on the official slates and Friendly-25.

Each candidate transforms the saved reference ``p3_robust_native`` raw
forecasts (no refit): team-level kicking concentration, team-strength
calibration, or both. Fantasy scoring, squads and friendly metrics reuse the
unchanged evaluators.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model.history import past_matches
from model.unified.friendly_eval import score_stats, summarise_stats
from model.unified.kicking import concentrate_kicking
from model.unified.rolling_eval import ROOT, evaluate, expected_points, season_summary
from model.unified.v4.context import add_candidate_context, pre_match_context, team_strength_scale
from research.context_experiment import load_store
from research.friendly_experiment import load_raw
from research.reweight_eval import cached_slates


def transform(raw, frame, store, cutoff, spec):
    if spec.get('beta'):
        state = pre_match_context(past_matches(store, cutoff))[1]
        edges = add_candidate_context(frame, state)['ctx__elo_edge']
        raw = [team_strength_scale(p, float(e), spec['beta']) for p, e in zip(raw, edges)]
    if spec.get('gamma'):
        raw = concentrate_kicking(raw, spec['gamma'])
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--friendly', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True, help='JSON {name: {gamma?, beta?}}')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.candidates.read_text())
    store, _ = load_store(args.base)
    rows = []
    for slate in cached_slates(args.base):
        raw = load_raw(args.base/'models'/slate.name/'p3_robust_native.jsonl')
        rows.append(evaluate(slate, 'p3_robust_native', expected_points(raw, slate.competition), args.output/'official'))
        rows.append(evaluate(slate, 'empirical_baseline', slate.baseline, args.output/'official'))
        for name, spec in specs.items():
            values = expected_points(transform(raw, slate.candidates, store, slate.cutoff, spec), slate.competition)
            rows.append(evaluate(slate, name, values, args.output/'official'))
    official = pd.DataFrame(rows)
    official.to_csv(args.output/'official_metrics.csv', index=False)
    print(season_summary(official)[['competition', 'season', 'engine', 'mae', 'team_points', 'scored_teams']].to_string(index=False), flush=True)

    manifest = json.loads((ROOT/'data'/'unified'/'friendly25'/'fixtures.json').read_text())
    fixtures = pd.DataFrame(manifest['fixtures'])
    fixtures['kickoff'] = pd.to_datetime(fixtures.kickoff, utc=True)
    stats = []
    for day, group in fixtures.groupby(fixtures.kickoff.dt.date):
        directory = args.friendly/'days'/str(day)
        truth = pd.read_pickle(directory/'truth.pkl')
        raw = load_raw(directory/'p3_robust_native.jsonl')
        stats.append(score_stats(truth, raw, engine='p3_robust_native'))
        stats.append(score_stats(truth, load_raw(directory/'empirical_raw.jsonl'), engine='empirical_raw'))
        for name, spec in specs.items():
            stats.append(score_stats(truth, transform(raw, truth, store, group.kickoff.min(), spec), engine=name))
    stats = pd.concat(stats, ignore_index=True)
    stats.to_csv(args.output/'friendly_stat_metrics.csv', index=False)
    summary, per_stat = summarise_stats(stats, fixtures.fixture_id.astype(str))
    per_stat.to_csv(args.output/'friendly_per_stat.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
