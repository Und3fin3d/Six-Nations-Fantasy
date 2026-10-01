"""Score the protocol's candidates C1-C3 on the 2023 development slates and all four benchmarks.

See ``research/HILLCLIMB_PROTOCOL_2026-10-01.md``. Inputs are the saved robust
P3 forecasts and empirical fantasy baseline forecasts from the reference runs;
no model is refitted. The blend weight is chosen on Six Nations 2023 only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.friendly_eval import score_stats, summarise_stats
from model.unified.rolling_eval import evaluate, expected_points, season_summary
from model.unified.v4.context import add_candidate_context, pre_match_context, team_strength_scale
from research.context_experiment import load_store
from research.dev_six_nations_2023 import development_slates
from research.friendly_experiment import load_raw
from research.reweight_eval import cached_slates

WEIGHTS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TIE = 0.005


def calibrated(raw, frame, state, beta):
    edges = add_candidate_context(frame, state)['ctx__elo_edge']
    return [team_strength_scale(p, float(edge), beta) for p, edge in zip(raw, edges)]


def choose_weight(dev: pd.DataFrame, engine: str) -> float:
    """Lowest mean 2023 round MAE; ties within TIE go to the larger robust weight."""
    mae = {w: dev[dev.engine.eq(f'{engine}_w{w}')].mae.mean() for w in WEIGHTS}
    best = min(mae.values())
    return max(w for w, value in mae.items() if value <= best + TIE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True, help='dev_six_nations_2023 output')
    parser.add_argument('--friendly', type=Path, required=True, help='friendly_experiment output with reference forecasts')
    parser.add_argument('--beta', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    beta = json.loads(args.beta.read_text())
    store, _ = load_store(args.base)
    forecasts = {}

    def points(slate, raw_dir):
        raw = load_raw(raw_dir/slate.name/'p3_robust_native.jsonl')
        state = pre_match_context(past_matches(store, slate.cutoff))[1]
        return (expected_points(raw, slate.competition),
                expected_points(calibrated(raw, slate.candidates, state, beta), slate.competition))

    dev_rows = []
    for slate in development_slates(store):
        slate.baseline = pd.read_csv(args.dev/slate.name/'empirical_baseline_predictions.csv').predicted.to_numpy(float)
        robust, team = points(slate, args.dev/'models')
        for name, values in (('robust', robust), ('team', team)):
            for w in WEIGHTS:
                dev_rows.append(evaluate(slate, f'{name}_w{w}', w*values + (1-w)*slate.baseline, args.output/'dev'))
    dev = pd.DataFrame(dev_rows)
    dev.to_csv(args.output/'dev_metrics.csv', index=False)
    chosen = {'C1': choose_weight(dev, 'robust'), 'C3': choose_weight(dev, 'team')}
    print('2023 development MAE by weight:')
    print(dev.assign(w=dev.engine.str.split('_w').str[1]).pivot_table(
        index='w', columns=dev.engine.str.split('_w').str[0], values='mae').round(4).to_string())
    print('chosen weights:', chosen, flush=True)

    rows = []
    for slate in cached_slates(args.base):
        robust, team = points(slate, args.base/'models')
        candidates = {'p3_robust_native': robust, 'empirical_baseline': slate.baseline,
                      'C1': chosen['C1']*robust + (1-chosen['C1'])*slate.baseline,
                      'C2': team, 'C3': chosen['C3']*team + (1-chosen['C3'])*slate.baseline}
        for name, values in candidates.items():
            rows.append(evaluate(slate, name, values, args.output/'official'))
    official = pd.DataFrame(rows)
    official.to_csv(args.output/'official_metrics.csv', index=False)
    print(season_summary(official)[['competition', 'season', 'engine', 'mae', 'team_points', 'scored_teams']].to_string(index=False))

    stats = []
    manifest = json.loads((Path(__file__).resolve().parents[1]/'data'/'unified'/'friendly25'/'fixtures.json').read_text())
    fixtures = pd.DataFrame(manifest['fixtures'])
    fixtures['kickoff'] = pd.to_datetime(fixtures.kickoff, utc=True)
    for day, group in fixtures.groupby(fixtures.kickoff.dt.date):
        directory = args.friendly/'days'/str(day)
        truth = pd.read_pickle(directory/'truth.pkl')
        raw = load_raw(directory/'p3_robust_native.jsonl')
        state = pre_match_context(past_matches(store, group.kickoff.min()))[1]
        stats.append(score_stats(truth, raw, engine='p3_robust_native'))
        stats.append(score_stats(truth, calibrated(raw, truth, state, beta), engine='C2'))
        stats.append(score_stats(truth, load_raw(directory/'empirical_raw.jsonl'), engine='empirical_raw'))
    stats = pd.concat(stats, ignore_index=True)
    stats.to_csv(args.output/'friendly_stat_metrics.csv', index=False)
    summary, per_stat = summarise_stats(stats, fixtures.fixture_id.astype(str))
    per_stat.to_csv(args.output/'friendly_per_stat.csv', index=False)
    print(summary.to_string(index=False))
    (args.output/'chosen_weights.json').write_text(json.dumps(chosen, indent=2)+'\n')


if __name__ == '__main__':
    main()
