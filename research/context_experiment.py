"""Refit only the v4 trees of robust P3 on the 13 official slates.

Reuses a completed ``model.unified.rolling_eval --native-categories`` output:
its store, prior-only training features, empirical baseline forecasts and the
fitted robust empirical component. Only the tree component changes, so any
difference is attributable to the variant. Cache-only; no API requests.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.raw_benchmark.blend import EventWeightedBlend
from model.unified.raw_benchmark.config import EXTENDED_EVENTS, STABLE_EVENTS
from model.unified.raw_benchmark.features import build_frozen_feature_frames
from model.unified.rolling_eval import (DATA, evaluate, expected_points, official_slates,
                                        season_summary)
from model.unified.v4.context import add_candidate_context, add_training_context, pre_match_context
from model.unified.v4.gbdt import V4GBDT

KEY = ['fixture_id', 'player_id', 'team']


def load_store(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    store = pd.read_csv(base/'inputs'/'player_match.csv', low_memory=False,
                        dtype={key: str for key in KEY}, parse_dates=['date', 'match_at'])
    prepared = pd.read_pickle(base/'inputs'/'training_features.pkl')
    return store, prepared


def fit_v4(features: pd.DataFrame, variant: dict) -> V4GBDT:
    return V4GBDT(events=(*STABLE_EVENTS, *EXTENDED_EVENTS), weighting='natural', pool_player_id=True,
                  player_effects=True, native_categories=True,
                  n_estimators=int(variant.get('n_estimators', 180)),
                  num_leaves=int(variant.get('num_leaves', 23)),
                  learning_rate=float(variant.get('learning_rate', .045)),
                  min_child_samples=int(variant.get('min_child_samples', 35)),
                  random_state=int(variant.get('seed', 17))).fit(features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', default='{"context": true}')
    parser.add_argument('--slates', nargs='*')
    args = parser.parse_args()
    variant = json.loads(args.variant)
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'variant.json').write_text(json.dumps(variant, indent=2, sort_keys=True)+'\n')
    store, prepared = load_store(args.base)
    context_table, _ = pre_match_context(store) if variant.get('context') else (None, None)
    config = json.loads((DATA/'unified'/'p3_hillclimb'/'config.json').read_text())
    slates = official_slates(store, ('six_nations', 'ncr'))
    if args.slates:
        slates = [s for s in slates if s.name in args.slates]
    results = []
    for slate in slates:
        start = time.monotonic()
        train = past_matches(store, slate.cutoff).sort_values(['date', 'fixture_id', 'team', 'player_id'])
        features, candidates = build_frozen_feature_frames(
            train, slate.candidates, v4=True, prepared_train=prepared.loc[train.index])
        if variant.get('context'):
            features = add_training_context(features, context_table)
            _, state = pre_match_context(train)
            candidates = add_candidate_context(candidates, state)
        base_dir = args.base/'models'/slate.name
        baseline = pd.read_csv(args.base/slate.name/'empirical_baseline_predictions.csv')
        results.append(evaluate(slate, 'empirical_baseline', baseline.predicted.to_numpy(float), args.output))
        reference = EventWeightedBlend.load(base_dir/'p3_robust_native.pkl')
        v4 = fit_v4(features, variant)
        model = EventWeightedBlend(reference.empirical, v4, weight_v4=config['default_weight_v4'],
                                   event_weights_v4=config['event_weights_v4'])
        raw = model.predict_frame(candidates)
        model_dir = args.output/'models'/slate.name
        model_dir.mkdir(parents=True, exist_ok=True)
        v4_path = model_dir/'v4.pkl'
        v4.save(v4_path)
        (model_dir/'candidate.jsonl').write_text(''.join(json.dumps(p.to_dict())+'\n' for p in raw))
        results.append(evaluate(slate, 'candidate', expected_points(raw, slate.competition), args.output))
        reference_points = pd.read_csv(args.base/slate.name/'p3_robust_native_predictions.csv').predicted
        results.append(evaluate(slate, 'p3_robust_native', reference_points.to_numpy(float), args.output))
        metrics = pd.DataFrame(results)
        metrics.to_csv(args.output/'metrics.csv', index=False)
        print(metrics[metrics.slate.eq(slate.name)][['engine', 'mae', 'team_points']].to_string(index=False), flush=True)
        print(f'{slate.name} finished in {time.monotonic()-start:.1f}s', flush=True)
    summary = season_summary(pd.DataFrame(results))
    summary.to_csv(args.output/'summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
