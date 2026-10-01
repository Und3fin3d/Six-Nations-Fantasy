"""Score alternative robust P3 event weights on the official slates without refitting.

Re-blends the components saved by ``research.export_components`` with a given
weight set (same moment-matched blend as ``EventWeightedBlend``), then scores
fantasy MAE and selected squads with the unchanged rolling_eval evaluator.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from model.unified.contracts import RawPrediction
from model.unified.raw_benchmark.blend import _blend_distribution
from model.unified.rolling_eval import DATA, evaluate, expected_points, official_slates, season_summary
from research.context_experiment import load_store


def load_raw(path: Path) -> list[RawPrediction]:
    return [RawPrediction.from_dict(json.loads(line)) for line in path.read_text().splitlines()]


def reblend(empirical: list[RawPrediction], tree: list[RawPrediction], default: float,
            weights: dict[str, float]) -> list[RawPrediction]:
    output = []
    for left, right in zip(empirical, tree):
        if (left.fixture_id, left.player_id, left.team) != (right.fixture_id, right.player_id, right.team):
            raise ValueError('component rows are misaligned')
        events = {}
        for event in sorted(set(left.events) | set(right.events)):
            a, b = right.events.get(event), left.events.get(event)
            events[event] = b if a is None else a if b is None else _blend_distribution(
                a, b, weights.get(event, default), target=event)
        minutes = _blend_distribution(right.minutes, left.minutes, weights.get('minutes', default), target='minutes')
        output.append(RawPrediction(right.fixture_id, right.player_id, right.player_name, right.team,
                                    right.opponent, right.position, right.is_forward, events, minutes))
    return output


def cached_slates(base: Path):
    path = base/'slates.pkl'
    if path.exists():
        return pickle.loads(path.read_bytes())
    store, _ = load_store(base)
    slates = official_slates(store, ('six_nations', 'ncr'))
    for slate in slates:
        slate.baseline = pd.read_csv(base/slate.name/'empirical_baseline_predictions.csv').predicted.to_numpy(float)
    path.write_bytes(pickle.dumps(slates))
    return slates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--weights', type=Path, help='JSON {event: weight_v4}; default is the P3 config')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--components', type=Path, help='directory with <slate>/{empirical,v4}.jsonl')
    args = parser.parse_args()
    config = json.loads((DATA/'unified'/'p3_hillclimb'/'config.json').read_text())
    weights = dict(config['event_weights_v4'])
    if args.weights:
        weights.update(json.loads(args.weights.read_text()))
    components = args.components or args.base/'components'
    results = []
    for slate in cached_slates(args.base):
        directory = components/slate.name
        if not (directory/'v4.jsonl').exists():
            continue
        raw = reblend(load_raw(directory/'empirical.jsonl'), load_raw(directory/'v4.jsonl'),
                      config['default_weight_v4'], weights)
        results.append(evaluate(slate, 'candidate', expected_points(raw, slate.competition), args.output))
        reference = pd.read_csv(args.base/slate.name/'p3_robust_native_predictions.csv').predicted
        results.append(evaluate(slate, 'p3_robust_native', reference.to_numpy(float), args.output))
        results.append(evaluate(slate, 'empirical_baseline', slate.baseline, args.output))
    metrics = pd.DataFrame(results)
    metrics.to_csv(args.output/'metrics.csv', index=False)
    print(season_summary(metrics).to_string(index=False))


if __name__ == '__main__':
    main()
