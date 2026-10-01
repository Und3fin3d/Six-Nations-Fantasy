"""Positional-bias and super-sub research: generate status-aware forecasts.

For every evaluation unit (official slate, current-rubric development slate,
archived international block, Friendly-25 day) this refits the robust
empirical component with status-aware rates
(``model.unified.raw_benchmark.status_rates``) at the unit's own lock, on
strictly earlier history, and combines it with the unchanged saved tree
component:

* official slates and development slates have saved tree forecasts, so the
  P3 blend is recomputed exactly (``EventWeightedBlend`` arithmetic);
* archived blocks and Friendly-25 days only keep the blended forecast, so the
  blend mean is shifted by ``(1 - w_event) * (status - plain)`` empirical
  means, with both empirical fits made here at the same lock, and the
  dispersion keeps its variance-to-mean shape.

Cache-only: reads the research store and saved forecasts; no API requests.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.raw_benchmark.folds import masked_candidates
from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from model.unified.raw_benchmark.status_rates import StatusAwareEmpiricalEventModel
from model.unified.rolling_eval import DATA
from research.context_experiment import load_store
from research.reweight_eval import cached_slates, reblend

KEY = ['fixture_id', 'player_id', 'team']


def load_raw(path: Path) -> list[RawPrediction]:
    return [RawPrediction.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines()]


def save_raw(path: Path, raw: list[RawPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(p.to_dict())+'\n' for p in raw))


def _shift(distribution: EventDistribution, mean: float) -> EventDistribution:
    mean = max(float(mean), 0.0)
    old = distribution.mean
    ratio = mean/old if old > 1e-12 else 1.0
    if distribution.family == 'bernoulli':
        return EventDistribution('bernoulli', min(mean, 1.0), 1.0)
    if distribution.family == 'negative_binomial':
        # Same variance/mean shape: var' = ratio * var.
        return EventDistribution('negative_binomial', mean, max(distribution.dispersion*ratio, 0.05))
    return EventDistribution(distribution.family, mean, max(distribution.dispersion*ratio*ratio, 1e-6))


def shift_blend(reference: list[RawPrediction], plain: list[RawPrediction], status: list[RawPrediction],
                default: float, weights: dict[str, float]) -> list[RawPrediction]:
    """Blend-mean shift by the empirical share of each event."""
    output = []
    for ref, old, new in zip(reference, plain, status):
        if (ref.fixture_id, ref.player_id, ref.team) != (old.fixture_id, old.player_id, old.team) or \
                (old.fixture_id, old.player_id, old.team) != (new.fixture_id, new.player_id, new.team):
            raise ValueError('forecast rows are misaligned')
        events = {}
        for event, dist in ref.events.items():
            if event in old.events and event in new.events:
                share = 1.0 - weights.get(event, default)
                events[event] = _shift(dist, dist.mean + share*(new.events[event].mean - old.events[event].mean))
            else:
                events[event] = dist
        output.append(RawPrediction(ref.fixture_id, ref.player_id, ref.player_name, ref.team, ref.opponent,
                                    ref.position, ref.is_forward, events, ref.minutes,
                                    {**ref.metadata, 'status_rates': True}))
    return output


def _fit_predict(train, cutoff, candidates, plain_too: bool):
    status = StatusAwareEmpiricalEventModel(asof=cutoff).fit(train)
    plain = RobustEmpiricalEventModel(asof=cutoff).fit(train) if plain_too else None
    return (status.predict_frame(candidates), plain.predict_frame(candidates) if plain else None,
            status.status_factors)


def generate(runs: Path, output: Path, sets: list[str], fallback: Path | None = None) -> None:
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    config = json.loads((DATA/'unified'/'p3_hillclimb'/'config.json').read_text())
    default, weights = config['default_weight_v4'], config['event_weights_v4']
    store, _ = load_store(runs/'base')
    factors = {}
    units = []
    if 'official' in sets:
        for slate in cached_slates(runs/'base'):
            comp = runs/'base'/'components'/slate.name
            if not (comp/'v4.jsonl').exists() and fallback is not None:
                comp = fallback/slate.name
            units.append(('official', slate.name, slate.cutoff, slate.candidates, runs/'base'/'models'/slate.name/'p3_robust_native.jsonl', comp))
    if 'devcr' in sets:
        for slate in pickle.loads((runs/'devcr'/'slates.pkl').read_bytes()):
            directory = runs/'devcr'/'models'/slate.name
            units.append(('devcr', slate.name, slate.cutoff, slate.candidates, directory/'p3_robust_native.jsonl', directory))
    if 'blocks' in sets:
        for directory in sorted(glob.glob(str(runs.parent/'raw'/'comparison-raw-*'))):
            results = Path(directory)/'results'
            cutoff = pd.Timestamp(json.loads((results/'run_manifest.json').read_text())['cutoff'])
            reference = load_raw(results/'p3_robust_native.jsonl')
            keys = pd.DataFrame([(p.fixture_id, p.player_id, p.team, p.position, p.is_forward) for p in reference],
                                columns=KEY+['position', 'is_forward'])
            rows = keys.merge(store.drop(columns=['position', 'is_forward']), on=KEY, how='left', validate='one_to_one')
            units.append(('blocks', Path(directory).name.removeprefix('comparison-raw-'), cutoff,
                          masked_candidates(rows.drop(columns=['team_score', 'opp_score'], errors='ignore')),
                          results/'p3_robust_native.jsonl', None))
    if 'friendly' in sets:
        manifest = json.loads((DATA/'unified'/'friendly25'/'fixtures.json').read_text())
        fixtures = pd.DataFrame(manifest['fixtures'])
        fixtures['kickoff'] = pd.to_datetime(fixtures.kickoff, utc=True)
        for day, group in fixtures.groupby(fixtures.kickoff.dt.date):
            directory = runs/'f25_ctx'/'days'/str(day)
            truth = pd.read_pickle(directory/'truth.pkl')
            units.append(('friendly', str(day), group.kickoff.min(),
                          masked_candidates(truth.drop(columns=['team_score', 'opp_score'], errors='ignore')),
                          directory/'p3_robust_native.jsonl', None))
    for kind, name, cutoff, candidates, reference_path, components in units:
        path = output/kind/name/'status.jsonl'
        if path.exists():
            continue
        start = time.monotonic()
        train = past_matches(store, cutoff)
        reference = load_raw(reference_path)
        by_key = {(p.fixture_id, p.player_id, p.team): p for p in reference}
        keys = list(zip(candidates.fixture_id.astype(str), candidates.player_id.astype(str), candidates.team.astype(str)))
        reference = [by_key[k] for k in keys]
        status, plain, unit_factors = _fit_predict(train, cutoff, candidates, components is None)
        if components is not None:
            raw = reblend(status, load_raw(components/'v4.jsonl'), default, weights)
        else:
            raw = shift_blend(reference, plain, status, default, weights)
        save_raw(path, raw)
        factors[f'{kind}/{name}'] = {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in unit_factors.items()}
        print(f'{kind}/{name}: {len(raw)} rows, {time.monotonic()-start:.1f}s', flush=True)
        (output/'factors.json').write_text(json.dumps(
            {**(json.loads((output/'factors.json').read_text()) if (output/'factors.json').exists() else {}), **factors},
            indent=1, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True, help='scratch runs directory (base/, devcr/, f25_ctx/; raw/ beside it)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sets', nargs='+', default=['devcr', 'blocks', 'official', 'friendly'],
                        choices=['devcr', 'blocks', 'official', 'friendly'])
    parser.add_argument('--components-fallback', type=Path,
                        help='directory with <slate>/{empirical,v4}.jsonl for slates missing from base/components')
    args = parser.parse_args()
    generate(args.runs, args.output, args.sets, args.components_fallback)


if __name__ == '__main__':
    main()
