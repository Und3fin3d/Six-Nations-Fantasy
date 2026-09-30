"""Raw-stat evaluation of robust P3 variants on the latest 25 cached friendlies.

Friendly-25 extends the frozen Friendly-15 rule (latest completed cache
competition 30 fixtures, excluding fantasy-labelled fixtures) to 25 games as of
the research store date. Both complete teamsheets are scored. All fixtures on
one UTC calendar day share the day's earliest kickoff as their lock, so one
fit serves the day; every engine uses the same locks and history.

Engines: ``empirical_raw`` (raw empirical control), ``p3_robust_native``
(unchanged reference) and ``candidate`` (the variant under test).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd

from model.history import past_matches
from model.unified.contracts import RawPrediction
from model.unified.friendly_cohort import complete_fixture_cohort
from model.unified.friendly_eval import score_stats, select_fixtures, summarise_stats, write_manifest
from model.unified.raw_benchmark.blend import EventWeightedBlend
from model.unified.raw_benchmark.empirical import EmpiricalEventModel
from model.unified.raw_benchmark.features import build_frozen_feature_frames
from model.unified.raw_benchmark.folds import masked_candidates
from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from model.unified.rolling_eval import DATA
from model.unified.v4.context import add_candidate_context, add_training_context, pre_match_context
from research.context_experiment import fit_v4, load_store

MANIFEST = DATA/'unified'/'friendly25'/'fixtures.json'
ASOF = '2026-09-22'


def forbidden_fixtures(store: pd.DataFrame) -> set[str]:
    labels = pd.read_csv(DATA/'model_targets.csv', dtype={'fixture_id': str})
    forbidden = set(labels.loc[labels['official_pts'].notna(), 'fixture_id'])
    forbidden.update(store.loc[pd.to_numeric(store['competition_id_cache']).eq(696), 'fixture_id'].astype(str))
    return forbidden


def save_raw(path: Path, raw: list[RawPrediction]) -> None:
    path.write_text(''.join(json.dumps(p.to_dict(), sort_keys=True)+'\n' for p in raw))


def load_raw(path: Path) -> list[RawPrediction]:
    return [RawPrediction.from_dict(json.loads(line)) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='rolling_eval output with inputs/')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', default='{"context": true}')
    parser.add_argument('--reference', action='store_true', help='also fit empirical_raw and p3_robust_native')
    args = parser.parse_args()
    variant = json.loads(args.variant)
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    store, prepared = load_store(args.base)
    manifest = select_fixtures(store, DATA/'cache', forbidden=forbidden_fixtures(store), asof=ASOF, count=25)
    write_manifest(MANIFEST, manifest)
    config = json.loads((DATA/'unified'/'p3_hillclimb'/'config.json').read_text())
    context_table = pre_match_context(store)[0] if variant.get('context') else None
    fixtures = pd.DataFrame(manifest['fixtures'])
    fixtures['kickoff'] = pd.to_datetime(fixtures.kickoff, utc=True)
    fixtures['day'] = fixtures.kickoff.dt.date
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'variant.json').write_text(json.dumps(variant, indent=2, sort_keys=True)+'\n')
    metrics = []
    for day, group in fixtures.groupby('day', sort=True):
        start = time.monotonic()
        cutoff = group.kickoff.min()
        truth = pd.concat([complete_fixture_cohort(store, f, DATA/'cache') for f in group.fixture_id],
                          ignore_index=True)
        candidates = masked_candidates(truth.drop(columns=['team_score', 'opp_score'], errors='ignore'))
        train = past_matches(store, cutoff).sort_values(['date', 'fixture_id', 'team', 'player_id'])
        if set(train.fixture_id.astype(str)) & set(candidates.fixture_id.astype(str)):
            raise ValueError('current fixture leaked into history')
        features, candidate_features = build_frozen_feature_frames(
            train, candidates, v4=True, prepared_train=prepared.loc[train.index])
        directory = args.output/'days'/str(day)
        directory.mkdir(parents=True, exist_ok=True)
        truth.to_pickle(directory/'truth.pkl')
        robust = RobustEmpiricalEventModel(asof=cutoff).fit(train)
        engines = {}
        if args.reference:
            engines['empirical_raw'] = EmpiricalEventModel(asof=cutoff).fit(train).predict_frame(candidate_features)
            reference = fit_v4(features, {})
            engines['p3_robust_native'] = EventWeightedBlend(
                robust, reference, weight_v4=config['default_weight_v4'],
                event_weights_v4=config['event_weights_v4']).predict_frame(candidate_features)
        if variant.get('context'):
            features = add_training_context(features, context_table)
            candidate_features = add_candidate_context(candidate_features, pre_match_context(train)[1])
        tree = fit_v4(features, variant)
        engines['candidate'] = EventWeightedBlend(
            robust, tree, weight_v4=config['default_weight_v4'],
            event_weights_v4=config['event_weights_v4']).predict_frame(candidate_features)
        for engine, raw in engines.items():
            save_raw(directory/f'{engine}.jsonl', raw)
            metrics.append(score_stats(truth, raw, engine=engine))
        pd.concat(metrics, ignore_index=True).to_csv(args.output/'stat_metrics.csv', index=False)
        print(f'{day}: {len(group)} fixtures, {len(truth)} players, {time.monotonic()-start:.1f}s', flush=True)
    stats = pd.concat(metrics, ignore_index=True)
    summary, per_stat = summarise_stats(stats, fixtures.fixture_id.astype(str))
    summary.to_csv(args.output/'summary.csv', index=False)
    per_stat.to_csv(args.output/'per_stat.csv', index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
