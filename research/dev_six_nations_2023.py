"""Six Nations 2023 development slates for choosing decision-level settings.

Rounds 1-4 of 2023 carry official fantasy labels (round 5 has none) and
precede every evaluation season (Six Nations 2025/2026, NCR 2026 and the
Friendly-25 fixtures). Each round refits the empirical fantasy baseline and
robust P3 on strictly earlier history, exactly as ``model.unified.rolling_eval``
does for the official slates. 2023 labels follow an older rubric, so these
slates are used only to choose settings, never as evaluation results.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from model.empirical_unified import project_candidates
from model.history import past_matches
from model.unified.raw_benchmark.blend import EventWeightedBlend
from model.unified.raw_benchmark.features import build_frozen_feature_frames
from model.unified.raw_benchmark.folds import masked_candidates
from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from model.unified.rolling_eval import (DATA, KEY, POS, Slate, evaluate, expected_points,
                                        validate_six_nations_pool)
from research.context_experiment import fit_v4, load_store

SEASON = 2023


def development_slates(store: pd.DataFrame) -> list[Slate]:
    labels = pd.read_csv(DATA/'model_targets.csv', dtype={'fixture_id': str, 'player_id': str})
    labels = labels[labels.season.eq(SEASON)][KEY+['official_pts', 'canonical_pos']]
    six = store[store.competition_id_cache.eq(1266) & store.calendar_year.eq(SEASON)]
    joined = six.merge(labels, on=KEY, how='left', validate='one_to_one')
    # Substitutes without a prior start fall back to the fantasy catalogue
    # position, which is published before the lock.
    unknown = joined.position.eq('Unknown') & joined.canonical_pos.isin(list(POS))
    joined.loc[unknown, 'position'] = joined.loc[unknown, 'canonical_pos']
    joined.loc[unknown, 'is_forward'] = joined.loc[unknown, 'position'].isin(
        ['Prop', 'Hooker', 'Second-row', 'Back-row'])
    joined = joined.drop(columns='canonical_pos')
    slates = []
    for round_no, rows in joined.groupby('round'):
        if rows.official_pts.notna().sum() == 0:
            continue
        rows = rows.sort_values(KEY).reset_index(drop=True)
        validate_six_nations_pool(rows, f'{SEASON}/round {round_no}')
        ids = np.arange(1, len(rows)+1)
        pool = pd.DataFrame({'id': ids, 'name': rows.player_name, 'team': rows.team,
                             'pos': rows.position.map(POS), 'hemi': 1, 'value': 0.0,
                             'status': np.where(rows.started, 'P', 'B')})
        actual = rows.official_pts.to_numpy(float)
        team_actuals = {str(i): (float(p), float(m), str(s)) for i, p, m, s in zip(ids, actual, rows.minutes, pool.status)}
        candidates = masked_candidates(rows.drop(columns=['official_pts', 'team_score', 'opp_score'], errors='ignore'))
        candidates['label_row_id'] = ids
        slates.append(Slate(f'six_nations_{SEASON}_r{int(round_no)}', 'six_nations', SEASON, int(round_no),
                            pd.to_datetime(rows.match_at, utc=True).min(), candidates, pool, actual,
                            team_actuals, False, 'development only; older rubric; no prices'))
    return slates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='rolling_eval output with inputs/')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    store, prepared = load_store(args.base)
    wr = pd.read_csv(DATA/'wr_rankings.csv', parse_dates=['snapshot_date'])
    config = json.loads((DATA/'unified'/'p3_hillclimb'/'config.json').read_text())
    results = []
    for slate in development_slates(store):
        start = time.monotonic()
        train = past_matches(store, slate.cutoff).sort_values(['date', 'fixture_id', 'team', 'player_id'])
        if set(train.fixture_id.astype(str)) & set(slate.candidates.fixture_id.astype(str)):
            raise ValueError(f'{slate.name}: evaluation fixture entered training')
        features, candidates = build_frozen_feature_frames(
            train, slate.candidates, v4=True, prepared_train=prepared.loc[train.index])
        baseline = project_candidates(slate.candidates, train, slate.competition, wr, asof=slate.cutoff)
        slate.baseline = baseline.set_index('label_row_id').reindex(slate.candidates.label_row_id).predicted_points.to_numpy(float)
        results.append(evaluate(slate, 'empirical_baseline', slate.baseline, args.output))
        robust = EventWeightedBlend(RobustEmpiricalEventModel(asof=slate.cutoff).fit(train), fit_v4(features, {}),
                                    weight_v4=config['default_weight_v4'], event_weights_v4=config['event_weights_v4'])
        raw = robust.predict_frame(candidates)
        directory = args.output/'models'/slate.name
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'p3_robust_native.jsonl').write_text(''.join(json.dumps(p.to_dict())+'\n' for p in raw))
        results.append(evaluate(slate, 'p3_robust_native', expected_points(raw, slate.competition), args.output))
        metrics = pd.DataFrame(results)
        metrics.to_csv(args.output/'metrics.csv', index=False)
        print(metrics[metrics.slate.eq(slate.name)][['engine', 'mae', 'team_points']].to_string(index=False), flush=True)
        print(f'{slate.name} finished in {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
