"""Six Nations 2023-2024 development slates scored under the current fantasy rubric.

These slates precede every evaluation season (Six Nations 2025/2026, NCR 2026
and the Friendly-25 fixtures) and exist only to choose settings.

Labels, in order of preference:

1. 2023 rounds 1-4: the official per-player statistics
   (``data/official_player_match.csv``) rescored with the 2025 rubric. The
   2023 file lacks defenders beaten, offloads and conceded penalties, which are
   taken from the API, and scrums won, which is approximated as the team's
   scrums won times the forward's share of minutes. The same construction
   reproduces 2025 official points with r = 0.99 (exact when every official
   column is present).
2. Otherwise ``target_pts`` from ``data/model_targets.csv`` (API statistics
   under the current rubric plus estimated official-only components; r = 0.97
   against official points in 2025-26).

Players who did not play score zero. Each round refits robust P3 and the
empirical fantasy baseline on strictly earlier history, exactly as
``model.unified.rolling_eval`` does for the official slates, and saves the raw
forecasts plus both P3 components.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from compare_api_official import norm_key
from model.empirical_unified import project_candidates
from model.history import past_matches
from model.unified.raw_benchmark.blend import EventWeightedBlend
from model.unified.raw_benchmark.features import build_frozen_feature_frames
from model.unified.raw_benchmark.folds import masked_candidates
from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from model.unified.rolling_eval import DATA, KEY, POS, Slate, evaluate, expected_points, validate_six_nations_pool
from research.context_experiment import fit_v4, load_store

SEASONS = (2023, 2024)
JOIN = ['season', 'round', 'team', 'key']


def rescored_official(targets: pd.DataFrame) -> pd.Series:
    """Current-rubric points from official 2023 statistics; NaN where unmatched."""
    official = pd.read_csv(DATA/'official_player_match.csv')
    official = official[~official.duplicated(JOIN, keep=False)]
    frame = targets.assign(key=targets.player_name.map(norm_key))
    unique = ~frame.duplicated(JOIN, keep=False)
    joined = frame[unique].reset_index().merge(official, on=JOIN, how='left', validate='one_to_one').set_index('index')
    number = lambda column: pd.to_numeric(joined[column], errors='coerce')
    api = lambda column: pd.to_numeric(joined[f'y_{column}'], errors='coerce').fillna(0)
    forward = joined.is_forward.astype(bool)
    scrums = np.where(forward, joined.team_scrums_won*joined.min_share, 0.0)
    points = (np.where(forward, 15, 10)*number('T') + 4*number('As') + 2*number('C') + 3*number('Pen')
              + 4*number('DG') + 2*api('defenders_beaten') + 2*api('offload') + number('Ta') + 5*number('BS')
              - api('penalties_conceded') - 5*number('YC') - 8*number('RC') + 7*number('50-22') + 7*number('LS')
              + np.floor(number('MC')/10) + 15*number('POTM') + scrums)
    return points.reindex(targets.index)


def current_rubric_labels(season: int) -> pd.DataFrame:
    targets = pd.read_csv(DATA/'model_targets.csv', dtype={'fixture_id': str, 'player_id': str})
    targets = targets[targets.season.eq(season)].copy()
    rescored = rescored_official(targets) if season == 2023 else pd.Series(np.nan, index=targets.index)
    targets['label'] = rescored.fillna(targets.target_pts)
    targets['label_source'] = np.where(rescored.notna(), 'official_rescored', 'target_pts')
    targets.loc[targets.minutes.eq(0), ['label', 'label_source']] = (0.0, 'did_not_play')
    return targets[KEY+['label', 'label_source', 'canonical_pos']]


def development_slates(store: pd.DataFrame, season: int) -> list[Slate]:
    labels = current_rubric_labels(season)
    six = store[store.competition_id_cache.eq(1266) & store.calendar_year.eq(season)]
    joined = six.merge(labels, on=KEY, how='left', validate='one_to_one')
    if joined.label.isna().any():
        raise ValueError(f'{season}: unlabelled teamsheet rows')
    # Substitutes without a prior start fall back to the fantasy catalogue
    # position, which is published before the lock.
    unknown = joined.position.eq('Unknown') & joined.canonical_pos.isin(list(POS))
    joined.loc[unknown, 'position'] = joined.loc[unknown, 'canonical_pos']
    joined.loc[unknown, 'is_forward'] = joined.loc[unknown, 'position'].isin(['Prop', 'Hooker', 'Second-row', 'Back-row'])
    slates = []
    for round_no, rows in joined.drop(columns='canonical_pos').groupby('round'):
        rows = rows.sort_values(KEY).reset_index(drop=True)
        validate_six_nations_pool(rows, f'{season}/round {round_no}')
        ids = np.arange(1, len(rows)+1)
        pool = pd.DataFrame({'id': ids, 'name': rows.player_name, 'team': rows.team, 'pos': rows.position.map(POS),
                             'hemi': 1, 'value': 0.0, 'status': np.where(rows.started, 'P', 'B')})
        actual = rows.label.to_numpy(float)
        team_actuals = {str(i): (float(p), float(m), str(s)) for i, p, m, s in zip(ids, actual, rows.minutes, pool.status)}
        candidates = masked_candidates(rows.drop(columns=['label', 'label_source', 'team_score', 'opp_score'], errors='ignore'))
        candidates['label_row_id'] = ids
        slate = Slate(f'six_nations_{season}_r{int(round_no)}', 'six_nations', season, int(round_no),
                      pd.to_datetime(rows.match_at, utc=True).min(), candidates, pool, actual, team_actuals, False,
                      'development only; current-rubric labels; no prices')
        slate.label_source = rows.label_source.to_numpy()
        slates.append(slate)
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
    args.output.mkdir(parents=True, exist_ok=True)
    slates = [slate for season in SEASONS for slate in development_slates(store, season)]
    results = []
    for slate in slates:
        start = time.monotonic()
        directory = args.output/'models'/slate.name
        if (directory/'p3_robust_native.jsonl').exists():
            slate.baseline = pd.read_csv(args.output/slate.name/'empirical_baseline_predictions.csv').predicted.to_numpy(float)
            continue
        train = past_matches(store, slate.cutoff).sort_values(['date', 'fixture_id', 'team', 'player_id'])
        if set(train.fixture_id.astype(str)) & set(slate.candidates.fixture_id.astype(str)):
            raise ValueError(f'{slate.name}: evaluation fixture entered training')
        features, candidates = build_frozen_feature_frames(
            train, slate.candidates, v4=True, prepared_train=prepared.loc[train.index])
        baseline = project_candidates(slate.candidates, train, slate.competition, wr, asof=slate.cutoff)
        slate.baseline = baseline.set_index('label_row_id').reindex(slate.candidates.label_row_id).predicted_points.to_numpy(float)
        results.append(evaluate(slate, 'empirical_baseline', slate.baseline, args.output))
        empirical = RobustEmpiricalEventModel(asof=slate.cutoff).fit(train)
        tree = fit_v4(features, {})
        robust = EventWeightedBlend(empirical, tree, weight_v4=config['default_weight_v4'],
                                    event_weights_v4=config['event_weights_v4'])
        directory.mkdir(parents=True, exist_ok=True)
        for name, raw in (('p3_robust_native', robust.predict_frame(candidates)),
                          ('empirical', empirical.predict_frame(candidates)), ('v4', tree.predict_frame(candidates))):
            (directory/f'{name}.jsonl').write_text(''.join(json.dumps(p.to_dict())+'\n' for p in raw))
            if name == 'p3_robust_native':
                results.append(evaluate(slate, name, expected_points(raw, slate.competition), args.output))
        metrics = pd.DataFrame(results)
        metrics.to_csv(args.output/'metrics.csv', mode='a', header=not (args.output/'metrics.csv').exists(), index=False)
        results = []
        print(metrics[['slate', 'engine', 'mae', 'team_points']].to_string(index=False), flush=True)
        print(f'{slate.name} finished in {time.monotonic()-start:.1f}s', flush=True)
    (args.output/'slates.pkl').write_bytes(pickle.dumps(slates))


if __name__ == '__main__':
    main()
