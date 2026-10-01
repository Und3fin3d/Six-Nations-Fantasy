"""Development-only probe of v4 tree capacity on archived raw blocks.

For each named block of the PR #24 raw comparison, refit only the v4 trees at
the block's frozen cutoff and blend them with the archived robust empirical
component means (recovered exactly from the archived forecasts) using the
fixed P3 event weights. Reports fantasy-point MAE/MSE under core Six Nations
and NCR rubrics computed from API events, and the count-stat MAE used by the
friendly benchmark. Intended for pre-2025 blocks only; cache-only.
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
from model.unified.friendly_eval import COUNT_TARGETS
from model.unified.raw_benchmark.features import build_frozen_feature_frames
from model.unified.raw_benchmark.folds import masked_candidates
from model.unified.rolling_eval import DATA
from research.context_experiment import fit_v4, load_store

KEY = ['fixture_id', 'player_id', 'team']
SIX = {'try_assists': 4, 'conversion_goals': 2, 'penalty_goals': 3, 'drop_goals_converted': 4,
       'defenders_beaten': 2, 'offload': 2, 'tackles': 1, 'penalties_conceded': -1,
       'yellow_cards': -5, 'red_cards': -8, 'metres': 0.1}
NCR = {'try_assists': 5, 'conversion_goals': 2, 'missed_conversion_goals': -1, 'penalty_goals': 3,
       'missed_penalty_goals': -1, 'drop_goals_converted': 5, 'defenders_beaten': 2, 'offload': 2,
       'clean_breaks': 3, 'tackles': 1, 'missed_tackles': -1, 'turnovers_conceded': -1,
       'penalties_conceded': -1, 'yellow_cards': -5, 'red_cards': -10}


def means(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        p = json.loads(line)
        row = {k: p[k] for k in KEY} | {e: d['mean'] for e, d in p['events'].items()}
        row['minutes'] = p['minutes']['mean']
        rows.append(row)
    return pd.DataFrame(rows).set_index(KEY)


def fantasy(values: dict, forward: np.ndarray, rubric: dict) -> np.ndarray:
    tries = (np.where(forward, 15, 10) if rubric is SIX else 12) * values['tries']
    return tries + sum(weight * values[e] for e, weight in rubric.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True, help='unpacked raw comparison evidence')
    parser.add_argument('--blocks', nargs='+', required=True)
    parser.add_argument('--variants', nargs='+', required=True, help='JSON variant per entry')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    store, prepared = load_store(args.base)
    config = json.loads((DATA/'unified'/'p3_hillclimb'/'config.json').read_text())
    weight = lambda e: config['event_weights_v4'].get(e, config['default_weight_v4'])
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for block in args.blocks:
        if not block.startswith(('six_nations_202', 'autumn', 'summer', 'rugby', 'pacific')) or '2025' in block or '2026' in block:
            raise ValueError(f'{block}: development blocks must precede 2025')
        results = args.archive/f'comparison-raw-{block}'/'results'
        cutoff = pd.Timestamp(json.loads((results/'run_manifest.json').read_text())['cutoff'])
        roles = pd.read_csv(results/'candidate_roles.csv', dtype={k: str for k in KEY})
        empirical, rolling, robust = (means(results/f'{n}.jsonl') for n in
                                      ('empirical_event', 'p3_rolling_native', 'p3_robust_native'))
        events = [e for e in robust.columns if e != 'minutes']
        archived_v4 = 2*rolling[events] - empirical[events]
        w = pd.Series({e: weight(e) for e in events})
        robust_empirical = (robust[events] - archived_v4*w)/(1 - w)
        truth = store.merge(roles[KEY], on=KEY)
        truth = truth.drop(columns=['position', 'is_forward']).merge(
            roles[KEY+['position', 'is_forward']], on=KEY)
        truth = truth.set_index(KEY).loc[robust.index].reset_index()
        candidates = masked_candidates(truth.drop(columns=['team_score', 'opp_score'], errors='ignore'))
        train = past_matches(store, cutoff).sort_values(['date', 'fixture_id', 'team', 'player_id'])
        features, candidate_features = build_frozen_feature_frames(
            train, candidates, v4=True, prepared_train=prepared.loc[train.index])
        forward = truth.is_forward.astype(bool).to_numpy()
        actual = {e: pd.to_numeric(truth[e], errors='coerce').to_numpy(float) for e in events}
        complete = np.ones(len(truth), bool)
        for e in set(SIX) | set(NCR) | {'tries'}:
            complete &= truth[f'available__{e}'].fillna(False).astype(bool).to_numpy() & np.isfinite(actual[e])
        for spec in args.variants:
            variant = json.loads(spec)
            start = time.monotonic()
            if variant.get('archived'):
                tree = archived_v4
            else:
                raw = fit_v4(features, variant).predict_frame(candidate_features)
                tree = pd.DataFrame([{e: p.events[e].mean for e in events if e in p.events} for p in raw],
                                    index=robust.index)
            blend = {e: (w[e]*tree[e] + (1 - w[e])*robust_empirical[e]).to_numpy() for e in events}
            row = {'block': block, 'variant': spec, 'seconds': time.monotonic() - start}
            for name, rubric in (('six', SIX), ('ncr', NCR)):
                error = (fantasy(blend, forward, rubric) - fantasy(actual, forward, rubric))[complete]
                row[f'{name}_mae'], row[f'{name}_mse'] = np.abs(error).mean(), (error**2).mean()
            stat = []
            for e in COUNT_TARGETS:
                mask = truth[f'available__{e}'].fillna(False).astype(bool).to_numpy() & np.isfinite(actual[e])
                stat.append(np.abs(blend[e][mask] - actual[e][mask]).mean())
            row['count_mae'] = float(np.mean(stat))
            mask = truth['available__metres'].fillna(False).astype(bool).to_numpy() & np.isfinite(actual['metres'])
            row['metres_mae'] = float(np.abs(blend['metres'][mask] - actual['metres'][mask]).mean())
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output/'probe.csv', index=False)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == '__main__':
    main()
