import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd


KEY = ['slate', 'fixture_id', 'player_id', 'team']
PAIRS = {
    'role_calibration': 'p3_robust_native',
    'kicking_role_gate': 'p3_robust_native',
    'equal_incumbent': 'incumbent',
    'regularised_incumbent': 'incumbent',
    'equal_empirical_baseline': 'empirical_baseline',
    'regularised_empirical_baseline': 'empirical_baseline',
}


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def compare(args):
    files = {n: f'{args.run}/{n}.csv' for n in ('forecasts', 'squads', 'metrics')}
    frames = {n: pd.read_csv(p, dtype={'fixture_id': str, 'player_id': str}) for n, p in files.items()}
    f, s, m = [frames[n] for n in ('forecasts', 'squads', 'metrics')]
    os.makedirs(args.output, exist_ok=False)
    rows, strata = [], []
    for candidate, comparator in PAIRS.items():
        a = f[f.engine.eq(candidate)].drop(columns='engine')
        b = f[f.engine.eq(comparator)].drop(columns='engine')
        pair = a.merge(b, on=KEY, how='outer', suffixes=('_candidate', '_comparator'), validate='one_to_one', indicator=True)
        if not pair._merge.eq('both').all() or not np.allclose(pair.actual_candidate, pair.actual_comparator, equal_nan=True):
            raise ValueError('Unequal historical forecast support')
        for slate, group in pair.groupby('slate'):
            changed = np.abs(group.predicted_candidate - group.predicted_comparator)
            cs = s[s.slate.eq(slate) & s.engine.eq(candidate)]
            bs = s[s.slate.eq(slate) & s.engine.eq(comparator)]
            cm = m[m.slate.eq(slate) & m.engine.eq(candidate)].iloc[0]
            bm = m[m.slate.eq(slate) & m.engine.eq(comparator)].iloc[0]
            row = dict(slate=slate, candidate=candidate, comparator=comparator, n=len(group),
                       changed_forecasts=int((changed > 1e-8).sum()), mean_absolute_change=float(changed.mean()),
                       maximum_absolute_change=float(changed.max()),
                       changed_xv_players=len(set(cs.loc[~cs.is_sub, 'id']) - set(bs.loc[~bs.is_sub, 'id'])),
                       changed_captain=bool(set(cs.loc[cs.is_capt, 'id']) != set(bs.loc[bs.is_capt, 'id'])),
                       changed_sub=bool(set(cs.loc[cs.is_sub, 'id']) != set(bs.loc[bs.is_sub, 'id'])))
            row.update({f'delta_{col}': cm[col] - bm[col] for col in ('mae', 'total', 'xv', 'captain_extra', 'super_sub')})
            rows.append(row)
            for (position, status), g in group.groupby(['position_candidate', 'status_candidate']):
                left, right = g.predicted_candidate.to_numpy(), g.predicted_comparator.to_numpy()
                i, j = np.triu_indices(len(g), 1)
                reversals = ((left[i] - left[j]) * (right[i] - right[j]) < -1e-10).sum()
                valid = g.actual_candidate.notna()
                strata.append(dict(slate=slate, candidate=candidate, comparator=comparator, position=position,
                                   status=status, pairs=len(i), ranking_reversals=int(reversals),
                                   candidate_bias=float((g.loc[valid, 'predicted_candidate'] - g.loc[valid, 'actual_candidate']).mean()),
                                   comparator_bias=float((g.loc[valid, 'predicted_comparator'] - g.loc[valid, 'actual_candidate']).mean())))
    table = pd.DataFrame(rows)
    table.to_csv(f'{args.output}/round_changes.csv', index=False)
    pd.DataFrame(strata).to_csv(f'{args.output}/within_role_rank_changes.csv', index=False)
    summary = table.groupby(['candidate', 'comparator'], as_index=False).agg(
        rounds=('slate', 'nunique'), changed_forecasts=('changed_forecasts', 'sum'),
        maximum_absolute_change=('maximum_absolute_change', 'max'), changed_xv_players=('changed_xv_players', 'sum'),
        changed_captains=('changed_captain', 'sum'), changed_subs=('changed_sub', 'sum'),
        delta_xv=('delta_xv', 'sum'), delta_captain=('delta_captain_extra', 'sum'), delta_sub=('delta_super_sub', 'sum'),
        delta_total=('delta_total', lambda values: values.sum(min_count=len(values))))
    summary.to_csv(f'{args.output}/change_summary.csv', index=False)
    with open(f'{args.output}/manifest.json', 'w') as handle:
        json.dump(dict(input_hashes={n: digest(p) for n, p in files.items()},
                       source_sha256=digest(__file__), models_fitted=0, new_candidates_scored=0,
                       interpretation='Diagnostic recomputation of already completed failed methods; no untouched validation'), handle, indent=2)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    compare(parser.parse_args())
