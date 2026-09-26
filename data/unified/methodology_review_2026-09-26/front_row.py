import hashlib
import json
import sys

import numpy as np
import pandas as pd

from model.ncr_eval import load_actuals, team_points
from model.ncr_project import optimise
from interpolate import pool_for_slate, verify_archived_squad

ENGINES = ('p3_robust_native', 'p3_rolling_native', 'p3_weighted_native')
KEY = ['fixture_id', 'player_id', 'team']


def forecast_delta(path, candidates):
    with open(path) as stream:
        raw = [json.loads(line) for line in stream]
    keys = pd.DataFrame([{key: row[key] for key in KEY} for row in raw])
    if not keys.equals(candidates[KEY].astype(str)):
        raise ValueError(f'Raw forecast identity/order differs: {path}')
    return np.array([
        0.0 if row['position'] in ('Prop', 'Hooker') else 2 * row['events']['scrums_won']['mean']
        for row in raw
    ])


def select(pool, kwargs, points):
    pool = pool.assign(starter_exp=points, supersub_exp=points * np.where(pool.status.eq('B'), 3.0, 0.5))
    return optimise(pool, **kwargs)[0]


def result_row(slate, engine, variant, points, pool, squad, actuals):
    roles = {role: squad.loc[squad[column]].iloc[0] for role, column in [('captain', 'is_capt'), ('super_sub', 'is_sub')]}
    row = dict(slate=slate, engine=engine, variant=variant, n=len(pool),
               mae=float(np.abs(points - pool.actual).mean()), team_points=team_points(squad, actuals),
               cost=float(squad.value.sum()))
    for role, player in roles.items():
        row[f'{role}_id'], row[f'{role}_name'] = int(player.id), player['name']
    return row


def run_slate(repo, evidence, output, slate, candidates):
    pool, kwargs = pool_for_slate(repo, slate, candidates)
    directory = f'{evidence}/comparison-slate-{slate}/results'
    feed = f'{repo}/data/ncr/feeds/players_gw{slate[-1]}.json'
    actuals = load_actuals(feed)
    rows, files, deltas = [], [feed], []
    for engine in ENGINES:
        prediction_path = f'{directory}/{slate}/{engine}_predictions.csv'
        raw_path = f'{directory}/models/{slate}/{engine}.jsonl'
        archive = f'{directory}/{slate}/{engine}_squad.csv'
        predictions = pd.read_csv(prediction_path)
        if not predictions.id.equals(pool.id) or not np.allclose(predictions.actual, pool.actual):
            raise ValueError(f'Forecast IDs or outcomes differ: {prediction_path}')
        delta = forecast_delta(raw_path, candidates)
        deltas.append(candidates.assign(engine=engine, original=predictions.predicted,
                                        removed_scrum_points=delta, corrected=predictions.predicted-delta))
        original = select(pool, kwargs, predictions.predicted)
        verify_archived_squad(original, archive)
        variants = [('archived_scoring', predictions.predicted, original),
                    ('front_row_only', predictions.predicted-delta, select(pool, kwargs, predictions.predicted-delta))]
        for variant, points, squad in variants:
            row = result_row(slate, engine, variant, points, pool, squad, actuals)
            row['common_players'] = len(set(original.id) & set(squad.id))
            row['players_entered'] = '; '.join(squad.loc[~squad.id.isin(original.id), 'name'])
            row['players_removed'] = '; '.join(original.loc[~original.id.isin(squad.id), 'name'])
            rows.append(row)
            squad.to_csv(f'{output}/{slate}_{engine}_{variant}_squad.csv', index=False)
        files.extend([prediction_path, raw_path, archive])
    return rows, files, deltas


def main(repo, base):
    output, evidence = f'{base}/scoring', f'{base}/replay'
    candidate_path = f'{base}/squad/candidate_predictions.csv'
    candidates = pd.read_csv(candidate_path, dtype={key: str for key in KEY})
    candidates = candidates[candidates.engine.eq('empirical_baseline') & candidates.slate.str.startswith('ncr_')]
    rows, deltas, files = [], [], [candidate_path, __file__, f'{base}/squad/interpolate.py']
    for slate, frame in candidates.groupby('slate'):
        results, sources, changes = run_slate(repo, evidence, output, slate, frame.sort_values('id').reset_index(drop=True))
        rows.extend(results)
        files.extend(sources)
        deltas.extend(changes)
    result = pd.DataFrame(rows)
    result.to_csv(f'{output}/front_row_rounds.csv', index=False)
    pd.concat(deltas, ignore_index=True).to_csv(f'{output}/front_row_prediction_changes.csv', index=False)
    summary = result.groupby(['engine', 'variant']).agg(mae=('mae', 'mean'), team_points=('team_points', 'sum')).reset_index()
    summary.to_csv(f'{output}/front_row_summary.csv', index=False)
    files.extend(f'{repo}/{path}' for path in ['model/ncr_project.py', 'model/ncr_eval.py', 'data/ncr/ncr_gw1_projections.csv', 'data/ncr/ncr_gw2_projections.csv'])
    hashes = {path: hashlib.sha256(open(path, 'rb').read()).hexdigest() for path in files}
    manifest = dict(source_commit='93f9670f38ed1699c0029192ac38227392b329fb',
                    interpretation='Retrospective NCR scoring-contract diagnostic; current official application rule confirms front-row allocation; July publication not independently archived; no refits.',
                    change='Remove 2*E[scrums_won] only for non-Prop/Hooker P3 predictions; preserve all other predictions, pools, prices, constraints and official outcomes.',
                    source_sha256=hashes)
    with open(f'{output}/front_row_manifest.json', 'w') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print(summary.to_string(index=False))
    print(result[['slate', 'engine', 'variant', 'team_points', 'mae', 'captain_name', 'super_sub_name', 'common_players']].to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
