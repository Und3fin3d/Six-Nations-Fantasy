import json
import os
import sys

import numpy as np
import pandas as pd

from model.ncr_project import optimise

from analyse import ENGINES, components

POSITION = {'Prop': 'Prop', 'Hooker': 'Hooker', 'Second-row': 'Lock', 'Back-row': 'Loose Forward', 'Scrum-half': 'Scrum Half', 'Fly-half': 'Fly Half', 'Centre': 'Centre', 'Back-three': 'Back Three'}


def pool_for_slate(repo, slate, candidates):
    pool = candidates[['id', 'player_name', 'team', 'position', 'status', 'actual']].rename(columns={'player_name': 'name'}).copy()
    pool['pos'] = pool.position.map(POSITION)
    if slate.startswith('six_nations'):
        pool['value'], pool['hemi'] = 0.0, 1
        return pool, dict(budget=np.inf, max_nation=4, max_hemi=None)
    gw = int(slate[-1])
    with open(f'{repo}/data/ncr/feeds/players_gw{gw}.json') as stream:
        feed = pd.DataFrame(json.load(stream)['Data']['Value']['Players'])
    feed['id'] = pd.to_numeric(feed.id).astype(int)
    feed = feed.set_index('id')
    values = pd.to_numeric(feed.value)
    if gw < 3:
        saved = pd.read_csv(f'{repo}/data/ncr/ncr_gw{gw}_projections.csv').set_index('id')
        values = pd.to_numeric(saved.value).reindex(feed.index).fillna(values)
    pool['value'], pool['hemi'] = pool.id.map(values), pool.id.map(pd.to_numeric(feed.hemisphere))
    return pool, {}


def run_slate(repo, base, slate, candidates):
    baseline = candidates[candidates.engine.eq(ENGINES[0])].sort_values('id').reset_index(drop=True)
    robust = candidates[candidates.engine.eq(ENGINES[1])].sort_values('id').reset_index(drop=True)
    pool, kwargs = pool_for_slate(repo, slate, baseline)
    previous_ids, rows, members = set(), [], []
    for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pool['starter_exp'] = (1 - weight) * baseline.predicted + weight * robust.predicted
        pool['supersub_exp'] = pool.starter_exp * np.where(pool.status.eq('B'), 3.0, 0.5)
        squad, _, result = optimise(pool, **kwargs)
        row = dict(slate=slate, robust_weight=weight, **components(squad))
        row.update(predicted_total=components(squad, 'starter_exp')['total'], unknown_selected=int(squad.actual.isna().sum()))
        row['captain'] = squad.loc[squad.is_capt, 'name'].iloc[0]
        row['super_sub'] = squad.loc[squad.is_sub, 'name'].iloc[0]
        row['common_with_previous'] = len(set(squad.id) & previous_ids) if previous_ids else np.nan
        previous_ids = set(squad.id)
        rows.append(row)
        members.extend(squad.assign(slate=slate, robust_weight=weight).to_dict('records'))
        verify_endpoint(base, slate, weight, squad)
    return rows, members


def verify_endpoint(base, slate, weight, squad):
    if weight not in [0.0, 1.0]:
        return
    engine = ENGINES[int(weight)]
    path = f'{base}/evidence/comparison-slate-{slate}/results/{slate}/{engine}_squad.csv'
    verify_archived_squad(squad, path)


def verify_archived_squad(squad, path):
    archived = pd.read_csv(path).sort_values('id').reset_index(drop=True)
    actual = squad.sort_values('id').reset_index(drop=True)
    columns = ['id', 'is_sub', 'is_capt']
    if not archived[columns].equals(actual[columns]):
        raise ValueError(f'Original selection differs from archive: {path}')
    if not np.allclose(archived[['starter_exp', 'supersub_exp', 'value', 'hemi']], actual[['starter_exp', 'supersub_exp', 'value', 'hemi']]):
        raise ValueError(f'Original pool differs from archive: {path}')


def main(repo, base):
    candidates = pd.read_csv(f'{base}/candidate_predictions.csv')
    rows, members = [], []
    for slate, frame in candidates.groupby('slate'):
        results, selected = run_slate(repo, base, slate, frame)
        rows.extend(results)
        members.extend(selected)
        print(slate, flush=True)
    table = pd.DataFrame(rows)
    split = table.slate.str.rsplit('_', n=2, expand=True)
    table['competition'], table['season'] = split[0], split[1]
    table.to_csv(f'{base}/interpolation_rounds.csv', index=False)
    pd.DataFrame(members).to_csv(f'{base}/interpolation_members.csv', index=False)
    totals = table.groupby(['competition', 'season', 'robust_weight']).agg(total=('total', lambda x: x.sum(min_count=len(x))), scored_rounds=('total', 'count'), unknown_selected=('unknown_selected', 'sum')).reset_index()
    totals.to_csv(f'{base}/interpolation_seasons.csv', index=False)
    print(totals.to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
