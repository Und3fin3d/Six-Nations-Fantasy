import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ENGINES = ['empirical_baseline', 'p3_robust_native']
KEY = ['fixture_id', 'player_id', 'team']


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def statuses(repo, slate, rows):
    if slate.startswith('ncr'):
        gw = int(slate[-1])
        feed = pd.DataFrame(read_json(f'{repo}/data/ncr/feeds/players_gw{gw}.json')['Data']['Value']['Players'])
        feed['id'] = pd.to_numeric(feed.id).astype(int)
        status = feed.set_index('id').player_status
        if gw < 3:
            saved = pd.read_csv(f'{repo}/data/ncr/ncr_gw{gw}_projections.csv').set_index('id')
            status = saved.status
        return rows.id.map(status).fillna('OUT')
    result = {}
    for fixture in rows.fixture_id.unique():
        cache = read_json(f'{repo}/data/cache/match_{fixture}.json')['results']
        for side in ['home', 'away']:
            for player in cache[side]['teamsheet']:
                key = str(fixture), str(player['player_id']), cache['match'][f'{side}_team']
                result[key] = 'B' if player['substitute'] else 'P'
    return pd.Series([result[tuple(row)] for row in rows[KEY].to_numpy()], index=rows.index)


def verify_identity(repo, slate, frame):
    if slate.startswith('six_nations'):
        ordered = frame.sort_values(KEY).reset_index(drop=True)
        if not np.array_equal(ordered.id.to_numpy(), np.arange(1, len(frame) + 1)):
            raise ValueError(f'{slate}: IDs disagree with sorted fixture/player/team keys')
        return
    crosswalk = pd.read_csv(f'{repo}/data/ncr/ncr_player_crosswalk.csv').set_index('fantasy_id')
    api_ids = frame.id.map(crosswalk.api_player_id)
    expected_ids = pd.Series([str(int(api_id)) if pd.notna(api_id) else f'fantasy_{fantasy_id}' for fantasy_id, api_id in zip(frame.id, api_ids)])
    fixtures = pd.read_csv(f'{repo}/data/ncr/ncr_fixtures.csv')
    fixtures = fixtures[pd.to_numeric(fixtures.gameday).eq(int(slate[-1]))]
    fixture_by_team = {team: str(row.match_id) for row in fixtures.itertuples() for team in [row.home, row.away]}
    if not frame.player_id.eq(expected_ids).all() or not frame.fixture_id.eq(frame.team.map(fixture_by_team)).all():
        raise ValueError(f'{slate}: fantasy IDs disagree with crosswalk or fixture keys')


def read_slate(repo, job):
    slate = os.path.basename(job).removeprefix('comparison-slate-')
    directory = f'{job}/results/{slate}'
    raw = [json.loads(line) for line in open(f'{job}/results/models/{slate}/p3_robust_native.jsonl')]
    columns = [*KEY, 'player_name', 'position', 'is_forward']
    metadata = pd.DataFrame([{k: row[k] for k in columns} for row in raw])
    frames, squads = [], {}
    for engine in ENGINES:
        frame = pd.read_csv(f'{directory}/{engine}_predictions.csv')
        if len(frame) != len(metadata) or frame.id.duplicated().any() or metadata.duplicated(KEY).any():
            raise ValueError(f'{slate}: candidate alignment failed')
        frame = pd.concat([frame, metadata], axis=1)
        verify_identity(repo, slate, frame)
        frame['status'] = statuses(repo, slate, frame)
        frame['slate'], frame['engine'] = slate, engine
        frame['competition'], frame['season'] = slate.rsplit('_', 2)[:2]
        squad = pd.read_csv(f'{directory}/{engine}_squad.csv')
        selected = squad.merge(frame, on='id', validate='one_to_one', suffixes=('', '_candidate'))
        if not selected.team.eq(selected.team_candidate).all() or not selected.status.eq(selected.status_candidate).all():
            raise ValueError(f'{slate}: squad metadata alignment failed')
        if not np.allclose(selected.starter_exp, selected.predicted):
            raise ValueError(f'{slate}: squad forecast mismatch')
        frames.append(frame)
        squads[engine] = selected
    if not frames[0].actual.equals(frames[1].actual):
        raise ValueError(f'{slate}: models have different outcomes')
    return pd.concat(frames, ignore_index=True), squads


def components(squad, values='actual'):
    xv = squad.loc[~squad.is_sub, values].sum(min_count=15)
    captain = squad.loc[squad.is_capt, values].sum(min_count=1)
    sub = 3 * squad.loc[squad.is_sub, values].sum(min_count=1)
    return {'xv': xv, 'captain_extra': captain, 'super_sub': sub, 'total': xv + captain + sub}


def squad_row(slate, engine, squad, candidate_frame):
    row = dict(slate=slate, engine=engine, **components(squad))
    for role, mask in [('captain', squad.is_capt), ('super_sub', squad.is_sub)]:
        player = squad.loc[mask].iloc[0]
        row[f'{role}_name'] = player['name']
        row[f'{role}_actual'] = player.actual
        row[f'{role}_predicted'] = player.predicted
    row['predicted_total'] = components(squad, 'predicted')['total']
    row['unknown_selected'] = int(squad.actual.isna().sum())
    row['n_selected'], row['n_xv'] = len(squad), int((~squad.is_sub).sum())
    for forecaster in ENGINES:
        prediction = candidate_frame[candidate_frame.engine.eq(forecaster)].set_index('id').predicted
        copy = squad.assign(cross_prediction=squad.id.map(prediction))
        row[f'objective_under_{forecaster}'] = components(copy, 'cross_prediction')['total']
    return row


def differences(slate, squads):
    baseline, robust = [squads[engine] for engine in ENGINES]
    components_diff = {key: components(robust)[key] - components(baseline)[key] for key in components(robust)}
    base_xv, robust_xv = [set(s.loc[~s.is_sub, 'id']) for s in [baseline, robust]]
    row = dict(slate=slate, **components_diff, common_squad=len(set(baseline.id) & set(robust.id)), common_xv=len(base_xv & robust_xv))
    row['same_captain'] = baseline.loc[baseline.is_capt, 'id'].iloc[0] == robust.loc[robust.is_capt, 'id'].iloc[0]
    row['same_super_sub'] = baseline.loc[baseline.is_sub, 'id'].iloc[0] == robust.loc[robust.is_sub, 'id'].iloc[0]
    return row


def contributions(slate, squads):
    rows = []
    baseline_ids = set(squads[ENGINES[0]].id)
    robust_ids = set(squads[ENGINES[1]].id)
    for engine, squad in squads.items():
        for row in squad.to_dict('records'):
            multiplier = 3 if row['is_sub'] else (2 if row['is_capt'] else 1)
            item = {key: row[key] for key in [*KEY, 'id', 'name', 'position', 'status', 'actual', 'predicted', 'is_sub', 'is_capt']}
            item.update(slate=slate, engine=engine, multiplier=multiplier, contribution=multiplier * row['actual'])
            item['in_both_squads'] = row['id'] in baseline_ids & robust_ids
            rows.append(item)
    return rows


def error_row(frame):
    known = frame.dropna(subset=['actual'])
    error = known.predicted - known.actual
    return dict(n=len(frame), n_labelled=len(known), predicted_mean=known.predicted.mean(), actual_mean=known.actual.mean(), bias=error.mean(), mae=error.abs().mean(), rmse=np.sqrt(np.square(error).mean()), spearman=known.predicted.corr(known.actual, method='spearman'))


def group_errors(frame, group):
    return pd.DataFrame([{**dict(zip(group, keys)), **error_row(part)} for keys, part in frame.groupby(group)])


def calibration_bins(frame):
    rows = []
    for keys, part in frame.groupby(['slate', 'engine']):
        part = part.dropna(subset=['actual']).copy()
        part['bin'] = pd.qcut(part.predicted.rank(method='first'), 5, labels=False) + 1
        for bucket, selected in part.groupby('bin'):
            rows.append(dict(slate=keys[0], engine=keys[1], bin=int(bucket), min_prediction=selected.predicted.min(), max_prediction=selected.predicted.max(), **error_row(selected)))
    return pd.DataFrame(rows)


def squad_swaps(frame, selected):
    keys = ['slate', 'id']
    table = frame[frame.engine.eq(ENGINES[0])].set_index(keys).drop(columns='engine')
    table = table.rename(columns={'predicted': 'baseline_prediction'})
    table['robust_prediction'] = frame[frame.engine.eq(ENGINES[1])].set_index(keys).predicted
    for engine, label in zip(ENGINES, ['baseline', 'robust']):
        members = selected[selected.engine.eq(engine)].set_index(keys)
        table[f'{label}_xv'] = (~members.is_sub).reindex(table.index).fillna(False).astype(int)
        table[f'{label}_captain_extra'] = members.is_capt.reindex(table.index).fillna(False).astype(int)
        table[f'{label}_super_sub'] = 3 * members.is_sub.reindex(table.index).fillna(False).astype(int)
    for role in ['xv', 'captain_extra', 'super_sub']:
        table[f'{role}_weight_difference'] = table[f'robust_{role}'] - table[f'baseline_{role}']
    weights = [f'{role}_weight_difference' for role in ['xv', 'captain_extra', 'super_sub']]
    table = table[table[weights].ne(0).any(axis=1)].copy()
    for role in ['xv', 'captain_extra', 'super_sub']:
        table[f'{role}_difference'] = table[f'{role}_weight_difference'] * table.actual
    table['total_difference'] = table[[f'{role}_difference' for role in ['xv', 'captain_extra', 'super_sub']]].sum(axis=1, min_count=3)
    return table.reset_index()


def verify_totals(repo, squad, selected):
    published = pd.read_csv(f'{repo}/data/unified/complete_comparison/results/official_metrics.csv')
    merged = squad.merge(published, on=['slate', 'engine'], validate='one_to_one')
    error = np.abs(merged.total - merged.team_points).max()
    if len(merged) != 26 or error != 0:
        raise ValueError('Reconstructed totals differ from published evidence')
    return dict(compared_totals=len(merged), max_total_error=error, unknown_selected=int(squad.unknown_selected.sum()), bench_in_xv=int((selected.status.eq('B') & ~selected.is_sub).sum()))


def write_diagnostics(repo, base):
    frames, squad_rows, diffs, selected = [], [], [], []
    for job in sorted(glob.glob(f'{base}/evidence/comparison-slate-*')):
        frame, squads = read_slate(repo, job)
        slate = frame.slate.iloc[0]
        frames.append(frame)
        squad_rows.extend(squad_row(slate, engine, squad, frame) for engine, squad in squads.items())
        diffs.append(differences(slate, squads))
        selected.extend(contributions(slate, squads))
    frame = pd.concat(frames, ignore_index=True)
    squad = pd.DataFrame(squad_rows)
    diff = pd.DataFrame(diffs)
    for table in [squad, diff]:
        split = table.slate.str.rsplit('_', n=2, expand=True)
        table['competition'], table['season'] = split[0], split[1]
    selected = pd.DataFrame(selected)
    verification = verify_totals(repo, squad, selected)
    write_tables(base, frame, squad, diff, selected)
    metadata = dict(repo=repo, commit='93f9670f38ed1699c0029192ac38227392b329fb', python=sys.version, pandas=pd.__version__, numpy=np.__version__)
    metadata['verification'] = verification
    archive = f'{repo}/data/unified/complete_comparison/evidence/official.tar.gz'
    with open(archive, 'rb') as stream:
        metadata['official_archive_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    with open(f'{base}/diagnostic_manifest.json', 'w') as stream:
        json.dump(metadata, stream, indent=2)


def write_tables(base, frame, squad, diff, selected):
    tables = {'candidate_predictions': frame, 'squad_components': squad, 'round_differences': diff, 'selected_players': selected}
    tables['season_differences'] = diff.groupby(['competition', 'season'])[['xv', 'captain_extra', 'super_sub', 'total']].sum().reset_index()
    tables['season_error_by_status'] = group_errors(frame, ['competition', 'season', 'engine', 'status'])
    tables['season_error_by_position'] = group_errors(frame, ['competition', 'season', 'engine', 'position'])
    tables['round_error'] = group_errors(frame, ['competition', 'season', 'slate', 'engine'])
    tables['round_calibration_bins'] = calibration_bins(frame)
    tables['squad_swaps'] = squad_swaps(frame, selected)
    tables['xv_position_differences'] = tables['squad_swaps'].groupby(['slate', 'position']).xv_difference.sum().reset_index()
    for name, table in tables.items():
        table.to_csv(f'{base}/{name}.csv', index=False)
    print(tables['season_differences'].to_string(index=False))
    print(diff.to_string(index=False))


if __name__ == '__main__':
    write_diagnostics(sys.argv[1], sys.argv[2])
