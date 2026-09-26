import argparse
import glob
import hashlib
import json
import os
import pickle

import numpy as np
import pandas as pd

from model.unified.raw_benchmark.config import DIRECT_EVENTS
from model.unified.scoring import scorer_for
from research.conditional_duration_inputs import digest, write_json


def numeric(value):
    if value is None or value == '':
        return np.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def read_feed(path):
    players = json.load(open(path))['Data']['Value']['Players']
    rows = []
    for player in players:
        rows.append(dict(id=int(float(player['id'])), feed_points=numeric(player.get('cur_gd_points')),
                         feed_played=numeric(player.get('min_in_game')),
                         feed_status=player.get('player_status') or None,
                         feed_current_round=numeric(player.get('cur_gameday_id')),
                         feed_gameday=numeric(player.get('gameday_id')),
                         feed_has_played=numeric(player.get('has_played'))))
    frame = pd.DataFrame(rows)
    if frame.id.duplicated().any():
        raise ValueError(f'Duplicate official feed IDs: {path}')
    return frame


def observed(row, event):
    return pd.notna(row.get(f'available__{event}')) and bool(row.get(f'available__{event}')) and pd.notna(row.get(event))


def base_pool(unit):
    slate, labels = unit['slate'], unit['raw_labels']
    pool = slate.pool.copy().reset_index(drop=True)
    pool['slate'], pool['competition'], pool['season'], pool['round'] = slate.name, slate.competition, slate.season, slate.round
    pool['official_points'] = slate.actual
    pool['source_key_matched'] = labels.source_key_matched.to_numpy(bool)
    pool['source_fixture_id'] = labels.fixture_id.to_numpy()
    pool['source_player_id'] = labels.player_id.to_numpy()
    pool['source_started'] = labels.started.where(labels.source_key_matched).to_numpy()
    pool['source_minutes'] = labels.minutes.where(labels.available__minutes.fillna(False)).to_numpy(float)
    activity = []
    for _, row in labels.iterrows():
        activity.append(any(observed(row, event) and float(row[event]) > 0 for event in DIRECT_EVENTS))
    pool['source_direct_activity'] = activity
    pool['source_entered_evidence'] = pool.source_started.eq(True) | pool.source_minutes.gt(0) | pool.source_direct_activity
    return pool


def role_outcomes(pool):
    rows = []
    for row in pool.itertuples(index=False):
        y = row.official_points
        conflict = False
        reasons = []
        if row.competition == 'ncr':
            entered = row.feed_played > 0 if np.isfinite(row.feed_played) else None
            starter = True if row.feed_status == 'P' else False if row.feed_status == 'B' else None
            if entered is False and row.source_entered_evidence:
                conflict = True
                reasons.append('feed_unused_conflicts_with_source_activity')
            if pd.notna(row.source_started) and starter is not None and bool(row.source_started) != starter:
                conflict = True
                reasons.append('feed_start_status_conflicts_with_source_teamsheet')
            if np.isfinite(row.feed_current_round) and row.feed_current_round != row.round:
                conflict = True
                reasons.append('official_feed_current_round_mismatch')
            if not np.isfinite(row.feed_points) or not np.isfinite(y) or abs(row.feed_points-y) > 1e-9:
                conflict = True
                reasons.append('official_point_missing_or_parser_disagreement')
        else:
            entered = True if row.source_entered_evidence else None
            starter = bool(row.source_started) if pd.notna(row.source_started) else None
        possible = [0.0, .5, 3.0]
        if starter is True:
            possible = [.5]
        elif starter is False:
            possible = [0.0, 3.0]
        if entered is False:
            possible = [0.0]
        elif entered is True and starter is not None:
            possible = [.5 if starter else 3.0]
        if conflict:
            possible = [0.0, .5, 3.0]
        values = np.array(possible) * y
        known = bool(np.isfinite(y) and (len(possible) == 1 or y == 0))
        rows.append(dict(slate=row.slate, id=row.id, actual_role_supersub_points=float(values[0]) if known else np.nan,
                         possible_min=float(np.min(values)) if np.isfinite(y) else np.nan,
                         possible_max=float(np.max(values)) if np.isfinite(y) else np.nan,
                         role_outcome_known=known, source_conflict=conflict,
                         outcome_reason=';'.join(reasons) if reasons else ('known_role_and_entry' if len(possible) == 1 else 'zero_points' if y == 0 else 'entry_or_role_unknown'),
                         original_assigned_bench_points=3*y))
    return pd.DataFrame(rows)


def replay_saved_policies(pool, squads, metrics, output):
    rows = []
    for (slate, engine), selected in squads.groupby(['slate', 'engine']):
        data = selected[['id', 'is_sub', 'is_capt']].merge(pool[pool.slate.eq(slate)], on='id', validate='one_to_one')
        if len(data) != 16 or data.is_sub.sum() != 1 or data.is_capt.sum() != 1:
            raise ValueError('Archived squad is incomplete')
        xv = data.loc[~data.is_sub, 'official_points'].sum(min_count=15)
        captain = data.loc[data.is_capt, 'official_points'].sum(min_count=1)
        original_sub = data.loc[data.is_sub, 'original_assigned_bench_points'].sum(min_count=1)
        actual_sub = data.loc[data.is_sub, 'actual_role_supersub_points'].sum(min_count=1)
        original_total, audited_total = xv + captain + original_sub, xv + captain + actual_sub
        old = metrics[metrics.slate.eq(slate) & metrics.engine.eq(engine)]
        if len(old) != 1 or not np.allclose(original_total, old.total.iloc[0], rtol=0, atol=1e-8, equal_nan=True):
            raise ValueError('Original outcome policy did not reproduce before auditing roles')
        rows.append(dict(slate=slate, engine=engine, original_total=original_total, actual_role_total=audited_total,
                         delta=audited_total-original_total, original_sub=original_sub, actual_role_sub=actual_sub,
                         selected_outcome_unknown=int(data.official_points.isna().sum()),
                         selected_sub_unknown=int(data.loc[data.is_sub, 'actual_role_supersub_points'].isna().sum()),
                         selected_source_conflicts=int(data.source_conflict.sum()),
                         sub_name=data.loc[data.is_sub, 'name'].iloc[0],
                         sub_reason=data.loc[data.is_sub, 'outcome_reason'].iloc[0]))
    pd.DataFrame(rows).to_csv(f'{output}/archived_policy_outcome_audit.csv', index=False)


def repair_ncr_diagnostics(units, args):
    old_players = pd.read_csv(f'{args.diagnostics}/players.csv')
    old_events = pd.read_csv(f'{args.diagnostics}/event_components.csv')
    fixed_players, fixed_events = [], []
    for unit in units:
        slate = unit['slate']
        if slate.competition != 'ncr':
            continue
        labels = unit['raw_labels'].copy()
        labels['id'] = slate.pool.id.to_numpy()
        labels['pos'] = slate.pool.pos.to_numpy()
        label_index = labels.set_index('id')
        events = old_events[old_events.slate.eq(slate.name)].copy()
        for index, event in events.iterrows():
            label = label_index.loc[event.id]
            known = observed(label, event.event)
            actual = float(label[event.event]) if known else np.nan
            scorer = scorer_for('ncr', version='ncr_front_row_v2')
            weight = 12.0 if event.event == 'tries' else scorer.scrum_weight(label.pos) if event.event == 'scrums_won' else scorer.weights.get(event.event, 0.0)
            events.loc[index, ['observed', 'actual', 'actual_component', 'component_error']] = [known, actual, weight*actual, event.predicted_component-weight*actual]
        players = old_players[old_players.slate.eq(slate.name)].copy()
        for index, player in players.iterrows():
            label = label_index.loc[player.id]
            minutes = float(label.minutes) if observed(label, 'minutes') else np.nan
            activity = any(observed(label, event) and float(label[event]) > 0 for event in DIRECT_EVENTS)
            group = events[events.engine.eq(player.engine) & events.id.eq(player.id)]
            known = group.observed.eq(True)
            known_pred = group.loc[known, 'predicted_component'].sum()
            known_actual = group.loc[known, 'actual_component'].sum()
            players.loc[index, ['actual_minutes', 'minute_error', 'observed_activity', 'known_raw_score',
                                'known_component_prediction', 'observed_component_error', 'unresolved_reconciliation_error']] = [
                minutes, player.predicted_minutes-minutes, activity, known_actual, known_pred,
                known_pred-known_actual, player.point_error-(known_pred-known_actual)]
            players.loc[index, 'minutes_provenance'] = label.get('provenance__minutes', 'unknown')
            players.loc[index, 'unobserved_scoring_events'] = ';'.join(group.loc[~known, 'event'])
            players.loc[index, 'source_key_matched'] = bool(label.source_key_matched)
        fixed_players.append(players)
        fixed_events.append(events)
    players = pd.concat(fixed_players, ignore_index=True)
    events = pd.concat(fixed_events, ignore_index=True)
    players.to_csv(f'{args.output}/ncr_players_corrected_join.csv', index=False)
    events.to_csv(f'{args.output}/ncr_event_components_corrected_join.csv', index=False)
    events.groupby(['engine', 'status', 'event'], as_index=False).agg(
        rows=('id', 'size'), observed=('actual', 'count'), predicted=('predicted', 'mean'), actual=('actual', 'mean'),
        weighted_bias=('component_error', 'mean'), weighted_mae=('component_error', lambda values: values.abs().mean())
    ).to_csv(f'{args.output}/ncr_event_summary_corrected_join.csv', index=False)
    players.groupby(['engine', 'status'], as_index=False).agg(
        rows=('id', 'size'), known_duration=('actual_minutes', 'count'),
        point_mae=('point_error', lambda values: values.abs().mean()), point_bias=('point_error', 'mean'),
        minutes_mae=('minute_error', lambda values: values.abs().mean()), minutes_bias=('minute_error', 'mean'),
        unresolved_reconciliation_mae=('unresolved_reconciliation_error', lambda values: values.abs().mean())
    ).to_csv(f'{args.output}/ncr_exposure_summary_corrected_join.csv', index=False)


def run(args):
    os.makedirs(args.output, exist_ok=False)
    manifest = json.load(open(f'{args.prepared}/manifest.json'))
    units, input_hashes = [], {}
    for path in sorted(glob.glob(f'{args.prepared}/units/*.pkl')):
        name = path.removeprefix(args.prepared + '/')
        if digest(path) != manifest['outputs'][name]:
            raise ValueError('Prepared units changed before outcome audit')
        input_hashes[path] = digest(path)
        with open(path, 'rb') as handle:
            units.extend(unit for unit in pickle.load(handle) if unit['kind'] == 'fantasy')
    if len(units) != 13:
        raise ValueError('All thirteen official pools are required')
    pools, labels = [], []
    for unit in units:
        pool = base_pool(unit)
        if unit['slate'].competition == 'ncr':
            path = f'data/ncr/feeds/players_gw{unit["slate"].round}.json'
            input_hashes[path] = digest(path)
            pool = pool.merge(read_feed(path), on='id', how='left', validate='one_to_one')
        pools.append(pool)
        label = unit['raw_labels'].copy()
        label['slate'], label['id'] = unit['name'], unit['slate'].pool.id.to_numpy()
        labels.append(label)
    pool = pd.concat(pools, ignore_index=True)
    if len(pool) != 2138 or pool.duplicated(['slate', 'id']).any():
        raise ValueError('Complete original candidate population changed')
    pool = pool.merge(role_outcomes(pool), on=['slate', 'id'], validate='one_to_one')
    pool.to_csv(f'{args.output}/full_pool_outcomes.csv', index=False)
    pd.concat(labels, ignore_index=True).to_csv(f'{args.output}/bridged_raw_labels.csv', index=False)
    squads_path, metrics_path = f'{args.roles}/squads.csv', f'{args.roles}/metrics.csv'
    input_hashes.update({path: digest(path) for path in (squads_path, metrics_path, __file__)})
    replay_saved_policies(pool, pd.read_csv(squads_path), pd.read_csv(metrics_path), args.output)
    snapshot_rows = []
    first = read_feed('data/ncr/feeds/players_gw1.json')
    alternate_path = 'data/ncr/feeds/players_post_gw1.json'
    if os.path.exists(alternate_path):
        input_hashes[alternate_path] = digest(alternate_path)
        alternate = read_feed(alternate_path)
        comparison = first.merge(alternate, on='id', how='outer', suffixes=('_registered', '_alternate'), indicator=True, validate='one_to_one')
        for column in first.columns.drop('id'):
            a, b = comparison[f'{column}_registered'], comparison[f'{column}_alternate']
            changed = ~(a.eq(b) | (a.isna() & b.isna()))
            for _, row in comparison[changed].iterrows():
                snapshot_rows.append(dict(id=row.id, field=column, registered=row[f'{column}_registered'], alternate=row[f'{column}_alternate'], support=row['_merge']))
    pd.DataFrame(snapshot_rows, columns=['id', 'field', 'registered', 'alternate', 'support']).to_csv(f'{args.output}/gw1_snapshot_differences.csv', index=False)
    pool.groupby(['slate', 'competition'], as_index=False).agg(
        candidates=('id', 'size'), unknown_official_points=('official_points', lambda values: values.isna().sum()),
        matched_raw_keys=('source_key_matched', 'sum'), known_duration=('source_minutes', 'count'),
        source_conflicts=('source_conflict', 'sum'), unknown_supersub_outcomes=('role_outcome_known', lambda values: (~values).sum()),
        differing_known_sub_points=('actual_role_supersub_points', 'count')
    ).to_csv(f'{args.output}/pool_summary.csv', index=False)
    for path in ('players.csv', 'event_components.csv'):
        input_hashes[f'{args.diagnostics}/{path}'] = digest(f'{args.diagnostics}/{path}')
    repair_ncr_diagnostics(units, args)
    if any(digest(path) != value for path, value in input_hashes.items()):
        raise ValueError('Outcome audit inputs changed during execution')
    write_json(f'{args.output}/manifest.json', dict(input_hashes=input_hashes,
               status='descriptive_correctness_audit_not_candidate_selection', objective='active_unfinished',
               outcome_rule='Sub receives triple official points only after entering as a replacement, half if actually starting, zero if not playing. Unknown role or entry and conflicting sources remain unknown unless all possible outcomes equal zero.',
               source_policy='NCR registered round feed min_in_game is an entry flag, not duration. Missing values are not coerced to zero. Canonical direct positive activity supports entry, and source conflicts are retained. Six entry uses known start, positive observed duration or direct activity; zero-duration inactivity alone does not prove unused.',
               original_scoring_unchanged=True, candidate_fits=0, rugby_api_calls=0,
               limitations='Current public rules and retrospective source roles do not establish original publication time. This audit does not independently verify historical prices. Older diagnostics remain intact; repaired NCR raw rows replace only previously unmatched diagnostic labels, never forecast inputs.'))
    print(pd.read_csv(f'{args.output}/pool_summary.csv').to_string(index=False), flush=True)
    print(pd.read_csv(f'{args.output}/archived_policy_outcome_audit.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--roles', required=True)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
