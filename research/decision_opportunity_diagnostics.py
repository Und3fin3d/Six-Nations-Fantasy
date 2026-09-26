import argparse
import glob
import hashlib
import json
import os
import tarfile

import numpy as np
import pandas as pd

from model.history import past_matches
from model.role_selection import SelectionRules, verify_selection
from model.unified.raw_benchmark.config import DIRECT_EVENTS
from model.unified.rolling_eval import KEY, official_slates, expected_points
from model.unified.scoring import scorer_for
from research.decision_replay import archived_forecasts


ORIGINAL_SHA = '2b70331c0cf215def67831cd27c987362e2ec8330b2afb41c8d204f48ef17313'
CORRECTED_SHA = 'e4b20637e2b514d622bb35cc7c43762e0eddb4f3b4e50f1ca2e39155ae96e6be'


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def restore_official_evidence(destination):
    source = 'data/unified/complete_comparison/evidence'
    manifest = json.load(open(f'{source}/manifest.json'))
    expected = next(row for row in manifest['bundles'] if row['path'] == 'official.tar.gz')
    archive_path = f'{source}/official.tar.gz'
    if digest(archive_path) != expected['sha256']:
        raise ValueError('Original official evidence bundle changed')
    packed = f'{destination}/packed'
    os.makedirs(packed, exist_ok=False)
    with tarfile.open(archive_path) as archive:
        archive.extractall(packed, filter='data')
    files_by_hash = {digest(path): path for path in glob.glob(f'{packed}/**/*.tar.gz', recursive=True)}
    for job in manifest['jobs']:
        if job['bundle'] != 'official.tar.gz':
            continue
        path = files_by_hash.get(job['sha256'])
        if path is None:
            raise ValueError(f'Missing exact inner evidence bundle: {job["name"]}')
        target = f'{destination}/{job["name"]}'
        os.makedirs(target, exist_ok=False)
        with tarfile.open(path) as archive:
            archive.extractall(target, filter='data')
    return destination


def minute_band(minutes, active):
    if not np.isfinite(minutes):
        return 'unknown'
    if minutes == 0:
        return 'zero_with_observed_activity' if active else 'zero_without_observed_activity'
    if minutes < 10:
        return 'positive_under_10'
    if minutes < 30:
        return '10_to_under_30'
    if minutes < 60:
        return '30_to_under_60'
    return '60_plus'


def prior_opportunity(history):
    history = history[history.competition_level.eq('international')].copy()
    rows = []
    for (fixture, team), group in history.groupby(['fixture_id', 'team']):
        complete = len(group) >= 15 and group['available__tries'].fillna(False).all() and group.tries.notna().all()
        if complete:
            rows.append(dict(fixture_id=fixture, team=team, opponent=group.opponent.iloc[0],
                             date=group.date.iloc[0], tries=float(group.tries.sum())))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}, {}, 0
    frame = frame.sort_values(['date', 'fixture_id', 'team'])
    attacking = {team: float(group.tries.ewm(span=8, adjust=False).mean().iloc[-1]) for team, group in frame.groupby('team')}
    conceding = {team: float(group.tries.ewm(span=8, adjust=False).mean().iloc[-1]) for team, group in frame.groupby('opponent')}
    return attacking, conceding, len(frame)


def feasible_exchanges(pool, selected, rules, slate, engine):
    selected = selected.copy()
    records = []
    identities = set(selected.id)
    captain = selected.loc[selected.is_capt].iloc[0]
    for _, alternative in selected.loc[~selected.is_sub & selected.status.isin(rules.captain_status)].iterrows():
        records.append(dict(slate=slate, engine=engine, role='captain_extra', removed_id=captain.id,
                            alternative_id=alternative.id, removed_name=captain['name'], alternative_name=alternative['name'],
                            predicted_delta=alternative.point_forecast-captain.point_forecast,
                            realised_delta=alternative.actual-captain.actual,
                            outcome_known=bool(np.isfinite(alternative.actual) and np.isfinite(captain.actual))))
    for _, removed in selected.iterrows():
        if removed.is_sub:
            alternatives = pool[~pool.id.isin(identities) & pool.status.isin(rules.supersub_status)]
            role, multiplier = 'super_sub', 3
        else:
            alternatives = pool[~pool.id.isin(identities) & pool.pos.eq(removed.pos)]
            if removed.is_capt:
                alternatives = alternatives[alternatives.status.isin(rules.captain_status)]
            role, multiplier = ('xv_with_captain', 2) if removed.is_capt else ('xv', 1)
        for _, alternative in alternatives.iterrows():
            replacement = alternative.to_frame().T.copy()
            replacement['is_sub'] = bool(removed.is_sub)
            replacement['is_capt'] = bool(removed.is_capt)
            candidate = pd.concat([selected[selected.id.ne(removed.id)], replacement], ignore_index=True)
            candidate['is_sub'] = candidate.is_sub.astype(bool)
            candidate['is_capt'] = candidate.is_capt.astype(bool)
            try:
                verify_selection(candidate, rules)
            except ValueError:
                continue
            records.append(dict(slate=slate, engine=engine, role=role, removed_id=removed.id,
                                alternative_id=alternative.id, removed_name=removed['name'], alternative_name=alternative['name'],
                                predicted_delta=multiplier*(alternative.point_forecast-removed.point_forecast),
                                realised_delta=multiplier*(alternative.actual-removed.actual),
                                outcome_known=bool(np.isfinite(alternative.actual) and np.isfinite(removed.actual))))
    return records


def run(args):
    from model_env_preflight import check
    errors = check()
    if errors:
        raise RuntimeError('; '.join(errors))
    for path, expected in ((args.original_store, ORIGINAL_SHA), (args.corrected_store, CORRECTED_SHA)):
        if digest(path) != expected:
            raise ValueError(f'Input store changed: {path}')
    os.makedirs(args.output, exist_ok=False)
    evidence = restore_official_evidence(f'{args.output}/original_archives')
    original = pd.read_csv(args.original_store, dtype={key: str for key in KEY}, low_memory=False)
    corrected = pd.read_csv(args.corrected_store, dtype={key: str for key in KEY}, low_memory=False)
    if corrected.duplicated(KEY).any() or not corrected[KEY].equals(original[KEY]):
        raise ValueError('Corrected raw-label support changed')
    lookup = corrected.set_index(KEY)
    role_pools = pd.read_csv(f'{args.roles}/pools.csv', dtype={f'key_{key}': str for key in KEY})
    squads = pd.read_csv(f'{args.roles}/squads.csv')
    metrics = pd.read_csv(f'{args.roles}/metrics.csv')
    source_files = [__file__, args.original_store, args.corrected_store,
                    f'{args.roles}/pools.csv', f'{args.roles}/squads.csv', f'{args.roles}/metrics.csv',
                    'model/role_selection.py', 'model/unified/scoring.py', 'model/unified/rolling_eval.py',
                    'research/decision_replay.py', 'data/unified/complete_comparison/evidence/manifest.json']
    before = {path: digest(path) for path in source_files}
    player_rows, event_rows, exchange_rows, context_rows = [], [], [], []
    for slate in official_slates(original, ('six_nations', 'ncr')):
        print(slate.name, flush=True)
        points, raw, files, checks = archived_forecasts(slate, evidence)
        before.update({path: digest(path) for path in files})
        labels = lookup.reindex(pd.MultiIndex.from_frame(slate.candidates[KEY].astype(str))).reset_index()
        if not labels[KEY].equals(slate.candidates[KEY].astype(str).reset_index(drop=True)):
            raise ValueError('Outcome keys changed')
        pool = role_pools[role_pools.slate.eq(slate.name)].reset_index(drop=True)
        if not np.array_equal(pool.id, slate.pool.id) or not np.allclose(pool.actual, slate.actual, equal_nan=True):
            raise ValueError('Role-study full candidate pool changed')
        scorer = scorer_for(slate.competition, season=slate.season)
        rules = SelectionRules(budget=np.inf, max_nation=4, max_hemi=None) if slate.competition == 'six_nations' else SelectionRules()
        history = past_matches(corrected, slate.cutoff)
        attack, defence, historical_team_games = prior_opportunity(history)
        for engine, forecasts in raw.items():
            current_points = expected_points(forecasts, slate.competition, season=slate.season)
            if engine == 'p3_robust_native' and not np.allclose(current_points, pool[engine], atol=1e-9, rtol=0):
                raise ValueError('Scoring versions or robust forecasts changed')
            for index, prediction in enumerate(forecasts):
                row = labels.iloc[index]
                activity = any(bool(row.get(f'available__{event}', False)) and pd.notna(row.get(event)) and row[event] > 0 for event in DIRECT_EVENTS)
                available_minutes = bool(row.available__minutes) and pd.notna(row.minutes)
                minutes = float(row.minutes) if available_minutes else np.nan
                point = dict(slate=slate.name, competition=slate.competition, season=slate.season, engine=engine,
                             id=pool.id.iloc[index], name=pool['name'].iloc[index], team=prediction.team,
                             opponent=prediction.opponent, position=prediction.position, status=pool.status.iloc[index],
                             actual=slate.actual[index], predicted=current_points[index],
                             point_error=current_points[index]-slate.actual[index],
                             actual_minutes=minutes, predicted_minutes=prediction.minutes.mean,
                             minute_error=prediction.minutes.mean-minutes, minute_band=minute_band(minutes, activity),
                             minutes_provenance=row.get('provenance__minutes', 'unknown'),
                             observed_activity=activity, prior_team_tries=attack.get(prediction.team, np.nan),
                             prior_opponent_tries_conceded=defence.get(prediction.opponent, np.nan),
                             context_history_team_games=historical_team_games)
                known_actual_points, known_predicted_points, unknown_events = 0.0, 0.0, []
                weights = dict(scorer.weights)
                weights['tries'] = 12 if slate.competition == 'ncr' else (15 if prediction.is_forward else 10)
                weights['scrums_won'] = scorer.scrum_weight(prediction.position) if slate.competition == 'ncr' else 1.0
                if slate.competition == 'six_nations':
                    weights['metres'] = 0.1
                for event, weight in weights.items():
                    if weight == 0:
                        continue
                    observed = bool(row.get(f'available__{event}', False)) and pd.notna(row.get(event))
                    mean = prediction.events[event].mean if event in prediction.events else 0.0
                    actual = float(row[event]) if observed else np.nan
                    if event == 'metres':
                        other = sum(w * (prediction.events[e].mean if e in prediction.events else 0.0)
                                    for e, w in weights.items() if e != 'metres')
                        predicted_points = float(current_points[index]-other)
                        actual_points = float(np.floor(actual / 10)) if observed else np.nan
                    else:
                        predicted_points, actual_points = weight * mean, weight * actual
                    event_rows.append(dict(slate=slate.name, engine=engine, id=point['id'], status=point['status'],
                                           team=prediction.team, opponent=prediction.opponent, event=event,
                                           observed=observed, predicted=mean, actual=actual,
                                           predicted_component=predicted_points, actual_component=actual_points,
                                           component_error=predicted_points-actual_points))
                    if observed:
                        known_actual_points += actual_points
                        known_predicted_points += predicted_points
                    else:
                        unknown_events.append(event)
                point.update(known_raw_score=known_actual_points, known_component_prediction=known_predicted_points,
                             observed_component_error=known_predicted_points-known_actual_points,
                             unresolved_reconciliation_error=(current_points[index]-slate.actual[index])-(known_predicted_points-known_actual_points),
                             unobserved_scoring_events=';'.join(unknown_events))
                player_rows.append(point)
        for engine in metrics[metrics.slate.eq(slate.name)].engine:
            if engine in ('incumbent', 'empirical_baseline', 'equal_incumbent', 'p3_robust_native'):
                base = engine
            elif slate.competition == 'ncr':
                base = 'incumbent'
            else:
                base = 'p3_robust_native' if engine.startswith('p3_') else 'incumbent'
            candidate_pool = slate.pool.copy()
            candidate_pool['actual'] = slate.actual
            candidate_pool['point_forecast'] = pool[base].to_numpy(float)
            known = np.isfinite(slate.actual)
            recorded = metrics[metrics.slate.eq(slate.name) & metrics.engine.eq(engine)]
            reproduced_mae = np.abs(candidate_pool.point_forecast.to_numpy()[known]-slate.actual[known]).mean()
            if len(recorded) != 1 or not np.isclose(reproduced_mae, recorded.mae.iloc[0], atol=1e-9, rtol=0):
                raise ValueError('Diagnostic point-head attribution changed')
            selected_keys = squads[squads.slate.eq(slate.name) & squads.engine.eq(engine)][['id', 'is_sub', 'is_capt']]
            selected = candidate_pool.merge(selected_keys, on='id', how='inner', validate='one_to_one')
            verify_selection(selected, rules)
            exchange_rows.extend(feasible_exchanges(candidate_pool, selected, rules, slate.name, engine))
        pd.DataFrame(player_rows).to_csv(f'{args.output}/players.csv', index=False)
        pd.DataFrame(event_rows).to_csv(f'{args.output}/event_components.csv', index=False)
        pd.DataFrame(exchange_rows).to_csv(f'{args.output}/feasible_exchanges.csv', index=False)
    players = pd.DataFrame(player_rows)
    events = pd.DataFrame(event_rows)
    exchanges = pd.DataFrame(exchange_rows)
    exposure = players.groupby(['competition', 'season', 'engine', 'status', 'minute_band'], as_index=False).agg(
        rows=('id', 'size'), known_points=('actual', 'count'), actual_points=('actual', 'mean'),
        predicted_points=('predicted', 'mean'), point_bias=('point_error', 'mean'),
        point_mae=('point_error', lambda value: value.abs().mean()), minutes_bias=('minute_error', 'mean'),
        minutes_mae=('minute_error', lambda value: value.abs().mean()))
    exposure.to_csv(f'{args.output}/exposure_summary.csv', index=False)
    events.groupby(['engine', 'status', 'event'], as_index=False).agg(
        rows=('id', 'size'), observed=('actual', 'count'), predicted=('predicted', 'mean'), actual=('actual', 'mean'),
        weighted_bias=('component_error', 'mean'), weighted_mae=('component_error', lambda value: value.abs().mean())
    ).to_csv(f'{args.output}/event_summary.csv', index=False)
    for (slate, engine, team), group in players.groupby(['slate', 'engine', 'team']):
        context_rows.append(dict(slate=slate, engine=engine, team=team, opponent=group.opponent.iloc[0], rows=len(group),
                                 observed=int(group.actual.notna().sum()),
                                 mean_point_error=float(group.point_error.mean()),
                                 mean_minutes_error=float(group.minute_error.mean()),
                                 prior_team_tries=group.prior_team_tries.iloc[0],
                                 prior_opponent_tries_conceded=group.prior_opponent_tries_conceded.iloc[0]))
    contexts = pd.DataFrame(context_rows)
    contexts.to_csv(f'{args.output}/context_summary.csv', index=False)
    correlations = []
    for engine, group in contexts.groupby('engine'):
        for feature in ('prior_team_tries', 'prior_opponent_tries_conceded'):
            observed = group.dropna(subset=[feature, 'mean_point_error'])
            correlations.append(dict(engine=engine, feature=feature, team_rounds=len(observed),
                                     descriptive_spearman=observed[feature].corr(observed.mean_point_error, method='spearman')))
    pd.DataFrame(correlations).to_csv(f'{args.output}/context_correlations.csv', index=False)
    exchange_summary = exchanges.groupby(['slate', 'engine', 'role'], as_index=False).agg(
        feasible_exchanges=('alternative_id', 'size'), known_outcomes=('outcome_known', 'sum'),
        best_known_realised_delta=('realised_delta', 'max'), best_predicted_point_delta=('predicted_delta', 'max'))
    exchange_summary.to_csv(f'{args.output}/exchange_summary.csv', index=False)
    if any(digest(path) != value for path, value in before.items()):
        raise ValueError('Inputs changed during descriptive analysis')
    with open(f'{args.output}/manifest.json', 'w') as handle:
        json.dump(dict(source_hashes=before, objective='active_unfinished', models_fitted=0,
                       new_candidates_evaluated=0, evidence='All previously inspected historical outcomes; descriptive only',
                       exchange_definition='All feasible single-player changes, holding other assignments fixed; a replaced captain transfers that role to the incoming eligible player. Captain-only alternatives hold the XV fixed. Unknown outcomes remain unknown. These are hindsight diagnostics, not deployable selectors. Predicted deltas use the named point forecast, not the native rank-utility objective.',
                       reconciliation='Unobserved scoring events are not imputed as observed zero. The unresolved component includes missing events and provider/rules reconciliation; it is not automatically a scoring defect.',
                       context='Prior-only exponentially weighted international team attacking and opponent-conceding tries, one observation per complete team-fixture, span eight; correlation is descriptive, not causal or independent validation.'), handle, indent=2)
    print(exposure.to_string(index=False), flush=True)
    print(pd.DataFrame(correlations).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-store', required=True)
    parser.add_argument('--corrected-store', required=True)
    parser.add_argument('--roles', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
