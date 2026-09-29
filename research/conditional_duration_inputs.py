import argparse
import glob
import hashlib
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.features import build_pit_features
from model.unified.raw_benchmark.folds import masked_candidates
from model.unified.rolling_eval import KEY, official_slates
from model.unified.schema import EVENTS
from model.unified.v4.features import add_v4_base_stats


REGISTRATION = 'research/DECISION_CONDITIONAL_DURATION_REGISTRATION_2026-09-26.json'
REGISTERED_COMMIT = '01d7848f61f414fef4527e89fda3fc9181058ce5'


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, values):
    with open(path, 'w') as handle:
        json.dump(values, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def read_store(path):
    return pd.read_csv(path, dtype={key: str for key in KEY}, parse_dates=['date', 'match_at'], low_memory=False)


def source_identity_bridge(slate, store):
    keys = slate.candidates[KEY].astype(str).copy()
    audit = []
    if slate.competition == 'ncr':
        fixtures = store[store.competition_level.eq('international')].drop_duplicates(['fixture_id', 'team', 'opponent'])
        dates = pd.to_datetime(fixtures.match_at, utc=True).dt.normalize()
        for synthetic, rows in slate.candidates.groupby('fixture_id', sort=False):
            teams = set(rows.team.astype(str)) | set(rows.opponent.astype(str))
            if len(teams) != 2:
                raise ValueError('An NCR fixture does not identify exactly two teams')
            date = pd.to_datetime(rows.date, utc=True).dt.normalize().iloc[0]
            matches = fixtures[fixtures.team.isin(teams) & fixtures.opponent.isin(teams) & (dates - date).abs().le(pd.Timedelta(days=3))]
            options = matches.fixture_id.astype(str).unique()
            if len(options) > 1:
                raise ValueError(f'Ambiguous source fixture for fantasy fixture {synthetic}: {options.tolist()}')
            actual_id = options[0] if len(options) else None
            audit.append(dict(slate=slate.name, candidate_fixture_id=str(synthetic), source_fixture_id=actual_id,
                              teams=' / '.join(sorted(teams)), match_count=len(options), rule='unique same teams within three calendar days; no scores consulted'))
            if actual_id is not None:
                keys.loc[slate.candidates.fixture_id.astype(str).eq(str(synthetic)), 'fixture_id'] = actual_id
    else:
        audit = [dict(slate=slate.name, candidate_fixture_id=fixture, source_fixture_id=fixture,
                      teams='', match_count=1, rule='exact canonical fixture identity') for fixture in keys.fixture_id.unique()]
    lookup = store.set_index(KEY)
    matched = pd.MultiIndex.from_frame(keys).isin(lookup.index)
    labels = lookup.reindex(pd.MultiIndex.from_frame(keys)).reset_index()
    labels['source_key_matched'] = matched
    labels['candidate_fixture_id'] = slate.candidates.fixture_id.astype(str).to_numpy()
    return labels, audit


def check_mask(candidates):
    for event in ('minutes', *EVENTS):
        if candidates[event].notna().any() or candidates[f'available__{event}'].fillna(False).any():
            raise ValueError(f'Evaluation outcome entered pre-match candidates: {event}')


def native_feature_changes(original, corrected, slates):
    from model.data import load
    from model.unified.rolling_eval import ROOT
    from research.incumbent_features import IncumbentFeatureHistory

    native = load()
    histories = [IncumbentFeatureHistory(ROOT, frame, native) for frame in (original, corrected)]
    differences = []
    for slate in slates:
        if slate.competition != 'six_nations':
            continue
        visible = native[(native.season < slate.season) | (native.season.eq(slate.season) & native['round'].le(slate.round))].copy()
        frames = [history.lock_features(visible.copy(), slate)[0] for history in histories]
        for column in frames[0].columns:
            a, b = frames[0][column], frames[1][column]
            changed = ~(a.eq(b) | (a.isna() & b.isna()))
            if changed.any():
                differences.append(dict(slate=slate.name, column=column, changed_rows=int(changed.sum()),
                                         changed_current_rows=int((changed & visible.season.eq(slate.season) & visible['round'].eq(slate.round)).sum())))
    return pd.DataFrame(differences, columns=['slate', 'column', 'changed_rows', 'changed_current_rows'])


def run(args):
    from model_env_preflight import check
    failures = check()
    if failures:
        raise RuntimeError('; '.join(failures))
    registration = json.load(open(REGISTRATION))
    if digest(args.original_store) != registration['stores']['original_sha256'] or digest(args.corrected_store) != registration['stores']['corrected_sha256']:
        raise ValueError('Registered stores changed')
    os.makedirs(args.output, exist_ok=False)
    original, corrected = read_store(args.original_store), read_store(args.corrected_store)
    if not original[KEY].equals(corrected[KEY]) or corrected.duplicated(KEY).any():
        raise ValueError('Store population changed')
    slates = official_slates(original, ('six_nations', 'ncr'))
    units, bridges, cutoff_audit = [], [], []
    for slate in slates:
        labels, audit = source_identity_bridge(slate, corrected)
        bridges.extend(audit)
        candidates = slate.candidates.copy()
        check_mask(candidates)
        known_source = set(labels.loc[labels.source_key_matched, 'fixture_id'].astype(str))
        training = past_matches(corrected, slate.cutoff)
        leaked_fixtures = known_source & set(training.fixture_id.astype(str))
        source_times = pd.to_datetime(labels.match_at, utc=True, errors='coerce')
        earliest = source_times.min()
        cutoff_audit.append(dict(unit=slate.name, cutoff=slate.cutoff.isoformat(),
                                 earliest_source_kickoff=earliest.isoformat() if pd.notna(earliest) else None,
                                 source_key_matches=int(labels.source_key_matched.sum()), rows=len(labels),
                                 source_fixtures_in_training=';'.join(sorted(leaked_fixtures)),
                                 cutoff_after_known_kickoff=bool(pd.notna(earliest) and slate.cutoff > earliest)))
        units.append(dict(name=slate.name, kind='fantasy', cutoff=slate.cutoff, candidates=candidates,
                          source_fixture_ids=sorted(known_source), raw_labels=labels, slate=slate))
    pd.DataFrame(bridges).to_csv(f'{args.output}/fixture_bridge.csv', index=False)
    pd.DataFrame(cutoff_audit).to_csv(f'{args.output}/cutoff_audit.csv', index=False)
    if any(row['source_fixtures_in_training'] or row['cutoff_after_known_kickoff'] for row in cutoff_audit):
        raise ValueError('Existing fantasy cutoff violates source fixture chronology; preserve audit and correct independently before fitting')
    friendly_ids = registration['evaluation']['friendly15_ids']
    friendly = corrected[corrected.fixture_id.isin(friendly_ids)].copy()
    if friendly.fixture_id.nunique() != 15 or len(friendly) != 690 or not friendly.groupby('fixture_id').size().eq(46).all():
        raise ValueError('Fixed Friendly-15 cohort is not exactly 15 complete 46-player teamsheets')
    for fixture in friendly_ids:
        labels = friendly[friendly.fixture_id.eq(fixture)].sort_values(KEY).reset_index(drop=True)
        candidates = masked_candidates(labels.drop(columns=['team_score', 'opp_score', 'official_pts'], errors='ignore'))
        check_mask(candidates)
        cutoff = pd.to_datetime(labels.match_at, utc=True).min()
        units.append(dict(name=f'friendly_{fixture}', kind='friendly', cutoff=cutoff, candidates=candidates,
                          source_fixture_ids=[fixture], raw_labels=labels, slate=None))
    os.makedirs(f'{args.output}/units')
    jobs = []
    for index, (cutoff, group) in enumerate(pd.Series(units).groupby(lambda i: units[i]['cutoff'])):
        name = f'lock_{index:02d}'
        values = group.tolist()
        with open(f'{args.output}/units/{name}.pkl', 'wb') as handle:
            pickle.dump(values, handle, protocol=5)
        jobs.append(dict(job=name, cutoff=cutoff.isoformat(), units=[value['name'] for value in values]))
    write_json(f'{args.output}/jobs.json', jobs)
    write_json(f'{args.output}/matrix.json', {'include': [{'job': row['job']} for row in jobs]})
    print(pd.DataFrame(cutoff_audit).to_string(index=False), flush=True)
    print('Checking native comparator feature effects', flush=True)
    native_feature_changes(original, corrected, slates).to_csv(f'{args.output}/native_feature_changes.csv', index=False)
    print('Preparing corrected prior-only full-model feature store', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        prepared = add_v4_base_stats(build_pit_features(corrected))
    if not prepared[KEY].equals(corrected[KEY]):
        raise ValueError('Prepared feature store key order changed')
    prepared.to_pickle(f'{args.output}/training_features.pkl')
    source_paths = [REGISTRATION, __file__, args.original_store, args.corrected_store,
                    *glob.glob('model/**/*.py', recursive=True), *glob.glob('research/incumbent*.py'),
                    *glob.glob('data/ncr/*.csv'), *glob.glob('data/ncr/feeds/*.json'),
                    'data/model_targets.csv', 'data/unified/p3_hillclimb/config.json']
    manifest = dict(registration=REGISTRATION, registered_commit=REGISTERED_COMMIT,
                    source_hashes={path: digest(path) for path in sorted(set(source_paths))},
                    outputs={path.removeprefix(args.output + '/'): digest(path) for path in glob.glob(f'{args.output}/**/*', recursive=True) if os.path.isfile(path)},
                    objective='active_unfinished', units=len(units), fantasy_units=13, friendly_units=15,
                    models_fitted=0, fixture_bridge='Used for outcome attribution and training-exclusion audit only; candidate forecast identities remain unchanged')
    write_json(f'{args.output}/manifest.json', manifest)
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-store', required=True)
    parser.add_argument('--corrected-store', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
