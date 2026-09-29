import argparse
import gc
import glob
import gzip
import hashlib
import json
import os
import pickle
import shutil
import time
import warnings

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.conditional_duration import ConditionalDurationGBDT
from model.unified.raw_benchmark.blend import EventWeightedBlend
from model.unified.raw_benchmark.config import EXTENDED_EVENTS, STABLE_EVENTS
from model.unified.raw_benchmark.empirical import EmpiricalEventModel
from model.unified.raw_benchmark.features import build_frozen_feature_frames
from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from model.unified.rolling_eval import KEY, expected_points, official_slates
from model.unified.v4.gbdt import V4GBDT
from research.conditional_duration_inputs import REGISTRATION, check_mask, digest, read_store, write_json
from research.conditional_duration_metrics import OUTCOME_AUDIT, duration_measurements, floor_sensitivity, load_outcome_audit, raw_measurements, select_points, write_forecasts
from research.decision_replay import corrected_empirical
from research.duration_domain_model import DOMAINS, DurationDomainGBDT, duration_mask


ADDENDUM = 'research/DECISION_CONDITIONAL_DURATION_SUPPORT_ADDENDUM_2026-09-26.json'
DOMAIN_ADDENDUM = 'research/DECISION_DURATION_DOMAIN_ADDENDUM_2026-09-26.json'
CONFIG = 'data/unified/p3_hillclimb/config.json'
EVENTS = (*STABLE_EVENTS, *EXTENDED_EVENTS)


def conditional_names(args):
    name = f'conditional_duration_{args.duration_domain}_v2' if args.duration_domain else 'conditional_duration_v1'
    return name, f'p3_{name}'


def model_context(args, train, cutoff):
    paths = sorted(set(glob.glob('model/**/*.py', recursive=True) + [REGISTRATION, ADDENDUM, CONFIG]))
    return dict(store_sha256=digest(args.store), prepared_features_sha256=digest(f'{args.prepared}/training_features.pkl'),
                cutoff=cutoff.isoformat(), training_rows=len(train),
                training_keys_sha256=hashlib.sha256(train[KEY].to_csv(index=False).encode()).hexdigest(),
                source_hashes={path: digest(path) for path in paths})


def domain_context(context, args, frame):
    valid = duration_mask(frame, args.duration_domain)
    paths = [DOMAIN_ADDENDUM, 'research/duration_domain_model.py', 'research/conditional_duration_run.py']
    return dict(context, duration_domain=args.duration_domain,
                duration_support_sha256=hashlib.sha256(frame.loc[valid, KEY].to_csv(index=False).encode()).hexdigest(),
                domain_sources={path: digest(path) for path in paths})


def fit_or_restore(name, model, frame, context, args):
    destination = f'{args.output}/models/{name}.pkl.gz'
    stamp = f'{args.output}/models/{name}.json'
    source = f'{args.reuse_models}/{name}.pkl.gz' if args.reuse_models else None
    source_stamp = f'{args.reuse_models}/{name}.json' if args.reuse_models else None
    if source and os.path.exists(source) and os.path.exists(source_stamp):
        old = json.load(open(source_stamp))
        if old['context'] != context or old['model_sha256'] != digest(source):
            raise ValueError(f'Cannot resume {name}: model context or binary changed')
        print(f'{name}: restore verified fitted checkpoint', flush=True)
        with gzip.open(source, 'rb') as handle:
            restored = pickle.load(handle)
        if not isinstance(restored, type(model)):
            raise TypeError('Restored model class differs from the registered class')
        shutil.copyfile(source, destination)
        shutil.copyfile(source_stamp, stamp)
        return restored
    print(f'{name}: fit {len(frame):,} prior rows', flush=True)
    started = time.monotonic()
    fitted = model.fit(frame)
    with open(destination, 'wb') as handle:
        handle.write(gzip.compress(pickle.dumps(fitted, protocol=5), compresslevel=3, mtime=0))
    write_json(stamp, dict(name=name, context=context, model_sha256=digest(destination),
                           fit_seconds=time.monotonic()-started))
    return fitted


def fit_models(train, training_features, context, args):
    empirical = fit_or_restore('empirical_event', EmpiricalEventModel(asof=context['cutoff']), train, context, args)
    robust = fit_or_restore('robust_empirical_event', RobustEmpiricalEventModel(asof=context['cutoff']), train, context, args)
    parameters = dict(events=EVENTS, weighting='natural', pool_player_id=True, player_effects=True, native_categories=True)
    direct = fit_or_restore('v4_corrected_full_refit', V4GBDT(**parameters), training_features, context, args)
    joint = training_features.copy()
    duration_observed = joint.available__minutes.fillna(False) & joint.minutes.notna()
    for event in EVENTS:
        joint[f'available__{event}'] = joint[f'available__{event}'].fillna(False) & duration_observed
    supported = fit_or_restore('v4_joint_support_control', V4GBDT(**parameters), joint, context, args)
    del joint
    gc.collect()
    models = dict(empirical_event=empirical, robust_empirical_event=robust,
                  v4_corrected_full_refit=direct, v4_joint_support_control=supported)
    conditional_name, blend_name = conditional_names(args)
    if args.duration_domain:
        extended = domain_context(context, args, training_features)
        if args.duration_domain == 'bounded':
            joint = training_features.copy()
            valid = duration_mask(joint, args.duration_domain)
            joint['available__minutes'] = valid
            for event in EVENTS:
                joint[f'available__{event}'] = joint[f'available__{event}'].fillna(False) & valid
            models['v4_bounded_support_control'] = fit_or_restore(
                'v4_bounded_support_control', V4GBDT(**parameters), joint, extended, args)
            del joint
            gc.collect()
        conditional = fit_or_restore(conditional_name, DurationDomainGBDT(duration_domain=args.duration_domain, **parameters),
                                     training_features, extended, args)
    else:
        conditional = fit_or_restore(conditional_name, ConditionalDurationGBDT(**parameters), training_features, context, args)
    configuration = json.load(open(CONFIG))
    weights = dict(weight_v4=configuration['default_weight_v4'], event_weights_v4=configuration['event_weights_v4'])
    models['p3_corrected_full_refit'] = EventWeightedBlend(robust, direct, **weights)
    models[conditional_name] = conditional
    models[blend_name] = EventWeightedBlend(robust, conditional, **weights)
    return models, configuration


def verified_prepared(args):
    from model_env_preflight import check
    errors = check()
    if errors:
        raise RuntimeError('; '.join(errors))
    manifest = json.load(open(f'{args.prepared}/manifest.json'))
    registration = json.load(open(REGISTRATION))
    if digest(args.store) != registration['stores']['corrected_sha256']:
        raise ValueError('Corrected training store differs from the registration')
    for path, expected in manifest['source_hashes'].items():
        if path.startswith('model/') or path == REGISTRATION:
            if digest(path) != expected:
                raise ValueError(f'Prepared model sources changed: {path}')
    for name in (f'units/{args.job}.pkl', 'training_features.pkl', 'jobs.json', 'native_feature_changes.csv'):
        if digest(f'{args.prepared}/{name}') != manifest['outputs'][name]:
            raise ValueError(f'Prepared input changed: {name}')
    with open(f'{args.prepared}/units/{args.job}.pkl', 'rb') as handle:
        units = pickle.load(handle)
    if not units or len({unit['cutoff'] for unit in units}) != 1:
        raise ValueError('A model group must have exactly one chronological lock')
    return units, manifest


def corrected_fantasy_points(unit, corrected_store, wr):
    slate = unit['slate']
    if slate.competition != 'ncr':
        return corrected_empirical(slate, corrected_store, wr)
    fresh = next(item for item in official_slates(corrected_store, ('ncr',)) if item.name == slate.name)
    columns = ['id', 'team', 'pos', 'status', 'value', 'hemi']
    if not fresh.pool[columns].equals(slate.pool[columns]) or not fresh.candidates[KEY].equals(slate.candidates[KEY]):
        raise ValueError('Corrected-history incumbent changed the candidate pool; review before comparing')
    if not np.allclose(fresh.actual, slate.actual, equal_nan=True):
        raise ValueError('Corrected-history incumbent outcomes changed')
    return fresh.baseline


def record_unit(unit, models, configuration, candidates, store, wr, args, outcome_audit=None):
    directory = f'{args.output}/units/{unit["name"]}'
    os.makedirs(directory, exist_ok=False)
    unit['raw_labels'].to_csv(f'{directory}/raw_labels.csv', index=False)
    unit['candidates'][KEY + ['position', 'started']].to_csv(f'{directory}/candidate_keys.csv', index=False)
    predictions, raw_metrics, fantasy_metrics = {}, [], []
    conditional_name, blend_name = conditional_names(args)
    for name, model in models.items():
        print(f'{unit["name"]}/{name}: predict complete pool', flush=True)
        prediction = model.predict_frame(candidates)
        write_forecasts(unit, name, prediction, directory)
        predictions[name] = prediction
        raw_metrics.extend(raw_measurements(unit, name, prediction))
        if unit['kind'] == 'fantasy':
            points = expected_points(prediction, unit['slate'].competition, season=unit['slate'].season)
            fantasy_metrics.append(select_points(unit, name, points, directory, outcome_audit=outcome_audit))
        pd.DataFrame(raw_metrics).to_csv(f'{directory}/raw_metrics.csv', index=False)
        if fantasy_metrics:
            pd.DataFrame(fantasy_metrics).to_csv(f'{directory}/fantasy_metrics.csv', index=False)
    pd.DataFrame(duration_measurements(unit, predictions[conditional_name], open_tail=bool(args.duration_domain))).to_csv(
        f'{directory}/duration_calibration.csv', index=False)
    if unit['kind'] == 'fantasy':
        empirical = corrected_fantasy_points(unit, store, wr)
        fantasy_metrics.append(select_points(unit, 'empirical_fantasy_corrected', empirical, directory, outcome_audit=outcome_audit))
        weight = configuration['event_weights_v4'].get('metres', configuration['default_weight_v4'])
        alternatives, sensitivity = floor_sensitivity(unit, predictions, weight, conditional_name, blend_name)
        for engine, points in alternatives.items():
            fantasy_metrics.append(select_points(unit, engine, points, directory, outcome_audit=outcome_audit))
        pd.DataFrame(sensitivity).to_csv(f'{directory}/floor_sensitivity.csv', index=False)
        pd.DataFrame(fantasy_metrics).to_csv(f'{directory}/fantasy_metrics.csv', index=False)
    write_json(f'{directory}/completion.json', dict(unit=unit['name'], raw_engines=list(models),
               fantasy_engines=[row['engine'] for row in fantasy_metrics], rows=len(candidates),
               candidate_keys_sha256=digest(f'{directory}/candidate_keys.csv'),
               output_hashes={os.path.basename(path): digest(path) for path in glob.glob(f'{directory}/*')},
               objective='active_unfinished'))
    return raw_metrics, fantasy_metrics


def run(args):
    units, prepared_manifest = verified_prepared(args)
    os.makedirs(args.output, exist_ok=False)
    os.makedirs(f'{args.output}/models')
    os.makedirs(f'{args.output}/units')
    store = read_store(args.store)
    cutoff = units[0]['cutoff']
    train = past_matches(store, cutoff).sort_values(['date', 'fixture_id', 'team', 'player_id'])
    forbidden = {fixture for unit in units for fixture in unit['source_fixture_ids']}
    forbidden.update(fixture for unit in units for fixture in unit['candidates'].fixture_id.astype(str))
    if set(train.fixture_id.astype(str)) & forbidden:
        raise ValueError('Evaluation fixture is present in model fitting history')
    for unit in units:
        check_mask(unit['candidates'])
    context = model_context(args, train, cutoff)
    executing_sources = [__file__, 'research/conditional_duration_metrics.py', 'research/conditional_duration_inputs.py',
                         'research/decision_role_policy.py', 'research/decision_replay.py',
                         'data/wr_rankings.csv', 'data/rp_compstats.csv', *glob.glob('data/ncr/*.csv'),
                         *glob.glob('data/ncr/feeds/*.json')]
    if args.duration_domain:
        executing_sources.extend([DOMAIN_ADDENDUM, 'research/duration_domain_model.py', OUTCOME_AUDIT])
    sources = {path: digest(path) for path in sorted(set(executing_sources))}
    write_json(f'{args.output}/manifest.json', dict(job=args.job, context=context, executing_sources=sources,
               duration_domain=args.duration_domain or 'registered_v1',
               prepared_manifest_sha256=digest(f'{args.prepared}/manifest.json'), units=[unit['name'] for unit in units],
               scoring=dict(six_nations_2025='six_nations_legacy_v1', six_nations_2026='six_nations_2026_v2', ncr='ncr_front_row_v2'),
               outcome_contract='actual_role_entry_v1' if args.duration_domain else 'assigned_bench_v1',
               native_comparator_feature_audit='Required in prepared/native_feature_changes.csv before any improvement claim',
               objective='active_unfinished', reused_historical_outcomes=True))
    prepared = pd.read_pickle(f'{args.prepared}/training_features.pkl')
    training_features = prepared.loc[train.index].copy()
    if not training_features[KEY].reset_index(drop=True).equals(train[KEY].reset_index(drop=True)):
        raise ValueError('Prepared training features do not match the chronological history')
    del prepared
    gc.collect()
    models, configuration = fit_models(train, training_features, context, args)
    conditional = models[conditional_names(args)[0]]
    pd.DataFrame(conditional.training_support).to_csv(f'{args.output}/conditional_training_support.csv', index=False)
    write_json(f'{args.output}/duration_training_classes.json', dict(counts=conditional.duration_counts.tolist(),
               centres=conditional.duration_centres.tolist(), variances=conditional.duration_variances.tolist()))
    if args.duration_domain:
        pd.DataFrame(conditional.duration_exclusions,
                     columns=['fixture_id', 'player_id', 'team', 'date', 'minutes', 'competition_level', 'started']).to_csv(
            f'{args.output}/excluded_duration_sources.csv', index=False)
    wr = pd.read_csv('data/wr_rankings.csv', parse_dates=['snapshot_date'])
    outcome_audit = load_outcome_audit() if args.duration_domain else None
    raw_metrics, fantasy_metrics = [], []
    for unit in units:
        _, features = build_frozen_feature_frames(train, unit['candidates'], v4=True, prepared_train=training_features)
        rows, fantasy = record_unit(unit, models, configuration, features, store, wr, args, outcome_audit)
        raw_metrics.extend(rows)
        fantasy_metrics.extend(fantasy)
        pd.DataFrame(raw_metrics).to_csv(f'{args.output}/raw_metrics.csv', index=False)
        if fantasy_metrics:
            pd.DataFrame(fantasy_metrics).to_csv(f'{args.output}/fantasy_metrics.csv', index=False)
    if sources != {path: digest(path) for path in sources} or context != model_context(args, train, cutoff):
        raise ValueError('Inputs or model sources changed during execution')
    write_json(f'{args.output}/completion.json', dict(job=args.job, units=[unit['name'] for unit in units],
               model_artifact_hashes={os.path.basename(path): digest(path) for path in glob.glob(f'{args.output}/models/*')},
               raw_rows=len(raw_metrics), fantasy_rows=len(fantasy_metrics), objective_completed=False,
               training_rows=len(train), analysis_completed=True, duration_domain=args.duration_domain or 'registered_v1'))
    if fantasy_metrics:
        print(pd.DataFrame(fantasy_metrics).to_string(index=False), flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--store', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reuse-models')
    parser.add_argument('--duration-domain', choices=DOMAINS)
    run(parser.parse_args())
