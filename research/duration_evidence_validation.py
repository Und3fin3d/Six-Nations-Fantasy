import json
import os
import pickle

import numpy as np
import pandas as pd

from model.unified.contracts import RawPrediction
from model.unified.rolling_eval import KEY
from research.conditional_duration_inputs import digest
from research.conditional_duration_metrics import raw_measurements


RUN = 36273533404
PREPARED_RUN = 36258801997
CONTROL_ENGINES = ('empirical_event', 'robust_empirical_event', 'v4_corrected_full_refit',
                   'v4_joint_support_control', 'p3_corrected_full_refit')


def close(actual, expected, label, *, tolerance=1e-8):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-10, atol=tolerance, equal_nan=True):
        raise ValueError(f'Research evidence does not reproduce: {label}')
    finite = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.0


def verify_hashes(directory, hashes):
    for name, expected in hashes.items():
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or digest(path) != expected:
            raise ValueError(f'Incomplete or modified recorded output: {path}')


def prepared_jobs(directory):
    manifest = json.load(open(f'{directory}/manifest.json'))
    verify_hashes(directory, {name: value for name, value in manifest['outputs'].items()
                              if name == 'jobs.json' or name.startswith('units/')})
    jobs = json.load(open(f'{directory}/jobs.json'))
    units = [name for job in jobs for name in job['units']]
    if len(jobs) != 27 or len(units) != 28 or len(set(units)) != 28:
        raise ValueError('The registered 27-lock / 28-unit cohort is required')
    changed = pd.read_csv(f'{directory}/native_feature_changes.csv')
    if not changed.empty:
        raise ValueError('Native comparator features changed; a corrected native replay is required')
    if digest(f'{directory}/native_feature_changes.csv') != manifest['outputs']['native_feature_changes.csv']:
        raise ValueError('Native feature audit changed')
    return jobs, manifest


def read_units(directory, job):
    with open(f'{directory}/units/{job["job"]}.pkl', 'rb') as handle:
        units = pickle.load(handle)
    if [unit['name'] for unit in units] != job['units']:
        raise ValueError('Prepared lock group membership changed')
    return units


def verify_study(path, job, domain):
    completion = json.load(open(f'{path}/completion.json'))
    manifest = json.load(open(f'{path}/manifest.json'))
    if completion['job'] != job['job'] or completion['units'] != job['units'] or completion['duration_domain'] != domain:
        raise ValueError('Study completion does not match the required lock/domain')
    if manifest['context']['cutoff'] != job['cutoff'] or manifest['duration_domain'] != domain:
        raise ValueError('Study lock or registered duration policy changed')
    for source, value in manifest['context']['source_hashes'].items():
        if digest(source) != value:
            raise ValueError(f'Fitted model source no longer matches: {source}')
    verify_hashes(f'{path}/models', completion['model_artifact_hashes'])
    conditional = f'conditional_duration_{domain}_v2'
    engines = [*CONTROL_ENGINES, conditional, f'p3_{conditional}']
    if domain == 'bounded':
        engines.append('v4_bounded_support_control')
    for unit in job['units']:
        record = json.load(open(f'{path}/units/{unit}/completion.json'))
        if set(record['raw_engines']) != set(engines):
            raise ValueError('A registered raw candidate or control is missing')
        verify_hashes(f'{path}/units/{unit}', record['output_hashes'])
    return manifest, engines


def read_forecasts(directory, engine, unit):
    path = f'{directory}/{engine}.jsonl'
    forecasts = [RawPrediction.from_dict(json.loads(line)) for line in open(path) if line.strip()]
    keys = pd.DataFrame([{key: getattr(prediction, key) for key in KEY} for prediction in forecasts])
    if not keys.equals(unit['candidates'][KEY].astype(str).reset_index(drop=True)):
        raise ValueError('Forecast file changed the full registered player pool or key order')
    saved = pd.read_csv(f'{directory}/raw_metrics.csv')
    saved = saved[saved.engine.eq(engine)].set_index('event').sort_index()
    measured = pd.DataFrame(raw_measurements(unit, engine, forecasts)).set_index('event').sort_index()
    if not saved.index.equals(measured.index):
        raise ValueError('Raw metric event support changed')
    numeric = ['rows', 'observed', 'predicted', 'scored', 'missing_prediction', 'absolute_error_sum',
               'squared_error_sum', 'error_sum', 'mae', 'bias', 'rmse']
    error = close(saved[numeric], measured[numeric], f'{unit["name"]}/{engine}/raw_metrics', tolerance=1e-7)
    return forecasts, measured.reset_index(), error


def distribution_variance(distribution):
    if distribution.family == 'bernoulli':
        return distribution.mean * (1 - distribution.mean)
    if distribution.family == 'negative_binomial':
        return distribution.mean + distribution.mean**2 / distribution.dispersion
    return distribution.dispersion


def distribution_parameter(family, mean, variance):
    if family == 'bernoulli':
        return 1.0
    if family == 'negative_binomial':
        return max(mean*mean/max(variance-mean, 1e-6), 0.05)
    return max(variance, 1e-8)


def verify_conditional(forecasts):
    max_mean_error, max_parameter_error, max_variance_error = 0.0, 0.0, 0.0
    reciprocal_roundoff = []
    for prediction in forecasts:
        metadata = prediction.metadata
        probability = np.asarray(metadata['duration_probabilities'], dtype=float)
        centres = np.asarray(metadata['duration_centres'], dtype=float)
        variances = np.asarray(metadata['duration_variances'], dtype=float)
        if probability.shape != (10,) or not np.isfinite(probability).all() or np.any(probability < 0):
            raise ValueError('Invalid saved duration distribution')
        close(probability.sum(), 1.0, 'duration probability sum')
        minute_mean = float(probability @ centres)
        minute_variance = max(float(probability @ (variances + centres**2) - minute_mean**2), 1e-8)
        max_mean_error = max(max_mean_error, close(prediction.minutes.mean, minute_mean, 'integrated minutes'))
        max_parameter_error = max(max_parameter_error, close(prediction.minutes.dispersion, minute_variance, 'minute variance'))
        if set(prediction.events) != set(metadata['conditional_event_means']):
            raise ValueError('Conditional event support differs from its marginal predictions')
        for event, distribution in prediction.events.items():
            means = np.asarray(metadata['conditional_event_means'][event], dtype=float)
            within = np.asarray(metadata['conditional_event_variances'][event], dtype=float)
            if means.shape != (10,) or within.shape != (10,) or not np.isfinite(means).all() or not np.isfinite(within).all():
                raise ValueError('Incomplete conditional event distribution')
            mean = float(probability @ means)
            variance = max(float(probability @ (within + means**2) - mean**2), 1e-8)
            source_mean = float((probability*means).sum())
            source_variance = max(float((probability*(within+means*means)).sum()-source_mean*source_mean), 1e-8)
            source_parameter = distribution_parameter(distribution.family, source_mean, source_variance)
            parameter = distribution_parameter(distribution.family, mean, variance)
            max_mean_error = max(max_mean_error, close(distribution.mean, mean, f'{event}/independent integrated mean'))
            max_parameter_error = max(max_parameter_error, close(distribution.dispersion, source_parameter, f'{event}/original arithmetic parameter'))
            if distribution.family == 'negative_binomial':
                expected_variance = mean + min(mean*mean/0.05, max(variance-mean, 1e-6))
            elif distribution.family == 'bernoulli':
                expected_variance = mean*(1-mean)
            else:
                expected_variance = variance
            error = close(distribution_variance(distribution), expected_variance, f'{event}/independent effective variance')
            max_variance_error = max(max_variance_error, error)
            if not np.isclose(distribution.dispersion, parameter, rtol=1e-10, atol=1e-8):
                reciprocal_roundoff.append(dict(fixture_id=prediction.fixture_id, player_id=prediction.player_id,
                    team=prediction.team, event=event, family=distribution.family, stored_parameter=distribution.dispersion,
                    independent_parameter=parameter, original_arithmetic_parameter=source_parameter,
                    source_mean=source_mean, independent_mean=mean, source_variance=source_variance,
                    independent_variance=variance, effective_variance_error=error))
    return dict(rows=len(forecasts), maximum_mean_error=max_mean_error, maximum_parameter_error=max_parameter_error,
                maximum_effective_variance_error=max_variance_error, reciprocal_roundoff_cases=len(reciprocal_roundoff),
                reciprocal_roundoff_evidence=json.dumps(reciprocal_roundoff, sort_keys=True, allow_nan=False))


def verify_blend(empirical, tree, blended, configuration):
    maximum = 0.0
    for left, right, output in zip(empirical, tree, blended):
        if set(output.events) != set(left.events) | set(right.events):
            raise ValueError('P3 changed component event support')
        for event in ('minutes', *output.events):
            a = left.minutes if event == 'minutes' else left.events.get(event)
            b = right.minutes if event == 'minutes' else right.events.get(event)
            actual = output.minutes if event == 'minutes' else output.events[event]
            if a is None or b is None:
                only = a if b is None else b
                maximum = max(maximum, close([actual.mean, actual.dispersion], [only.mean, only.dispersion], 'single-component blend'))
                continue
            weight = configuration['event_weights_v4'].get(event, configuration['default_weight_v4'])
            mean = (1-weight)*a.mean + weight*b.mean
            variance = ((1-weight)*(distribution_variance(a)+(a.mean-mean)**2)
                        + weight*(distribution_variance(b)+(b.mean-mean)**2))
            parameter = 1.0 if actual.family == 'bernoulli' else (
                max(mean**2/max(variance-mean, 1e-6), 0.05) if actual.family == 'negative_binomial' else max(variance, 1e-6))
            maximum = max(maximum, close([actual.mean, actual.dispersion], [mean, parameter], f'fixed P3/{event}'))
    return dict(rows=len(blended), maximum_blend_error=maximum)
