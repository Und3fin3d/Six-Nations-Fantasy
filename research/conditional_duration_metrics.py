import dataclasses
import hashlib
import json
import os

import numpy as np
import pandas as pd

from model.role_selection import SelectionRules, optimise_roles
from model.unified.conditional_duration import ConditionalDurationGBDT
from model.unified.contracts import EventDistribution
from model.unified.raw_benchmark.config import EXTENDED_EVENTS, STABLE_EVENTS
from model.unified.rolling_eval import KEY, expected_points
from research.decision_role_policy import measure


TARGETS = ('minutes', *STABLE_EVENTS, *EXTENDED_EVENTS)
OUTCOME_AUDIT = 'data/unified/decision_continuation_2026-09-26/run-36272929019/outcomes/full_pool_outcomes.csv'
OUTCOME_BLOB = '6cd79c7f10252d73d95f29a937d3dda2fe212172'


def load_outcome_audit():
    content = open(OUTCOME_AUDIT, 'rb').read()
    blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
    if blob != OUTCOME_BLOB:
        raise ValueError('The independently reviewed actual-role outcome ledger changed')
    frame = pd.read_csv(OUTCOME_AUDIT)
    if len(frame) != 2138 or frame.duplicated(['slate', 'id']).any():
        raise ValueError('The actual-role ledger does not cover the complete registered pools')
    return frame


def raw_measurements(unit, engine, forecasts):
    labels = unit['raw_labels']
    rows = []
    for event in TARGETS:
        actual = pd.to_numeric(labels[event], errors='coerce').to_numpy(float)
        observed = labels[f'available__{event}'].fillna(False).to_numpy(bool) & np.isfinite(actual)
        predicted = np.array([prediction.minutes.mean if event == 'minutes' else
                              prediction.events[event].mean if event in prediction.events else np.nan
                              for prediction in forecasts])
        scored = observed & np.isfinite(predicted)
        error = predicted[scored] - actual[scored]
        complete = int(scored.sum()) == int(observed.sum())
        rows.append(dict(unit=unit['name'], kind=unit['kind'], engine=engine, event=event,
                         rows=len(labels), observed=int(observed.sum()), predicted=int(np.isfinite(predicted).sum()),
                         scored=int(scored.sum()), missing_prediction=int((observed & ~np.isfinite(predicted)).sum()),
                         absolute_error_sum=float(np.abs(error).sum()), squared_error_sum=float(np.square(error).sum()),
                         error_sum=float(error.sum()),
                         mae=float(np.abs(error).mean()) if len(error) and complete else np.nan,
                         bias=float(error.mean()) if len(error) and complete else np.nan,
                         rmse=float(np.sqrt(np.square(error).mean())) if len(error) and complete else np.nan))
    return rows


def duration_measurements(unit, forecasts, *, open_tail=False):
    labels = unit['raw_labels']
    minutes = pd.to_numeric(labels.minutes, errors='coerce').to_numpy(float)
    observed = labels.available__minutes.fillna(False).to_numpy(bool) & np.isfinite(minutes)
    actual_class = np.full(len(minutes), -1)
    labeler = ConditionalDurationGBDT._duration_labels
    if open_tail:
        from research.duration_domain_model import DurationDomainGBDT
        labeler = DurationDomainGBDT._duration_labels
    actual_class[observed] = labeler(minutes[observed])
    rows = []
    for index, prediction in enumerate(forecasts):
        probability = np.array(prediction.metadata['duration_probabilities'])
        label = actual_class[index]
        record = dict(unit=unit['name'], kind=unit['kind'], fixture_id=prediction.fixture_id,
                      player_id=prediction.player_id, team=prediction.team, position=prediction.position,
                      started=bool(unit['candidates'].started.iloc[index]), observed=bool(observed[index]),
                      actual_minutes=float(minutes[index]), predicted_minutes=float(prediction.minutes.mean),
                      actual_class=int(label), probability_zero=float(probability[0]),
                      probability_under_10=float(probability[:2].sum()), probability_at_least_30=float(probability[4:].sum()))
        if open_tail:
            record['above_original_training_domain'] = bool(observed[index] and minutes[index] > 80)
        if observed[index]:
            onehot = np.eye(10)[label]
            record['multiclass_brier'] = float(np.square(probability - onehot).sum())
            record['log_loss'] = float(-np.log(max(probability[label], 1e-12)))
            record['long_appearance_brier'] = float((probability[4:].sum() - (minutes[index] >= 30)) ** 2)
        rows.append(record)
    return rows


def audited_role_measurement(slate, metrics, selected, audit):
    full = audit[audit.slate.eq(slate.name)].set_index('id').reindex(slate.pool.id)
    if len(full) != len(slate.pool) or full.role_outcome_known.isna().any():
        raise ValueError('Actual-role ledger is incomplete for this candidate pool')
    if not np.allclose(full.official_points, slate.actual, rtol=0, atol=0, equal_nan=True):
        raise ValueError('Actual-role ledger changed the original official point labels')
    selected = selected.copy()
    for column in ('actual_role_supersub_points', 'possible_min', 'possible_max', 'role_outcome_known', 'source_conflict'):
        selected[column] = selected.id.map(full[column])
    sub = selected.loc[selected.is_sub]
    if len(sub) != 1:
        raise ValueError('Exactly one selected super-sub is required by the outcome contract')
    metrics['assigned_role_total'] = metrics['total']
    metrics['assigned_role_super_sub'] = metrics['super_sub']
    metrics['super_sub'] = sub.actual_role_supersub_points.iloc[0]
    metrics['total'] = metrics['xv'] + metrics['captain_extra'] + metrics['super_sub']
    metrics['unknown_role_selected'] = int(not bool(sub.role_outcome_known.iloc[0]))
    metrics['selected_source_conflicts'] = int(selected.source_conflict.sum())
    metrics['total_lower_role_bound'] = metrics['xv'] + metrics['captain_extra'] + sub.possible_min.iloc[0]
    metrics['total_upper_role_bound'] = metrics['xv'] + metrics['captain_extra'] + sub.possible_max.iloc[0]
    metrics['outcome_contract'] = 'actual_role_entry_v1'
    return metrics, selected


def select_points(unit, engine, values, directory, *, outcome_audit=None):
    slate = unit['slate']
    values = np.asarray(values, dtype=float)
    if values.shape != slate.actual.shape or not np.isfinite(values).all():
        raise ValueError(f'{slate.name}/{engine}: incomplete finite point forecasts')
    rules = SelectionRules(budget=np.inf, max_nation=4, max_hemi=None) if slate.competition == 'six_nations' else SelectionRules()
    selected, diagnostics = optimise_roles(slate.pool, values, values, values * 3, rules=rules)
    metrics, selected = measure(slate, engine, values, selected)
    if outcome_audit is not None:
        metrics, selected = audited_role_measurement(slate, metrics, selected, outcome_audit)
    pool = slate.pool.copy()
    pool['predicted'], pool['actual'] = values, slate.actual
    for key in KEY:
        pool[f'key_{key}'] = slate.candidates[key].astype(str).to_numpy()
    pool.to_csv(f'{directory}/{engine}_pool.csv', index=False)
    selected.to_csv(f'{directory}/{engine}_squad.csv', index=False)
    with open(f'{directory}/{engine}_selection.json', 'w') as handle:
        json.dump(diagnostics, handle, indent=2, allow_nan=False)
    return metrics


def only_metres_points(predictions):
    metres_only = [dataclasses.replace(prediction, events={'metres': prediction.events['metres']}) for prediction in predictions]
    return expected_points(metres_only, 'six_nations', season=2026)


def duration_mixture_floor(conditional):
    components, weights = [], []
    counts = []
    for prediction in conditional:
        probability = prediction.metadata['duration_probabilities']
        means = prediction.metadata['conditional_event_means']['metres']
        variances = prediction.metadata['conditional_event_variances']['metres']
        count = 0
        for weight, mean, variance in zip(probability, means, variances):
            if weight <= 0:
                continue
            components.append(dataclasses.replace(prediction, events={'metres': EventDistribution('lognormal', mean, variance)}))
            weights.append(weight)
            count += 1
        counts.append(count)
    values = only_metres_points(components)
    result, start = [], 0
    for count in counts:
        end = start + count
        result.append(float(np.dot(values[start:end], weights[start:end])))
        start = end
    return np.array(result)


def floor_sensitivity(unit, forecasts_by_engine, metre_tree_weight,
                      conditional_engine='conditional_duration_v1', blend_engine='p3_conditional_duration_v1'):
    if unit['slate'].competition != 'six_nations':
        return {}, []
    conditional = forecasts_by_engine[conditional_engine]
    raw_floor = duration_mixture_floor(conditional)
    empirical_floor = only_metres_points(forecasts_by_engine['robust_empirical_event'])
    alternatives, audit = {}, []
    for engine in (conditional_engine, blend_engine):
        predictions = forecasts_by_engine[engine]
        alternative_floor = raw_floor if engine == conditional_engine else (
            (1 - metre_tree_weight) * empirical_floor + metre_tree_weight * raw_floor)
        ordinary_floor = only_metres_points(predictions)
        means = np.array([prediction.events['metres'].mean for prediction in predictions])
        if np.any(np.abs(alternative_floor - ordinary_floor) > 1 + 1e-7):
            raise ValueError('Declared equal-mean floor expectation sensitivity bound failed')
        for values in (alternative_floor, ordinary_floor):
            if np.any(values > means / 10 + 1e-7) or np.any(values < means / 10 - 1 - 1e-7):
                raise ValueError('Floor expectation does not match the recorded metre mean')
        points = expected_points(predictions, 'six_nations', season=unit['slate'].season)
        alternatives[f'{engine}_mixture_floor'] = points - ordinary_floor + alternative_floor
        for index, prediction in enumerate(predictions):
            audit.append(dict(unit=unit['name'], engine=engine, player_id=prediction.player_id, team=prediction.team,
                              metres_mean=means[index], moment_matched_floor=ordinary_floor[index],
                              duration_mixture_floor=alternative_floor[index], delta=alternative_floor[index]-ordinary_floor[index]))
    return alternatives, audit


def write_forecasts(unit, engine, forecasts, directory):
    observed_keys = pd.DataFrame([{key: getattr(prediction, key) for key in KEY} for prediction in forecasts])
    expected_keys = unit['candidates'][KEY].astype(str).reset_index(drop=True)
    if not observed_keys.equals(expected_keys) or len(forecasts) != len(expected_keys):
        raise ValueError('Raw forecast candidate keys changed')
    path = f'{directory}/{engine}.jsonl'
    with open(path, 'w') as handle:
        for forecast in forecasts:
            handle.write(json.dumps(forecast.to_dict(), allow_nan=False, sort_keys=True) + '\n')
    return path
