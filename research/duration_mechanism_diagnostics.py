import argparse
import dataclasses
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from model.unified.contracts import EventDistribution
from model.unified.raw_benchmark.blend import _blend_distribution
from model.unified.rolling_eval import expected_points
from research.conditional_duration_inputs import digest, write_json
from research.duration_domain_model import DurationDomainGBDT
from research.duration_evidence_validation import RUN, distribution_parameter, prepared_jobs, read_forecasts, read_units
from research.ncr_rubric_completion import expected_ncr_points


CONFIG = 'data/unified/p3_hillclimb/config.json'


def long_probability(prediction):
    if 'duration_probabilities' in prediction.metadata:
        return float(sum(prediction.metadata['duration_probabilities'][4:]))
    minutes = prediction.minutes
    if minutes.mean <= 0:
        return 0.0
    sigma2 = np.log1p(minutes.dispersion/minutes.mean**2)
    return float(lognorm.sf(30, np.sqrt(sigma2), scale=np.exp(np.log(minutes.mean)-sigma2/2)))


def conditional_at_class(prediction, label):
    events = {}
    for event, original in prediction.events.items():
        mean = prediction.metadata['conditional_event_means'][event][label]
        variance = prediction.metadata['conditional_event_variances'][event][label]
        events[event] = EventDistribution(original.family, mean, distribution_parameter(original.family, mean, variance))
    return dataclasses.replace(prediction, events=events, metadata={'diagnostic': 'unavailable_at_lock_actual_duration_class'})


def blend_at_class(empirical, tree, configuration):
    events = {}
    for event in set(empirical.events) | set(tree.events):
        left, right = tree.events.get(event), empirical.events.get(event)
        if left is None or right is None:
            events[event] = right if left is None else left
        else:
            weight = configuration['event_weights_v4'].get(event, configuration['default_weight_v4'])
            events[event] = _blend_distribution(left, right, weight, target=event)
    return dataclasses.replace(tree, events=events)


def point_values(unit, predictions, rubric):
    slate = unit['slate']
    if slate.competition == 'ncr' and rubric == 'missing_attack_v3':
        return expected_ncr_points(predictions)
    return expected_points(predictions, slate.competition, season=slate.season)


def probability_rows(unit, forecasts, domain):
    labels = unit['raw_labels']
    minutes = pd.to_numeric(labels.minutes, errors='coerce').to_numpy(float)
    observed = labels.available__minutes.fillna(False).to_numpy(bool) & np.isfinite(minutes)
    rows = []
    for engine, predictions in forecasts.items():
        for index, prediction in enumerate(predictions):
            probability = long_probability(prediction)
            known = bool(observed[index])
            target = float(minutes[index] >= 30) if known else np.nan
            rows.append(dict(unit=unit['name'], kind=unit['kind'], domain=domain, engine=engine,
                cohort='friendly15' if unit['kind'] == 'friendly' else f'{unit["slate"].competition}_{unit["slate"].season}',
                fixture_id=prediction.fixture_id, player_id=prediction.player_id, team=prediction.team,
                position=prediction.position, started=bool(unit['candidates'].started.iloc[index]),
                observed=known, actual_minutes=float(minutes[index]), predicted_minutes=prediction.minutes.mean,
                probability_at_least_30=probability, actual_at_least_30=target,
                brier=(probability-target)**2 if known else np.nan,
                log_loss=-(target*np.log(max(probability, 1e-12))+(1-target)*np.log(max(1-probability, 1e-12))) if known else np.nan))
    return rows


def point_rows(unit, forecasts, domain, configuration):
    conditional_name = f'conditional_duration_{domain}_v2'
    tree = forecasts[conditional_name]
    empirical = forecasts['robust_empirical_event']
    minutes = pd.to_numeric(unit['raw_labels'].minutes, errors='coerce').to_numpy(float)
    observed = unit['raw_labels'].available__minutes.fillna(False).to_numpy(bool) & np.isfinite(minutes)
    indices = np.flatnonzero(observed)
    classes = DurationDomainGBDT._duration_labels(minutes[observed])
    hypothetical = [conditional_at_class(tree[index], int(label)) for index, label in zip(indices, classes)]
    blended = [blend_at_class(empirical[index], prediction, configuration) for index, prediction in zip(indices, hypothetical)]
    rows = []
    for name, oracle in ((conditional_name, hypothetical), ('p3_'+conditional_name, blended)):
        for rubric in ('original_v2', 'missing_attack_v3'):
            marginal = point_values(unit, forecasts[name], rubric)
            hindsight = np.full(len(tree), np.nan)
            hindsight[indices] = point_values(unit, oracle, rubric)
            for index, player in enumerate(unit['slate'].pool.itertuples(index=False)):
                actual = unit['slate'].actual[index]
                rows.append(dict(slate=unit['name'], domain=domain, engine=name, rubric=rubric,
                    cohort=f'{unit["slate"].competition}_{unit["slate"].season}',
                    id=player.id, name=player.name, team=player.team, pos=player.pos, status=player.status,
                    minutes_observed=bool(observed[index]), actual_minutes=float(minutes[index]),
                    predicted=marginal[index], actual=actual, oracle_duration_points=hindsight[index],
                    exposure_information_difference=hindsight[index]-marginal[index],
                    remaining_point_residual=actual-hindsight[index], total_point_residual=actual-marginal[index],
                    interpretation='Hindsight duration-bin diagnostic only; empirical blend component unchanged; never an eligible forecast or squad'))
    return rows


def auc_probability(group):
    positive = group.loc[group.actual_at_least_30.eq(1), 'probability_at_least_30'].to_numpy()
    negative = group.loc[group.actual_at_least_30.eq(0), 'probability_at_least_30'].to_numpy()
    if not len(positive) or not len(negative):
        return np.nan
    difference = positive[:, None] - negative[None, :]
    return float((np.sum(difference > 0)+0.5*np.sum(difference == 0))/difference.size)


def summaries(probability, points, output):
    records, reliability = [], []
    for keys, group in probability.groupby(['domain', 'cohort', 'engine', 'started']):
        known = group[group.observed]
        records.append(dict(zip(['domain', 'cohort', 'engine', 'started'], keys)) | dict(
            rows=len(group), observed=len(known), brier=float(known.brier.mean()), log_loss=float(known.log_loss.mean()),
            event_fraction=float(known.actual_at_least_30.mean()), mean_probability=float(known.probability_at_least_30.mean()),
            auc=auc_probability(known), minutes_mae=float((known.predicted_minutes-known.actual_minutes).abs().mean())))
        for decile in range(10):
            part = known[np.minimum(9, np.floor(known.probability_at_least_30*10)).eq(decile)]
            reliability.append(dict(zip(['domain', 'cohort', 'engine', 'started'], keys)) | dict(
                probability_decile=decile, rows=len(part), mean_probability=float(part.probability_at_least_30.mean()),
                event_fraction=float(part.actual_at_least_30.mean())))
    pd.DataFrame(records).to_csv(f'{output}/exposure_probability_summary.csv', index=False)
    pd.DataFrame(reliability).to_csv(f'{output}/exposure_reliability.csv', index=False)
    records = []
    for keys, group in points.groupby(['domain', 'cohort', 'engine', 'rubric', 'status']):
        valid = np.isfinite(group[['actual', 'predicted', 'oracle_duration_points']].to_numpy(float)).all(axis=1)
        known = group[valid]
        records.append(dict(zip(['domain', 'cohort', 'engine', 'rubric', 'status'], keys)) | dict(
            full_pool_rows=len(group), jointly_observed_rows=len(known), excluded_unknown_rows=int((~valid).sum()),
            marginal_mae=float((known.predicted-known.actual).abs().mean()),
            hindsight_duration_mae=float((known.oracle_duration_points-known.actual).abs().mean()),
            marginal_mse=float(((known.predicted-known.actual)**2).mean()),
            hindsight_duration_mse=float(((known.oracle_duration_points-known.actual)**2).mean()),
            exposure_difference_bias=float(known.exposure_information_difference.mean()),
            remaining_point_residual_bias=float(known.remaining_point_residual.mean()),
            interpretation='Matched observed-duration/point subset diagnostic, not deployable improvement or causal attribution'))
    pd.DataFrame(records).to_csv(f'{output}/hindsight_duration_summary.csv', index=False)


def run(args):
    os.makedirs(args.output, exist_ok=False)
    jobs, _ = prepared_jobs(args.prepared)
    configuration = json.load(open(CONFIG))
    probability, points, completed = [], [], []
    for job in jobs:
        artifact = f'{args.artifacts}/conditional-duration-domain-{job["job"]}-{RUN}'
        if not all(os.path.exists(f'{artifact}/study-{domain}/completion.json') for domain in ('bounded', 'source_tail')):
            continue
        for unit in read_units(args.prepared, job):
            for domain in ('bounded', 'source_tail'):
                directory = f'{artifact}/study-{domain}/units/{unit["name"]}'
                names = ('robust_empirical_event', 'v4_corrected_full_refit', 'p3_corrected_full_refit',
                         f'conditional_duration_{domain}_v2', f'p3_conditional_duration_{domain}_v2')
                forecasts = {name: read_forecasts(directory, name, unit)[0] for name in names}
                probability.extend(probability_rows(unit, forecasts, domain))
                if unit['kind'] == 'fantasy':
                    points.extend(point_rows(unit, forecasts, domain, configuration))
        completed.append(job['job'])
    probability, points = pd.DataFrame(probability), pd.DataFrame(points)
    probability.to_csv(f'{args.output}/exposure_probabilities.csv', index=False)
    points.to_csv(f'{args.output}/hindsight_duration_points.csv', index=False)
    if len(completed) == 27:
        summaries(probability, points, args.output)
    write_json(f'{args.output}/manifest.json', dict(source_sha256=digest(__file__), config_sha256=digest(CONFIG),
        fitted_run=RUN, completed_locks=completed, full_cohort=len(completed)==27,
        registered_mechanism='Exposure discrimination and conditional event adequacy, not a new fitted candidate',
        diagnostic='Actual duration bins are unavailable at the lock. No oracle squad is optimised. Unknown durations and points are retained.',
        baseline_probability='Existing marginal lognormal duration contract versus classifier mass at duration >=30, without recalibration',
        selection_or_fit_performed=False, candidate_admission=False, objective='active_unfinished'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
