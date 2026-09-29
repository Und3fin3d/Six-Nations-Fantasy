import argparse
import json
import os

import numpy as np
import pandas as pd

from model.role_selection import SelectionRules, optimise_roles, verify_selection
from model.unified.rolling_eval import KEY, expected_points
from research.conditional_duration_inputs import digest, write_json
from research.conditional_duration_metrics import audited_role_measurement, duration_mixture_floor, load_outcome_audit, only_metres_points
from research.decision_role_policy import measure, signature
from research.duration_decision_summary import sensitivity_receipt, summarize_decisions, summarize_raw
from research.duration_evidence_validation import CONTROL_ENGINES, RUN, close, prepared_jobs, read_forecasts, read_units, verify_blend, verify_conditional, verify_hashes, verify_study
from research.ncr_rubric_completion import complete_native_points, expected_ncr_points, score_ncr_events


ROLE_ROOT = 'data/unified/decision_continuation_2026-09-26/run-36257541020/role_policy'
CONFIG = 'data/unified/p3_hillclimb/config.json'
METRICS = ['n', 'n_labelled', 'unknown_selected', 'total', 'xv', 'captain_extra', 'super_sub', 'mae', 'bias', 'rmse']
RUBRICS = ('original_v2', 'missing_attack_v3')


def rules_for(slate):
    return SelectionRules(budget=np.inf, max_nation=4, max_hemi=None) if slate.competition == 'six_nations' else SelectionRules()


def cohort(unit):
    return 'friendly15' if unit['kind'] == 'friendly' else f'{unit["slate"].competition}_{unit["slate"].season}'


def evaluate_vector(slate, engine, points, audit, *, saved=None, selected=None):
    points = np.asarray(points, dtype=float)
    if points.shape != slate.actual.shape or not np.isfinite(points).all():
        raise ValueError('The decision comparison requires complete finite utility pools')
    rules = rules_for(slate)
    if selected is None:
        selected, diagnostics = optimise_roles(slate.pool, points, points, 3*points, rules=rules)
    else:
        diagnostics = dict(mode='retained_verified_policy', **verify_selection(selected, rules))
    measured, selected = measure(slate, engine, points, selected)
    measured, selected = audited_role_measurement(slate, measured, selected, audit)
    if saved is not None:
        old = pd.read_csv(f'{saved}/{engine}_squad.csv')
        if not signature(selected).equals(signature(old)):
            raise ValueError(f'{slate.name}/{engine}: saved legal squad does not reproduce')
        pool = pd.read_csv(f'{saved}/{engine}_pool.csv')
        if not np.array_equal(pool.id, slate.pool.id):
            raise ValueError('Saved prediction pool changed candidate order')
        close(pool.predicted, points, f'{slate.name}/{engine}/point forecasts')
        close(pool.actual, slate.actual, f'{slate.name}/{engine}/official outcomes')
        metrics = pd.read_csv(f'{saved}/fantasy_metrics.csv').set_index('engine').loc[engine]
        close([measured[name] for name in METRICS], metrics[METRICS], f'{slate.name}/{engine}/squad metrics')
    return measured, selected, diagnostics


def reference_data():
    completion = json.load(open(f'{ROLE_ROOT}/completion.json'))
    verify_hashes(ROLE_ROOT, completion['output_hashes'])
    metrics = pd.read_csv(f'{ROLE_ROOT}/metrics.csv')
    squads = pd.read_csv(f'{ROLE_ROOT}/squads.csv')
    pools = pd.read_csv(f'{ROLE_ROOT}/pools.csv', dtype={f'key_{key}': str for key in KEY})
    if metrics.duplicated(['slate', 'engine']).any() or len(metrics) != 91:
        raise ValueError('All original role-policy results must be retained')
    return metrics, squads, pools


def reference_rows(slate, empirical, references, audit, directory):
    old_metrics, old_squads, old_pools = references
    pool = old_pools[old_pools.slate.eq(slate.name)].reset_index(drop=True)
    if len(pool) != len(slate.pool) or not np.array_equal(pool.id, slate.pool.id):
        raise ValueError('Native reference candidate pool changed')
    for key in KEY:
        if not pool[f'key_{key}'].equals(slate.candidates[key].astype(str).reset_index(drop=True)):
            raise ValueError('Native reference identity alignment changed')
    close(pool.actual, slate.actual, 'Native reference original outcomes')
    records, selections = [], []
    for old in old_metrics[old_metrics.slate.eq(slate.name)].itertuples(index=False):
        selected = old_squads[old_squads.slate.eq(slate.name) & old_squads.engine.eq(old.engine)].copy()
        point_name = 'p3_robust_native' if old.engine == 'p3_legal_joint_role_transfer' else (
            'incumbent' if old.engine.startswith('incumbent_legal') else old.engine)
        measured, selected, _ = evaluate_vector(slate, old.engine, pool[point_name].to_numpy(float), audit, selected=selected)
        close([measured[name] for name in METRICS], [getattr(old, name) for name in METRICS], 'Retained reference metrics')
        records.append(dict(measured, rubric='original_v2'))
        if old.engine in ('incumbent', 'incumbent_legal_sequential'):
            if slate.competition == 'ncr':
                points = complete_native_points(pool.incumbent.to_numpy(float), empirical)
                new, chosen, diagnostic = evaluate_vector(slate, old.engine, points, audit)
                chosen.to_csv(f'{directory}/{old.engine}_completed_squad.csv', index=False)
                write_json(f'{directory}/{old.engine}_completed_selection.json', diagnostic)
                selections.append(signature(selected))
                records.append(dict(new, rubric='missing_attack_v3'))
            else:
                records.append(dict(measured, rubric='missing_attack_v3'))
    if slate.competition == 'ncr':
        if len(selections) != 2 or not selections[0].equals(selections[1]):
            raise ValueError('The declared NCR incumbent policies no longer coincide')
        pool[['id', 'incumbent']].assign(completed_incumbent=complete_native_points(pool.incumbent, empirical)).to_csv(
            f'{directory}/native_completed_pool.csv', index=False)
    return records


def point_alternatives(unit, forecasts, configuration, rubric):
    slate = unit['slate']
    values = {}
    for name, predictions in forecasts.items():
        values[name] = expected_ncr_points(predictions) if slate.competition == 'ncr' and rubric == 'missing_attack_v3' else (
            expected_points(predictions, slate.competition, season=slate.season))
    conditional = next(name for name in forecasts if name.startswith('conditional_duration_'))
    blended = 'p3_' + conditional
    mixture = duration_mixture_floor(forecasts[conditional])
    empirical_floor = only_metres_points(forecasts['robust_empirical_event'])
    weight = configuration['event_weights_v4'].get('metres', configuration['default_weight_v4'])
    for name, floor in ((conditional, mixture), (blended, (1-weight)*empirical_floor+weight*mixture)):
        original_floor = only_metres_points(forecasts[name])
        if np.any(np.abs(floor-original_floor) > 1+1e-7):
            raise ValueError('Equal-mean metre-floor sensitivity exceeded one point per player')
        if slate.competition == 'ncr':
            adjusted = expected_ncr_points(forecasts[name], metre_floor=floor) if rubric == 'missing_attack_v3' else values[name]
        else:
            adjusted = values[name] + floor-original_floor
        values[name+'_mixture_floor'] = adjusted
    return values


def verify_ncr_additions(forecasts):
    rows = []
    for engine, predictions in forecasts.items():
        old, new = expected_ncr_points(predictions, version='ncr_front_row_v2'), expected_ncr_points(predictions)
        floor = only_metres_points(predictions)
        kicks = np.array([prediction.events['fifty_22'].mean for prediction in predictions])
        maximum = close(new-old, floor+2*kicks, 'NCR expected missing terms')
        for prediction in predictions:
            events = {event: np.array([distribution.mean]) for event, distribution in prediction.events.items()}
            deterministic = score_ncr_events(events, position=prediction.position) - score_ncr_events(
                events, position=prediction.position, version='ncr_front_row_v2')
            maximum = max(maximum, close(deterministic, np.floor(events['metres']/10)+2*events['fifty_22'], 'NCR deterministic missing terms'))
        rows.append(dict(engine=engine, rows=len(predictions), maximum_addition_error=maximum,
                         mean_metres_addition=float(floor.mean()), mean_fifty22_addition=float((2*kicks).mean())))
    return rows


def player_decisions(unit, predictions, points, selected, domain, rubric, engine):
    slate = unit['slate']
    chosen = set(selected.id)
    captain = set(selected.loc[selected.is_capt, 'id'])
    supersub = set(selected.loc[selected.is_sub, 'id'])
    observed = unit['raw_labels'].available__minutes.fillna(False).to_numpy(bool)
    return [dict(slate=slate.name, domain=domain, rubric=rubric, engine=engine,
                 id=player.id, name=player.name, team=player.team, pos=player.pos, status=player.status,
                 predicted=float(points[index]), actual=float(slate.actual[index]),
                 predicted_minutes=float(predictions[index].minutes.mean),
                 actual_minutes=unit['raw_labels'].minutes.iloc[index], minutes_observed=bool(observed[index]),
                 selected=player.id in chosen, captain=player.id in captain, supersub=player.id in supersub)
            for index, player in enumerate(slate.pool.itertuples(index=False))]


def review_fantasy(unit, forecasts, directory, output, domain, configuration, audit, references):
    slate = unit['slate']
    os.makedirs(output, exist_ok=False)
    records, diagnostics, prediction_rows = [], [], []
    old_empirical = pd.read_csv(f'{directory}/empirical_fantasy_corrected_pool.csv')
    if not np.array_equal(old_empirical.id, slate.pool.id):
        raise ValueError('Corrected-history native pool changed')
    for rubric in RUBRICS:
        points_by_engine = point_alternatives(unit, forecasts, configuration, rubric)
        native = old_empirical.predicted.to_numpy(float)
        if slate.competition == 'ncr' and rubric == 'missing_attack_v3':
            native = complete_native_points(native, forecasts['empirical_event'])
        points_by_engine['empirical_fantasy_corrected'] = native
        for engine, points in points_by_engine.items():
            saved = directory if rubric == 'original_v2' and not (slate.competition == 'ncr' and engine.endswith('_mixture_floor')) else None
            result, selected, diagnostic = evaluate_vector(slate, engine, points, audit, saved=saved)
            records.append(dict(result, rubric=rubric, domain=domain))
            diagnostics.append(dict(slate=slate.name, engine=engine, rubric=rubric, domain=domain, **diagnostic))
            if rubric == 'missing_attack_v3' and slate.competition == 'ncr':
                selected.to_csv(f'{output}/{engine}_completed_squad.csv', index=False)
            if engine in forecasts:
                prediction_rows.extend(player_decisions(unit, forecasts[engine], points, selected, domain, rubric, engine))
    records.extend(dict(row, domain=domain) for row in reference_rows(slate, forecasts['empirical_event'], references, audit, output))
    pd.DataFrame(records).to_csv(f'{output}/metrics.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(f'{output}/player_decisions.csv', index=False)
    write_json(f'{output}/selection_verification.json', diagnostics)
    return records


def run(args):
    from model_env_preflight import check
    failures = check()
    if failures:
        raise RuntimeError('; '.join(failures))
    os.makedirs(args.output, exist_ok=False)
    jobs, prepared_manifest = prepared_jobs(args.prepared)
    configuration = json.load(open(CONFIG))
    audit, references = load_outcome_audit(), reference_data()
    status, verification, raw, metrics, calibration, scoring = [], [], [], [], [], []
    sources = [__file__, 'research/duration_evidence_validation.py', 'research/duration_decision_summary.py',
               'research/ncr_rubric_completion.py', 'research/conditional_duration_metrics.py', CONFIG]
    source_hashes = {path: digest(path) for path in sources}
    for job in jobs:
        artifact = f'{args.artifacts}/conditional-duration-domain-{job["job"]}-{RUN}'
        available = all(os.path.isfile(f'{artifact}/study-{domain}/completion.json') for domain in ('bounded', 'source_tail'))
        status.append(dict(job=job['job'], units=';'.join(job['units']), paired_complete=available))
        pd.DataFrame(status).to_csv(f'{args.output}/execution_coverage.csv', index=False)
        if not available:
            continue
        units = read_units(args.prepared, job)
        controls = {}
        for domain in ('bounded', 'source_tail'):
            study = f'{artifact}/study-{domain}'
            manifest, engines = verify_study(study, job, domain)
            if manifest['prepared_manifest_sha256'] != digest(f'{args.prepared}/manifest.json'):
                raise ValueError('The fitted models used a different prepared source store')
            for unit in units:
                directory = f'{study}/units/{unit["name"]}'
                output = f'{args.output}/units/{domain}/{unit["name"]}'
                forecasts = {}
                for engine in engines:
                    predictions, rows, error = read_forecasts(directory, engine, unit)
                    forecasts[engine] = predictions
                    raw.append(rows.assign(domain=domain, cohort=cohort(unit)))
                    verification.append(dict(job=job['job'], unit=unit['name'], domain=domain, engine=engine, raw_metric_max_error=error))
                    if engine in CONTROL_ENGINES:
                        key = (unit['name'], engine)
                        value = digest(f'{directory}/{engine}.jsonl')
                        if domain == 'bounded':
                            controls[key] = value
                        elif controls.get(key) != value:
                            raise ValueError('An original control changed across the paired domain experiments')
                conditional = f'conditional_duration_{domain}_v2'
                verification.append(dict(job=job['job'], unit=unit['name'], domain=domain, engine=conditional,
                                         **verify_conditional(forecasts[conditional])))
                for tree, blend in ((conditional, 'p3_'+conditional), ('v4_corrected_full_refit', 'p3_corrected_full_refit')):
                    verification.append(dict(job=job['job'], unit=unit['name'], domain=domain, engine=blend,
                        **verify_blend(forecasts['robust_empirical_event'], forecasts[tree], forecasts[blend], configuration)))
                calibration.append(pd.read_csv(f'{directory}/duration_calibration.csv').assign(domain=domain, cohort=cohort(unit)))
                if unit['kind'] == 'fantasy':
                    metrics.extend(review_fantasy(unit, forecasts, directory, output, domain, configuration, audit, references))
                    if unit['slate'].competition == 'ncr':
                        scoring.extend(dict(row, unit=unit['name'], domain=domain) for row in verify_ncr_additions(forecasts))
                print(f'{job["job"]}/{domain}/{unit["name"]}: complete pools and calculations verified', flush=True)
    pd.DataFrame(verification).to_csv(f'{args.output}/verification.csv', index=False)
    if raw:
        raw = pd.concat(raw, ignore_index=True)
        raw.to_csv(f'{args.output}/raw_metrics.csv', index=False)
    pd.DataFrame(metrics).to_csv(f'{args.output}/metrics.csv', index=False)
    if calibration:
        pd.concat(calibration, ignore_index=True).to_csv(f'{args.output}/duration_calibration.csv', index=False)
    pd.DataFrame(scoring).to_csv(f'{args.output}/ncr_scoring_verification.csv', index=False)
    complete = all(row['paired_complete'] for row in status)
    if complete:
        summarize_raw(raw, args.output)
        tables = summarize_decisions(pd.DataFrame(metrics), args.output)
        sensitivity_receipt(tables['admission'], args.output)
        print(tables['admission'].query('comparator in ["incumbent", "incumbent_legal_sequential"]').to_string(index=False), flush=True)
    if source_hashes != {path: digest(path) for path in sources}:
        raise ValueError('Review implementation changed during execution')
    write_json(f'{args.output}/completion.json', dict(source_hashes=source_hashes, fitted_run=RUN,
        prepared_manifest_sha256=digest(f'{args.prepared}/manifest.json'), paired_locks_complete=sum(row['paired_complete'] for row in status),
        required_locks=27, full_comparison=complete, admission_computed=complete, objective_completed=False,
        unknown_outcomes='Retained; all selected-role outcomes attached after selection',
        uncertainty='Reused historical evidence, correlated small samples and multiple candidate history; not confirmatory future superiority',
        objective='active_unfinished'))
    if not complete and not args.allow_partial:
        raise RuntimeError('Partial evidence checkpoint saved; full registered comparison not yet executable')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    run(parser.parse_args())
