import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.raw_benchmark.config import STABLE_EVENTS
from model.unified.rolling_eval import KEY
from model.unified.scoring import NationsChampionshipScorer, SixNationsScorer
from research.conditional_duration_inputs import digest, write_json
from research.duration_evidence_validation import prepared_jobs, read_units


FORWARDS = {'Prop', 'Hooker', 'Lock', 'Loose Forward'}
POSITIONS = FORWARDS | {'Scrum Half', 'Fly Half', 'Centre', 'Back Three'}


def core_spec(competition, season, rubric):
    if rubric not in ('original_v2', 'missing_attack_v3'):
        raise ValueError('Unknown registered rubric')
    if competition == 'six_nations':
        scorer = SixNationsScorer(season=season)
        events = set(scorer.weights) & set(STABLE_EVENTS)
        events.update(('tries', 'metres'))
    elif competition == 'ncr':
        scorer = NationsChampionshipScorer(version='ncr_front_row_v2')
        events = set(scorer.weights) & set(STABLE_EVENTS)
        if rubric == 'missing_attack_v3':
            events.add('metres')
    else:
        raise ValueError('Core utility supervision is only defined for the declared fantasy rubrics')
    return scorer, tuple(sorted(events))


def core_targets(frame, competition, season, rubric):
    scorer, events = core_spec(competition, season, rubric)
    valid = frame.position.isin(POSITIONS).to_numpy(bool)
    for event in events:
        values = pd.to_numeric(frame[event], errors='coerce').to_numpy(float)
        valid &= frame[f'available__{event}'].fillna(False).to_numpy(bool) & np.isfinite(values) & (values >= 0)
    target = pd.Series(np.nan, index=frame.index, dtype=float)
    for position in POSITIONS:
        indices = frame.index[valid & frame.position.eq(position).to_numpy()]
        if not len(indices):
            continue
        recorded = {event: frame.loc[indices, event].to_numpy(float) for event in events}
        values = scorer.score_samples(recorded, is_forward=position in FORWARDS, position=position)
        if competition == 'ncr' and rubric == 'missing_attack_v3':
            values = values + np.floor(recorded['metres']/10)
        target.loc[indices] = values
    if not np.isfinite(target.loc[valid]).all():
        raise ValueError('Complete core event supervision produced nonfinite utility labels')
    return target, valid, events


def key_hash(frame):
    return hashlib.sha256(frame[KEY].to_csv(index=False).encode()).hexdigest()


def run(args):
    os.makedirs(args.output, exist_ok=False)
    jobs, manifest = prepared_jobs(args.prepared)
    path = f'{args.prepared}/training_features.pkl'
    if digest(path) != manifest['outputs']['training_features.pkl']:
        raise ValueError('Prior-only prepared training features changed')
    frame = pd.read_pickle(path)
    outputs, support = [], []
    for job in jobs:
        for unit in read_units(args.prepared, job):
            if unit['kind'] != 'fantasy':
                continue
            slate = unit['slate']
            train = past_matches(frame, unit['cutoff']).sort_values(['date', 'fixture_id', 'team', 'player_id'])
            forbidden = set(unit['source_fixture_ids']) | set(unit['candidates'].fixture_id.astype(str))
            if set(train.fixture_id.astype(str)) & forbidden:
                raise ValueError('Evaluation fixture entered utility supervision')
            rubrics = ('original_v2', 'missing_attack_v3') if slate.competition == 'ncr' else ('original_v2',)
            for rubric in rubrics:
                target, valid, events = core_targets(train, slate.competition, slate.season, rubric)
                observed = target.loc[valid]
                for event in events:
                    own = train[f'available__{event}'].fillna(False).to_numpy(bool) & train[event].notna().to_numpy()
                    support.append(dict(unit=unit['name'], rubric=rubric, event=event,
                        original_event_rows=int(own.sum()), complete_core_rows=int(valid.sum()),
                        support_identical=bool(np.array_equal(own, valid)), excluded_for_joint_support=int((own & ~valid).sum()),
                        joint_support_keys_sha256=key_hash(train.loc[valid])))
                record = dict(unit=unit['name'], job=job['job'], competition=slate.competition, season=slate.season,
                    rubric=rubric, cutoff=unit['cutoff'].isoformat(), rows=len(train), supervised_rows=int(valid.sum()),
                    unknown_core_rows=int((~valid).sum()), core_events=';'.join(events),
                    negative_targets=int(observed.lt(0).sum()), zero_targets=int(observed.eq(0).sum()),
                    mean=float(observed.mean()), variance=float(observed.var(ddof=0)),
                    minimum=float(observed.min()), median=float(observed.median()),
                    percentile90=float(observed.quantile(0.90)), percentile99=float(observed.quantile(0.99)), maximum=float(observed.max()),
                    training_keys_sha256=key_hash(train), supervised_keys_sha256=key_hash(train.loc[valid]))
                outputs.append(record)
                grouped = train.loc[valid, ['competition_level', 'position', 'started']].copy()
                grouped['utility_core'] = observed
                grouped.groupby(['competition_level', 'position', 'started'], dropna=False).utility_core.agg(
                    ['size', 'mean', 'std', 'min', 'max']).reset_index().to_csv(
                        f'{args.output}/{unit["name"]}_{rubric}_supervision.csv', index=False)
                excluded = train.loc[~valid, KEY + ['date', 'competition_level', 'position', 'started']].copy()
                excluded['missing_core_events'] = [
                    ';'.join(event for event in events if not bool(train.at[index, f'available__{event}'])
                             or not np.isfinite(float(train.at[index, event])) or float(train.at[index, event]) < 0)
                    for index in excluded.index]
                excluded.to_csv(f'{args.output}/{unit["name"]}_{rubric}_excluded.csv', index=False)
    pd.DataFrame(outputs).to_csv(f'{args.output}/supervision.csv', index=False)
    pd.DataFrame(support).to_csv(f'{args.output}/matched_support_requirements.csv', index=False)
    if digest(path) != manifest['outputs']['training_features.pkl']:
        raise ValueError('Read-only feature cache changed during supervision audit')
    write_json(f'{args.output}/manifest.json', dict(source_sha256=digest(__file__),
        training_features_sha256=manifest['outputs']['training_features.pkl'], fitted_candidates=0,
        source_cutoffs='Existing past_matches availability delay and per-unit exclusion of evaluation fixture identities',
        targets='Signed observed stable-event point sum only; not complete official points. Extended-event forecasts would remain explicit separate contributions.',
        reason='Both duration primaries failed the complete 2025 development comparison; XV/captain errors remain. Audit whether direct conditional-mean utility supervision can be fitted without zero-filling missing events.',
        support_control='Every non-identical support mask requires an identically supported raw-event control before attributing gain to the utility loss.',
        raw_friendly15='A future decision-only utility head would retain and report the exact fixed raw forecast cohort, not relabel its raw accuracy as newly improved.',
        registrations_changed=False, candidate_selected=False, objective='active_unfinished'))
    print(pd.DataFrame(outputs).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
