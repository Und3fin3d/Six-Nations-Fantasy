import itertools

import numpy as np
import pandas as pd

from model.unified.raw_benchmark.config import STABLE_EVENTS
from research.decision_summary import admissions, paired_rows, paired_summary


REQUIRED = ('incumbent', 'incumbent_legal_sequential')
REFERENCES = (*REQUIRED, 'empirical_baseline', 'p3_robust_native', 'equal_incumbent',
              'v4_corrected_full_refit', 'v4_joint_support_control', 'v4_bounded_support_control',
              'p3_corrected_full_refit', 'empirical_fantasy_corrected', 'empirical_event', 'robust_empirical_event')


def influence_rows(pairs):
    rows = []
    for (candidate, comparator, competition), group in pairs.groupby(['candidate', 'comparator', 'competition']):
        group = group.sort_values('slate')
        values = group.delta_total.to_numpy(float)
        for count in (1, 2):
            for removed in itertools.combinations(range(len(group)), count):
                kept = np.delete(values, removed)
                if not len(kept):
                    continue
                rows.append(dict(candidate=candidate, comparator=comparator, competition=competition,
                    removed_count=count, removed_slates=';'.join(group.slate.iloc[list(removed)]),
                    remaining_rounds=len(kept), known_rounds=int(np.isfinite(kept).sum()),
                    remaining_total=float(kept.sum()), remaining_mean=float(kept.mean())))
    return pd.DataFrame(rows)


def summarize_decisions(metrics, output):
    all_pairs, all_admission, all_influence, all_seasons, all_competitions, balanced = [], [], [], [], [], []
    for (domain, rubric), table in metrics.groupby(['domain', 'rubric']):
        if table.duplicated(['engine', 'slate']).any() or table.groupby('engine').slate.nunique().ne(13).any():
            raise ValueError('Every reported decision method requires all thirteen unique slates')
        candidates = [name for name in table.engine.unique() if 'conditional_duration_' in name]
        references = [name for name in REFERENCES if name in set(table.engine)]
        if not set(REQUIRED).issubset(references):
            raise ValueError('Both required incumbent policies must remain in the comparison')
        pairs = paired_rows(table, [(candidate, reference) for candidate in candidates for reference in references])
        all_pairs.append(pairs.assign(domain=domain, rubric=rubric))
        all_admission.append(admissions(pairs).assign(domain=domain, rubric=rubric))
        all_influence.append(influence_rows(pairs).assign(domain=domain, rubric=rubric))
        seasons = paired_summary(pairs, ['competition', 'season', 'candidate', 'comparator'])
        all_seasons.append(seasons.assign(domain=domain, rubric=rubric))
        all_competitions.append(paired_summary(pairs, ['competition', 'candidate', 'comparator']).assign(domain=domain, rubric=rubric))
        for (candidate, comparator), group in seasons.groupby(['candidate', 'comparator']):
            six = group[group.competition.eq('six_nations')]
            balanced.append(dict(domain=domain, rubric=rubric, candidate=candidate, comparator=comparator,
                                 tournament_count=len(group), known_tournaments=int(group['mean'].notna().sum()),
                                 equal_tournament_mean=float(group['mean'].mean()) if len(group) == 3 and group['mean'].notna().all() else np.nan,
                                 equal_six_nations_season_mean=float(six['mean'].mean()) if len(six) == 2 and six['mean'].notna().all() else np.nan))
    tables = dict(paired_rounds=pd.concat(all_pairs, ignore_index=True),
                  admission=pd.concat(all_admission, ignore_index=True),
                  round_removal=pd.concat(all_influence, ignore_index=True),
                  paired_seasons=pd.concat(all_seasons, ignore_index=True),
                  paired_competitions=pd.concat(all_competitions, ignore_index=True),
                  equal_tournament_weight=pd.DataFrame(balanced))
    for name, frame in tables.items():
        frame.to_csv(f'{output}/{name}.csv', index=False)
    return tables


def summarize_raw(raw, output):
    if raw.duplicated(['domain', 'engine', 'unit', 'event']).any():
        raise ValueError('Duplicate raw evaluation cells')
    aggregate = []
    for keys, group in raw.groupby(['domain', 'cohort', 'engine', 'event']):
        observed, scored = int(group.observed.sum()), int(group.scored.sum())
        complete = group.missing_prediction.eq(0).all() and observed == scored and observed > 0
        aggregate.append(dict(zip(['domain', 'cohort', 'engine', 'event'], keys)) | dict(
            units=len(group), observed=observed, scored=scored, missing_prediction=int(group.missing_prediction.sum()),
            mae=float(group.absolute_error_sum.sum()/observed) if complete else np.nan,
            bias=float(group.error_sum.sum()/observed) if complete else np.nan,
            rmse=float(np.sqrt(group.squared_error_sum.sum()/observed)) if complete else np.nan,
            equal_unit_mae=float(group.mae.mean()) if group.mae.notna().all() else np.nan))
    pd.DataFrame(aggregate).to_csv(f'{output}/raw_event_summary.csv', index=False)
    counts = set(STABLE_EVENTS) - {'metres'}
    if len(counts) != 22:
        raise ValueError('Registered Friendly-15 headline event definition changed')
    rows = []
    for (domain, engine), group in raw[raw.kind.eq('friendly')].groupby(['domain', 'engine']):
        if group.unit.nunique() != 15:
            raise ValueError('Friendly-15 cannot be replaced by a partial cohort')
        core = group[group.event.isin(counts)]
        complete = len(core) == 330 and core.mae.notna().all() and core.missing_prediction.eq(0).all()
        row = dict(domain=domain, engine=engine, games=group.unit.nunique(),
                   count_event_cells=len(core), known_count_event_cells=int(core.mae.notna().sum()),
                   observed_count_labels=int(core.observed.sum()), scored_count_labels=int(core.scored.sum()),
                   equal_fixture_event_count_mae=float(core.mae.mean()) if complete else np.nan)
        for event in ('metres', 'minutes'):
            part = group[group.event.eq(event)]
            row[f'{event}_observed'] = int(part.observed.sum())
            row[f'{event}_scored'] = int(part.scored.sum())
            row[f'{event}_equal_fixture_mae'] = float(part.mae.mean()) if len(part) == 15 and part.mae.notna().all() else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(f'{output}/friendly15.csv', index=False)


def sensitivity_receipt(admission, output):
    indexed = admission.set_index(['domain', 'rubric', 'candidate', 'comparator'])
    rows = []
    for prefix in ('', 'p3_'):
        for comparator in REQUIRED:
            for rubric in ('original_v2', 'missing_attack_v3'):
                primary = f'{prefix}conditional_duration_bounded_v2'
                sensitivity = f'{prefix}conditional_duration_source_tail_v2'
                main = indexed.loc[('bounded', rubric, primary, comparator)]
                tail = indexed.loc[('source_tail', rubric, sensitivity, comparator)]
                floor = indexed.loc[('bounded', rubric, primary+'_mixture_floor', comparator)]
                tail_floor = indexed.loc[('source_tail', rubric, sensitivity+'_mixture_floor', comparator)]
                direction = lambda item: bool(item.development_delta > 0 and item.audit_delta >= 0)
                rows.append(dict(candidate=primary, comparator=comparator, rubric=rubric,
                    primary_six_nations_criteria=bool(main.six_nations_research_priority),
                    primary_shared_criteria=bool(main.shared_research_priority),
                    source_tail_development_delta=tail.development_delta, source_tail_audit_delta=tail.audit_delta,
                    mixture_floor_development_delta=floor.development_delta, mixture_floor_audit_delta=floor.audit_delta,
                    source_tail_floor_development_delta=tail_floor.development_delta, source_tail_floor_audit_delta=tail_floor.audit_delta,
                    season_directions_survive_all_sensitivities=all(direction(item) for item in (tail, floor, tail_floor)),
                    ncr_directions_survive_all_sensitivities=all(bool(item.ncr_delta >= 0) for item in (main, tail, floor, tail_floor)),
                    interpretation='Registered research criteria and directional sensitivity only; not future superiority or automatic promotion'))
    pd.DataFrame(rows).to_csv(f'{output}/sensitivity_receipt.csv', index=False)
