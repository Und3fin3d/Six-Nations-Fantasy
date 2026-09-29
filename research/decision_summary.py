import argparse
import json

import numpy as np
import pandas as pd

PAIRS = {
    'p3_robust_native': 'incumbent',
    'p3_weighted_native': 'incumbent',
    'p3_rolling_native': 'incumbent',
    'equal_incumbent': 'incumbent',
    'regularised_incumbent': 'incumbent',
    'equal_empirical_baseline': 'empirical_baseline',
    'regularised_empirical_baseline': 'empirical_baseline',
    'role_calibration': 'p3_robust_native',
    'kicking_role_gate': 'p3_robust_native',
}


def paired_rows(metrics, comparisons=None):
    rows = []
    if comparisons is None:
        comparisons = [*PAIRS.items(), ('role_calibration', 'incumbent'), ('kicking_role_gate', 'incumbent')]
    for candidate, comparator in comparisons:
        baseline = metrics[metrics.engine.eq(comparator)].set_index('slate')
        selected = metrics[metrics.engine.eq(candidate)].set_index('slate')
        if set(selected.index) != set(baseline.index):
            raise ValueError(f'Unequal slate support: {candidate}/{comparator}')
        for slate, row in selected.iterrows():
            other = baseline.loc[slate]
            if row.n != other.n or row.n_labelled != other.n_labelled:
                raise ValueError(f'Unequal forecast support: {slate}')
            item = dict(slate=slate, competition=row.competition, season=int(row.season),
                        candidate=candidate, comparator=comparator, unknown_selected=row.unknown_selected+other.unknown_selected)
            item.update({f'delta_{name}': row[name]-other[name] for name in ('total', 'xv', 'captain_extra', 'super_sub', 'mae')})
            rows.append(item)
    return pd.DataFrame(rows)


def uncertainty(values):
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        return dict(mean=np.nan, lower=np.nan, upper=np.nan, min_leave_one_out=np.nan)
    rng = np.random.default_rng(20260926)
    draws = data[rng.integers(0, len(data), size=(20000, len(data)))].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    leave = (data.sum() - data) / (len(data) - 1) if len(data) > 1 else np.array([np.nan])
    return dict(mean=data.mean(), lower=lower, upper=upper, min_leave_one_out=np.min(leave))


def paired_summary(pairs, grouping):
    rows = []
    for keys, group in pairs.groupby(grouping):
        row = dict(zip(grouping, keys))
        row.update(uncertainty(group.delta_total))
        row.update(rounds=len(group), known_rounds=int(group.delta_total.notna().sum()),
                   total=group.delta_total.sum(min_count=len(group)), wins=int(group.delta_total.gt(0).sum()),
                   losses=int(group.delta_total.lt(0).sum()), unknown_selected=int(group.unknown_selected.sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def admissions(pairs):
    rows = []
    for (candidate, comparator), group in pairs.groupby(['candidate', 'comparator']):
        six = group[group.competition.eq('six_nations')]
        development, audit = [six[six.season.eq(year)] for year in (2025, 2026)]
        dsum = development.delta_total.sum(min_count=5)
        asum = audit.delta_total.sum(min_count=5)
        pooled = uncertainty(six.delta_total)
        ncr = group[group.competition.eq('ncr')]
        nsum = ncr.delta_total.sum(min_count=3)
        improves_dev = bool(dsum > 0)
        preserves_audit = bool(asum >= 0)
        dispersed = bool(pooled['min_leave_one_out'] > 0)
        rows.append(dict(candidate=candidate, comparator=comparator, development_delta=dsum,
                         audit_delta=asum, pooled_min_leave_one_out=pooled['min_leave_one_out'],
                         ncr_delta=nsum, improves_development=improves_dev, preserves_audit=preserves_audit,
                         gain_survives_round_removal=dispersed,
                         six_nations_research_priority=improves_dev and preserves_audit and dispersed,
                         shared_research_priority=improves_dev and preserves_audit and dispersed and bool(nsum >= 0)))
    return pd.DataFrame(rows)


def position_diagnostics(forecasts):
    rows = []
    for (slate, engine, status, position), group in forecasts.groupby(['slate', 'engine', 'status', 'position']):
        known = group.dropna(subset=['actual'])
        error = known.predicted-known.actual
        rank = known.predicted.corr(known.actual, method='spearman') if len(known) > 2 and known.actual.nunique() > 1 else np.nan
        rows.append(dict(slate=slate, engine=engine, status=status, position=position, n=len(group),
                         labelled=len(known), mae=error.abs().mean(), bias=error.mean(), within_position_rank=rank))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    metrics = pd.read_csv(f'{args.run}/metrics.csv')
    if metrics.groupby('engine').slate.nunique().ne(13).any():
        raise ValueError('The registered official comparison requires all thirteen slates for every candidate')
    pairs = paired_rows(metrics)
    pairs.to_csv(f'{args.run}/paired_rounds.csv', index=False)
    paired_summary(pairs, ['competition', 'season', 'candidate', 'comparator']).to_csv(f'{args.run}/paired_seasons.csv', index=False)
    paired_summary(pairs, ['competition', 'candidate', 'comparator']).to_csv(f'{args.run}/paired_competitions.csv', index=False)
    decisions = admissions(pairs)
    decisions.to_csv(f'{args.run}/admission.csv', index=False)
    forecasts = pd.read_csv(f'{args.run}/forecasts.csv')
    position_diagnostics(forecasts).to_csv(f'{args.run}/position_diagnostics.csv', index=False)
    with open(f'{args.run}/summary_method.json', 'w') as stream:
        json.dump(dict(interval='Descriptive paired round bootstrap, 20000 resamples, seed20260926, percentile95%; small correlated historical sample, not a confirmatory interval',
                       admission='Predeclared development>0, audit>=0, pooled gain survives every single round removal; shared route also NCR>=0',
                       interpretation='Admission selects research priority only. Calibration and gate remedies must also be assessed against actual incumbent.'), stream, indent=2)
    print(decisions.to_string(index=False))


if __name__ == '__main__':
    main()
