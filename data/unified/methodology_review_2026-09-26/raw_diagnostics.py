import json
import os
import sys
from datetime import datetime

import pandas as pd

from model.unified.raw_benchmark.config import STABLE_EVENTS
from model.unified.scoring import NationsChampionshipScorer, SixNationsScorer


def raw_statistics(source, output):
    events = pd.read_csv(f'{source}/results/raw_events.csv')
    stable = events[events.cohort.eq('all') & events.target.isin(STABLE_EVENTS)]
    pivot = stable.pivot(index=['fold', 'target'], columns='engine', values='relative_loss').reset_index()
    rows = []
    for rubric, scorer in [('six_nations', SixNationsScorer()), ('ncr', NationsChampionshipScorer())]:
        scored = set(scorer.weights) | {'tries'}
        if rubric == 'six_nations':
            scored.add('metres')
        for scoring in [True, False]:
            targets = [target for target in STABLE_EVENTS if (target in scored) == scoring]
            subset = pivot[pivot.target.isin(targets)]
            for comparator in ['p3_rolling_native', 'p3_weighted_native', 'empirical_event']:
                difference = subset.p3_robust_native - subset[comparator]
                rows.append(dict(rubric=rubric, scored=scoring, events=len(targets), targets=','.join(targets),
                    comparator=comparator, mean_loss_difference=difference.mean(),
                    contribution_to_23_event_difference=difference.sum() / len(pivot)))
    pd.DataFrame(rows).to_csv(f'{output}/raw_scoring_relevance.csv', index=False)
    targets = pivot.groupby('target')[['p3_robust_native', 'p3_rolling_native']].mean()
    targets['difference'] = targets.p3_robust_native - targets.p3_rolling_native
    targets.reset_index().to_csv(f'{output}/raw_event_differences.csv', index=False)
    minutes = events[events.target.eq('minutes') & events.cohort.isin(['all', 'starter', 'bench'])]
    columns = ['mae', 'actual_mean', 'predicted_mean', 'calibration_slope']
    minutes.groupby(['engine', 'cohort'])[columns].mean().to_csv(f'{output}/raw_minutes_summary.csv')


def elapsed(start, end):
    return (datetime.fromisoformat(end.replace('Z', '+00:00')) - datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds()


def workflow_runtime(source, output):
    with open(f'{source}/evidence/workflow.json') as stream:
        workflow = json.load(stream)
    rows = []
    for job in workflow['jobs']:
        fit = sum(elapsed(step['startedAt'], step['completedAt']) for step in job['steps']
            if 'run comparison' in step['name'].lower())
        rows.append(dict(job=job['name'], seconds=elapsed(job['startedAt'], job['completedAt']), comparison_seconds=fit))
    pd.DataFrame(rows).to_csv(f'{output}/workflow_runtime.csv', index=False)


if __name__ == '__main__':
    repo, output = sys.argv[1:]
    os.makedirs(output, exist_ok=True)
    source = f'{repo}/data/unified/complete_comparison'
    raw_statistics(source, output)
    workflow_runtime(source, output)
