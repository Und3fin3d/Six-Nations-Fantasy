from dataclasses import replace

import numpy as np
import pandas as pd

from model.history import past_matches, utc_cutoff

GOAL_EVENTS = ('conversion_goals', 'missed_conversion_goals', 'penalty_goals', 'missed_penalty_goals')


def kicking_history(store, cutoff):
    history = past_matches(store, cutoff)
    available = history[[f'available__{event}' for event in GOAL_EVENTS]].fillna(False).all(axis=1)
    values = history[list(GOAL_EVENTS)].apply(pd.to_numeric, errors='coerce')
    minutes = pd.to_numeric(history.minutes, errors='coerce')
    valid = available & values.notna().all(axis=1) & minutes.gt(0) & history.available__minutes.fillna(False)
    observed = history.loc[valid, ['player_id', 'date', 'competition_level']].copy()
    age = (utc_cutoff(cutoff) - pd.to_datetime(observed.date, utc=True)).dt.total_seconds() / 86400
    confidence = observed.competition_level.map({'international': 1.0, 'club': 0.55}).fillna(0.0)
    observed['exposure'] = minutes[valid] * np.power(0.5, age / 420.0) * confidence
    observed['attempts'] = values.loc[valid].sum(axis=1)
    return observed.groupby('player_id')[['exposure', 'attempts']].sum()


def apply_kicking_gate(predictions, history):
    adjusted, audit = [], []
    for prediction in predictions:
        exposure, attempts = 0.0, 0.0
        if prediction.player_id in history.index:
            exposure, attempts = history.loc[prediction.player_id, ['exposure', 'attempts']]
        factor = 220.0 / (220.0 + exposure) if attempts == 0 else 1.0
        events = dict(prediction.events)
        for event in GOAL_EVENTS:
            if event in events:
                events[event] = replace(events[event], mean=events[event].mean * factor)
        adjusted.append(replace(prediction, events=events))
        audit.append({'player_id': prediction.player_id, 'fixture_id': prediction.fixture_id,
                      'team': prediction.team, 'exposure': exposure, 'attempts': attempts, 'factor': factor})
    return adjusted, audit


def calibrate_roles(current, earlier, prediction):
    if earlier.empty:
        return prediction.copy(), pd.DataFrame()
    groups = earlier.groupby(['position', 'status']).residual.agg(['sum', 'count'])
    groups['correction'] = (groups['sum'] / (groups['count'] + 20.0)).clip(-5.0, 5.0)
    keys = pd.MultiIndex.from_frame(current[['position', 'status']])
    correction = groups.correction.reindex(keys).fillna(0.0).to_numpy()
    return prediction + correction, groups.reset_index()


def combination_weight(earlier, comparator):
    if earlier.empty:
        return 0.5
    valid = earlier.dropna(subset=['actual', 'robust', comparator])
    if valid.empty:
        return 0.5
    delta = (valid.robust - valid[comparator]).to_numpy()
    residual = (valid.actual - valid[comparator]).to_numpy()
    penalty = 60.0 * np.square(delta).mean()
    denominator = np.square(delta).sum() + penalty
    return float(np.clip((delta @ residual + 0.5 * penalty) / denominator, 0, 1)) if denominator else 0.5


def bench_history_diagnostic(store, cutoff):
    history = past_matches(store, cutoff)
    bench = history[history.competition_level.eq('international') & ~history.started].copy()
    bench['known_minutes'] = bench.available__minutes.fillna(False) & bench.minutes.notna()
    bench['zero_minutes'] = bench.known_minutes & bench.minutes.eq(0)
    bench['played'] = bench.known_minutes & bench.minutes.gt(0)
    return bench.groupby('position').agg(rows=('player_id', 'size'), known_minutes=('known_minutes', 'sum'),
                                        zero_minutes=('zero_minutes', 'sum'), played=('played', 'sum')).reset_index()
