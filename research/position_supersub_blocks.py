"""Raw-event and observable-points metrics on archived international blocks and Friendly-25.

Compares the reference robust P3 forecasts with the status-aware forecasts
written by ``research.position_supersub``. Truth is the research store (API
statistics). Reported per block and per starter/bench group:

* event-level mean ratio (forecast / actual), MSE and Poisson deviance;
* count-stat MAE with the unchanged Friendly evaluator definition
  (per fixture and stat, equal weights);
* observable fantasy points under the current Six Nations rubric (tries,
  assists, goal kicking, defenders beaten, offloads, tackles, breakdown steals
  from the API turnover proxy, conceded penalties, cards, metres), with
  ``E[floor(metres/10)]`` taken from the forecast distribution.

Pre-2025 blocks are development data; 2025-26 blocks overlap the evaluation
seasons.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.unified.friendly_eval import COUNT_TARGETS, score_stats, summarise_stats
from model.unified.rolling_eval import ROOT, expected_points
from research.context_experiment import load_store
from research.position_supersub import load_raw

KEY = ['fixture_id', 'player_id', 'team']
EVENTS = ('tries', 'try_assists', 'defenders_beaten', 'offload', 'tackles', 'tackle_turnover', 'metres', 'runs',
          'clean_breaks', 'missed_tackles', 'penalties_conceded', 'turnovers_conceded', 'passes', 'rucks_won')


def observable_points(frame: pd.DataFrame) -> np.ndarray:
    n = lambda c: pd.to_numeric(frame[c], errors='coerce').fillna(0).to_numpy(float)
    return (np.where(frame.is_forward.astype(bool), 15, 10)*n('tries') + 4*n('try_assists') + 2*n('conversion_goals')
            + 3*n('penalty_goals') + 4*n('drop_goals_converted') + 2*n('defenders_beaten') + 2*n('offload')
            + n('tackles') + 5*n('tackle_turnover') - n('penalties_conceded') - 5*n('yellow_cards') - 8*n('red_cards')
            + np.floor(n('metres')/10))


def observable_expected(raw) -> np.ndarray:
    """Six Nations expected points restricted to API-observable events."""
    keep = {'tries', 'try_assists', 'conversion_goals', 'penalty_goals', 'drop_goals_converted', 'defenders_beaten',
            'offload', 'tackles', 'tackle_turnover', 'penalties_conceded', 'yellow_cards', 'red_cards', 'metres'}
    from dataclasses import replace
    trimmed = [replace(p, events={e: d for e, d in p.events.items() if e in keep}) for p in raw]
    return expected_points(trimmed, 'six_nations')


def event_rows(truth: pd.DataFrame, engines: dict, unit: str) -> pd.DataFrame:
    rows = []
    started = truth.started.astype(bool).to_numpy()
    for engine, raw in engines.items():
        for event in EVENTS:
            available = truth[f'available__{event}'].fillna(False).astype(bool).to_numpy()
            actual = pd.to_numeric(truth[event], errors='coerce').to_numpy(float)
            pred = np.array([p.events[event].mean if event in p.events else np.nan for p in raw])
            ok = available & np.isfinite(actual) & np.isfinite(pred)
            for status, mask in (('start', started), ('bench', ~started), ('all', np.ones_like(started))):
                m = ok & mask
                if not m.any():
                    continue
                mu = np.maximum(pred[m], 1e-9); y = actual[m]
                deviance = 2*np.where(y > 0, y*np.log(np.maximum(y, 1e-12)/mu), 0) - 2*(y-mu)
                rows.append({'unit': unit, 'engine': engine, 'event': event, 'status': status, 'n': int(m.sum()),
                             'pred_sum': float(pred[m].sum()), 'actual_sum': float(y.sum()),
                             'sse': float(((pred[m]-y)**2).sum()), 'deviance': float(deviance.sum()),
                             'abs': float(np.abs(pred[m]-y).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--set', choices=['blocks', 'friendly'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store, _ = load_store(args.runs/'base')
    events, players, stats = [], [], []
    if args.set == 'blocks':
        units = [(Path(d).name.removeprefix('comparison-raw-'), Path(d)/'results'/'p3_robust_native.jsonl')
                 for d in sorted(glob.glob(str(args.runs.parent/'raw'/'comparison-raw-*')))]
    else:
        manifest = json.loads((ROOT/'data'/'unified'/'friendly25'/'fixtures.json').read_text())
        days = sorted({pd.Timestamp(f['kickoff']).date() for f in manifest['fixtures']})
        units = [(str(day), args.runs/'f25_ctx'/'days'/str(day)/'p3_robust_native.jsonl') for day in days]
    for unit, reference_path in units:
        reference = load_raw(reference_path)
        status = load_raw(args.status/args.set/unit/'status.jsonl')
        by_key = {(p.fixture_id, p.player_id, p.team): p for p in reference}
        keys = [(p.fixture_id, p.player_id, p.team) for p in status]
        reference = [by_key[k] for k in keys]
        if args.set == 'friendly':
            truth = pd.read_pickle(args.runs/'f25_ctx'/'days'/unit/'truth.pkl')
            truth[KEY] = truth[KEY].astype(str)
            truth = truth.set_index(KEY).loc[keys].reset_index()
        else:
            frame = pd.DataFrame(keys, columns=KEY)
            truth = frame.merge(store, on=KEY, how='left', validate='one_to_one')
            truth['is_forward'] = [p.is_forward for p in status]
            truth['position'] = [p.position for p in status]
        engines = {'p3_robust_native': reference, 'status_rates': status}
        shrunk_path = args.status/args.set/unit/'shrunk.jsonl'
        if shrunk_path.exists():
            engines['status_shrunk'] = load_raw(shrunk_path)
        events.append(event_rows(truth, engines, unit))
        actual = observable_points(truth)
        for engine, raw in engines.items():
            players.append(pd.DataFrame({'unit': unit, 'engine': engine, 'fixture_id': truth.fixture_id.astype(str),
                                         'player_id': truth.player_id.astype(str), 'position': truth.position,
                                         'started': truth.started.astype(bool), 'minutes': truth.minutes,
                                         'predicted': observable_expected(raw), 'actual': actual}))
            stats.append(score_stats(truth, raw, engine=engine).assign(unit=unit))
        print(f'{unit} done', flush=True)
    events = pd.concat(events, ignore_index=True); players = pd.concat(players, ignore_index=True)
    stats = pd.concat(stats, ignore_index=True)
    events.to_csv(args.output/'event_metrics.csv', index=False)
    players.to_pickle(args.output/'players.pkl')
    stats.to_csv(args.output/'count_stats.csv', index=False)
    if args.set == 'friendly':
        summary, _ = summarise_stats(stats.drop(columns='unit'), stats.fixture_id.astype(str).unique())
        summary.to_csv(args.output/'friendly_summary.csv', index=False)
        print(summary.to_string(index=False))
    agg = events.groupby(['engine', 'event', 'status'])[['n', 'pred_sum', 'actual_sum', 'sse', 'deviance']].sum()
    agg['ratio'] = agg.pred_sum/agg.actual_sum
    print(agg.round(3).to_string())
    count = stats[stats.target.isin(COUNT_TARGETS)].groupby('engine').mae.mean()
    print('count-stat MAE', count.round(5).to_dict())


if __name__ == '__main__':
    main()
