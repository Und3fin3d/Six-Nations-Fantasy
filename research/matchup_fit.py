"""Fit and check the matchup calibration on archived international blocks.

Uses the exactly recovered robust P3 per-event component means of the 22
archived PR #24 tournament blocks (``components_raw.pkl``, produced by
``research.export_components``) and the research store. Coefficients are fitted
only on blocks whose cutoff precedes 2025. Development quality is reported
leave-one-block-out on those blocks; later blocks are reported only with the
frozen all-development fit, as a confirmation.

Player-level metrics use only API-observable events: current Six Nations and
NCR rubric points restricted to observed events, the friendly count-stat MAE
(equal weight per fixture and count statistic) and metres MAE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.matchup import fit_matchup_coefficients, matchup_features
from model.unified.raw_benchmark.config import STABLE_EVENTS
from research.context_experiment import load_store

KEY = ['fixture_id', 'player_id', 'team']
EVENTS = tuple(STABLE_EVENTS)
DEVELOPMENT_END = pd.Timestamp('2025-01-01', tz='UTC')
SIX = {'try_assists': 4, 'conversion_goals': 2, 'penalty_goals': 3, 'drop_goals_converted': 4,
       'defenders_beaten': 2, 'offload': 2, 'tackles': 1, 'tackle_turnover': 5, 'penalties_conceded': -1,
       'yellow_cards': -5, 'red_cards': -8}
NCR = {'tries': 12, 'try_assists': 5, 'conversion_goals': 2, 'missed_conversion_goals': -1, 'penalty_goals': 3,
       'missed_penalty_goals': -1, 'drop_goals_converted': 5, 'defenders_beaten': 2, 'offload': 2,
       'clean_breaks': 3, 'tackles': 1, 'missed_tackles': -1, 'tackle_turnover': 4, 'turnovers_conceded': -1,
       'penalties_conceded': -1, 'yellow_cards': -5, 'red_cards': -10}


def block_frames(components: pd.DataFrame, store: pd.DataFrame, cutoffs: pd.Series) -> pd.DataFrame:
    """Player rows with robust means (``p_``), truth (``y_``) and matchup features."""
    flat = pd.DataFrame({k: components[(k, '')].to_numpy() for k in KEY+['block', 'is_forward']})
    for event in (*EVENTS, 'tackle_turnover'):
        flat[f'p_{event}'] = components[('rob', event)].to_numpy(float)
    columns = KEY+['opponent', 'home_away', 'match_at', 'started']
    truth = store[columns + [e for e in (*EVENTS, 'tackle_turnover')] +
                  [f'available__{e}' for e in (*EVENTS, 'tackle_turnover')]].copy()
    for event in (*EVENTS, 'tackle_turnover'):
        observed = truth[f'available__{event}'].fillna(False).astype(bool)
        truth[f'y_{event}'] = pd.to_numeric(truth[event], errors='coerce').where(observed)
    truth = truth[columns + [f'y_{e}' for e in (*EVENTS, 'tackle_turnover')]]
    frame = flat.merge(truth, on=KEY, how='left', validate='one_to_one')
    parts = []
    for block, rows in frame.groupby('block', sort=True):
        cutoff = cutoffs[block]
        history = past_matches(store, cutoff)
        features = matchup_features(rows.reset_index(drop=True), history, cutoff, EVENTS)
        rows = pd.concat([rows.reset_index(drop=True), features], axis=1)
        rows['cutoff'] = cutoff
        parts.append(rows)
        print(f'{block}: {len(rows)} players', flush=True)
    return pd.concat(parts, ignore_index=True)


def team_rows(players: pd.DataFrame) -> pd.DataFrame:
    grouped = players.groupby(['block', 'fixture_id', 'team'], sort=False)
    out = grouped[['edge'] + [f'opp__{e}' for e in EVENTS]].first()
    for event in EVENTS:
        valid = players[f'y_{event}'].notna()
        out[f'p_{event}'] = players[f'p_{event}'].where(valid).groupby(
            [players.block, players.fixture_id, players.team], sort=False).sum(min_count=1)
        out[f'y_{event}'] = players[f'y_{event}'].groupby(
            [players.block, players.fixture_id, players.team], sort=False).sum(min_count=1)
    out = out.reset_index()
    out['cutoff'] = out.block.map(players.groupby('block').cutoff.first())
    return out


def shrink(coefficients: dict, factor: float) -> dict:
    return {e: {k: factor*v for k, v in c.items()} for e, c in coefficients.items()}


def apply(players: pd.DataFrame, coefficients: dict) -> pd.DataFrame:
    adjusted = players.copy()
    for event, coef in coefficients.items():
        eta = coef['edge']*adjusted['edge']/400.0 + coef['opp']*adjusted[f'opp__{event}']
        adjusted[f'p_{event}'] = adjusted[f'p_{event}']*np.exp(eta)
    return adjusted


def metrics(players: pd.DataFrame) -> dict:
    forward = players.is_forward.astype(bool).to_numpy()
    p = lambda e: players[f'p_{e}'].fillna(0.0)
    y = lambda e: players[f'y_{e}'].fillna(0.0)
    p6 = sum(w*p(e) for e, w in SIX.items()) + np.where(forward, 15, 10)*p('tries') + np.maximum(p('metres')/10 - .45, 0)
    y6 = sum(w*y(e) for e, w in SIX.items()) + np.where(forward, 15, 10)*y('tries') + np.floor(y('metres')/10)
    pn, yn = sum(w*p(e) for e, w in NCR.items()), sum(w*y(e) for e, w in NCR.items())
    counts = []
    for event in EVENTS:
        if event == 'metres':
            continue
        valid = players[f'y_{event}'].notna()
        error = (players.loc[valid, f'p_{event}'] - players.loc[valid, f'y_{event}']).abs()
        counts.append(error.groupby(players.loc[valid, 'fixture_id']).mean())
    valid = players.y_metres.notna()
    metres = (players.loc[valid, 'p_metres'] - players.loc[valid, 'y_metres']).abs().groupby(
        players.loc[valid, 'fixture_id']).mean().mean()
    return {'six_mae': float(np.abs(p6 - y6).mean()), 'ncr_mae': float(np.abs(pn - yn).mean()),
            'count_mae': float(pd.concat(counts).mean()), 'metres_mae': float(metres), 'players': len(players)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='rolling_eval output with inputs/')
    parser.add_argument('--components', type=Path, required=True, help='components_raw.pkl')
    parser.add_argument('--cutoffs', type=Path, required=True, help='pickle with block and cutoff columns')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ridge', type=float, default=1.0)
    parser.add_argument('--shrink', type=float, default=0.75,
                        help='multiplier on fitted coefficients; 0.75 chosen leave-one-block-out on development blocks')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store, _ = load_store(args.base)
    cutoffs = pd.read_pickle(args.cutoffs)[['block', 'cutoff']].drop_duplicates().set_index('block').cutoff
    players = block_frames(pd.read_pickle(args.components), store, cutoffs)
    players.to_pickle(args.output/'players.pkl')
    teams = team_rows(players)
    development = teams.cutoff < DEVELOPMENT_END
    fitted = fit_matchup_coefficients(teams[development], EVENTS, ridge=args.ridge)
    frozen = shrink(fitted, args.shrink)
    (args.output/'matchup_coefficients_unshrunk.json').write_text(json.dumps(fitted, indent=1, sort_keys=True)+'\n')
    (args.output/'matchup_coefficients.json').write_text(json.dumps(frozen, indent=1, sort_keys=True)+'\n')
    edge_only = {e: {'edge': c['edge'], 'opp': 0.0} for e, c in
                 fit_matchup_coefficients(teams[development].assign(**{f'opp__{e}': 0.0 for e in EVENTS}),
                                          EVENTS, ridge=args.ridge).items()}
    rows = []
    for block, group in players.groupby('block', sort=True):
        is_dev = group.cutoff.iloc[0] < DEVELOPMENT_END
        if is_dev:
            train = teams[development & teams.block.ne(block)]
            coef = shrink(fit_matchup_coefficients(train, EVENTS, ridge=args.ridge), args.shrink)
            coef_edge = {e: {'edge': c['edge'], 'opp': 0.0} for e, c in fit_matchup_coefficients(
                train.assign(**{f'opp__{e}': 0.0 for e in EVENTS}), EVENTS, ridge=args.ridge).items()}
        else:
            coef, coef_edge = frozen, edge_only
        for name, adjusted in (('reference', group), ('edge_only', apply(group, coef_edge)),
                               ('matchup', apply(group, coef))):
            rows.append({'block': block, 'development': is_dev, 'variant': name, **metrics(adjusted)})
    result = pd.DataFrame(rows)
    result.to_csv(args.output/'block_metrics.csv', index=False)
    summary = result.groupby(['development', 'variant'])[['six_mae', 'ncr_mae', 'count_mae', 'metres_mae']].mean()
    summary.to_csv(args.output/'summary.csv')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
