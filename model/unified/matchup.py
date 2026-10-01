"""Team-level matchup calibration for raw-event forecasts (research only).

Robust P3 forecasts each player independently. Team-level residuals of its
summed forecasts still depend on the matchup: on the pre-match Elo edge (see
``v4.context``) and on what the opponent has historically conceded for that
event. A possession-heavy opponent makes our side tackle more; a side that has
conceded many metres lets us make more.

Each event mean is multiplied by::

    exp(b_edge[e] * edge / 400 + b_opp[e] * log(opp_conceded[e] / mean[e]))

``opp_conceded`` is a recency-weighted, shrunken average of the event total
the opponent's opponents made in internationals before the lock. The
coefficients come from team-level Poisson fits on the international tournament
blocks that finished before 2025. The calibration is competition-independent:
it uses only international history before the lock and does not know the
fantasy rules. Minutes and the event families' shapes are unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import EventDistribution, RawPrediction
from .v4.context import add_candidate_context, pre_match_context

HALFLIFE_DAYS = 365.0
WINDOW_DAYS = 1095
PRIOR_MATCHES = 3.0


def team_event_totals(history: pd.DataFrame, events: tuple[str, ...]) -> pd.DataFrame:
    """One row per international (fixture, team): own totals and the opponent's.

    Unobserved player values stay missing; a team total is missing when no
    player in it has the event observed. ``a_<event>`` is what the opponent
    made in the same fixture, i.e. what this team conceded.
    """
    frame = history[history['competition_level'].eq('international')]
    columns = {}
    for event in events:
        values = pd.to_numeric(frame[event], errors='coerce')
        available = frame.get(f'available__{event}')
        if available is not None:
            values = values.where(available.fillna(False).astype(bool))
        columns[event] = values
    data = pd.DataFrame(columns, index=frame.index)
    data['fixture_id'] = frame['fixture_id'].astype(str).to_numpy()
    data['team'] = frame['team'].astype(str).to_numpy()
    data['opponent'] = frame['opponent'].astype(str).to_numpy()
    data['match_at'] = pd.to_datetime(frame['match_at'], utc=True).to_numpy()
    grouped = data.groupby(['fixture_id', 'team'], sort=False)
    totals = grouped[list(events)].sum(min_count=1)
    totals['opponent'] = grouped['opponent'].first()
    totals['match_at'] = grouped['match_at'].first()
    totals = totals.reset_index()
    against = totals[['fixture_id', 'team', *events]].rename(
        columns={'team': 'opponent', **{e: f'a_{e}' for e in events}})
    return totals.merge(against, on=['fixture_id', 'opponent'], how='left', validate='one_to_one')


def conceded_profiles(totals: pd.DataFrame, cutoff: pd.Timestamp, events: tuple[str, ...], *,
                      halflife: float = HALFLIFE_DAYS, window: int = WINDOW_DAYS,
                      prior: float = PRIOR_MATCHES) -> dict[str, pd.Series]:
    """Per event, each team's shrunken conceded total relative to the pooled mean.

    Only fixtures that kicked off before ``cutoff`` and within ``window`` days
    of it are used, weighted by ``0.5 ** (age / halflife)``. Each team is
    shrunk towards the pooled mean with ``prior`` matches of weight.
    """
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
    recent = totals[(totals['match_at'] < cutoff) & (totals['match_at'] >= cutoff - pd.Timedelta(days=window))]
    weight = 0.5 ** ((cutoff - recent['match_at']).dt.total_seconds() / 86400.0 / halflife)
    profiles = {}
    for event in events:
        values = recent[f'a_{event}']
        observed = values.notna()
        if not observed.any():
            profiles[event] = pd.Series(dtype=float)
            continue
        mean = float(np.sum(weight[observed]*values[observed]) / np.sum(weight[observed]))
        if mean <= 0:
            profiles[event] = pd.Series(dtype=float)
            continue
        numerator = (weight*values.fillna(0.0)).groupby(recent['team']).sum()
        denominator = (weight*observed).groupby(recent['team']).sum()
        profiles[event] = (numerator + prior*mean) / (denominator + prior) / mean
    return profiles


def matchup_features(candidates: pd.DataFrame, history: pd.DataFrame, cutoff: pd.Timestamp,
                     events: tuple[str, ...]) -> pd.DataFrame:
    """Per candidate row: the Elo edge and log conceded ratio of the opponent per event."""
    state = pre_match_context(history)[1]
    edge = add_candidate_context(candidates, state)['ctx__elo_edge'].to_numpy(float)
    profiles = conceded_profiles(team_event_totals(history, events), cutoff, events)
    opponents = candidates['opponent'].astype(str)
    features = {'edge': edge}
    for event in events:
        ratio = opponents.map(profiles[event]).fillna(1.0).to_numpy(float)
        features[f'opp__{event}'] = np.log(np.clip(ratio, 1e-6, None))
    return pd.DataFrame(features, index=candidates.index)


def matchup_scale(prediction: RawPrediction, features: Mapping[str, float],
                  coefficients: Mapping[str, Mapping[str, float]]) -> RawPrediction:
    """Rescale event means; lognormal variance scales with the squared factor."""
    events = {}
    for name, dist in prediction.events.items():
        coef = coefficients.get(name, {})
        eta = coef.get('edge', 0.0)*features['edge']/400.0 + coef.get('opp', 0.0)*features.get(f'opp__{name}', 0.0)
        factor = float(np.exp(eta))
        if dist.family == 'bernoulli':
            events[name] = EventDistribution(dist.family, min(dist.mean*factor, 1.0), dist.dispersion)
        elif dist.family == 'lognormal':
            events[name] = EventDistribution(dist.family, dist.mean*factor, dist.dispersion*factor**2)
        else:
            events[name] = EventDistribution(dist.family, dist.mean*factor, dist.dispersion)
    return replace(prediction, events=events)


def calibrate_matchups(raw: list[RawPrediction], candidates: pd.DataFrame, history: pd.DataFrame,
                       cutoff: pd.Timestamp, coefficients: Mapping[str, Mapping[str, float]]) -> list[RawPrediction]:
    if len(raw) != len(candidates):
        raise ValueError('forecasts and candidates differ in length')
    keys = list(zip(candidates['fixture_id'].astype(str), candidates['player_id'].astype(str),
                    candidates['team'].astype(str)))
    if keys != [(p.fixture_id, p.player_id, p.team) for p in raw]:
        raise ValueError('forecasts and candidates are misaligned')
    events = tuple(sorted(coefficients))
    features = matchup_features(candidates.reset_index(drop=True), history, cutoff, events)
    return [matchup_scale(p, row, coefficients) for p, row in zip(raw, features.to_dict('records'))]


def fit_matchup_coefficients(teams: pd.DataFrame, events: tuple[str, ...], *, ridge: float = 1.0,
                             minimum_total: float = 30.0) -> dict[str, dict[str, float]]:
    """Team-level Poisson fits: y = p * exp(b_edge * edge/400 + b_opp * opp), ridge-penalised.

    ``teams`` has one row per (fixture, team) with ``edge``, ``opp__<event>``,
    ``p_<event>`` (summed forecast means) and ``y_<event>`` (observed totals).
    Metres are fitted in units of 10 m so the ridge penalty is comparable.
    """
    from scipy.optimize import minimize

    output = {}
    for event in events:
        block = teams[['edge', f'opp__{event}', f'p_{event}', f'y_{event}']].dropna()
        block = block[block[f'p_{event}'] > 0]
        if block[f'y_{event}'].sum() <= minimum_total:
            output[event] = {'edge': 0.0, 'opp': 0.0}
            continue
        scale = 0.1 if event == 'metres' else 1.0
        p = block[f'p_{event}'].to_numpy(float)*scale
        y = block[f'y_{event}'].to_numpy(float)*scale
        x = np.column_stack([block['edge'].to_numpy(float)/400.0, block[f'opp__{event}'].to_numpy(float)])

        def loss(b):
            mu = p*np.exp(x @ b)
            return float(np.sum(mu - y*np.log(mu)) + 0.5*ridge*np.sum(b**2))

        b = minimize(loss, np.zeros(2), method='BFGS').x
        output[event] = {'edge': float(b[0]), 'opp': float(b[1])}
    return output
