"""Per-player expected and realised points for every engine and rubric.

Engines (all from the same lock-time robust P3 fit; no refits):

``p3_robust``          reference raw forecasts;
``empirical_baseline`` empirical fantasy baseline, observable rubric configs;
``c2k15``              team-strength (Elo edge) calibration + kicking gamma 1.5;
``h1_oct1``            1 October candidate: 0.75 * c2k15 + 0.25 * baseline;
``mk``                 matchup calibration + kicking gamma 1.5 (no blend);
``h2``                 round-2 candidate: 0.7 * mk + 0.3 * baseline;
``sk``                 robust P3 with the status-aware, within-position-shrunk
                       empirical component (``status_rates``; same trees and weights);
``sk_mk``              SK + matchup calibration + kicking gamma 1.5;
``sk_h2``              0.7 * sk_mk + 0.3 * baseline;
``null3_a``/``null3_b`` reference points times a fixed per-player factor
                       exp(N(0, 0.03)): equally informed copies that differ from
                       the reference about as much as the candidates do. They
                       calibrate how large a difference pure noise produces.

Raw-event metrics (``model.unified.raw_benchmark.metrics.event_metrics``) are
written for the three raw-level engines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.data import ROOT
from model.unified.kicking import concentrate_kicking
from model.unified.matchup import matchup_scale
from model.unified.raw_benchmark.config import EXTENDED_EVENTS, STABLE_EVENTS
from model.unified.raw_benchmark.metrics import NaiveComparator, event_metrics
from model.unified.v4.context import team_strength_scale

from .decision import pool_positions
from .fit import FIT_NAME, KEY, load_raw, lock_name, write_atomic
from .rubrics import RUBRICS, actual_points, expected_points
from .slates import COMPETITIONS
from .status import SK_NAME

ENGINES = ('p3_robust', 'empirical_baseline', 'c2k15', 'h1_oct1', 'mk', 'h2', 'sk', 'sk_mk', 'sk_h2',
           'null3_a', 'null3_b')
REFERENCE = 'p3_robust'
GAMMA = 1.5
H1_WEIGHT, H2_WEIGHT = 0.75, 0.7
NULL_SD = 0.03
BETA_PATH = ROOT/'research'/'hillclimb_2026-10-01'/'team_strength_beta.json'
MATCHUP_PATH = ROOT/'research'/'hillclimb_2026-10-02'/'matchup_coefficients.json'


def raw_engines(reference, context: pd.DataFrame, sk=None) -> dict:
    beta = json.loads(BETA_PATH.read_text())
    coefficients = json.loads(MATCHUP_PATH.read_text())
    records = context.to_dict('records')
    c2 = [team_strength_scale(p, float(row['elo_edge']), beta) for p, row in zip(reference, records)]
    matchup = [matchup_scale(p, row, coefficients) for p, row in zip(reference, records)]
    raws = {'p3_robust': reference, 'c2k15': concentrate_kicking(c2, GAMMA),
            'mk': concentrate_kicking(matchup, GAMMA)}
    if sk is not None:
        raws['sk'] = sk
        raws['sk_mk'] = concentrate_kicking([matchup_scale(p, row, coefficients) for p, row in zip(sk, records)], GAMMA)
    return raws


def null_factor(keys: pd.DataFrame, label: str) -> np.ndarray:
    """Fixed multiplicative error per player-match, reproducible from the keys alone."""
    text = (keys['fixture_id'].astype(str) + '|' + keys['player_id'].astype(str) + '|' + label)
    seeds = pd.util.hash_pandas_object(text, index=False).to_numpy(np.uint64)
    uniform = (seeds >> np.uint64(11)).astype(float)/float(1 << 53)
    from scipy.stats import norm
    return np.exp(NULL_SD*norm.ppf(np.clip(uniform, 1e-12, 1 - 1e-12)))


def load_players(output: Path, name: str) -> pd.DataFrame | None:
    path = output/'players'/f'{name}.pkl'
    return pd.read_pickle(path) if path.exists() else None


def score_lock(store: pd.DataFrame, manifest: dict, output: Path, lock: pd.Timestamp, engines, rubrics) -> pd.DataFrame:
    name = lock_name(lock)
    directory = output/'locks'/name
    candidates = pd.read_pickle(directory/'candidates.pkl')
    reference = load_raw(directory/f'{FIT_NAME}.jsonl.gz')
    context = pd.read_pickle(directory/'context.pkl')
    baseline = pd.read_pickle(directory/'baseline.pkl')
    if not (len(candidates) == len(reference) == len(context) == len(baseline)):
        raise ValueError(f'{name}: saved lock artefacts differ in length')
    if not baseline['label_row_id'].equals(candidates['label_row_id']):
        raise ValueError(f'{name}: baseline rows are misaligned')
    truth = candidates[KEY].merge(store, on=KEY, how='left', validate='one_to_one')
    if truth['minutes'].isna().any():
        raise ValueError(f'{name}: candidate rows missing from the store')
    slate_info = pd.DataFrame(manifest['slates']).set_index('slate')
    sk_path = directory/f'{SK_NAME}.jsonl.gz'
    if any(e.startswith('sk') for e in engines) and not sk_path.exists():
        raise FileNotFoundError(f'{name}: run the sk stage before scoring SK engines')
    raws = raw_engines(reference, context, load_raw(sk_path) if sk_path.exists() else None)
    frame = candidates[['slate', *KEY, 'player_name', 'started', 'jersey', 'position', 'is_forward']].copy()
    frame['lock'] = name
    frame['competition'] = truth['competition_id_cache'].astype(int).map(COMPETITIONS).to_numpy()
    frame['slate_competition'] = frame['slate'].map(slate_info['competition'])
    frame['family'] = frame['slate'].map(slate_info['family'])
    frame['block'] = frame['slate'].map(slate_info['block'])
    frame['minutes'] = truth['minutes'].to_numpy(float)
    frame['pos'] = pool_positions(candidates['position'], candidates['jersey'])
    frame['expected_minutes'] = [p.minutes.mean for p in reference]
    factors = {label: null_factor(candidates, label) for label in ('null3_a', 'null3_b')}
    for rubric_name in rubrics:
        rubric = RUBRICS[rubric_name]
        frame[f'actual__{rubric_name}'] = actual_points(truth.assign(is_forward=candidates['is_forward'].to_numpy()), rubric)
        points = {engine: expected_points(raw, rubric) for engine, raw in raws.items()}
        points['empirical_baseline'] = baseline[rubric_name].to_numpy(float)
        points['h1_oct1'] = H1_WEIGHT*points['c2k15'] + (1 - H1_WEIGHT)*points['empirical_baseline']
        points['h2'] = H2_WEIGHT*points['mk'] + (1 - H2_WEIGHT)*points['empirical_baseline']
        if 'sk_mk' in points:
            points['sk_h2'] = H2_WEIGHT*points['sk_mk'] + (1 - H2_WEIGHT)*points['empirical_baseline']
        for label, factor in factors.items():
            points[label] = points[REFERENCE]*factor
        for engine in engines:
            frame[f'pred__{engine}__{rubric_name}'] = points[engine]

    # Raw-event accuracy for the raw-level engines, with the existing metrics.
    events_path = output/'events'/f'{name}.csv'
    if not events_path.exists():
        train = past_matches(store, lock)
        naive = NaiveComparator.fit(train, ('minutes', *STABLE_EVENTS, *EXTENDED_EVENTS))
        evaluation = truth.copy()
        evaluation[['position', 'is_forward']] = candidates[['position', 'is_forward']].to_numpy()
        evaluation['career_matches'] = evaluation['player_id'].map(train.groupby('player_id')['fixture_id'].nunique()).fillna(0)
        tables = [event_metrics(evaluation, raw, naive, engine=engine, fold=name) for engine, raw in raws.items()]
        events_path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(events_path, lambda tmp: pd.concat(tables, ignore_index=True).to_csv(tmp, index=False))
    return frame


def score_stage(store: pd.DataFrame, manifest: dict, output: Path, locks, engines, rubrics) -> None:
    engines = list(engines or ENGINES)
    rubrics = list(rubrics or RUBRICS)
    (output/'players').mkdir(parents=True, exist_ok=True)
    for lock in manifest['locks']:
        name = lock_name(pd.Timestamp(lock))
        if locks and name not in locks:
            continue
        path = output/'players'/f'{name}.pkl'
        if path.exists():
            continue
        if not (output/'locks'/name/'fit.json').exists():
            print(f'score: lock {name} not fitted yet; skipped', flush=True)
            continue
        frame = score_lock(store, manifest, output, pd.Timestamp(lock), engines, rubrics)
        write_atomic(path, lambda tmp: frame.to_pickle(tmp, compression=None))
        print(f'score: lock {name}: {len(frame)} rows', flush=True)
