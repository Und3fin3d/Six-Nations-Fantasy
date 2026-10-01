"""One robust P3 fit per monthly lock, forecasting every slate in that month.

Outputs per lock directory (``<output>/locks/<YYYY-MM>/``), each written
atomically so an interrupted run resumes at the first incomplete lock:

``candidates.pkl``    candidate rows (keys, slate, roles after the lock-time
                      position rebuild, status) in forecast order;
``p3_robust.jsonl.gz`` robust P3 raw forecasts (reference engine);
``context.pkl``       pre-lock matchup features (Elo edge, opponent conceded
                      ratios) used by the post-processing candidates;
``baseline.pkl``      empirical fantasy baseline per observable rubric;
``fit.json``          training cohort summary and timings.
"""
from __future__ import annotations

import gzip
import json
import os
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from model.empirical_unified import CONFIGS, project_candidates
from model.history import past_matches
from model.unified.contracts import RawPrediction
from model.unified.matchup import matchup_features
from model.unified.raw_benchmark.blend import EventWeightedBlend
from model.unified.raw_benchmark.features import build_frozen_feature_frames
from model.unified.raw_benchmark.folds import masked_candidates
from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from model.unified.v4.context import add_candidate_context, pre_match_context

from .rubrics import BASELINE_CONFIGS, strip_potm

KEY = ['fixture_id', 'player_id', 'team']
SORT = ['date', 'fixture_id', 'team', 'player_id']
OUTCOME_COLUMNS = ['team_score', 'opp_score', 'result', 'margin', 'fantasy_pts_recon']
ROLE_COLUMNS = [*KEY, 'slate', 'player_name', 'opponent', 'home_away', 'started', 'jersey', 'position',
                'position_source', 'is_forward', 'competition_level', 'hemisphere', 'match_at', 'label_row_id']
FIT_NAME = 'p3_robust'


def lock_name(lock: pd.Timestamp) -> str:
    return pd.Timestamp(lock).strftime('%Y-%m')


def write_atomic(path: Path, writer) -> None:
    tmp = path.with_name(path.name + '.tmp')
    writer(tmp)
    os.replace(tmp, path)


def save_raw(path: Path, predictions: list[RawPrediction]) -> None:
    def writer(tmp):
        with gzip.open(tmp, 'wt') as handle:
            for p in predictions:
                handle.write(json.dumps(p.to_dict()) + '\n')
    write_atomic(path, writer)


def load_raw(path: Path) -> list[RawPrediction]:
    with gzip.open(path, 'rt') as handle:
        return [RawPrediction.from_dict(json.loads(line)) for line in handle]


@contextmanager
def lightgbm_threads():
    """Honour ``OMP_NUM_THREADS`` in LightGBM.

    The model code passes ``n_jobs=-1``, which LightGBM >= 4 turns into every
    logical CPU regardless of ``OMP_NUM_THREADS``; on a shared machine the
    oversubscribed OpenMP threads slow a fit by an order of magnitude. Results
    are unchanged in kind (``deterministic=True`` for a fixed thread count).
    """
    from lightgbm.sklearn import LGBMModel
    threads = int(os.environ.get('OMP_NUM_THREADS', '2'))
    original = LGBMModel._process_n_jobs
    LGBMModel._process_n_jobs = lambda self, n_jobs: threads
    try:
        yield threads
    finally:
        LGBMModel._process_n_jobs = original


def register_baseline_configs() -> None:
    """Expose the observable rubric configs through the baseline's config seam (in-process only)."""
    for name, config in BASELINE_CONFIGS.items():
        CONFIGS.setdefault(f'bigbench_{name}', config)


@contextmanager
def memoised_baseline(candidates: pd.DataFrame):
    """Compute the baseline's player profiles and RugbyPass table once per lock.

    ``project_candidates`` rebuilds every international player's recency-weighted
    profile on each call (~20 s). Within one lock the history is identical, so
    the profiles are cached per rubric config. The only input that varies by
    slate is the try weight of the slate's own players, taken from their
    candidate rows; the cache uses the candidate rows of every slate in the
    lock instead. NCR and SRP try weights do not depend on role, so their
    profiles are unchanged; for the Six Nations rubric a profile changes only
    for a player whose forward/back role differs between a slate of the same
    month and his latest recorded role, and then only through the
    RugbyPass calibration median. The positional base remains per slate.
    """
    import model.empirical_unified as empirical_unified
    original_profiles, original_rp = empirical_unified.profiles, empirical_unified.rp_prior
    cache: dict = {}
    roles = candidates[['player_id', 'is_forward']].drop_duplicates('player_id', keep='last')

    def profiles(hist, club, tryw_by_pid, cfg):
        key = ('profiles', id(cfg))
        if key not in cache:
            weights = dict(tryw_by_pid)
            weights.update({str(p): cfg['try_weight'](f) for p, f in zip(roles.player_id, roles.is_forward)})
            cache[key] = original_profiles(hist, club, weights, cfg)
        return cache[key]

    def rp_prior(asof, cfg):
        key = ('rp', id(cfg), pd.Timestamp(asof))
        if key not in cache:
            cache[key] = original_rp(asof, cfg)
        return cache[key]

    empirical_unified.profiles, empirical_unified.rp_prior = profiles, rp_prior
    try:
        yield
    finally:
        empirical_unified.profiles, empirical_unified.rp_prior = original_profiles, original_rp


def lock_candidates(store: pd.DataFrame, fixture_slate: pd.Series) -> pd.DataFrame:
    rows = store[store['fixture_id'].astype(str).isin(fixture_slate.index)].copy()
    rows['slate'] = rows['fixture_id'].astype(str).map(fixture_slate)
    rows = rows.sort_values(['slate', 'match_at', 'fixture_id', 'team', 'player_id']).reset_index(drop=True)
    if rows.duplicated(KEY).any():
        raise ValueError('duplicate (fixture_id, player_id, team) candidate rows')
    return rows


def empirical_baselines(candidates: pd.DataFrame, train: pd.DataFrame, wr: pd.DataFrame,
                        lock: pd.Timestamp) -> pd.DataFrame:
    """Observable-rubric empirical baseline per slate (its positional base is cohort-relative)."""
    register_baseline_configs()
    baseline = pd.DataFrame({'label_row_id': candidates['label_row_id'].to_numpy()})
    for rubric in BASELINE_CONFIGS:
        values = pd.Series(np.nan, index=candidates['label_row_id'].to_numpy())
        for _, cohort in candidates.groupby('slate', sort=False):
            projected = project_candidates(cohort, train, f'bigbench_{rubric}', wr, asof=lock)
            values.loc[projected['label_row_id'].to_numpy()] = projected['predicted_points'].to_numpy(float)
        if values.isna().any():
            raise ValueError(f'{lock}: incomplete empirical baseline for {rubric}')
        baseline[rubric] = strip_potm(values.to_numpy(float), candidates['started'].astype(bool).to_numpy())
    return baseline


def fit_lock(store: pd.DataFrame, prepared: pd.DataFrame, lock: pd.Timestamp, fixture_slate: pd.Series,
             directory: Path, config: dict, wr: pd.DataFrame) -> dict:
    from research.context_experiment import fit_v4

    directory.mkdir(parents=True, exist_ok=True)
    done = directory/'fit.json'
    if done.exists():
        return json.loads(done.read_text())
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    register_baseline_configs()
    start = time.monotonic()
    timings = {}
    train = past_matches(store, lock).sort_values(SORT)
    rows = lock_candidates(store, fixture_slate)
    if set(train['fixture_id'].astype(str)) & set(rows['fixture_id'].astype(str)):
        raise ValueError(f'{lock}: evaluation fixture entered training')
    if pd.to_datetime(train['match_at'], utc=True).max() >= lock - pd.Timedelta(hours=3):
        raise AssertionError('training row is not available before the lock')
    masked = masked_candidates(rows.drop(columns=OUTCOME_COLUMNS, errors='ignore'))
    masked['label_row_id'] = np.arange(1, len(masked) + 1)
    features, candidates = build_frozen_feature_frames(train, masked, v4=True, prepared_train=prepared.loc[train.index])
    timings['features'] = time.monotonic() - start

    # Pre-lock matchup context for the post-processing candidates.
    t = time.monotonic()
    coefficients = json.loads((Path(__file__).resolve().parents[2]/'research'/'hillclimb_2026-10-02'/
                               'matchup_coefficients.json').read_text())
    context = matchup_features(candidates.reset_index(drop=True), train, lock, tuple(sorted(coefficients)))
    context['elo_edge'] = add_candidate_context(candidates.reset_index(drop=True),
                                                pre_match_context(train)[1])['ctx__elo_edge'].to_numpy(float)
    write_atomic(directory/'context.pkl', lambda tmp: context.to_pickle(tmp, compression=None))
    timings['context'] = time.monotonic() - t

    # Empirical fantasy baseline, per slate (its positional base is cohort-relative).
    t = time.monotonic()
    with memoised_baseline(candidates):
        baseline = empirical_baselines(candidates, train, wr, lock)
    write_atomic(directory/'baseline.pkl', lambda tmp: baseline.to_pickle(tmp, compression=None))
    timings['baseline'] = time.monotonic() - t

    t = time.monotonic()
    empirical = RobustEmpiricalEventModel(asof=lock).fit(train)
    with lightgbm_threads():
        tree = fit_v4(features, {})
    robust = EventWeightedBlend(empirical, tree, weight_v4=config['default_weight_v4'],
                                event_weights_v4=config['event_weights_v4'])
    timings['fit'] = time.monotonic() - t
    t = time.monotonic()
    raw = robust.predict_frame(candidates)
    keys = list(zip(candidates['fixture_id'].astype(str), candidates['player_id'].astype(str),
                    candidates['team'].astype(str)))
    if keys != [(p.fixture_id, p.player_id, p.team) for p in raw]:
        raise ValueError(f'{lock}: forecast rows are misaligned with candidates')
    save_raw(directory/f'{FIT_NAME}.jsonl.gz', raw)
    roles = candidates[ROLE_COLUMNS].reset_index(drop=True)
    write_atomic(directory/'candidates.pkl', lambda tmp: roles.to_pickle(tmp, compression=None))
    timings['predict'] = time.monotonic() - t
    summary = {
        'lock': pd.Timestamp(lock).isoformat(), 'training_rows': int(len(train)),
        'training_fixtures': int(train['fixture_id'].nunique()),
        'training_match_at_max': pd.to_datetime(train['match_at'], utc=True).max().isoformat(),
        'candidate_rows': int(len(candidates)), 'candidate_fixtures': int(candidates['fixture_id'].nunique()),
        'slates': sorted(candidates['slate'].unique().tolist()),
        'seconds': {k: round(v, 1) for k, v in timings.items()},
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
    }
    done.write_text(json.dumps(summary, indent=1) + '\n')
    return summary
