"""SK engine: robust P3 with the status-aware, within-position-shrunk empirical component.

``model.unified.raw_benchmark.status_rates.ShrunkStatusEmpiricalEventModel``
replaces the robust empirical component; the tree component and the blend
weights are unchanged. Only the cheap empirical component is refitted per lock.

The lock fits saved only the blended reference, so the tree component is
recovered exactly from it. The plain robust empirical component is refitted
at the same lock (it is deterministic), and the ``EventWeightedBlend``
arithmetic is inverted per event:

    m_tree = (m_blend - (1 - w) m_emp) / w
    v_tree = (v_blend - (1 - w)(v_emp + (m_emp - m_blend)^2)) / w - (m_tree - m_blend)^2

and the tree is re-blended with the status-aware component using the same
moment-matching rule. Events present only in the empirical component take
the new component; events only in the tree are unchanged. Means are exact;
variances are exact except where the original negative-binomial dispersion
hit its 0.05 floor, which never affects expected points (only lognormal
metres variance enters the rubrics, and it is exact).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.history import past_matches
from model.unified.contracts import EventDistribution, RawPrediction
from model.unified.raw_benchmark.blend import _variance
from model.unified.raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from model.unified.schema import distribution_family

from .fit import FIT_NAME, SORT, load_raw, lock_name, save_raw

SK_NAME = 'sk_p3'


def _from_moments(family: str, mean: float, variance: float) -> EventDistribution:
    """Same family conversion as ``raw_benchmark.blend._blend_distribution``."""
    if family == 'bernoulli':
        return EventDistribution(family, min(max(mean, 0.0), 1.0), 1.0)
    if family == 'negative_binomial':
        return EventDistribution(family, max(mean, 0.0), max(mean*mean/max(variance - mean, 1e-6), 0.05))
    return EventDistribution(family, max(mean, 0.0), max(variance, 1e-6))


def reblend_one(blend: EventDistribution, plain: EventDistribution | None, new: EventDistribution | None,
                weight_v4: float, target: str) -> EventDistribution:
    if plain is None or new is None:
        return blend
    family = 'lognormal' if target == 'minutes' else distribution_family(target)
    w = float(weight_v4)
    if w >= 1.0:
        return blend
    if w <= 0.0:
        return new
    m_b, v_b = blend.mean, _variance(blend)
    m_e, v_e = plain.mean, _variance(plain)
    m_t = (m_b - (1 - w)*m_e)/w
    v_t = max((v_b - (1 - w)*(v_e + (m_e - m_b)**2))/w - (m_t - m_b)**2, 1e-9)
    m_n, v_n = new.mean, _variance(new)
    mean = w*m_t + (1 - w)*m_n
    variance = w*(v_t + (m_t - mean)**2) + (1 - w)*(v_n + (m_n - mean)**2)
    return _from_moments(family, mean, variance)


def reblend(blended: list[RawPrediction], plain: list[RawPrediction], new: list[RawPrediction],
            default_weight_v4: float, event_weights_v4: dict) -> list[RawPrediction]:
    output = []
    for b, p, n in zip(blended, plain, new, strict=True):
        if (b.fixture_id, b.player_id, b.team) != (n.fixture_id, n.player_id, n.team):
            raise ValueError('status-aware forecasts are misaligned')
        events = {}
        for event, dist in b.events.items():
            weight = event_weights_v4.get(event, default_weight_v4)
            plain_dist, new_dist = p.events.get(event), n.events.get(event)
            if plain_dist is not None and dist == plain_dist:  # empirical-only event
                events[event] = new_dist if new_dist is not None else dist
            else:
                events[event] = reblend_one(dist, plain_dist, new_dist, weight, event)
        minutes = reblend_one(b.minutes, p.minutes, n.minutes,
                              event_weights_v4.get('minutes', default_weight_v4), 'minutes')
        output.append(RawPrediction(b.fixture_id, b.player_id, b.player_name, b.team, b.opponent, b.position,
                                    b.is_forward, events, minutes, {**b.metadata, 'status_rates': 'shrunk'}))
    return output


def sk_lock(store: pd.DataFrame, lock: pd.Timestamp, directory: Path, config: dict) -> Path:
    from model.unified.raw_benchmark.status_rates import ShrunkStatusEmpiricalEventModel
    path = directory/f'{SK_NAME}.jsonl.gz'
    if path.exists():
        return path
    train = past_matches(store, lock).sort_values(SORT)
    candidates = pd.read_pickle(directory/'candidates.pkl')
    reference = load_raw(directory/f'{FIT_NAME}.jsonl.gz')
    plain = RobustEmpiricalEventModel(asof=lock).fit(train).predict_frame(candidates)
    shrunk = ShrunkStatusEmpiricalEventModel(asof=lock).fit(train).predict_frame(candidates)
    save_raw(path, reblend(reference, plain, shrunk, config['default_weight_v4'], config['event_weights_v4']))
    return path


def sk_stage(store: pd.DataFrame, manifest: dict, output: Path, locks, data_root: Path) -> None:
    config = json.loads((data_root/'unified'/'p3_hillclimb'/'config.json').read_text())
    for lock in manifest['locks']:
        name = lock_name(pd.Timestamp(lock))
        if locks and name not in locks:
            continue
        directory = output/'locks'/name
        if not (directory/'fit.json').exists():
            print(f'sk: lock {name} not fitted yet; skipped', flush=True)
            continue
        if (directory/f'{SK_NAME}.jsonl.gz').exists():
            continue
        sk_lock(store, pd.Timestamp(lock), directory, config)
        print(f'sk: lock {name} done', flush=True)
