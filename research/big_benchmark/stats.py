"""Paired bootstraps, within-slate correlations, calibration and power analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def paired_bootstrap(frame: pd.DataFrame, candidate: str, baseline: str, metric: str, *,
                     cluster: str, unit: str | None = None, n_boot: int = 4000, seed: int = 17) -> dict:
    """Bootstrap of mean(candidate - baseline) resampling whole clusters.

    ``unit`` (default ``cluster``) is the paired observation, e.g. a slate; when
    ``cluster`` is coarser (a block), whole blocks are resampled and the
    statistic stays the unit-weighted mean, so slates are weighted equally.
    """
    unit = unit or cluster
    keys = [unit] if unit == cluster else [cluster, unit]
    pivot = frame.pivot_table(index=keys, columns='engine', values=metric, aggfunc='mean')
    pivot = pivot.dropna(subset=[candidate, baseline])
    diff = (pivot[candidate] - pivot[baseline]).rename('diff').reset_index()
    if diff.empty:
        return {'n': 0, 'clusters': 0, 'mean': np.nan, 'sd': np.nan, 'p05': np.nan, 'p95': np.nan,
                'p025': np.nan, 'p975': np.nan, 'share_better': np.nan, 'share_positive': np.nan}
    sums = diff.groupby(cluster)['diff'].agg(['sum', 'size'])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(sums), (n_boot, len(sums)))
    samples = sums['sum'].to_numpy()[draw].sum(axis=1)/sums['size'].to_numpy()[draw].sum(axis=1)
    values = diff['diff'].to_numpy(float)
    return {'n': int(len(values)), 'clusters': int(len(sums)), 'mean': float(values.mean()),
            'sd': float(values.std(ddof=1)) if len(values) > 1 else np.nan,
            'p05': float(np.quantile(samples, .05)), 'p95': float(np.quantile(samples, .95)),
            'p025': float(np.quantile(samples, .025)), 'p975': float(np.quantile(samples, .975)),
            'share_better': float(np.mean(values < 0)), 'share_positive': float(np.mean(values > 0))}


def power_slates(sd: float, delta: float, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """Paired observations needed for a two-sided z-test to detect ``delta``."""
    if not np.isfinite(sd) or delta <= 0:
        return np.nan
    z = norm.ppf(1 - alpha/2) + norm.ppf(power)
    return float((z*sd/delta)**2)


def detectable(sd: float, n: float, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable mean difference with ``n`` paired observations."""
    z = norm.ppf(1 - alpha/2) + norm.ppf(power)
    return float(z*sd/np.sqrt(n)) if n > 0 else np.nan


def slate_correlations(players: pd.DataFrame, predicted: str, actual: str, by: str = 'slate') -> pd.DataFrame:
    rows = []
    for key, group in players.groupby(by, sort=False):
        group = group[np.isfinite(group[actual]) & np.isfinite(group[predicted])]
        if len(group) < 3 or group[predicted].std() < 1e-12 or group[actual].std() < 1e-12:
            continue
        rows.append({by: key, 'n': len(group), 'pearson': float(group[predicted].corr(group[actual])),
                     'spearman': float(group[predicted].corr(group[actual], method='spearman'))})
    return pd.DataFrame(rows)


def calibration_deciles(predicted: np.ndarray, actual: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({'predicted': predicted, 'actual': actual}).dropna()
    frame['decile'] = pd.qcut(frame['predicted'].rank(method='first'), bins, labels=False) + 1
    out = frame.groupby('decile').agg(n=('actual', 'size'), mean_predicted=('predicted', 'mean'),
                                      mean_actual=('actual', 'mean'), median_actual=('actual', 'median'))
    out['gap'] = out['mean_actual'] - out['mean_predicted']
    return out.reset_index()


def calibration_line(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(predicted) & np.isfinite(actual)
    slope, intercept = np.polyfit(predicted[mask], actual[mask], 1)
    return float(intercept), float(slope)
