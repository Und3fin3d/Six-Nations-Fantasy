from __future__ import annotations

import pandas as pd


def tie_aware_top_n(predicted: pd.Series, actual: pd.Series, n: int) -> tuple[float, float]:
    n = min(n, len(actual))
    selected = set(predicted.nlargest(n).index)
    cutoff = actual.nlargest(n).iloc[-1]
    above = set(actual[actual > cutoff].index)
    tied = set(actual[actual == cutoff].index)
    remaining = n - len(above)
    tie_weight = remaining / max(len(tied), 1)
    hits = len(selected & above) + tie_weight * len(selected & tied)
    capture = float(actual.loc[list(selected)].sum() / max(actual.nlargest(n).sum(), 1e-9))
    return float(hits / n), capture
