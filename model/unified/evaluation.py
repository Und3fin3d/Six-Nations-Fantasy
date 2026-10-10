from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    reasons: tuple[str, ...]


def promotion_gate(
    candidate: dict[str, dict[str, float]], incumbent: dict[str, dict[str, float]], *, noninferiority_margin: float = 0.02, required_capture_gain: float = 0.03
) -> PromotionDecision:
    reasons, gains = [], []
    for competition in sorted(incumbent):
        if competition not in candidate:
            reasons.append(f"missing candidate results for {competition}")
            continue
        c, b = candidate[competition], incumbent[competition]
        if c["points_mae"] > b["points_mae"] * (1 + noninferiority_margin):
            reasons.append(f"{competition} points MAE regressed by more than {noninferiority_margin:.0%}")
        for key in ("top_10_capture", "top_25_capture", "top_50_capture", "top_100_capture"):
            if key in b and key in c:
                if c[key] < b[key] - noninferiority_margin:
                    reasons.append(f"{competition} {key} regressed by more than {noninferiority_margin:.0%}")
                gains.append(c[key] - b[key])
    if not gains or float(np.mean(gains)) < required_capture_gain:
        reasons.append(f"mean top-N point-capture gain is below {required_capture_gain:.0%}")
    return PromotionDecision(not reasons, tuple(reasons))
