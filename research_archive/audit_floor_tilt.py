#!/usr/bin/env python3
"""Evidence gate: does a floor-tilted XV selector beat the incumbent value?

Experiment 18 showed that *boosting* the sparse spike components (removing the
component layer's volume deflation) HURTS value_team, because the deflation
acts as implicit variance-regularisation for quota-constrained selection.  The
symmetric, untested question: does *suppressing* spike content in the
selection score (a deliberate floor tilt, decoupled from target_pts_hat) help?

This audit re-picks the XV directly from the promoted 2025 prediction CSV under
a floor tilt

    sel_tilt(a) = target_pts_hat - a * spike_pts

where spike_pts is the exact point contribution of the high-variance
components (tries, assists, turnovers, defenders-beaten, offloads, kicks,
cards) and floor content (tackles, metres) is left untouched.  a=0 reproduces
the incumbent selection.  No retraining; this is the exact re-scoring the real
selector would perform.  We report round-robustness so a one-round fluke is
visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.evaluate import _labelled, _pick_xv, _supersub_pool, SUPERSUB_MULTIPLIER

# Spike components and their fantasy point weights (try weight is positional).
SPIKE_WEIGHTS = {
    "try_assists": 4.0, "conversion_goals": 2.0, "penalty_goals": 3.0,
    "drop_goals_converted": 4.0, "defenders_beaten": 2.0, "offload": 2.0,
    "tackle_turnover": 5.0, "penalties_conceded": -1.0,
    "yellow_cards": -5.0, "red_cards": -8.0,
}


def spike_points(pred: pd.DataFrame) -> np.ndarray:
    fwd = pred["is_forward"].to_numpy(bool)
    try_w = np.where(fwd, 15.0, 10.0)
    sp = try_w * pred["hat_tries"].to_numpy(float)
    for comp, w in SPIKE_WEIGHTS.items():
        sp = sp + w * pred[f"hat_{comp}"].to_numpy(float)
    return sp


def value_team_from(pred: pd.DataFrame, sel: np.ndarray) -> tuple[float, list]:
    """value_of_team but with an externally supplied selection score, captain
    and supersub left on their incumbent columns (decisions decoupled)."""
    lab = _labelled(pred).copy()
    lab["_sel"] = pd.Series(sel, index=pred.index).loc[lab.index].to_numpy()
    ratios = []
    for _, g in lab.groupby("round"):
        xv = _pick_xv(g, "_sel")
        if xv.empty:
            ratios.append(np.nan)
            continue
        base = float(xv["official_pts"].sum())
        cap = float(xv.loc[xv["captain_score"].idxmax(), "official_pts"]) \
            if "captain_score" in xv else float(xv.loc[xv["_sel"].idxmax(), "official_pts"])
        pool = _supersub_pool(g, set(xv["player_id"]))
        ss = float(pool.loc[pool["supersub_score"].idxmax(), "official_pts"]) \
            if ("supersub_score" in pool and not pool.empty) else 0.0
        picked = base + cap + SUPERSUB_MULTIPLIER * ss

        opt_xv = _pick_xv(g, "official_pts")
        opt_base = float(opt_xv["official_pts"].sum())
        opt_cap = float(opt_xv["official_pts"].max())
        opt_pool = _supersub_pool(g, set(opt_xv["player_id"]))
        opt_ss = float(opt_pool["official_pts"].max()) if not opt_pool.empty else 0.0
        optimal = opt_base + opt_cap + SUPERSUB_MULTIPLIER * opt_ss
        ratios.append(picked / optimal if optimal > 0 else np.nan)
    return float(np.nanmean(ratios)), ratios


def value_xv_from(pred: pd.DataFrame, sel: np.ndarray) -> float:
    lab = _labelled(pred).copy()
    lab["_sel"] = pd.Series(sel, index=pred.index).loc[lab.index].to_numpy()
    ratios = []
    for _, g in lab.groupby("round"):
        picked = float(_pick_xv(g, "_sel")["official_pts"].sum())
        optimal = float(_pick_xv(g, "official_pts")["official_pts"].sum())
        ratios.append(picked / optimal if optimal > 0 else np.nan)
    return float(np.nanmean(ratios))


def main() -> None:
    pred = pd.read_csv("research/dev_predictions_2025.csv")
    sp = spike_points(pred)
    target = pred["target_pts_hat"].to_numpy(float)

    base_val, base_ratios = value_team_from(pred, target)
    print(f"incumbent (a=0): value_team={base_val:.4f}  "
          f"value_xv={value_xv_from(pred, target):.4f}  "
          f"rounds={[round(r,3) for r in base_ratios]}")
    print(f"\nmean spike_pts per row = {sp.mean():.2f}  "
          f"(as share of target {sp.mean()/np.nanmean(target)*100:.1f}%)\n")

    print(f"{'alpha':>6} {'value_team':>11} {'value_xv':>9} {'rounds_up':>10} "
          f"{'per-round vs incumbent':>28}")
    for a in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        sel = target - a * sp
        val, ratios = value_team_from(pred, sel)
        vxv = value_xv_from(pred, sel)
        deltas = [r - b for r, b in zip(ratios, base_ratios)]
        up = sum(1 for d in deltas if d > 1e-9)
        down = sum(1 for d in deltas if d < -1e-9)
        print(f"{a:>6.2f} {val:>11.4f} {vxv:>9.4f} {f'{up}/5 (-{down})':>10} "
              f"  {[round(d,3) for d in deltas]}")


if __name__ == "__main__":
    main()
