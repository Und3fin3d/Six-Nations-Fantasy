#!/usr/bin/env python3
"""Winner's-curse / lower-confidence-bound selector audit.

Evidence (audit_full_picks.py): the model's WRONG XV picks are systematically
OVER-predicted tight forwards (Second-row pred 21.2 -> act 12.7, Hooker +7.7,
Prop +4.4).  This is selection-induced optimism: within a low-signal position
the top-ranked pick is disproportionately a noise-inflated one.

The principled correction is a pessimistic (lower-confidence-bound) selection:
demote picks whose prediction is UNRELIABLE (estimator uncertainty), not whose
OUTCOME is volatile (that was the closed floor-tilt).  uncertainty_proxy =
cold-start + low-minutes-history + XGB/LGBM-disagreement is PIT-available and
varies within position, so it can re-order the within-position top-k.

Two checks, on the promoted 2025 CSV (no retraining):
 1. Does uncertainty separate over-picks (bad) from the model's good picks and
    from the optimal picks it missed?
 2. LCB re-pick: sel_lcb = sel_score - lam * zscore_within_pos(uncertainty);
    measure value_team / value_xv and round-robustness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.evaluate import _labelled, _pick_xv, _supersub_pool, SUPERSUB_MULTIPLIER


def zscore_within_pos(df: pd.DataFrame, col: str) -> np.ndarray:
    z = np.zeros(len(df))
    vals = df[col].to_numpy(float)
    pos = df["canonical_pos"].to_numpy()
    for p in np.unique(pos):
        m = pos == p
        v = vals[m]
        s = v.std()
        z[m] = (v - v.mean()) / s if s > 1e-9 else 0.0
    return z


def value_from(pred: pd.DataFrame, sel: np.ndarray, base_ratios=None):
    lab = _labelled(pred).copy()
    lab["_sel"] = pd.Series(sel, index=pred.index).loc[lab.index].to_numpy()
    ratios, xvr = [], []
    for _, g in lab.groupby("round"):
        xv = _pick_xv(g, "_sel")
        base = float(xv["official_pts"].sum())
        cap = float(xv.loc[xv["captain_score"].idxmax(), "official_pts"])
        pool = _supersub_pool(g, set(xv["player_id"]))
        ss = float(pool.loc[pool["supersub_score"].idxmax(), "official_pts"]) if not pool.empty else 0.0
        picked = base + cap + SUPERSUB_MULTIPLIER * ss
        opt_xv = _pick_xv(g, "official_pts")
        opt_base = float(opt_xv["official_pts"].sum())
        opt_cap = float(opt_xv["official_pts"].max())
        opt_pool = _supersub_pool(g, set(opt_xv["player_id"]))
        opt_ss = float(opt_pool["official_pts"].max()) if not opt_pool.empty else 0.0
        optimal = opt_base + opt_cap + SUPERSUB_MULTIPLIER * opt_ss
        ratios.append(picked / optimal if optimal > 0 else np.nan)
        xvr.append(base / opt_base if opt_base > 0 else np.nan)
    return float(np.nanmean(ratios)), ratios, float(np.nanmean(xvr))


def main() -> None:
    pred = pd.read_csv("research/dev_predictions_2025.csv")
    lab = _labelled(pred)

    # ---- Check 1: uncertainty of over-picks vs good picks vs missed optimal
    up = pred["uncertainty_proxy"].to_numpy(float)
    print(f"uncertainty_proxy distribution: mean={up.mean():.2f} "
          f"values={pd.Series(up).value_counts().sort_index().to_dict()}")
    over_u, good_u, missed_u = [], [], []
    for rnd, g in lab.groupby("round"):
        xv = _pick_xv(g, "sel_score"); opt = _pick_xv(g, "official_pts")
        mids, oids = set(xv["player_id"]), set(opt["player_id"])
        over_u += g[g["player_id"].isin(mids - oids)]["uncertainty_proxy"].tolist()
        good_u += g[g["player_id"].isin(mids & oids)]["uncertainty_proxy"].tolist()
        missed_u += g[g["player_id"].isin(oids - mids)]["uncertainty_proxy"].tolist()
    print(f"\nmean uncertainty_proxy:")
    print(f"  over-picks (model in XV, not optimal): {np.mean(over_u):.3f}  (n={len(over_u)})")
    print(f"  good picks (model in XV, in optimal):  {np.mean(good_u):.3f}  (n={len(good_u)})")
    print(f"  missed  (optimal, not in model XV):    {np.mean(missed_u):.3f}  (n={len(missed_u)})")

    # by-position over-pick uncertainty (tight forwards vs rest)
    print("\nover-pick uncertainty by position:")
    rows = []
    for rnd, g in lab.groupby("round"):
        xv = _pick_xv(g, "sel_score"); opt = _pick_xv(g, "official_pts")
        over = g[g["player_id"].isin(set(xv["player_id"]) - set(opt["player_id"]))]
        for _, r in over.iterrows():
            rows.append((r["canonical_pos"], r["uncertainty_proxy"],
                         r["target_pts_hat"] - r["official_pts"]))
    od = pd.DataFrame(rows, columns=["pos", "unc", "over_err"])
    print(od.groupby("pos").agg(n=("unc","size"), mean_unc=("unc","mean"),
          over_err=("over_err","mean")).round(2).to_string())

    # ---- Check 2: LCB re-pick value sweep
    base_val, base_ratios, base_xv = value_from(pred, pred["sel_score"].to_numpy(float))
    print(f"\nincumbent: value_team={base_val:.4f} value_xv={base_xv:.4f} "
          f"rounds={[round(r,3) for r in base_ratios]}")
    uz = zscore_within_pos(pred, "uncertainty_proxy")
    print(f"\n{'lambda':>7} {'value_team':>11} {'value_xv':>9} {'rounds_up':>11}  per-round-delta")
    for lam in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]:
        sel = pred["sel_score"].to_numpy(float) - lam * uz
        val, ratios, vxv = value_from(pred, sel)
        deltas = [r - b for r, b in zip(ratios, base_ratios)]
        upn = sum(1 for d in deltas if d > 1e-9); dn = sum(1 for d in deltas if d < -1e-9)
        print(f"{lam:>7.2f} {val:>11.4f} {vxv:>9.4f} {f'{upn}/5 (-{dn})':>11}  {[round(d,3) for d in deltas]}")


if __name__ == "__main__":
    main()
