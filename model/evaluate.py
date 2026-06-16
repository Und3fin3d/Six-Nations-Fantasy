#!/usr/bin/env python3
"""model/evaluate.py  —  evaluation on SELECTION quality, not just MAE (Phase 6).

The objective is picking a good XV, so MAE is necessary but not sufficient.  A
lower-MAE model that picks worse XVs is not the winner.

Metrics (all on modern-labelled rows):
  points_mae            MAE(official_pts, target_pts_hat), overall + by position
  value_of_xv           sum(actual pts of model's chosen XV) / hindsight-optimal XV,
                        respecting the positional quota, averaged over rounds
  captain_hitrate       model's #1 pick == actual #1 (and within actual top-3)
  topn_overlap          fraction of model's top-N that are in the actual top-N
  spearman_within_pos   within-position rank correlation of prediction vs actual
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# fantasy XV positional quota (sums to 15) over canonical positions.
XV_QUOTA = {
    "Prop": 2, "Hooker": 1, "Second-row": 2, "Back-row": 3,
    "Scrum-half": 1, "Fly-half": 1, "Centre": 2, "Back-three": 3,
}


def _labelled(pred: pd.DataFrame) -> pd.DataFrame:
    return pred[pred["has_label"].astype(bool) & pred["is_modern"].astype(bool)].copy()


def points_mae(pred: pd.DataFrame, score_col: str = "target_pts_hat") -> dict:
    lab = _labelled(pred).dropna(subset=[score_col])
    err = (lab["official_pts"] - lab[score_col]).abs()
    by_pos = (lab.assign(e=err).groupby("canonical_pos")["e"].mean()
              .round(3).to_dict())
    return {"overall": float(err.mean()), "by_pos": by_pos}


def value_of_xv(pred: pd.DataFrame, score_col: str = "target_pts_hat") -> tuple[float, list]:
    lab = _labelled(pred)
    ratios = []
    for _, g in lab.groupby("round"):
        picked = optimal = 0.0
        for pos, k in XV_QUOTA.items():
            gp = g[g["canonical_pos"] == pos]
            if gp.empty:
                continue
            kk = min(k, len(gp))
            picked += gp.nlargest(kk, score_col)["official_pts"].sum()
            optimal += gp.nlargest(kk, "official_pts")["official_pts"].sum()
        ratios.append(picked / optimal if optimal > 0 else np.nan)
    return float(np.nanmean(ratios)), ratios


def captain_hitrate(pred: pd.DataFrame, score_col: str = "target_pts_hat") -> tuple[float, float]:
    lab = _labelled(pred)
    top1 = top3 = n = 0
    for _, g in lab.groupby("round"):
        if g.empty:
            continue
        cap = g.loc[g[score_col].idxmax()]
        act = g["official_pts"].sort_values(ascending=False).to_numpy()
        thr3 = act[min(2, len(act) - 1)]
        top1 += int(cap["official_pts"] >= act[0])
        top3 += int(cap["official_pts"] >= thr3)
        n += 1
    return (top1 / n, top3 / n) if n else (np.nan, np.nan)


def topn_overlap(pred: pd.DataFrame, n: int = 15, score_col: str = "target_pts_hat") -> float:
    lab = _labelled(pred)
    fr = []
    for _, g in lab.groupby("round"):
        k = min(n, len(g))
        pm = set(g.nlargest(k, score_col)["player_id"])
        am = set(g.nlargest(k, "official_pts")["player_id"])
        fr.append(len(pm & am) / k)
    return float(np.mean(fr)) if fr else np.nan


def spearman_within_pos(pred: pd.DataFrame, score_col: str = "target_pts_hat") -> float:
    lab = _labelled(pred).dropna(subset=[score_col])
    num = den = 0.0
    for _, g in lab.groupby("canonical_pos"):
        if g[score_col].nunique() < 3 or len(g) < 5:
            continue
        rho = spearmanr(g[score_col], g["official_pts"]).correlation
        if np.isfinite(rho):
            num += rho * len(g)
            den += len(g)
    return float(num / den) if den else np.nan


def evaluate(pred: pd.DataFrame, score_col: str = "target_pts_hat", *, points: bool = True) -> dict:
    """All metrics for one prediction frame.  `points=False` for ordering-only
    overlays (e.g. the rank head) where the score is not a calibrated point."""
    vx, _ = value_of_xv(pred, score_col)
    c1, c3 = captain_hitrate(pred, score_col)
    out = {
        "value_xv": vx, "capt_top1": c1, "capt_top3": c3,
        "top15": topn_overlap(pred, 15, score_col),
        "spearman_pos": spearman_within_pos(pred, score_col),
    }
    out["mae"] = points_mae(pred, score_col)["overall"] if points else np.nan
    return out


def format_table(rows: dict[str, dict], title: str) -> str:
    cols = ["mae", "value_xv", "top15", "capt_top1", "capt_top3", "spearman_pos"]
    head = f"{'engine':16s} " + " ".join(f"{c:>10s}" for c in cols)
    lines = [f"\n=== {title} ===", head, "-" * len(head)]
    for name, m in rows.items():
        cells = []
        for c in cols:
            v = m.get(c, np.nan)
            cells.append("       nan" if v is None or (isinstance(v, float) and np.isnan(v))
                         else f"{v:10.3f}")
        lines.append(f"{name:16s} " + " ".join(cells))
    return "\n".join(lines)
