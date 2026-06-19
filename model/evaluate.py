#!/usr/bin/env python3
"""model/evaluate.py  —  evaluation on SELECTION quality, not just MAE (Phase 6).

The objective is picking a good XV, so MAE is necessary but not sufficient.  A
lower-MAE model that picks worse XVs is not the winner.

Metrics (all on modern-labelled rows):
  points_mae            MAE(official_pts, target_pts_hat), overall + by position
  value_of_xv           sum(actual pts of model's chosen XV) / hindsight-optimal XV,
                        respecting the positional quota, averaged over rounds
  value_of_team         fantasy-team value: XV + model captain + model supersub,
                        divided by hindsight XV + captain + supersub
  captain_hit_rates     model's #1 pick lands inside actual top-N
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
CAPTAIN_MULTIPLIER = 2.0
SUPERSUB_MULTIPLIER = 3.0


def _labelled(pred: pd.DataFrame) -> pd.DataFrame:
    return pred[pred["has_label"].astype(bool) & pred["is_modern"].astype(bool)].copy()


def points_mae(pred: pd.DataFrame, score_col: str = "target_pts_hat") -> dict:
    lab = _labelled(pred).dropna(subset=[score_col])
    err = (lab["official_pts"] - lab[score_col]).abs()
    by_pos = (lab.assign(e=err).groupby("canonical_pos")["e"].mean()
              .round(3).to_dict())
    return {"overall": float(err.mean()), "by_pos": by_pos}


def bench_mae(pred: pd.DataFrame, score_col: str = "target_pts_hat") -> float:
    lab = _labelled(pred).dropna(subset=[score_col])
    if "started" not in lab.columns:
        return np.nan
    bench = lab[~lab["started"].astype(bool)]
    if bench.empty:
        return np.nan
    return float((bench["official_pts"] - bench[score_col]).abs().mean())


def _pick_xv(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    picked = []
    for pos, k in XV_QUOTA.items():
        gp = df[df["canonical_pos"] == pos]
        if gp.empty:
            continue
        kk = min(k, len(gp))
        picked.append(gp.nlargest(kk, score_col))
    if not picked:
        return df.iloc[0:0].copy()
    return pd.concat(picked)


def _supersub_pool(df: pd.DataFrame, selected_ids: set) -> pd.DataFrame:
    pool = df[~df["player_id"].isin(selected_ids)].copy()
    if "started" in pool.columns:
        bench = pool[~pool["started"].astype(bool)].copy()
        if not bench.empty:
            pool = bench
    return pool


def value_of_xv(pred: pd.DataFrame, score_col: str = "target_pts_hat") -> tuple[float, list]:
    lab = _labelled(pred)
    ratios = []
    for _, g in lab.groupby("round"):
        picked = float(_pick_xv(g, score_col)["official_pts"].sum())
        optimal = float(_pick_xv(g, "official_pts")["official_pts"].sum())
        ratios.append(picked / optimal if optimal > 0 else np.nan)
    return float(np.nanmean(ratios)), ratios


def value_of_team(
    pred: pd.DataFrame,
    score_col: str = "target_pts_hat",
    *,
    captain_score_col: str | None = None,
    supersub_score_col: str | None = None,
) -> tuple[float, list]:
    """Fantasy-team value with multiplier roles.

    The XV is picked by `score_col`.  The captain is picked from that XV by
    `captain_score_col` (default same score).  The supersub is picked from
    non-starters outside the XV by `supersub_score_col` (default same score).
    """
    lab = _labelled(pred)
    captain_score_col = captain_score_col or score_col
    supersub_score_col = supersub_score_col or score_col
    ratios = []
    for _, g in lab.groupby("round"):
        xv = _pick_xv(g, score_col)
        if xv.empty:
            ratios.append(np.nan)
            continue
        picked_base = float(xv["official_pts"].sum())
        captain_extra = float(
            xv.loc[xv[captain_score_col].idxmax(), "official_pts"]
        ) if captain_score_col in xv.columns else float(
            xv.loc[xv[score_col].idxmax(), "official_pts"]
        )
        pool = _supersub_pool(g, set(xv["player_id"]))
        supersub = 0.0
        if not pool.empty:
            ss_col = supersub_score_col if supersub_score_col in pool.columns else score_col
            supersub = float(pool.loc[pool[ss_col].idxmax(), "official_pts"])
        picked = picked_base + captain_extra + SUPERSUB_MULTIPLIER * supersub

        opt_xv = _pick_xv(g, "official_pts")
        opt_base = float(opt_xv["official_pts"].sum())
        opt_captain = float(opt_xv["official_pts"].max()) if not opt_xv.empty else 0.0
        opt_pool = _supersub_pool(g, set(opt_xv["player_id"]))
        opt_supersub = float(opt_pool["official_pts"].max()) if not opt_pool.empty else 0.0
        optimal = opt_base + opt_captain + SUPERSUB_MULTIPLIER * opt_supersub
        ratios.append(picked / optimal if optimal > 0 else np.nan)
    return float(np.nanmean(ratios)), ratios


def captain_hit_rates(
    pred: pd.DataFrame, score_col: str = "target_pts_hat", topns: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    lab = _labelled(pred)
    hits = {n: 0 for n in topns}
    total = 0
    for _, g in lab.groupby("round"):
        if g.empty:
            continue
        cap = g.loc[g[score_col].idxmax()]
        act = g["official_pts"].sort_values(ascending=False).to_numpy()
        for n in topns:
            threshold = act[min(n - 1, len(act) - 1)]
            hits[n] += int(cap["official_pts"] >= threshold)
        total += 1
    return {f"capt_top{n}": (hits[n] / total if total else np.nan) for n in topns}


def captain_hitrate(pred: pd.DataFrame, score_col: str = "target_pts_hat") -> tuple[float, float]:
    rates = captain_hit_rates(pred, score_col, (1, 3))
    return rates["capt_top1"], rates["capt_top3"]


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
    vt, _ = value_of_team(
        pred,
        score_col,
        supersub_score_col="supersub_score" if "supersub_score" in pred.columns else None,
        captain_score_col="captain_score" if "captain_score" in pred.columns else None,
    )
    capt = captain_hit_rates(pred, score_col, (1, 3, 5))
    out = {
        "value_team": vt,
        "value_xv": vx, **capt,
        "top15": topn_overlap(pred, 15, score_col),
        "top30": topn_overlap(pred, 30, score_col),
        "spearman_pos": spearman_within_pos(pred, score_col),
    }
    out["mae"] = points_mae(pred, score_col)["overall"] if points else np.nan
    out["bench_mae"] = bench_mae(pred, score_col) if points else np.nan
    return out


def format_table(rows: dict[str, dict], title: str) -> str:
    cols = [
        "mae", "bench_mae", "value_team", "value_xv", "top15", "top30",
        "capt_top1", "capt_top3", "capt_top5", "spearman_pos",
    ]
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
