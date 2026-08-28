#!/usr/bin/env python3
"""
model/ncr_eval.py — one standard scorer for NCR projections vs realised results.
================================================================================
Every NCR model comparison should route through `evaluate()` so the metric bundle
is consistent and ALWAYS includes MAE (raw + calibrated). Historically these
numbers were computed ad-hoc per experiment and MAE was easy to forget; this
closes that.

Metrics (over players who actually took the field, min_in_game>0):
  mae            mean |starter_exp − actual_pts|           (calibration-sensitive)
  cal_mae        MAE after a 2-param least-squares rescale  (scale-free ranking-ish)
  rho            Spearman(starter_exp, actual) — ranking quality
  team           fantasy team points incl. captain 2× and super-sub 3×/0.5×/0×
  xv             the optimiser's XV actual points, NO multipliers (pure selection)
  top15          overlap of model top-15 with hindsight top-15
  captain        (name, actual_pts) of the optimiser's captain pick
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
FEED = ROOT / "data" / "ncr" / "feeds" / "players_latest.json"


def load_actuals(feed_path: Path = FEED) -> dict:
    """{player_id(str): (points, minutes_played_flag, status)} from a results feed."""
    players = json.loads(Path(feed_path).read_text())["Data"]["Value"]["Players"]
    out = {}
    for p in players:
        k = str(int(float(p["id"])))
        out[k] = (float(p.get("cur_gd_points") or 0),
                  float(p.get("min_in_game") or 0),
                  p.get("player_status"))
    return out


def _key(x) -> str:
    return str(int(float(x)))


def team_points(squad: pd.DataFrame, actuals: dict) -> float:
    total = 0.0
    for r in squad.itertuples():
        pts, played, status = actuals.get(_key(r.id), (0.0, 0.0, None))
        if r.is_capt:
            mult = 2.0
        elif r.is_sub:
            mult = 3.0 if (status == "B" and played > 0) else (0.5 if played > 0 else 0.0)
        else:
            mult = 1.0
        total += mult * pts
    return total


def evaluate(proj: pd.DataFrame, actuals: dict, squad: pd.DataFrame | None = None) -> dict:
    """proj needs columns id, name, starter_exp, status (+ pos/team/value/hemi if
    `squad` not supplied, so this can optimise one). `squad` is an optimiser result
    (in_squad/is_sub/is_capt) if already computed."""
    d = proj.copy()
    d["k"] = d["id"].map(_key)
    d["actual"] = d["k"].map(lambda k: actuals.get(k, (np.nan, 0, None))[0])
    d["played"] = d["k"].map(lambda k: actuals.get(k, (0, 0, None))[1])

    live = d[(d["played"] > 0) & d["actual"].notna()]
    y = live["actual"].to_numpy(float)
    p = live["starter_exp"].to_numpy(float)
    mae = float(np.abs(p - y).mean()) if len(live) else np.nan
    if len(live) >= 3:
        A = np.column_stack([np.ones(len(p)), p])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        cal_mae = float(np.abs(A @ coef - y).mean())
    else:
        cal_mae = np.nan
    rho = spearmanr(p, y).correlation if len(live) >= 5 else np.nan

    top15 = set(d.nlargest(15, "starter_exp")["k"])
    opt15 = {k for k, _ in sorted(((k, actuals.get(k, (0,))[0]) for k in d["k"]),
                                  key=lambda x: -x[1])[:15]}
    overlap = len(top15 & opt15)

    out = {"mae": mae, "cal_mae": cal_mae, "rho": rho, "n": len(live),
           "top15": overlap}
    if squad is None and {"pos", "team", "value", "hemi"}.issubset(proj.columns):
        import model.ncr_project as NP
        squad, _, _ = NP.optimise(proj)
    if squad is not None:
        xv = squad[~squad.is_sub]
        out["xv"] = float(sum(actuals.get(_key(i), (0,))[0] for i in xv.id))
        out["team"] = team_points(squad, actuals)
        cap = squad[squad.is_capt]
        if len(cap):
            cr = cap.iloc[0]
            out["captain"] = f"{cr['name']} ({actuals.get(_key(cr.id), (0,))[0]:.0f})"
    return out


def fmt(label: str, m: dict) -> str:
    parts = [f"{label:<26}"]
    for k in ("mae", "cal_mae", "rho"):
        parts.append(f"{k} {m[k]:6.3f}" if m.get(k) == m.get(k) else f"{k}   nan")
    for k in ("xv", "team", "top15"):
        if k in m:
            parts.append(f"{k} {m[k]:.0f}" if k != "top15" else f"top15 {m[k]}/15")
    if "captain" in m:
        parts.append(f"capt {m['captain']}")
    return "  ".join(parts)
