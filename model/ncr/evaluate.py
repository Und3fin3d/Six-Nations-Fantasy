from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model.ncr.project import optimise
from model.unified.scoring import ncr_multiplier


def _key(x) -> str:
    return str(int(float(x)))


def load_actuals(feed_path: str = "data/ncr/feeds/players_latest.json") -> dict:
    players = json.load(open(feed_path))["Data"]["Value"]["Players"]
    if any(p.get("cur_gameday_id") not in (None, "") and _key(p["cur_gameday_id"]) != _key(p["gameday_id"]) for p in players):
        raise ValueError("fantasy feed points belong to a different gameweek")
    return {
        _key(p["id"]): (float(p.get("cur_gd_points") or 0), float(p.get("min_in_game") or 0), p.get("player_status"))
        for p in players
    }


def team_points(squad: pd.DataFrame, actuals: dict) -> float:
    total = 0.0
    for r in squad.itertuples():
        pts, _, status = actuals.get(_key(r.id), (0.0, 0.0, None))
        total += ncr_multiplier(status, is_capt=bool(r.is_capt), is_sub=bool(r.is_sub)) * pts
    return total


def evaluate(proj: pd.DataFrame, actuals: dict, squad: pd.DataFrame | None = None) -> dict:
    d = proj.copy()
    d["k"] = d["id"].map(_key)
    d["actual"] = d["k"].map(lambda k: actuals.get(k, (np.nan, 0, None))[0])
    live = d[np.isfinite(d["actual"]) & np.isfinite(d["starter_exp"])]
    y, p = live["actual"].to_numpy(float), live["starter_exp"].to_numpy(float)
    cal_mae = np.nan
    if len(live) >= 3:
        A = np.column_stack([np.ones(len(p)), p])
        cal_mae = float(np.abs(A @ np.linalg.lstsq(A, y, rcond=None)[0] - y).mean())
    opt15 = {k for k, _ in sorted(((k, actuals.get(k, (0,))[0]) for k in d["k"]), key=lambda x: -x[1])[:15]}
    out = {
        "mae": float(np.abs(p - y).mean()) if len(live) else np.nan, "cal_mae": cal_mae, "rho": spearmanr(p, y).correlation if len(live) >= 5 else np.nan,
        "n": len(live), "calibration_scope": "in_sample_diagnostic", "top15": len(set(d.nlargest(15, "starter_exp")["k"]) & opt15),
    }
    if squad is None and {"pos", "team", "value", "hemi"}.issubset(proj.columns):
        squad = optimise(proj)
    if squad is not None:
        out["xv"] = float(sum(actuals.get(_key(i), (0,))[0] for i in squad[~squad.is_sub].id))
        out["team"] = team_points(squad, actuals)
        cap = squad[squad.is_capt]
        if len(cap):
            out["captain"] = f"{cap.iloc[0]['name']} ({actuals.get(_key(cap.iloc[0].id), (0,))[0]:.0f})"
    return out


def fmt(label: str, m: dict) -> str:
    names = {"cal_mae": "cal_mae_in_sample"}
    parts = [f"{label:<26}"] + [f"{names.get(k, k)} {m[k]:6.3f}" if m.get(k) == m.get(k) else f"{names.get(k, k)}   nan" for k in ("mae", "cal_mae", "rho")]
    parts += [f"top15 {m[k]}/15" if k == "top15" else f"{k} {m[k]:.0f}" for k in ("xv", "team", "top15") if k in m]
    return "  ".join(parts + ([f"capt {m['captain']}"] if "captain" in m else []))
