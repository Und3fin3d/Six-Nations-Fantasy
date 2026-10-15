from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model.ncr.project import optimise


def _key(x) -> str:
    return str(int(float(x)))


def load_actuals(feed_path: str = "data/ncr/feeds/players_latest.json") -> dict:
    return {
        _key(p["id"]): (float(p.get("cur_gd_points") or 0), float(p.get("min_in_game") or 0), p.get("player_status"))
        for p in json.load(open(feed_path))["Data"]["Value"]["Players"]
    }


def team_points(squad: pd.DataFrame, actuals: dict) -> float:
    total = 0.0
    for r in squad.itertuples():
        pts, played, status = actuals.get(_key(r.id), (0.0, 0.0, None))
        total += (2.0 if r.is_capt else (3.0 if status == "B" and played > 0 else 0.5 if played > 0 else 0.0) if r.is_sub else 1.0) * pts
    return total


def evaluate(proj: pd.DataFrame, actuals: dict, squad: pd.DataFrame | None = None) -> dict:
    d = proj.copy()
    d["k"] = d["id"].map(_key)
    d["actual"] = d["k"].map(lambda k: actuals.get(k, (np.nan, 0, None))[0])
    d["played"] = d["k"].map(lambda k: actuals.get(k, (0, 0, None))[1])
    live = d[(d["played"] > 0) & d["actual"].notna()]
    y, p = live["actual"].to_numpy(float), live["starter_exp"].to_numpy(float)
    cal_mae = np.nan
    if len(live) >= 3:
        A = np.column_stack([np.ones(len(p)), p])
        cal_mae = float(np.abs(A @ np.linalg.lstsq(A, y, rcond=None)[0] - y).mean())
    opt15 = {k for k, _ in sorted(((k, actuals.get(k, (0,))[0]) for k in d["k"]), key=lambda x: -x[1])[:15]}
    out = {
        "mae": float(np.abs(p - y).mean()) if len(live) else np.nan, "cal_mae": cal_mae, "rho": spearmanr(p, y).correlation if len(live) >= 5 else np.nan,
        "n": len(live), "top15": len(set(d.nlargest(15, "starter_exp")["k"]) & opt15),
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
    parts = [f"{label:<26}"] + [f"{k} {m[k]:6.3f}" if m.get(k) == m.get(k) else f"{k}   nan" for k in ("mae", "cal_mae", "rho")]
    parts += [f"top15 {m[k]}/15" if k == "top15" else f"{k} {m[k]:.0f}" for k in ("xv", "team", "top15") if k in m]
    return "  ".join(parts + ([f"capt {m['captain']}"] if "captain" in m else []))
