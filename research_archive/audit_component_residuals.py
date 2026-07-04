#!/usr/bin/env python3
"""Residual audit: which components, positions, and roles drive 2025 dev error?

Reads research/dev_predictions_2025.csv (promoted-config predictions with
hat_* component columns) and joins realised y_* components.  Attributes the
point-scale residual (official - predicted) to components using the modern
scorer weights, sliced by position group, starter/bench, and designated-kicker
status.  Pure diagnostics: no model changes, no leakage risk (2025 labels are
the dev season).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.data import load

W = {
    "tackles": 1.0, "try_assists": 4.0, "conversion_goals": 2.0,
    "penalty_goals": 3.0, "drop_goals_converted": 4.0, "defenders_beaten": 2.0,
    "offload": 2.0, "tackle_turnover": 5.0, "penalties_conceded": -1.0,
    "yellow_cards": -5.0, "red_cards": -8.0,
}


def component_points(row_vals: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Point contribution per component (modern formula)."""
    out = pd.DataFrame(index=row_vals.index)
    isf = row_vals["is_forward"].astype(bool).to_numpy()
    try_val = np.where(isf, 15.0, 10.0)
    out["tries"] = try_val * row_vals[f"{prefix}tries"].to_numpy(float)
    out["metres"] = np.floor(row_vals[f"{prefix}metres"].to_numpy(float) / 10.0)
    for comp, w in W.items():
        out[comp] = w * row_vals[f"{prefix}{comp}"].to_numpy(float)
    return out


def main() -> None:
    df = load()
    pred = pd.read_csv("research/dev_predictions_2025.csv")
    keys = ["fixture_id", "player_id"]
    ycols = [c for c in df.columns if c.startswith("y_")]
    aux = df[keys + ycols + ["minutes", "role_goal_kicker_rate", "started"]].copy()
    aux = aux.rename(columns={"started": "started_store"})
    m = pred.merge(aux, on=keys, how="left", validate="one_to_one")
    m = m[m["has_label"].astype(bool)].copy()
    m["resid"] = m["official_pts"] - m["target_pts_hat"]

    print(f"rows={len(m)}  MAE={m['resid'].abs().mean():.4f}  bias={m['resid'].mean():+.4f}")

    # --- per-position bias/MAE, starters vs bench --------------------------
    m["seg"] = np.where(m["started"].astype(bool), "starter", "bench")
    print("\n=== residual by position x starter (official - pred) ===")
    g = m.groupby(["canonical_pos", "seg"])["resid"].agg(["mean", lambda s: s.abs().mean(), "count"])
    g.columns = ["bias", "mae", "n"]
    print(g.round(2).to_string())

    # --- component attribution ---------------------------------------------
    hat = component_points(
        m.rename(columns={f"hat_{c}": f"hat_{c}" for c in W}), "hat_")
    real = component_points(
        m.rename(columns={f"y_{c}": f"y_{c}" for c in W}), "y_")
    delta = real - hat  # positive => model underpredicts this component's points
    delta["position"] = m["canonical_pos"].to_numpy()
    delta["seg"] = m["seg"].to_numpy()

    print("\n=== mean point-residual per component by position (starters only) ===")
    d = delta[delta["seg"] == "starter"].groupby("position").mean(numeric_only=True)
    d["TOTAL_comp"] = d.sum(axis=1)
    print(d.round(2).to_string())

    print("\n=== mean |point-residual| per component (starters, all positions) ===")
    ad = (real - hat).abs()
    ad["seg"] = m["seg"].to_numpy()
    print(ad[ad["seg"] == "starter"].mean(numeric_only=True).round(3).sort_values(ascending=False).to_string())

    # --- minutes ------------------------------------------------------------
    m["min_err"] = m["minutes"] - m["minutes_hat"]
    print("\n=== minutes error by position x seg ===")
    gm = m.groupby(["canonical_pos", "seg"])["min_err"].agg(["mean", lambda s: s.abs().mean()])
    gm.columns = ["bias", "mae"]
    print(gm.round(1).to_string())

    # --- kicking deep dive ----------------------------------------------------
    kick_hat_pts = 2.0 * m["hat_conversion_goals"] + 3.0 * m["hat_penalty_goals"]
    kick_real_pts = 2.0 * m["y_conversion_goals"] + 3.0 * m["y_penalty_goals"]
    m["kick_hat_pts"] = kick_hat_pts
    m["kick_real_pts"] = kick_real_pts
    m["is_kicker"] = m["role_goal_kicker_rate"].fillna(0) > 0.5
    print("\n=== kicking points: predicted vs realised ===")
    for lab, sel in [("designated kicker (rate>0.5)", m["is_kicker"]),
                     ("other players", ~m["is_kicker"])]:
        s = m[sel]
        print(f"{lab:32s} n={len(s):4d} hat={s['kick_hat_pts'].mean():6.2f} "
              f"real={s['kick_real_pts'].mean():6.2f} "
              f"comp_mae={ (s['kick_real_pts']-s['kick_hat_pts']).abs().mean():6.2f}")

    # per fixture-team realised total kicking vs predicted total kicking
    ft = m.groupby(["fixture_id", "team"]).agg(
        hat=("kick_hat_pts", "sum"), real=("kick_real_pts", "sum")).reset_index()
    print("\nfixture-team kicking totals: "
          f"n={len(ft)} hat_mean={ft['hat'].mean():.2f} real_mean={ft['real'].mean():.2f} "
          f"corr={ft['hat'].corr(ft['real']):.3f} "
          f"mae={(ft['real']-ft['hat']).abs().mean():.2f}")
    ft["spread_real"] = ft["real"]
    print("realised fixture-team kicking spread: "
          f"std={ft['real'].std():.2f} p10={ft['real'].quantile(0.1):.1f} "
          f"p90={ft['real'].quantile(0.9):.1f}")
    print("predicted fixture-team kicking spread: "
          f"std={ft['hat'].std():.2f} p10={ft['hat'].quantile(0.1):.1f} "
          f"p90={ft['hat'].quantile(0.9):.1f}")


if __name__ == "__main__":
    main()
