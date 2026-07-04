#!/usr/bin/env python3
"""Sealed deployment protocol: train on 2023+2024+2025, test on 2026 held out
ENTIRELY (forward-chained, no 2026 in training). Do the adopted changes
(position minutes + shrink-0.20) increase or decrease the 2026 benchmarks vs
the original (global minutes, no shrink)?  Single deterministic split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.data import load
from model.baselines import rates_to_components, score_recon, predict_rates
from model.train_components import predict_rates_lgbm_only
from autoresearch_mixed import _minutes, _ALL_COMPS
from train_mixedfold import _pick_top_k, MODE

ORIGINAL = dict(minutes="global", shrink={})
IMPROVED = dict(minutes="position",
                shrink={c: 0.20 for c in _ALL_COMPS}, shrink_target="naive")


def predict(df, tr_idx, te_idx, cfg):
    rates = predict_rates_lgbm_only(df, tr_idx, te_idx, MODE)
    if cfg.get("shrink"):
        tgt = predict_rates(df, tr_idx, te_idx, MODE, cfg.get("shrink_target", "naive"))
        for comp, lam in cfg["shrink"].items():
            rates[comp] = (1 - lam) * rates[comp].to_numpy(float) + lam * tgt[comp].to_numpy(float)
    minutes = _minutes(df, tr_idx, te_idx, cfg["minutes"])
    return score_recon(rates_to_components(rates, minutes),
                       df.iloc[te_idx]["is_forward"].to_numpy())


def metrics(df, te_idx, recon_hat):
    te = df.iloc[te_idx].copy(); te["recon_hat"] = recon_hat
    mae = float(np.abs(te["recon_pts"].to_numpy(float) - recon_hat).mean())
    vxv, top15 = [], []
    for rnd, g in te.groupby("round"):
        opt = float(_pick_top_k(g, "recon_pts")["recon_pts"].sum())
        vxv.append(float(_pick_top_k(g, "recon_hat")["recon_pts"].sum()) / opt if opt > 0 else np.nan)
        picked = set(_pick_top_k(g, "recon_hat")["player_id"])
        best = set(g.nlargest(15, "recon_pts")["player_id"])
        top15.append(len(picked & best) / 15.0)
    return mae, float(np.nanmean(vxv)), float(np.nanmean(top15)), vxv


def main() -> None:
    df = load().reset_index(drop=True)
    ok = df["recon_pts"].notna().to_numpy() & (df["minutes"].fillna(0) > 0).to_numpy()
    tr_idx = np.where(ok & df["season"].isin([2023, 2024, 2025]).to_numpy())[0]
    te_idx = np.where(ok & (df["season"] == 2026).to_numpy())[0]
    assert 2026 not in set(df.iloc[tr_idx]["season"]), "2026 leaked into training"
    print(f"train = 2023+2024+2025 ({len(tr_idx)} rows), test = 2026 held out ({len(te_idx)} rows)\n")

    om, ov, ot, ovr = metrics(df, te_idx, predict(df, tr_idx, te_idx, ORIGINAL))
    im, iv, it, ivr = metrics(df, te_idx, predict(df, tr_idx, te_idx, IMPROVED))

    print("=== SEALED 2026 (forward-chained deployment; 2026 never trained on) ===")
    print(f"{'benchmark':>16} {'original':>10} {'improved':>10} {'delta':>10}  better?")
    print(f"{'recon MAE':>16} {om:>10.4f} {im:>10.4f} {im-om:>+10.4f}  {'YES' if im<om else 'no'} (lower=better)")
    print(f"{'value_xv':>16} {ov:>10.4f} {iv:>10.4f} {iv-ov:>+10.4f}  {'YES' if iv>ov else 'no'} (higher=better)")
    print(f"{'top15':>16} {ot:>10.4f} {it:>10.4f} {it-ot:>+10.4f}  {'YES' if it>ot else 'no'} (higher=better)")
    print(f"\nper-round value_xv (2026 R1-5):")
    print(f"  original: {[round(x,3) for x in ovr]}")
    print(f"  improved: {[round(x,3) for x in ivr]}")


if __name__ == "__main__":
    main()
