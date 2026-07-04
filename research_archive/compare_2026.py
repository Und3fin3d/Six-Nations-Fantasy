#!/usr/bin/env python3
"""Isolate the PROTOCOL effect on 2026: same from-scratch model, two ways.

  A) SEALED HOLDOUT : train on 2023+2024+2025, predict 2026 (never seen).
  B) MIXED-YEAR FOLD: 2026 rounds sit in CV folds trained partly on OTHER 2026
                      rounds (+ other years) — the friend's cross-year protocol.

Same model + same recon metric in both, so any gap is the protocol (i.e. how
much peeking at sibling 2026 rounds during training flatters 2026), not model
complexity. recon_pts = deterministic modern score of realised components.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.data import load
from model.baselines import rates_to_components, score_recon
from model.train_components import predict_rates_lgbm_only
from train_mixedfold import _minutes, _pick_top_k  # helpers (no argparse on import)

MODE = "post_team_sheet"


def predict_recon(df, tr_idx, te_idx):
    rates = predict_rates_lgbm_only(df, tr_idx, te_idx, MODE)
    minutes = _minutes(df, tr_idx, te_idx)
    comps = rates_to_components(rates, minutes)
    return score_recon(comps, df.iloc[te_idx]["is_forward"].to_numpy())


def metrics_2026(df, te_pos, recon_hat):
    te = df.iloc[te_pos].copy()
    te["recon_hat"] = recon_hat
    mae = float(np.abs(te["recon_pts"].to_numpy(float) - recon_hat).mean())
    vxv = []
    for rnd, g in te.groupby("round"):
        picked = float(_pick_top_k(g, "recon_hat")["recon_pts"].sum())
        optimal = float(_pick_top_k(g, "recon_pts")["recon_pts"].sum())
        vxv.append(picked / optimal if optimal > 0 else np.nan)
    return mae, float(np.nanmean(vxv)), vxv


def main() -> None:
    df = load().reset_index(drop=True)
    pool_mask = df["recon_pts"].notna().to_numpy() & (df["minutes"].fillna(0) > 0).to_numpy()
    is26 = (df["season"] == 2026).to_numpy() & pool_mask
    te26 = np.where(is26)[0]

    # --- A) sealed holdout: train on 2023+2024+2025, predict 2026 ---
    trA = np.where(df["season"].isin([2023, 2024, 2025]).to_numpy() & pool_mask)[0]
    reconA = predict_recon(df, trA, te26)
    maeA, vxvA, rA = metrics_2026(df, te26, reconA)

    # --- B) mixed-year folds: recreate the same whole-round folds; each 2026
    #        round predicted from the rest (incl. other 2026 rounds) ---
    pool = df[pool_mask].reset_index(drop=True)
    pool_pos = np.where(pool_mask)[0]
    rounds = pool[["season", "round"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(0)
    folds = 5
    fold_of = {tuple(r): int(i % folds)
               for i, r in enumerate(rng.permutation(rounds.to_numpy()))}
    pool_fold = np.array([fold_of[(s, r)] for s, r in
                          zip(pool["season"].to_numpy(), pool["round"].to_numpy())])
    reconB = np.full(len(df), np.nan)
    for f in range(folds):
        te_loc = np.where(pool_fold == f)[0]
        tr_loc = np.where(pool_fold != f)[0]
        te_idx, tr_idx = pool_pos[te_loc], pool_pos[tr_loc]
        rec = predict_recon(df, tr_idx, te_idx)
        reconB[te_idx] = rec
    maeB, vxvB, rB = metrics_2026(df, te26, reconB[te26])

    print("=== 2026-only, same from-scratch model, recon metric ===\n")
    print(f"{'protocol':28s} {'recon MAE':>10} {'recon value_xv':>15}")
    print(f"{'A sealed holdout (honest)':28s} {maeA:>10.4f} {vxvA:>15.4f}")
    print(f"{'B mixed-year folds':28s} {maeB:>10.4f} {vxvB:>15.4f}")
    print(f"{'B - A (protocol effect)':28s} {maeB-maeA:>+10.4f} {vxvB-vxvA:>+15.4f}")
    print(f"\nper-2026-round value_xv:")
    print(f"  A sealed: {[round(x,3) for x in rA]}")
    print(f"  B mixed : {[round(x,3) for x in rB]}")
    print("\nFor reference — CHAMPION full pipeline, OFFICIAL metric, sealed 2026")
    print("(promotion_report.json): MAE 7.302861, value_xv 0.718337.")
    print("Not comparable to the recon numbers above (recon excludes latent and")
    print("uses a simpler from-scratch model); shown only as the production anchor.")


if __name__ == "__main__":
    main()
