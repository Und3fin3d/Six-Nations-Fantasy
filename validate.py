#!/usr/bin/env python3
"""Unified model validation across three splits, so improvements can be judged
far better than the 5-round 2025 backtest alone.

  [1] 2025 dev backtest   full pipeline, standard split (train 2023+2024)  -> the
                          decision metrics value_team / value_xv / MAE / top15.
  [2] 2026 sealed OOF      full pipeline, train 2023+2024+2025 -> predict 2026
                          held out entirely (forward-chained deployment holdout).
  [3] Mixed-fold OOF        20 (year, round) blocks pooled across all four years,
                          GroupKFold on whole (season, round) groups with
                          round-level isolation (a test round is never in its own
                          training; sibling rounds incl. 2026's are). High-power
                          (~2,700 rows) recon-MAE + recon value_xv, per year.

Views [1]/[2] use the full config pipeline (validate selection/decision changes).
View [3] uses the config's component engine + minutes model scored on realised
recon points (validate component/MAE changes at high power).  recon_pts is the
deterministic modern-formula score of realised components (leakage-free, defined
for every year including 2023/2024 where fantasy labels are absent).

Usage:
  validate.py                      # validate the promoted champion
  validate.py --config best        # research best_config.json
  validate.py --config path.json   # an explicit config
  validate.py --no-mixed           # skip the (slow) mixed-fold view
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from model.data import load
from model.research import Config, _predict_config, build_minutes, PROMOTED_CONFIG, BEST_CONFIG
from model.baselines import rates_to_components, score_recon
from model.train_components import predict_rates_lgbm_only
from model.evaluate import (
    points_mae, value_of_team, value_of_xv, topn_overlap, XV_QUOTA)

DEV_YEARS = [2023, 2024, 2025]


# ---------------------------------------------------------------------------
def load_config(which: str) -> Config:
    if which in ("promoted", "best"):
        path = PROMOTED_CONFIG if which == "promoted" else BEST_CONFIG
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        return Config(**json.loads(path.read_text()))
    if which == "baseline":
        return Config()
    return Config(**json.loads(open(which).read_text()))


def _full_pipeline_metrics(df, cfg, season):
    pred, sel, _ = _predict_config(df, cfg, season)
    cap = "captain_score" if "captain_score" in pred.columns else None
    ss = "supersub_score" if "supersub_score" in pred.columns else None
    vt = value_of_team(pred, sel, captain_score_col=cap, supersub_score_col=ss)[0]
    return dict(value_team=vt, value_xv=value_of_xv(pred, sel)[0],
               mae=points_mae(pred)["overall"], top15=topn_overlap(pred, 15, sel))


def _pick_xv(g, col):
    picked = [g[g["canonical_pos"] == p].nlargest(min(k, (g["canonical_pos"] == p).sum()), col)
              for p, k in XV_QUOTA.items() if (g["canonical_pos"] == p).any()]
    return pd.concat(picked) if picked else g.iloc[0:0]


def _mixed_fold(df, cfg, folds=5, seeds=(0, 1, 2)):
    """Config-aware component recon over the all-years round-level OOF split."""
    ok = df["recon_pts"].notna().to_numpy() & (df["minutes"].fillna(0) > 0).to_numpy()
    pool = df[ok].reset_index(drop=True)
    pool_pos = np.where(ok)[0]
    yr = df["season"].to_numpy()
    mae_year = {y: [] for y in (2023, 2024, 2025, 2026)}
    vxv_all, mae_all = [], []
    for seed in seeds:
        rounds = pool[["season", "round"]].drop_duplicates().reset_index(drop=True)
        rng = np.random.default_rng(seed)
        fold_of = {tuple(r): i % folds for i, r in enumerate(rng.permutation(rounds.to_numpy()))}
        pfold = np.array([fold_of[(s, r)] for s, r in
                          zip(pool["season"].to_numpy(), pool["round"].to_numpy())])
        row_err = np.full(len(df), np.nan)
        for f in range(folds):
            te, tr = pool_pos[pfold == f], pool_pos[pfold != f]
            rates = predict_rates_lgbm_only(df, tr, te, "post_team_sheet")
            minutes = build_minutes(df, tr, te, cfg)
            rh = score_recon(rates_to_components(rates, minutes), df.iloc[te]["is_forward"].to_numpy())
            row_err[te] = np.abs(df.iloc[te]["recon_pts"].to_numpy(float) - rh)
            t = df.iloc[te].copy(); t["rh"] = rh
            for (y, r), g in t.groupby(["season", "round"]):
                opt = float(_pick_xv(g, "recon_pts")["recon_pts"].sum())
                if opt > 0:
                    vxv_all.append(float(_pick_xv(g, "rh")["recon_pts"].sum()) / opt)
        m = ~np.isnan(row_err)
        mae_all.append(row_err[m].mean())
        for y in mae_year:
            mae_year[y].append(row_err[m & (yr == y)].mean())
    n = len(vxv_all)
    return dict(recon_mae=float(np.mean(mae_all)),
                recon_vxv=float(np.mean(vxv_all)),
                recon_vxv_se=float(np.std(vxv_all, ddof=1) / np.sqrt(n)),
                mae_by_year={y: float(np.mean(v)) for y, v in mae_year.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="promoted",
                    help="promoted | best | baseline | path-to-config.json")
    ap.add_argument("--no-mixed", action="store_true", help="skip mixed-fold view (faster)")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    df = load().reset_index(drop=True)
    cfg = load_config(args.config)
    print(f"=== VALIDATION: {cfg.name} ===\n")

    dev = _full_pipeline_metrics(df, cfg, 2025)
    print(f"[1] 2025 dev backtest (full pipeline, 5 rounds)")
    print(f"    value_team {dev['value_team']:.4f}  value_xv {dev['value_xv']:.4f}  "
          f"MAE {dev['mae']:.4f}  top15 {dev['top15']:.3f}")

    seal = _full_pipeline_metrics(df, cfg, 2026)
    print(f"\n[2] 2026 sealed OOF (full pipeline, train 23-25 -> hold out 26)")
    print(f"    value_team {seal['value_team']:.4f}  value_xv {seal['value_xv']:.4f}  "
          f"MAE {seal['mae']:.4f}  top15 {seal['top15']:.3f}")

    if not args.no_mixed:
        mx = _mixed_fold(df, cfg, seeds=tuple(range(args.seeds)))
        print(f"\n[3] Mixed-fold OOF (all years, round-level isolation, {args.seeds} seeds, recon)")
        print(f"    recon MAE {mx['recon_mae']:.4f}   recon value_xv {mx['recon_vxv']:.4f} "
              f"(SE {mx['recon_vxv_se']:.4f}, 20 rounds/seed)")
        print(f"    recon MAE by year: " + "  ".join(
            f"{y} {mx['mae_by_year'][y]:.3f}" for y in (2023, 2024, 2025, 2026)))


if __name__ == "__main__":
    main()
