#!/usr/bin/env python3
"""Autoresearch loop on the mixed-year round-level-OOF split.

Metric = row-level recon-MAE (paired, ~2,718 obs -> high power), the only thing
this data can resolve.  Guardrail = value_xv (round-level, must not go clearly
negative).  Base model = from-scratch LGBM component rates x POSITION minutes
(iteration 1 win).  Each candidate overrides minutes mode and/or LGBM params;
we report the multi-seed paired dMAE vs base and append to a JSONL ledger.

Every result — win or null — is logged.  Run repeatedly; adopt a candidate into
BASE only when its paired dMAE is robustly negative across seeds (t <~ -2) with
value_xv not clearly hurt.
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, PoissonRegressor

import model.train_components as tc
from model.data import load
from model.baselines import MIN_MINUTES, rates_to_components, score_recon, predict_rates
from model.train_components import predict_rates_lgbm_only
from train_mixedfold import _pick_top_k, MODE

LEDGER = Path(__file__).parent / "research_mixed_ledger.jsonl"
POSITIONS = ["Prop", "Hooker", "Second-row", "Back-row",
             "Scrum-half", "Fly-half", "Centre", "Back-three"]

# BASE model = adopted wins so far: position minutes (iter 1) + uniform light
# shrinkage 0.20 toward the naive base rate (iter 3-4).  Candidates diff on top.
_ALL_COMPS = ["tries", "try_assists", "tackle_turnover", "defenders_beaten",
              "offload", "metres", "tackles"]
BASE = dict(minutes="position", lgbm={},
            shrink={c: 0.20 for c in _ALL_COMPS}, shrink_target="naive")


@contextmanager
def _lgbm_params(overrides: dict):
    old = tc._LGBM_BASE
    tc._LGBM_BASE = {**old, **overrides}
    try:
        yield
    finally:
        tc._LGBM_BASE = old


def _minutes(df, tr_idx, te_idx, mode):
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    pos_mean = tr.loc[tr["minutes"] >= MIN_MINUTES].groupby("canonical_pos")["minutes"].mean()
    glob = float(tr.loc[tr["minutes"] >= MIN_MINUTES, "minutes"].mean())

    def prior(rows):
        base = rows["form_minutes_recent"].to_numpy(float).copy()
        fill = rows["canonical_pos"].map(pos_mean).fillna(glob).to_numpy(float)
        return np.where(np.isnan(base), fill, base)

    def design(rows):
        cols = [rows[["started", "jersey", "is_forward"]].astype(float).to_numpy(),
                prior(rows)[:, None]]
        if mode in ("position", "position_interact", "position_bench", "poisson"):
            started = rows["started"].astype(float).to_numpy()
            for p in POSITIONS:
                d = (rows["canonical_pos"] == p).astype(float).to_numpy()
                cols.append((d * started)[:, None])
        if mode == "position_bench":
            # bench-specific position effects (front-row finishers ~50', etc.)
            bench = 1.0 - rows["started"].astype(float).to_numpy()
            for p in POSITIONS:
                d = (rows["canonical_pos"] == p).astype(float).to_numpy()
                cols.append((d * bench)[:, None])
        if mode == "position_interact":
            isf = rows["is_forward"].astype(float).to_numpy()
            st = rows["started"].astype(float).to_numpy()
            cols.append((st * isf)[:, None])
        return np.column_stack(cols)

    Xtr, ytr = design(tr), tr["minutes"].to_numpy(float)
    if mode == "poisson":
        reg = PoissonRegressor(alpha=1.0, max_iter=500).fit(Xtr, np.clip(ytr, 0, None))
    else:
        reg = Ridge(alpha=10.0).fit(Xtr, ytr)
    return np.clip(reg.predict(design(te)), 0, 80)


def _folds(pool, folds, seed):
    rounds = pool[["season", "round"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fold_of = {tuple(r): int(i % folds)
               for i, r in enumerate(rng.permutation(rounds.to_numpy()))}
    return np.array([fold_of[(s, r)] for s, r in
                     zip(pool["season"].to_numpy(), pool["round"].to_numpy())])


def _recon(df, pool_mask, cfg, folds=5, seed=0):
    """Row-level abs error + per-round value_xv for one config/seed."""
    pool = df[pool_mask].reset_index(drop=True)
    pool_pos = np.where(pool_mask)[0]
    pool_fold = _folds(pool, folds, seed)
    row_err = np.full(len(df), np.nan)
    per_round = {}
    shrink = cfg.get("shrink", {})
    bag = int(cfg.get("bag", 1))
    for f in range(folds):
        te_idx, tr_idx = pool_pos[pool_fold == f], pool_pos[pool_fold != f]
        if bag > 1:
            # average K LGBM component fits over different seeds (bagging =
            # model-variance reduction, distinct from shrinkage-to-prior).
            acc = None
            for k in range(bag):
                with _lgbm_params({**cfg.get("lgbm", {}), "random_state": 100 + k}):
                    r = predict_rates_lgbm_only(df, tr_idx, te_idx, MODE)
                acc = r if acc is None else acc + r
            rates = acc / bag
        else:
            with _lgbm_params(cfg.get("lgbm", {})):
                rates = predict_rates_lgbm_only(df, tr_idx, te_idx, MODE)
        if shrink:
            # Blend LGBM component rate toward a prior (variance regularisation
            # for near-unpredictable components).  Target: naive base rate or a
            # ridge component model (a smarter, still-regularised prior).
            target = predict_rates(df, tr_idx, te_idx, MODE, cfg.get("shrink_target", "naive"))
            for comp, lam in shrink.items():
                rates[comp] = (1.0 - lam) * rates[comp].to_numpy(float) \
                    + lam * target[comp].to_numpy(float)
        minutes = _minutes(df, tr_idx, te_idx, cfg["minutes"])
        rh = score_recon(rates_to_components(rates, minutes),
                         df.iloc[te_idx]["is_forward"].to_numpy())
        row_err[te_idx] = np.abs(df.iloc[te_idx]["recon_pts"].to_numpy(float) - rh)
        te = df.iloc[te_idx].copy(); te["recon_hat"] = rh
        for (yr, rnd), g in te.groupby(["season", "round"]):
            opt = float(_pick_top_k(g, "recon_pts")["recon_pts"].sum())
            per_round[(int(yr), int(rnd))] = (
                float(_pick_top_k(g, "recon_hat")["recon_pts"].sum()) / opt if opt > 0 else np.nan)
    return row_err, per_round


# ---- candidate queue (edit / extend between runs) --------------------------
# Batch 8: LGBM bagging (avg K seed fits) on the adopted base, and bag combined
# with lighter shrinkage (bagging already cuts variance, so 0.20 shrink may then
# over-regularise).  Distinct variance-reduction lever vs shrinkage-to-prior.
_ALL = ["tries", "try_assists", "tackle_turnover", "defenders_beaten",
        "offload", "metres", "tackles"]
CANDIDATES = {
    "bag5_shrink20": dict(minutes="position", bag=5, shrink={c: 0.20 for c in _ALL}, shrink_target="naive"),
    "bag5_shrink10": dict(minutes="position", bag=5, shrink={c: 0.10 for c in _ALL}, shrink_target="naive"),
    "bag5_noshrink": dict(minutes="position", bag=5, shrink={}, shrink_target="naive"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    df = load().reset_index(drop=True)
    pool_mask = df["recon_pts"].notna().to_numpy() & (df["minutes"].fillna(0) > 0).to_numpy()
    seeds = list(range(args.seeds))
    names = args.candidate or list(CANDIDATES)

    # base recon per seed (cache)
    base_err = {s: _recon(df, pool_mask, BASE, seed=s) for s in seeds}
    base_mae = np.nanmean([base_err[s][0][~np.isnan(base_err[s][0])].mean() for s in seeds])
    print(f"BASE (position minutes): recon MAE {base_mae:.4f}  over seeds {seeds}\n")

    for name in names:
        cfg = CANDIDATES[name]
        dmaes, tstats, dvals = [], [], []
        for s in seeds:
            be, bv = base_err[s]
            ce, cv = _recon(df, pool_mask, cfg, seed=s)
            m = ~np.isnan(be) & ~np.isnan(ce)
            d = (ce - be)[m]
            se = d.std(ddof=1) / np.sqrt(len(d))
            dmaes.append(d.mean()); tstats.append(d.mean() / se if se > 0 else 0.0)
            keys = sorted(set(bv) & set(cv))
            dvals.append(np.mean([cv[k] - bv[k] for k in keys]))
        rec = dict(name=name, cfg=cfg, seeds=seeds,
                   dMAE_mean=float(np.mean(dmaes)), dMAE_by_seed=[round(x, 4) for x in dmaes],
                   t_by_seed=[round(x, 2) for x in tstats],
                   dvalue_xv_mean=float(np.mean(dvals)), ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        verdict = ("WIN" if np.mean(dmaes) < 0 and max(tstats) < -2.0 and np.mean(dvals) > -0.01
                   else "null/reject")
        rec["verdict"] = verdict
        with open(LEDGER, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{name:28s} dMAE {np.mean(dmaes):+.4f}  t/seed {[round(x,2) for x in tstats]}  "
              f"dvalue_xv {np.mean(dvals):+.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
