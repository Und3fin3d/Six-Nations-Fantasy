#!/usr/bin/env python3
"""Autoresearch iteration on the mixed-year round-level-OOF split.

Candidate: position x started minutes dummies vs the global minutes ridge.

Experiment 17 (production) showed position minutes cut OOF minutes MAE +8.6%
but was REJECTED on the champion, because the champion's downstream stack
(front-row prior blend, bench two-stage pull, residual layer) had co-adapted
around the biased minutes model and double-corrected.  The from-scratch model
here has NO such downstream stack, so if the minutes fix is genuinely good it
should lower recon-MAE.  Evaluated on the 20-round mixed OOF with a PAIRED
row-level MAE comparison (same folds/rows) -> ~2,700 obs -> high power.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from model.data import load
from model.baselines import MIN_MINUTES, rates_to_components, score_recon
from model.train_components import predict_rates_lgbm_only
from train_mixedfold import _pick_top_k, MODE

POSITIONS = ["Prop", "Hooker", "Second-row", "Back-row",
             "Scrum-half", "Fly-half", "Centre", "Back-three"]


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
        if mode == "position":
            started = rows["started"].astype(float).to_numpy()
            for p in POSITIONS:
                d = (rows["canonical_pos"] == p).astype(float).to_numpy()
                cols.append((d * started)[:, None])
        return np.column_stack(cols)

    reg = Ridge(alpha=10.0).fit(design(tr), tr["minutes"].to_numpy(float))
    return np.clip(reg.predict(design(te)), 0, 80)


def run(df, pool_mask, mode, folds=5, seed=0):
    pool = df[pool_mask].reset_index(drop=True)
    pool_pos = np.where(pool_mask)[0]
    rounds = pool[["season", "round"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fold_of = {tuple(r): int(i % folds)
               for i, r in enumerate(rng.permutation(rounds.to_numpy()))}
    pool_fold = np.array([fold_of[(s, r)] for s, r in
                          zip(pool["season"].to_numpy(), pool["round"].to_numpy())])
    row_err = np.full(len(df), np.nan)
    per_round = {}
    for f in range(folds):
        te_idx = pool_pos[pool_fold == f]
        tr_idx = pool_pos[pool_fold != f]
        rates = predict_rates_lgbm_only(df, tr_idx, te_idx, MODE)
        minutes = _minutes(df, tr_idx, te_idx, mode)
        recon_hat = score_recon(rates_to_components(rates, minutes),
                                df.iloc[te_idx]["is_forward"].to_numpy())
        row_err[te_idx] = np.abs(df.iloc[te_idx]["recon_pts"].to_numpy(float) - recon_hat)
        te = df.iloc[te_idx].copy(); te["recon_hat"] = recon_hat
        for (yr, rnd), g in te.groupby(["season", "round"]):
            picked = float(_pick_top_k(g, "recon_hat")["recon_pts"].sum())
            optimal = float(_pick_top_k(g, "recon_pts")["recon_pts"].sum())
            per_round[(int(yr), int(rnd))] = picked / optimal if optimal > 0 else np.nan
    return row_err, per_round


def main() -> None:
    df = load().reset_index(drop=True)
    pool_mask = df["recon_pts"].notna().to_numpy() & (df["minutes"].fillna(0) > 0).to_numpy()
    idx = np.where(pool_mask)[0]

    eg, vg = run(df, pool_mask, "global")
    ep, vp = run(df, pool_mask, "position")

    # paired row-level MAE (same rows, both non-nan)
    m = ~np.isnan(eg) & ~np.isnan(ep)
    d = (ep - eg)[m]   # position - global; negative = position better
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"=== position vs global minutes, 20-round mixed OOF ===")
    print(f"recon MAE  global   = {eg[m].mean():.4f}")
    print(f"recon MAE  position = {ep[m].mean():.4f}")
    print(f"paired row-level delta (pos-glob) = {d.mean():+.4f}  "
          f"SE {se:.4f}  t = {d.mean()/se:+.2f}  (n={len(d)} rows)  "
          f"-> {'position BETTER' if d.mean()<0 else 'global better'}")
    rel = 100 * (1 - ep[m].mean() / eg[m].mean())
    print(f"relative MAE change: {rel:+.2f}%")

    # paired round-level value_xv
    keys = sorted(set(vg) & set(vp))
    dv = np.array([vp[k] - vg[k] for k in keys])
    sev = dv.std(ddof=1) / np.sqrt(len(dv))
    print(f"\nvalue_xv  global   = {np.nanmean([vg[k] for k in keys]):.4f}")
    print(f"value_xv  position = {np.nanmean([vp[k] for k in keys]):.4f}")
    print(f"paired round delta = {dv.mean():+.4f}  SE {sev:.4f}  t = {dv.mean()/sev:+.2f}  "
          f"(n={len(keys)} rounds)  up {int((dv>1e-9).sum())}/{len(keys)}")


if __name__ == "__main__":
    main()
