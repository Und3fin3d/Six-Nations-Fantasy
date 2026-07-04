#!/usr/bin/env python3
"""From-scratch model trained with a MIX OF YEARS in each test fold.

Cross-year cross-validation over ALL four years (2023-2026) with round-level
out-of-fold isolation.  Pool every fixture and run GroupKFold(5) over whole
(season, round) groups, so each test fold holds a mix of complete rounds from
several years.  Isolation is the standard OOF guarantee: a test round's own
rows are never in that fold's training set, but every other round is — incl.
sibling rounds of the same year (2026 R2 in test -> 2026 R2 out of training,
2026 R3 stays in).  Each round is predicted exactly once, held out of its own
fold.  Purpose: maximise independent test rounds (20) for statistical power.

Note: this touches 2026 rounds during training (for folds where they are not
the test round), so it is a generalisation CV, not the production protocol's
pristine sealed-2026 holdout.  Kept in this isolated worktree accordingly.

Model, from scratch but in the same fashion as production components:
  per-80 LGBM component rates x a ridge minutes model -> modern recon points.
Scored against realised recon_pts (deterministic modern formula; latent/POTM is
not modern-recoverable for 2023/2024 and is common noise, so excluded).
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from model.data import load
from model.baselines import MIN_MINUTES, rates_to_components, score_recon
from model.train_components import predict_rates_lgbm_only
from model.evaluate import XV_QUOTA

MODE = "post_team_sheet"
POSITIONS = list(XV_QUOTA)


def _minutes(df, tr_idx, te_idx):
    from sklearn.linear_model import Ridge
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]
    pos_mean = tr.loc[tr["minutes"] >= MIN_MINUTES].groupby("canonical_pos")["minutes"].mean()
    glob = float(tr.loc[tr["minutes"] >= MIN_MINUTES, "minutes"].mean())

    def prior(rows):
        base = rows["form_minutes_recent"].to_numpy(float).copy()
        fill = rows["canonical_pos"].map(pos_mean).fillna(glob).to_numpy(float)
        return np.where(np.isnan(base), fill, base)

    feats = ["started", "jersey", "is_forward"]
    Xtr = np.column_stack([tr[feats].astype(float).to_numpy(), prior(tr)[:, None]])
    Xte = np.column_stack([te[feats].astype(float).to_numpy(), prior(te)[:, None]])
    reg = Ridge(alpha=10.0).fit(Xtr, tr["minutes"].to_numpy(float))
    return np.clip(reg.predict(Xte), 0, 80)


def _pick_top_k(g, score_col):
    picked = []
    for pos, k in XV_QUOTA.items():
        gp = g[g["canonical_pos"] == pos]
        if not gp.empty:
            picked.append(gp.nlargest(min(k, len(gp)), score_col))
    return pd.concat(picked) if picked else g.iloc[0:0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load().reset_index(drop=True)
    # ISOLATION is round-level (standard out-of-fold): a test round's own rows
    # are never in that fold's training set, but every OTHER round is — incl.
    # sibling rounds of the SAME year.  So if 2026 R2 is in the test fold, 2026
    # R2 is excluded from training, while 2026 R3/R4/... stay in training.  All
    # four years participate; each round is predicted exactly once, held out of
    # its own fold.
    pool_mask = df["recon_pts"].notna().to_numpy() & (df["minutes"].fillna(0) > 0).to_numpy()
    pool = df[pool_mask].reset_index(drop=True)
    print(f"pooled fixtures across years {sorted(pool['season'].unique())}; "
          f"rows={len(pool)}  fixtures={pool['fixture_id'].nunique()}  "
          f"(round-level OOF isolation; each round held out of its own training)")

    # Fold on whole (season, round) groups so each test fold holds COMPLETE
    # rounds from a mix of years — the round-level XV decision is only valid on
    # a full squad pool, so we must never split a round's matches across folds.
    pool_pos = np.where(pool_mask)[0]  # positions into df
    rounds = pool[["season", "round"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    fold_of = {tuple(r): int(i % args.folds)
               for i, r in enumerate(rng.permutation(rounds.to_numpy()))}
    pool_fold = np.array([fold_of[(s, r)] for s, r in
                          zip(pool["season"].to_numpy(), pool["round"].to_numpy())])

    per_round = []
    row_abs_err = []
    fold_year_counts = []
    for f in range(args.folds):
        te_loc = np.where(pool_fold == f)[0]
        tr_loc = np.where(pool_fold != f)[0]
        tr_idx = pool_pos[tr_loc]
        te_idx = pool_pos[te_loc]
        rates = predict_rates_lgbm_only(df, tr_idx, te_idx, MODE)
        minutes = _minutes(df, tr_idx, te_idx)
        comps = rates_to_components(rates, minutes)
        recon_hat = score_recon(comps, df.iloc[te_idx]["is_forward"].to_numpy())

        te = df.iloc[te_idx].copy()
        te["recon_hat"] = recon_hat
        fold_year_counts.append(te.groupby("season").size().to_dict())
        # MAE vs realised recon
        row_abs_err.append(np.abs(te["recon_pts"].to_numpy(float) - recon_hat))
        # value_xv per (season, round)
        for (yr, rnd), g in te.groupby(["season", "round"]):
            picked = float(_pick_top_k(g, "recon_hat")["recon_pts"].sum())
            optimal = float(_pick_top_k(g, "recon_pts")["recon_pts"].sum())
            per_round.append(dict(year=int(yr), round=int(rnd),
                                  value_xv=picked / optimal if optimal > 0 else np.nan))

    pr = pd.DataFrame(per_round)
    mae = float(np.concatenate(row_abs_err).mean())
    n = len(pr)
    print(f"\nfold year mix (rows per year in each test fold):")
    for i, fc in enumerate(fold_year_counts):
        print(f"  fold {i}: {fc}")
    print(f"\n=== mixed-year-fold from-scratch model ===")
    print(f"recon MAE (per row, vs realised recon_pts): {mae:.4f}")
    print(f"recon value_xv: mean {pr.value_xv.mean():.4f}  SD {pr.value_xv.std(ddof=1):.4f}  "
          f"SE {pr.value_xv.std(ddof=1)/np.sqrt(n):.4f}  over {n} (year,round) blocks")
    print("\nvalue_xv by year:")
    print(pr.groupby("year")["value_xv"].agg(["mean", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
