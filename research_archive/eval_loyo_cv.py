#!/usr/bin/env python3
"""Leave-one-year-out CV over the dev years to raise evaluation power.

Motivation: value_team on 2025's 5 rounds has SE ~0.046, so no selection-score
candidate's ~0.005 gain can be resolved (Experiment 20).  The only modern-
labelled dev data is 2025, but 2023 and 2024 have full realised COMPONENT
stats, so their modern-formula RECON points (deterministic, leakage-free) can
serve as a realised-value target.  Latent (POTM/API-blind) is NOT recoverable
for 2023/2024 and is common noise to champion vs candidate, so we score on
recon points only — which also isolates the component-driven selection effect
the candidates actually move.

Protocol (2026 stays sealed and unused):
  for each test year Y in {2023, 2024, 2025}:
      train components on the OTHER dev years (leave-one-year-out)
      predict Y with the real pipeline (_predict_config train_mask override)
  realised points := recon_pts (modern formula of realised components)
  compare champion vs candidate on PAIRED per-round deltas over all ~15 rounds.

LOYO is a generalisation CV (2025 may train a 2023 fold — anti-causal), which
inflates absolute level equally for both configs; the PAIRED comparison is
unbiased.  Reported: recon value_xv (cleanest) and recon value_team.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from model.data import load
from model.research import Config, _predict_config, CANDIDATES, _load_base_config
from model.evaluate import XV_QUOTA, SUPERSUB_MULTIPLIER

DEV_YEARS = [2023, 2024, 2025]
SEALED = 2026


def _pick_top_k(g: pd.DataFrame, score_col: str) -> pd.DataFrame:
    picked = []
    for pos, k in XV_QUOTA.items():
        gp = g[g["canonical_pos"] == pos]
        if gp.empty:
            continue
        picked.append(gp.nlargest(min(k, len(gp)), score_col))
    return pd.concat(picked) if picked else g.iloc[0:0]


def _round_value(g: pd.DataFrame, sel: str, realised: str,
                 cap: str | None, ss: str | None) -> tuple[float, float]:
    """Return (value_xv, value_team) for one round using `realised` points."""
    xv = _pick_top_k(g, sel)
    opt = _pick_top_k(g, realised)
    xv_val = xv[realised].sum() / opt[realised].sum() if opt[realised].sum() > 0 else np.nan

    picked_base = float(xv[realised].sum())
    if cap and cap in xv:
        cap_extra = float(xv.loc[xv[cap].idxmax(), realised])
    else:
        cap_extra = float(xv.loc[xv[sel].idxmax(), realised])
    pool = g[~g["player_id"].isin(set(xv["player_id"]))]
    if "started" in pool:
        b = pool[~pool["started"].astype(bool)]
        pool = b if not b.empty else pool
    ss_pts = 0.0
    if not pool.empty:
        col = ss if (ss and ss in pool) else sel
        ss_pts = float(pool.loc[pool[col].idxmax(), realised])
    picked = picked_base + cap_extra + SUPERSUB_MULTIPLIER * ss_pts

    opt_base = float(opt[realised].sum())
    opt_cap = float(opt[realised].max())
    opt_pool = g[~g["player_id"].isin(set(opt["player_id"]))]
    if "started" in opt_pool:
        b = opt_pool[~opt_pool["started"].astype(bool)]
        opt_pool = b if not b.empty else opt_pool
    opt_ss = float(opt_pool[realised].max()) if not opt_pool.empty else 0.0
    optimal = opt_base + opt_cap + SUPERSUB_MULTIPLIER * opt_ss
    team_val = picked / optimal if optimal > 0 else np.nan
    return xv_val, team_val


def eval_config(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-round recon value_xv / value_team over LOYO dev folds."""
    rows = []
    for Y in DEV_YEARS:
        train_mask = df["season"].isin([y for y in DEV_YEARS if y != Y]).to_numpy()
        pred, sel_col, _ = _predict_config(df, cfg, Y, train_mask=train_mask)
        # join realised recon points (deterministic modern score of real comps)
        real = df[df["season"] == Y][["fixture_id", "player_id", "recon_pts", "minutes"]]
        pred = pred.merge(real, on=["fixture_id", "player_id"], how="left")
        pred = pred[pred["minutes"].fillna(0) > 0]  # matchday players who featured
        cap = "captain_score" if "captain_score" in pred else None
        ss = "supersub_score" if "supersub_score" in pred else None
        for rnd, g in pred.groupby("round"):
            vxv, vteam = _round_value(g, sel_col, "recon_pts", cap, ss)
            rows.append(dict(year=Y, round=int(rnd), value_xv=vxv, value_team=vteam))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", default=[],
                    help="candidate name(s) from CANDIDATES to compare vs champion")
    args = ap.parse_args()
    df = load()
    champ = _load_base_config("promoted")  # the real champion, with overlay/residual
    base = eval_config(df, champ).set_index(["year", "round"])
    n = len(base)
    print(f"=== LOYO recon-value over dev years {DEV_YEARS} (2026 sealed), {n} rounds ===")
    print(f"CHAMPION recon value_xv: mean {base.value_xv.mean():.4f}  "
          f"SD {base.value_xv.std(ddof=1):.4f}  SE {base.value_xv.std(ddof=1)/np.sqrt(n):.4f}")
    print(f"CHAMPION recon value_team: mean {base.value_team.mean():.4f}  "
          f"SD {base.value_team.std(ddof=1):.4f}  SE {base.value_team.std(ddof=1)/np.sqrt(n):.4f}")
    print(base.round(3).to_string())

    cmap = {c[0]: c[2] for c in CANDIDATES}
    for name in args.candidate:
        if name not in cmap:
            print(f"\n[skip] unknown candidate {name}"); continue
        cand = champ.delta(**cmap[name])
        cf = eval_config(df, cand).set_index(["year", "round"])
        dxv = (cf.value_xv - base.value_xv).dropna()
        dtm = (cf.value_team - base.value_team).dropna()
        def report(label, d):
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = d.mean() / se if se > 1e-12 else 0.0
            print(f"  {label}: mean delta {d.mean():+.4f}  paired SD {d.std(ddof=1):.4f}  "
                  f"SE {se:.4f}  t={t:+.2f}  up {int((d>1e-9).sum())}/{len(d)}  "
                  f"down {int((d<-1e-9).sum())}/{len(d)}")
        print(f"\n--- {name} vs champion (paired over {len(dxv)} rounds) ---")
        report("value_xv  ", dxv)
        report("value_team", dtm)


if __name__ == "__main__":
    main()
