#!/usr/bin/env python3
"""Decision-value decomposition: where does value_team lose its points?

For each 2025 round, split the (hindsight-optimal - model) fantasy point
deficit into the three decision classes: XV base points, captain extra (2x),
and supersub (3x).  Uses the exact evaluator picking logic so the numbers tie
out with value_of_team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.evaluate import (
    SUPERSUB_MULTIPLIER,
    XV_QUOTA,
    _labelled,
    _pick_xv,
    _supersub_pool,
)


def main() -> None:
    pred = pd.read_csv("research/dev_predictions_2025.csv")
    lab = _labelled(pred)
    rows = []
    for rnd, g in lab.groupby("round"):
        xv = _pick_xv(g, "sel_score")
        model_base = float(xv["official_pts"].sum())
        model_capt = float(xv.loc[xv["captain_score"].idxmax(), "official_pts"])
        pool = _supersub_pool(g, set(xv["player_id"]))
        model_ss = float(pool.loc[pool["supersub_score"].idxmax(), "official_pts"]) if not pool.empty else 0.0

        opt_xv = _pick_xv(g, "official_pts")
        opt_base = float(opt_xv["official_pts"].sum())
        opt_capt = float(opt_xv["official_pts"].max())
        opt_pool = _supersub_pool(g, set(opt_xv["player_id"]))
        opt_ss = float(opt_pool["official_pts"].max()) if not opt_pool.empty else 0.0

        rows.append(dict(
            round=int(rnd),
            xv_deficit=opt_base - model_base,
            capt_deficit=opt_capt - model_capt,
            ss_deficit=SUPERSUB_MULTIPLIER * (opt_ss - model_ss),
            model_total=model_base + model_capt + SUPERSUB_MULTIPLIER * model_ss,
            opt_total=opt_base + opt_capt + SUPERSUB_MULTIPLIER * opt_ss,
            model_capt=model_capt, opt_capt=opt_capt,
            model_ss=model_ss, opt_ss=opt_ss,
        ))
    t = pd.DataFrame(rows).set_index("round")
    t["total_deficit"] = t["opt_total"] - t["model_total"]
    print(t.round(1).to_string())
    print("\nshare of total deficit:")
    tot = t["total_deficit"].sum()
    for c in ("xv_deficit", "capt_deficit", "ss_deficit"):
        print(f"  {c:14s} {t[c].sum():7.1f}  ({100*t[c].sum()/tot:5.1f}%)")
    print(f"  total          {tot:7.1f}")

    # XV deficit by position quota slot: which positions lose the most?
    print("\nXV deficit by position (sum over rounds):")
    pos_rows = {}
    for rnd, g in lab.groupby("round"):
        xv = _pick_xv(g, "sel_score")
        opt = _pick_xv(g, "official_pts")
        for pos in XV_QUOTA:
            m = float(xv[xv["canonical_pos"] == pos]["official_pts"].sum())
            o = float(opt[opt["canonical_pos"] == pos]["official_pts"].sum())
            pos_rows[pos] = pos_rows.get(pos, 0.0) + (o - m)
    for pos, d in sorted(pos_rows.items(), key=lambda kv: -kv[1]):
        print(f"  {pos:12s} {d:7.1f}")


if __name__ == "__main__":
    main()
