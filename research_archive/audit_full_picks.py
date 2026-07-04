#!/usr/bin/env python3
"""Look at the actual picks. For each 2025 round: model XV vs optimal XV,
model captain/supersub vs optimal, and the swap list. Then aggregate patterns
across rounds that are NOT already-closed (floor/variance, back-row try
variance, captain ceiling). We are hunting for a distinct, fixable error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.evaluate import (
    _labelled, _pick_xv, _supersub_pool, XV_QUOTA, SUPERSUB_MULTIPLIER)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def main() -> None:
    pred = pd.read_csv("research/dev_predictions_2025.csv")
    lab = _labelled(pred)

    # ---- aggregate: model vs optimal XV membership, minutes accuracy on picks
    swap_rows, pick_rows = [], []
    for rnd, g in lab.groupby("round"):
        xv = _pick_xv(g, "sel_score")
        opt = _pick_xv(g, "official_pts")
        model_ids, opt_ids = set(xv["player_id"]), set(opt["player_id"])
        missed = opt[~opt["player_id"].isin(model_ids)]   # should've picked
        wrong = xv[~xv["player_id"].isin(opt_ids)]         # shouldn't have
        for _, r in wrong.iterrows():
            swap_rows.append(dict(round=int(rnd), kind="over-picked",
                pos=r["canonical_pos"], name=r["player_name"], team=r.get("team", "?"),
                started=bool(r["started"]), pred=r["target_pts_hat"], actual=r["official_pts"],
                minutes_hat=r.get("minutes_hat", np.nan)))
        for _, r in missed.iterrows():
            swap_rows.append(dict(round=int(rnd), kind="missed",
                pos=r["canonical_pos"], name=r["player_name"], team=r.get("team", "?"),
                started=bool(r["started"]), pred=r["target_pts_hat"], actual=r["official_pts"],
                minutes_hat=r.get("minutes_hat", np.nan)))
        for _, r in xv.iterrows():
            pick_rows.append(dict(round=int(rnd), pos=r["canonical_pos"],
                started=bool(r["started"]), pred=r["target_pts_hat"], actual=r["official_pts"],
                minutes_hat=r.get("minutes_hat", np.nan),
                minutes_act=r.get("minutes", np.nan)))

    swaps = pd.DataFrame(swap_rows)
    picks = pd.DataFrame(pick_rows)

    print("=" * 78)
    print("A. OVER-PICK ERROR: model's XV members who were NOT in the optimal XV")
    print("=" * 78)
    over = swaps[swaps.kind == "over-picked"]
    print(f"count={len(over)}  mean pred={over.pred.mean():.1f}  mean actual={over.actual.mean():.1f}"
          f"  over-prediction={over.pred.mean()-over.actual.mean():+.1f}")
    print("by position:")
    print(over.groupby("pos").agg(n=("name","size"), pred=("pred","mean"),
          actual=("actual","mean")).round(1).sort_values("n", ascending=False).to_string())
    print("\nstarted vs bench among over-picks:", over.started.value_counts().to_dict())

    print("\n" + "=" * 78)
    print("B. MINUTES on the model's XV picks — are we picking players who don't play?")
    print("=" * 78)
    picks["min_err"] = picks["minutes_hat"] - picks["minutes_act"]
    print(f"XV picks: mean minutes_hat={picks.minutes_hat.mean():.1f}  "
          f"mean minutes_act={picks.minutes_act.mean():.1f}  "
          f"mean |err|={picks.min_err.abs().mean():.1f}")
    low = picks[picks.minutes_act < 40]
    print(f"XV picks who played <40 min: {len(low)}/{len(picks)}  "
          f"(their mean predicted minutes was {low.minutes_hat.mean():.1f}; "
          f"mean actual pts {low.actual.mean():.1f})")

    print("\n" + "=" * 78)
    print("C. SUPERSUB — model pick vs optimal (supersub scores x3, so it matters)")
    print("=" * 78)
    ss_rows = []
    for rnd, g in lab.groupby("round"):
        xv = _pick_xv(g, "sel_score")
        pool = _supersub_pool(g, set(xv["player_id"]))
        m = pool.loc[pool["supersub_score"].idxmax()]
        o = pool.loc[pool["official_pts"].idxmax()]
        ss_rows.append(dict(round=int(rnd), model=m["player_name"], model_pos=m["canonical_pos"],
            model_act=m["official_pts"], opt=o["player_name"], opt_pos=o["canonical_pos"],
            opt_act=o["official_pts"], regret_x3=SUPERSUB_MULTIPLIER*(o["official_pts"]-m["official_pts"]),
            model_started=bool(m["started"])))
    ssdf = pd.DataFrame(ss_rows)
    print(ssdf.to_string(index=False))
    print(f"\ntotal supersub regret (x3) = {ssdf.regret_x3.sum():.0f}; "
          f"model supersub was a bench player in {ssdf.model_started.eq(False).sum()}/5 rounds")

    print("\n" + "=" * 78)
    print("D. Are the OVER-PICKS high-mean, high-minute 'safe' players who flopped?")
    print("=" * 78)
    print(over.sort_values("pred", ascending=False)[
        ["round","pos","name","team","started","minutes_hat","pred","actual"]].to_string(index=False))


if __name__ == "__main__":
    main()
