#!/usr/bin/env python3
"""Uniform cross-architecture benchmark: evaluate each branch's prediction
files with the SAME evaluator. MAE (target_pts_hat) and value_xv/top-N (sel_score)
are comparable across all; value_team is NOT (architecture models lack dedicated
captain/supersub heads), so it is omitted."""
from __future__ import annotations

import pandas as pd
from model.evaluate import points_mae, value_of_xv, topn_overlap

MODELS = {
    "6n / cdx (champion)":  "/Users/williamgreenfield/Documents/dev/6n",
    "bigmoves":             "/Users/williamgreenfield/Documents/dev/6n-bigmoves",
    "autoresearch-next-2":  "/Users/williamgreenfield/Documents/dev/6n-autoresearch-next-2",
    "mixedfold":            "/Users/williamgreenfield/Documents/dev/6n-mixedfold",
    "main":                 "/Users/williamgreenfield/Documents/dev/6n-main",
    "xgboost (genuine)":    "/Users/williamgreenfield/Documents/dev/6n-xgboost",
    "lstm (2026 stale)":    "/Users/williamgreenfield/Documents/dev/6n-lstm",
    "bayesian (2026 stale)":"/Users/williamgreenfield/Documents/dev/6n-bayesian",
}


def evaluate_csv(path):
    df = pd.read_csv(path)
    sel = "sel_score" if "sel_score" in df.columns else "target_pts_hat"
    return {
        "mae": points_mae(df)["overall"],
        "value_xv": value_of_xv(df, sel)[0],
        "top15": topn_overlap(df, 15, sel),
        "top30": topn_overlap(df, 30, sel),
    }


def main() -> None:
    for season in (2025, 2026):
        rows = []
        for name, wt in MODELS.items():
            f = f"{wt}/data/model_predictions_{season}.csv"
            try:
                m = evaluate_csv(f)
                rows.append((name, m))
            except Exception as e:
                rows.append((name, {"err": str(e)[:40]}))
        print(f"\n=== {season} {'(dev)' if season==2025 else '(sealed holdout)'} — uniform eval ===")
        print(f"{'model':30s} {'MAE':>8} {'value_xv':>9} {'top15':>7} {'top30':>7}")
        for name, m in rows:
            if "err" in m:
                print(f"{name:30s}  ERROR {m['err']}")
            else:
                print(f"{name:30s} {m['mae']:>8.4f} {m['value_xv']:>9.4f} {m['top15']:>7.3f} {m['top30']:>7.3f}")


if __name__ == "__main__":
    main()
