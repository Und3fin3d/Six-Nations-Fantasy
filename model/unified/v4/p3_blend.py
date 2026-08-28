"""P3 — the plan's single remaining structural idea (RESEARCH_PLAN.md §10.2):
a GLOBAL blend of gbdt_v4[B] with the empirical unified projection.

The blend weight is one global constant chosen on 6N-2025 LORO only — no
competition-specific weights, so the blend is itself a competition-independent
model and stays inside the one-model mandate. Quarantine S3 unchanged.

  dump  — refit gbdt_v4[B] per 6N-2025 LORO fold and save per-row predictions
          (p1 admission runs kept only fold metrics).
  sweep — grid the global weight over {0, .25, .5, .75, 1}; precommitted
          admission: the best blend must beat BOTH components on Spearman AND
          mean capture with MAE <= 1.005x the better component's, on 6N-2025.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ..benchmark_v2 import _group_metrics
from ..data import ROOT
from ..v3.benchmark import _cohort_predictions
from ..v3.harness import fold_for_block, strict_training_frame
from .experiments import build_store, selection_labels
from .gbdt import V4GBDT

OUT = ROOT / "data" / "unified" / "v4"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MAE_HOLD = 1.005


def dump() -> None:
    labels = selection_labels()
    store = build_store()
    frames = []
    for group_id, block in labels.groupby("group_id", sort=True):
        block = block.reset_index(drop=True)
        fold = fold_for_block(block, store)
        train = strict_training_frame(store, fold)
        model = V4GBDT(weighting="natural", pool_player_id=True, player_effects=True)
        print(f"[{group_id}] fitting B on {len(train):,} rows", flush=True)
        model.fit(train)
        predicted = _cohort_predictions(model, block, store, "six_nations")
        frames.append(predicted.assign(group_id=group_id))
    pd.concat(frames, ignore_index=True).to_csv(
        OUT / "p3_v4_loro_predictions.csv", index=False)
    print("wrote p3_v4_loro_predictions.csv")


def sweep() -> None:
    v4 = pd.read_csv(OUT / "p3_v4_loro_predictions.csv")
    emp = pd.read_csv(OUT / "empirical_unified_predictions.csv")
    emp = emp[emp["group_id"].str.startswith("6n-2025")]
    merged = v4.rename(columns={"predicted_points": "pred_v4"}).merge(
        emp.rename(columns={"predicted_points": "pred_emp",
                            "label_row_id": "_label_row_id"})
        [["group_id", "_label_row_id", "pred_emp"]],
        on=["group_id", "_label_row_id"])
    rows = []
    for w in WEIGHTS:
        merged["pred"] = w * merged["pred_v4"] + (1 - w) * merged["pred_emp"]
        fold_metrics = [
            _group_metrics(g.rename(columns={"official_pts": "actual"}),
                           "pred", actual_col="actual")
            for _, g in merged.groupby("group_id")
        ]
        agg = pd.DataFrame(fold_metrics).mean(numeric_only=True)
        capture = float(np.mean([agg[f"top_{n}_capture"] for n in (10, 25, 50, 100)]))
        rows.append({"w_v4": w, "mae": float(agg["mae"]),
                     "spearman": float(agg["spearman"]), "mean_capture": capture})
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(OUT / "p3_sweep.csv", index=False)
    print(sweep_df.to_string(index=False))

    v4_row = sweep_df[sweep_df.w_v4.eq(1.0)].iloc[0]
    emp_row = sweep_df[sweep_df.w_v4.eq(0.0)].iloc[0]
    interior = sweep_df[~sweep_df.w_v4.isin((0.0, 1.0))]
    best = interior.sort_values("mean_capture", ascending=False).iloc[0]
    better_mae = min(v4_row["mae"], emp_row["mae"])
    verdict = {
        "chosen_w_v4": float(best["w_v4"]),
        "blend": {k: float(best[k]) for k in ("mae", "spearman", "mean_capture")},
        "v4_alone": {k: float(v4_row[k]) for k in ("mae", "spearman", "mean_capture")},
        "empirical_alone": {k: float(emp_row[k]) for k in ("mae", "spearman", "mean_capture")},
        "admit": bool(
            best["spearman"] > max(v4_row["spearman"], emp_row["spearman"])
            and best["mean_capture"] > max(v4_row["mean_capture"], emp_row["mean_capture"])
            and best["mae"] <= better_mae * MAE_HOLD
        ),
    }
    (OUT / "p3_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    dump() if mode == "dump" else sweep()


if __name__ == "__main__":
    main()
