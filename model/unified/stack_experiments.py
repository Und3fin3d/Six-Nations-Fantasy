"""Head-to-head experiment: deterministic stage-1 vs the learned v2 stack.

Removes the two confounders the review flagged before declaring a winner:

1. **Out-of-fold stage-1 features.** Stage-2's training rows (6N 2025) are
   featured by a stage-1 fit strictly *before* 2025-02-01, so the anchor
   feature ``s1_exp_points`` is honest at train time as well as test time.
2. **Full-window stage-1 refits.** ``cli.py train`` silently discards the most
   recent 15% of dates as an unused validation tail; here every artifact is fit
   through its stated cutoff (suffix ``_full``), which also makes the per-GW
   re-cut real (the GW2 artifact actually contains GW1).

The stack config (blend weight x scoring head) is chosen by a predeclared sweep
on 2025 leave-one-round-out dev ONLY: maximise mean top-N capture subject to
dev MAE no more than 2% worse than stage-1's. The chosen config is then
evaluated once on the sealed 6N 2026 season and the NCR LOGO protocol.

Writes data/unified/stack_experiments.md (+ CSVs). Never touches production.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_v2 import _group_metrics
from .data import ROOT
from .gbdt import UniversalGBDT
from .labels import build_fantasy_labels
from .rank_stack import RankStack, build_stage2_features, load_store_features

DATA = ROOT / "data"
OUT = DATA / "unified"
MODELS = OUT / "models"

FULL_CUTOFFS = {
    "pre2025": "2025-02-01",   # OOF features for the 6N 2025 training rows
    "pre2026": "2026-02-01",   # honest features for the sealed 6N 2026 test
    "gw1": "2026-07-04",       # honest features for NCR GW1 (fixtures on the 4th excluded)
    "gw2": "2026-07-11",
}
BLENDS = (0.0, 0.25, 0.35, 0.5, 0.75, 1.0)
SCORE_MODES = ("rank+points", "rank_only", "points_only")
DEV_MAE_TOLERANCE = 1.02


def full_artifact(key: str) -> Path:
    return MODELS / f"unified_gbdt_full_{FULL_CUTOFFS[key]}.pkl"


def fit_full_stage1(store_feat: pd.DataFrame, key: str) -> UniversalGBDT:
    """Fit stage-1 on every row strictly before the cutoff (no tail discard)."""
    path = full_artifact(key)
    if path.exists():
        return UniversalGBDT.load(path)
    cutoff = pd.Timestamp(FULL_CUTOFFS[key])
    frame = store_feat[store_feat["date"] < cutoff]
    print(f"[stage1:{key}] fitting on {len(frame):,} rows "
          f"({frame.date.min().date()} -> {frame.date.max().date()})", flush=True)
    model = UniversalGBDT().fit(frame)
    model.save(path)
    return model


def scored(pred: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = pred.copy()
    if mode == "rank_only":
        out["stack_score"] = out["rank_component"]
    elif mode == "points_only":
        out["stack_score"] = out["points_component"]
    return out


def dev_sweep(feat_2025: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-round-out over 6N 2025 for every (blend, score mode) config."""
    rows = []
    groups = sorted(feat_2025["group_id"].unique())
    for blend, mode in itertools.product(BLENDS, SCORE_MODES):
        fold_metrics = []
        for held in groups:
            train = feat_2025[feat_2025.group_id != held]
            test = feat_2025[feat_2025.group_id == held].reset_index(drop=True)
            stack = RankStack(blend_s1=blend).fit(train)
            pred = scored(stack.predict(test), mode)
            fold_metrics.append(_group_metrics(pred, "stack_score", mae_col="expected_points"))
        agg = pd.DataFrame(fold_metrics).mean(numeric_only=True)
        captures = [agg[f"top_{n}_capture"] for n in (10, 25, 50, 100) if f"top_{n}_capture" in agg]
        rows.append({"blend_s1": blend, "score_mode": mode,
                     "dev_mae": float(agg["mae"]), "dev_spearman": float(agg["spearman"]),
                     "dev_mean_capture": float(np.mean(captures))})
    # Stage-1 baseline on identical folds.
    fold_metrics = [
        _group_metrics(feat_2025[feat_2025.group_id == held].reset_index(drop=True), "s1_exp_points")
        for held in groups
    ]
    agg = pd.DataFrame(fold_metrics).mean(numeric_only=True)
    captures = [agg[f"top_{n}_capture"] for n in (10, 25, 50, 100) if f"top_{n}_capture" in agg]
    rows.append({"blend_s1": np.nan, "score_mode": "stage1_baseline",
                 "dev_mae": float(agg["mae"]), "dev_spearman": float(agg["spearman"]),
                 "dev_mean_capture": float(np.mean(captures))})
    return pd.DataFrame(rows).sort_values("dev_mean_capture", ascending=False).reset_index(drop=True)


def choose_config(sweep: pd.DataFrame) -> tuple[float, str]:
    baseline = sweep[sweep.score_mode == "stage1_baseline"].iloc[0]
    candidates = sweep[(sweep.score_mode != "stage1_baseline")
                       & (sweep.dev_mae <= baseline.dev_mae * DEV_MAE_TOLERANCE)]
    if candidates.empty:
        candidates = sweep[sweep.score_mode != "stage1_baseline"]
    best = candidates.iloc[0]
    return float(best.blend_s1), str(best.score_mode)


def final_eval(feat_2025, feat_2026, feat_gw1, feat_gw2, blend: float, mode: str) -> pd.DataFrame:
    rows = []
    # Six Nations: train on all of 2025, test per 2026 round.
    stack = RankStack(blend_s1=blend).fit(feat_2025)
    pred = scored(stack.predict(feat_2026.reset_index(drop=True)), mode)
    for gid, block in pred.groupby("group_id"):
        block = block.reset_index(drop=True)
        rows.append({"competition": "six_nations", "model": "v2 stack (OOF+tuned)", "group": gid,
                     **_group_metrics(block, "stack_score", mae_col="expected_points")})
        rows.append({"competition": "six_nations", "model": "stage-1 full", "group": gid,
                     **_group_metrics(block, "s1_exp_points")})
    # NCR LOGO: GW1 from 6N-only training; GW2 adds GW1 labels.
    for gw, feat_test, extra in ((1, feat_gw1, []), (2, feat_gw2, [feat_gw1])):
        train = pd.concat([feat_2025, *extra], ignore_index=True)
        stack = RankStack(blend_s1=blend).fit(train)
        pred = scored(stack.predict(feat_test.reset_index(drop=True)), mode)
        rows.append({"competition": "ncr", "model": "v2 stack (OOF+tuned)", "group": f"gw{gw}",
                     **_group_metrics(pred, "stack_score", mae_col="expected_points")})
        rows.append({"competition": "ncr", "model": "stage-1 full", "group": f"gw{gw}",
                     **_group_metrics(pred, "s1_exp_points")})
    return pd.DataFrame(rows)


def render(sweep: pd.DataFrame, results: pd.DataFrame, blend: float, mode: str) -> str:
    def table(comp: str, incumbent: tuple[str, str] | None) -> str:
        block = results[results.competition == comp]
        agg = (block.groupby("model", as_index=False)
               .mean(numeric_only=True).drop(columns=["n"], errors="ignore"))
        lines = ["| Model | MAE | Spearman | Top10 cap | Top25 cap | Top50 cap | Top100 cap |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        rows = list(agg.itertuples(index=False))
        if incumbent:
            lines.append(incumbent[1])
        for r in rows:
            lines.append(f"| {r.model} | {r.mae:.2f} | {r.spearman:.3f} | "
                         f"{r.top_10_capture:.1%} | {r.top_25_capture:.1%} | "
                         f"{r.top_50_capture:.1%} | {r.top_100_capture:.1%} |")
        return "\n".join(lines)

    six_inc = ("6N champion", "| 6N champion (incumbent) | 7.30 | 0.666 | 71.5% | 77.1% | 83.5% | 95.2% |")
    ncr_inc = ("NCR incumbent", "| NCR incumbent | 8.90 | 0.581 | 62.7% | 71.5% | 74.2% | 80.4% |")
    top = sweep.head(8).to_string(index=False)
    return "\n".join([
        "# Unified head-to-head: stage-1 vs v2 stack (confounders removed)",
        "",
        "Stage-2 trains on OUT-OF-FOLD stage-1 features (pre-2025 artifact for the",
        "2025 training season); all stage-1 artifacts are full-window refits (no",
        "15% tail discard). Stack config chosen on 2025 leave-one-round-out dev",
        f"only: **blend_s1={blend}, score={mode}** (gate: dev MAE within 2% of",
        "stage-1). Incumbent rows are unchanged from benchmark_v2_full.md.",
        "",
        "## Dev sweep (6N 2025 LORO, top configs)",
        "", "```", top, "```", "",
        "## Six Nations 2026 (sealed holdout, per-round averaged)",
        "", table("six_nations", six_inc), "",
        "## NCR (leave-one-gameweek-out, 532-row parity cohort)",
        "", table("ncr", ncr_inc), "",
        "Same-cohort per-round metrics; small samples (5 + 2 rounds), no",
        "significance tests. Source CSVs: stack_experiments_dev.csv,",
        "stack_experiments_results.csv.",
    ])


def main() -> None:
    store_feat = load_store_features()
    stage1 = {key: fit_full_stage1(store_feat, key) for key in FULL_CUTOFFS}
    labels = build_fantasy_labels()
    six = labels[labels.competition == "six_nations"]
    print("[features] building 4 feature tables", flush=True)
    feat_2025 = build_stage2_features(six[six.season == 2025], store_feat, stage1["pre2025"])
    feat_2026 = build_stage2_features(six[six.season == 2026], store_feat, stage1["pre2026"])
    ncr_labels = labels[labels.competition == "ncr"]
    feat_gw1 = build_stage2_features(ncr_labels[ncr_labels["round"] == 1], store_feat, stage1["gw1"])
    feat_gw2 = build_stage2_features(ncr_labels[ncr_labels["round"] == 2], store_feat, stage1["gw2"])
    print("[sweep] 2025 LORO config sweep", flush=True)
    sweep = dev_sweep(feat_2025)
    sweep.to_csv(OUT / "stack_experiments_dev.csv", index=False)
    blend, mode = choose_config(sweep)
    print(f"[sweep] chosen blend_s1={blend} score_mode={mode}", flush=True)
    results = final_eval(feat_2025, feat_2026, feat_gw1, feat_gw2, blend, mode)
    results.to_csv(OUT / "stack_experiments_results.csv", index=False)
    report = render(sweep, results, blend, mode)
    (OUT / "stack_experiments.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
