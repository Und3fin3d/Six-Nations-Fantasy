"""P1/P2 admission experiments on the Six Nations 2025 LORO selection layer.

Quarantine (RESEARCH_PLAN.md S3): NCR GW1-2 and 6N 2026 are never read here.
Admission decisions (<=3): P1 variant choice, P1 admit, P2 admit — all on
6N-2025 leave-one-round-out official points, paired against the frozen v3
baseline fold predictions.

  A: V4GBDT + EB-shrunk rate features (per-fold moment-matched K) +
     level-split form + intl/club history counts (player_id retained)
  B: V4GBDT with pooled player_id + post-hoc EB player effects +
     level-split form + intl/club history counts
  P2: hurdle heads (tries, try_assists, clean_breaks, tackle_turnover)
      layered on the P1 winner (or on plain V4 base features if P1 dies).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..benchmark_v2 import _group_metrics
from ..data import ROOT
from ..features import build_pit_features
from ..labels import build_fantasy_labels
from ..v3.benchmark import _cohort_predictions
from ..v3.harness import attach_match_timestamps, fold_for_block, strict_training_frame
from .features import THIN_INTL_MATCHES, add_v4_base_stats, apply_eb_features, fit_shrinkage_k
from .gbdt import V4GBDT

DATA = ROOT / "data"
OUT = DATA / "unified" / "v4"
BENCH = DATA / "unified" / "v3" / "benchmark"

HURDLE_EVENTS = ("tries", "try_assists", "clean_breaks", "tackle_turnover")
MAE_HOLD = 1.005  # "without degrading overall MAE": <=0.5% worse than baseline


def selection_labels() -> pd.DataFrame:
    labels = build_fantasy_labels()
    return labels[labels["competition"].eq("six_nations")
                  & labels["season"].eq(2025)].copy()


def build_store() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "unified" / "player_match.csv", low_memory=False,
                      parse_dates=["date"])
    return add_v4_base_stats(build_pit_features(attach_match_timestamps(raw)))


def _config(name: str, store: pd.DataFrame, train: pd.DataFrame):
    if name == "A":
        k = fit_shrinkage_k(train)
        return V4GBDT(weighting="natural"), apply_eb_features(store, k), k
    if name == "B":
        return V4GBDT(weighting="natural", pool_player_id=True,
                      player_effects=True), store, None
    if name == "A+hurdle":
        k = fit_shrinkage_k(train)
        return (V4GBDT(weighting="natural", hurdle_events=HURDLE_EVENTS),
                apply_eb_features(store, k), k)
    if name == "B+hurdle":
        return (V4GBDT(weighting="natural", pool_player_id=True, player_effects=True,
                       hurdle_events=HURDLE_EVENTS), store, None)
    if name == "base+hurdle":
        return V4GBDT(weighting="natural", hurdle_events=HURDLE_EVENTS), store, None
    raise ValueError(name)


def _frozen_baseline() -> pd.DataFrame:
    predictions = pd.read_csv(BENCH / "predictions.csv")
    return predictions[predictions["engine"].eq("baseline")
                       & predictions["phase"].eq("selection")]


def _fold_metrics(predicted: pd.DataFrame, store_matched: pd.DataFrame) -> dict:
    metrics = _group_metrics(
        predicted.rename(columns={"official_pts": "actual"}),
        "predicted_points", actual_col="actual",
    )
    joined = predicted.merge(
        store_matched[["_label_row_id", "intl_prior_matches"]],
        on="_label_row_id", how="left",
    )
    thin = joined[pd.to_numeric(joined["intl_prior_matches"], errors="coerce")
                  .lt(THIN_INTL_MATCHES)]
    metrics["thin_n"] = int(len(thin))
    metrics["thin_mae"] = (
        float((thin["predicted_points"] - thin["official_pts"]).abs().mean())
        if len(thin) else np.nan
    )
    return metrics


def run(config_names: tuple[str, ...]) -> pd.DataFrame:
    labels = selection_labels()
    store = build_store()
    frozen = _frozen_baseline()
    from ..v3.cohorts import match_labels_to_store
    rows = []
    for group_id, block in labels.groupby("group_id", sort=True):
        block = block.reset_index(drop=True)
        fold = fold_for_block(block, store)
        train = strict_training_frame(store, fold)
        matched = match_labels_to_store(block, store)
        matched = matched[matched["store_matched"]]
        base_block = frozen[frozen["group_id"].eq(group_id)].copy()
        rows.append({"group_id": group_id, "config": "baseline(frozen)",
                     **_fold_metrics(base_block, matched)})
        for name in config_names:
            model, feat_store, k = _config(name, store, train)
            feat_train = strict_training_frame(feat_store, fold)
            print(f"[{group_id}] fitting {name} on {len(feat_train):,} rows", flush=True)
            model.fit(feat_train)
            predicted = _cohort_predictions(model, block, feat_store, "six_nations")
            record = {"group_id": group_id, "config": name,
                      **_fold_metrics(predicted, matched)}
            if k:
                record["k_tries"] = k.get("tries")
                record["k_tackles"] = k.get("tackles")
            rows.append(record)
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    cols = ["mae", "spearman", "top_10_capture", "top_25_capture",
            "top_50_capture", "top_100_capture", "thin_mae", "thin_n"]
    return results.groupby("config", as_index=False)[cols].mean(numeric_only=True)


def main() -> None:
    import sys
    configs = tuple(sys.argv[1:]) or ("A", "B")
    results = run(configs)
    summary = summarize(results)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(c.replace("+", "") for c in configs)
    results.to_csv(OUT / f"p1_folds_{suffix}.csv", index=False)
    summary.to_csv(OUT / f"p1_summary_{suffix}.csv", index=False)
    print(summary.to_string(index=False))
    base = summary[summary["config"].eq("baseline(frozen)")].iloc[0]
    verdicts = {}
    for name in configs:
        row = summary[summary["config"].eq(name)].iloc[0]
        capture = float(np.mean([row[f"top_{n}_capture"] for n in (10, 25, 50, 100)]))
        base_capture = float(np.mean([base[f"top_{n}_capture"] for n in (10, 25, 50, 100)]))
        verdicts[name] = {
            "mae_holds": bool(row["mae"] <= base["mae"] * MAE_HOLD),
            "spearman_improves": bool(row["spearman"] > base["spearman"]),
            "capture_improves": bool(capture > base_capture),
            "thin_n_total": int(results[results["config"].eq(name)]["thin_n"].sum()),
            "thin_mae": None if np.isnan(row["thin_mae"]) else float(row["thin_mae"]),
            "thin_mae_baseline": None if np.isnan(base["thin_mae"]) else float(base["thin_mae"]),
        }
    print(json.dumps(verdicts, indent=2))
    (OUT / f"p1_verdicts_{suffix}.json").write_text(json.dumps(verdicts, indent=2) + "\n")


if __name__ == "__main__":
    main()
