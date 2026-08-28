"""v5 benchmark: run :class:`ShrunkFormGBDT` through the *same* official-label
folds as the frozen v3 benchmark and emit per-fold MAE / Spearman / Pearson /
top-N capture, so v5 rows drop straight into the v3 baseline/gbdt_v3/incumbent
comparison (``data/unified/v3/benchmark/fold_metrics.csv``).

Design notes
------------
* One PIT-feature store is built once. It carries ``career_matches`` and
  ``ncr_gameday`` (needed by the cohort matcher) and is *idempotent* under a
  second ``build_pit_features`` call, so v5's internal feature rebuild is safe.
* v5's ``predict_frame`` recomputes point-in-time features from scratch, so it
  must see each evaluation player's history. For every fold we predict on
  ``strict_training_frame(train) + matched evaluation rows`` and align the
  returned predictions back to the cohort by ``(fixture_id, player_id)``.
* Nothing here is a promotion gate. It is the descriptive comparison the plan
  (`data/unified/v5/RESEARCH_PLAN.md` §8) calls the "development diagnostic".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..benchmark_v2 import _group_metrics
from ..data import ROOT
from ..features import build_pit_features
from ..labels import build_fantasy_labels
from ..scoring import scorer_for
from ..v3.benchmark import _incumbent
from ..v3.cohorts import match_labels_to_store
from ..v3.harness import attach_match_timestamps, fold_for_block, strict_training_frame
from .config import V5Config
from .model import ShrunkFormGBDT

DATA = ROOT / "data"
CUTS = (10, 25, 50, 100)
V3_METRICS = DATA / "unified" / "v3" / "benchmark" / "fold_metrics.csv"
OUT = DATA / "unified" / "v5" / "benchmark"


def _pearson(pred: pd.Series, actual: pd.Series) -> float:
    valid = pred.notna() & actual.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(pred[valid].to_numpy(float), actual[valid].to_numpy(float))[0, 1])


def _phase(block: pd.DataFrame) -> str:
    return (
        "selection" if block["competition"].iloc[0] == "six_nations"
        and int(block["season"].iloc[0]) == 2025 else "retrospective_reference"
    )


def _cohort_predictions(model: ShrunkFormGBDT, block: pd.DataFrame,
                        store: pd.DataFrame, fold, competition: str,
                        mc: int) -> pd.DataFrame:
    """v5 analogue of v3's ``_cohort_predictions`` with history-aware predict."""
    evaluation = match_labels_to_store(block, store)
    output = block.reset_index(drop=True).copy()
    output["_label_row_id"] = output.index
    matched = evaluation[evaluation["store_matched"]].copy()

    train = strict_training_frame(store, fold)
    # Select evaluation rows straight from the store so dtypes (player_id,
    # fixture_id) stay identical to train: `matched` casts keys to str during
    # the label merge, and a dtype mismatch would split each player's history
    # in v5's internal groupby rebuild (predicting everyone as a debutant).
    pairs = set(zip(matched["fixture_id"].astype(str), matched["player_id"].astype(str)))
    in_pairs = [
        (f, p) in pairs
        for f, p in zip(store["fixture_id"].astype(str), store["player_id"].astype(str))
    ]
    eval_rows = store[in_pairs]
    combined = pd.concat([train, eval_rows], ignore_index=True, sort=False)

    predictions = model.predict_frame(combined)
    by_key = {(str(p.fixture_id), str(p.player_id)): p for p in predictions}
    scorer = scorer_for(competition)
    points = []
    for i, (fixture, player) in enumerate(
        zip(matched["fixture_id"].astype(str), matched["player_id"].astype(str))
    ):
        prediction = by_key.get((fixture, player))
        points.append(
            scorer.score_prediction(prediction, n=mc, seed=701 + i).mean
            if prediction is not None else np.nan
        )
    matched["predicted_points"] = points

    values = matched.set_index("_label_row_id")["predicted_points"]
    output["predicted_points"] = output["_label_row_id"].map(values)
    output["career_matches"] = output["_label_row_id"].map(
        matched.set_index("_label_row_id")["career_matches"]
    )
    output["matched"] = output["predicted_points"].notna()
    # Full-cohort parity: unmatched rows get position medians (v3 convention).
    position_medians = (
        output[output["matched"]].groupby("position")["predicted_points"].median()
    )
    global_median = float(output["predicted_points"].median()) if output["matched"].any() else 0.0
    need = output["predicted_points"].isna()
    output.loc[need, "predicted_points"] = (
        output.loc[need, "position"].map(position_medians).fillna(global_median)
    )
    return output


def run(groups: tuple[str, ...] | None = None, mc: int = 800,
        config: V5Config | None = None) -> pd.DataFrame:
    config = config or V5Config()
    labels = build_fantasy_labels()
    raw = pd.read_csv(DATA / "unified" / "player_match.csv",
                      low_memory=False, parse_dates=["date"])
    store = build_pit_features(attach_match_timestamps(raw))

    OUT.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows = [], []
    for group_id, block in labels.groupby("group_id", sort=True):
        if groups and group_id not in groups:
            continue
        block = block.reset_index(drop=True)
        competition = str(block["competition"].iloc[0])
        phase = _phase(block)
        fold = fold_for_block(block, store)
        train = strict_training_frame(store, fold)
        print(f"[{phase}] v5 {group_id}: train {len(train):,}", flush=True)
        model = ShrunkFormGBDT(config=config).fit(train)
        predicted = _cohort_predictions(model, block, store, fold, competition, mc)
        predicted["incumbent_points"] = _incumbent(block).to_numpy()
        predicted["engine"] = "v5"
        predicted["phase"] = phase
        prediction_rows.append(predicted)

        metrics = _group_metrics(
            predicted.rename(columns={"official_pts": "actual"}),
            "predicted_points", actual_col="actual",
        )
        metrics["pearson"] = _pearson(predicted["predicted_points"], predicted["official_pts"])
        metric_rows.append({
            "competition": competition, "season": int(block["season"].iloc[0]),
            "round": int(block["round"].iloc[0]), "group_id": group_id,
            "phase": phase, "model": "v5", **metrics,
        })
        # Per-fold checkpoint so a long background run is observable / resumable.
        pd.DataFrame(metric_rows).to_csv(OUT / "fold_metrics_v5.csv", index=False)
        pd.concat(prediction_rows, ignore_index=True, sort=False).to_csv(
            OUT / "predictions_v5.csv", index=False)
        row = metric_rows[-1]
        print(f"    -> MAE {row['mae']:.2f}  rho {row['spearman']:.3f}  "
              f"top10 {row.get('top_10_capture', float('nan')):.1%}", flush=True)

    return pd.DataFrame(metric_rows)


def compare(v5_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate v5 with the frozen v3 baseline/gbdt_v3/incumbent metrics."""
    cols = ["mae", "spearman", *[f"top_{n}_capture" for n in CUTS]]
    frozen = pd.read_csv(V3_METRICS)
    combined = pd.concat([frozen, v5_metrics], ignore_index=True, sort=False)
    # Only phases/groups v5 actually covered, for a fair aggregate.
    covered = set(zip(v5_metrics["phase"], v5_metrics["competition"], v5_metrics["group_id"]))
    mask = [
        (p, c, g) in covered
        for p, c, g in zip(combined["phase"], combined["competition"], combined["group_id"])
    ]
    combined = combined[mask]
    agg = combined.groupby(["phase", "competition", "model"], as_index=False)[cols].mean()
    return agg.sort_values(["phase", "competition", "model"]).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--mc", type=int, default=800)
    args = parser.parse_args()
    result = run(tuple(args.groups) if args.groups else None, mc=args.mc)
    print(result.to_string(index=False))
