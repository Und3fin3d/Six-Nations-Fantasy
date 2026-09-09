"""Reconstructed-rubric MAE (raw and calibrated) for stored raw predictions.

The benchmark's `ranking_metrics` reports raw reconstructed-rubric MAE. This
adds a *calibrated* MAE: within each slate the predicted rubric scores are
affinely refit onto the actual scores (least squares) before taking the MAE.

That calibrated figure is an oracle-affine diagnostic, not a deployable
metric -- it uses the held-out slate's own actuals to choose the scale and
offset, so it is an upper bound on what pure recalibration could recover. It
is reported to separate "ranks the slate well but is mis-scaled" from "orders
the slate wrongly", which is exactly the distinction at issue for this engine.

Usage: python -m tools.rubric_mae <engine> [<engine> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "unified" / "raw_benchmark" / "allrugby_empirical"


def main() -> None:
    from model.unified.contracts import RawPrediction
    from model.unified.raw_benchmark.config import STABLE_EVENTS
    from model.unified.raw_benchmark.folds import build_folds, evaluation_frame
    from model.unified.raw_benchmark.metrics import _score_rows, _scorer_events
    from model.unified.scoring import NationsChampionshipScorer, SixNationsScorer

    engines = sys.argv[1:] or ["v1", "empirical_event"]
    store = pd.read_csv(
        WORK / "player_match.csv", low_memory=False, parse_dates=["date", "match_at"],
    )
    rows = []
    for engine in engines:
        directory = WORK / "predictions" / engine
        cached = {path.stem for path in directory.glob("*.jsonl")}
        folds = [fold for fold in build_folds(store) if fold.label in cached]
        for fold in folds:
            evaluation = evaluation_frame(store, fold).reset_index(drop=True)
            predictions = [
                RawPrediction.from_dict(json.loads(line))
                for line in (directory / f"{fold.label}.jsonl").read_text().splitlines()
                if line
            ]
            for rubric, scorer in (
                ("six_nations", SixNationsScorer()), ("ncr", NationsChampionshipScorer()),
            ):
                allowed = _scorer_events(scorer, STABLE_EVENTS)
                for slate_id, slate in evaluation.groupby("slate_id", sort=True):
                    local = slate.reset_index(drop=True)
                    local_predictions = [predictions[int(i)] for i in slate.index.to_numpy()]
                    _, actual, predicted = _score_rows(local, local_predictions, scorer, allowed)
                    if len(actual) < 10:
                        continue
                    raw = float(np.mean(np.abs(actual - predicted)))
                    if float(np.var(predicted)) < 1e-12:
                        calibrated = raw
                    else:
                        slope, intercept = np.polyfit(predicted, actual, 1)
                        calibrated = float(
                            np.mean(np.abs(actual - (intercept + slope * predicted)))
                        )
                    rows.append({
                        "engine": engine, "rubric": rubric, "fold": fold.label,
                        "slate_id": str(slate_id), "n": len(actual),
                        "mae_raw": raw, "mae_calibrated": calibrated,
                    })
            print(f"  {engine} {fold.label} done", flush=True)

    frame = pd.DataFrame(rows)
    summary = frame.groupby(["engine", "rubric"], as_index=False).agg(
        mae_raw=("mae_raw", "mean"), mae_calibrated=("mae_calibrated", "mean"),
        slates=("slate_id", "nunique"),
    )
    print("\n=== reconstructed-rubric MAE ===")
    print(summary.round(4).to_string(index=False))
    frame.to_csv(WORK / "rubric_mae.csv", index=False)


if __name__ == "__main__":
    main()
