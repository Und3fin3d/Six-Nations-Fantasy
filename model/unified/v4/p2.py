"""P2 admission: hurdle heads for sparse high-value events, on top of P1's B.

One B+hurdle fit per 6N-2025 LORO fold yields BOTH configurations' tail
diagnostics: the plain negative-binomial heads inside the hurdle model are
trained identically to variant B's (same features, same rows, same seed), so
the hurdle-vs-NB comparison is exact and paired.

Precommitted conjunctive kill (RESEARCH_PLAN.md §4):
  admit hurdle iff, averaged over the four events and five folds,
    (1) P(>=1) Brier improves,
    (2) |p90 coverage - 0.90| improves,
    (3) mean top-N capture does not regress by more than 0.2pp
        against variant B's already-frozen P1 result.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from ..data import ROOT
from ..v3.benchmark import _cohort_predictions
from ..v3.cohorts import match_labels_to_store
from ..v3.harness import fold_for_block, strict_training_frame
from .experiments import HURDLE_EVENTS, _fold_metrics, build_store, selection_labels
from .gbdt import V4GBDT

DATA = ROOT / "data"
OUT = DATA / "unified" / "v4"
CAPTURE_TOLERANCE = 0.002


def _nb_params(mean: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.clip(mean, 1e-9, None)
    var = np.maximum(var, mean + 1e-6)
    r = np.maximum(mean * mean / (var - mean), 0.05)
    return r, r / (r + mean)


def _tail_rows(model: V4GBDT, rows: pd.DataFrame) -> list[dict]:
    pooled = model._frame(rows)
    cat, numeric = model.encoder.transform(pooled)
    X = np.column_stack([cat, numeric])
    out = []
    for event in HURDLE_EVENTS:
        if event not in model.hurdle_models or event not in model.models:
            continue
        actual = pd.to_numeric(rows[event], errors="coerce").to_numpy(float)
        ok = np.isfinite(actual) & rows[f"available__{event}"].astype(bool).to_numpy()
        h_mean, h_var = model._hurdle_means(event, X)
        r_h, p_h = _nb_params(h_mean, h_var)
        plain_mean = np.clip(model.models[event].predict(X), 1e-9, None)
        plain_var = np.full_like(plain_mean, max(model.dispersion[event], 1e-4))
        r_p, p_p = _nb_params(plain_mean, plain_var)
        for name, r, p in (("hurdle", r_h, p_h), ("plain_nb", r_p, p_p)):
            p_ge1 = 1.0 - nbinom.pmf(0, r[ok], p[ok])
            p90 = nbinom.ppf(0.90, r[ok], p[ok])
            hit = (actual[ok] > 0).astype(float)
            out.append({
                "event": event, "head": name, "n": int(ok.sum()),
                "brier": float(np.mean((p_ge1 - hit) ** 2)),
                "p90_coverage": float(np.mean(actual[ok] <= p90)),
            })
    return out


def main() -> None:
    labels = selection_labels()
    store = build_store()
    fold_rows, tail_rows = [], []
    for group_id, block in labels.groupby("group_id", sort=True):
        block = block.reset_index(drop=True)
        fold = fold_for_block(block, store)
        train = strict_training_frame(store, fold)
        matched = match_labels_to_store(block, store)
        matched = matched[matched["store_matched"]]
        model = V4GBDT(weighting="natural", pool_player_id=True,
                       player_effects=True, hurdle_events=HURDLE_EVENTS)
        print(f"[{group_id}] fitting B+hurdle on {len(train):,} rows", flush=True)
        model.fit(train)
        predicted = _cohort_predictions(model, block, store, "six_nations")
        fold_rows.append({"group_id": group_id, "config": "B+hurdle",
                          **_fold_metrics(predicted, matched)})
        for record in _tail_rows(model, matched):
            tail_rows.append({"group_id": group_id, **record})
    folds = pd.DataFrame(fold_rows)
    tails = pd.DataFrame(tail_rows)
    folds.to_csv(OUT / "p2_folds.csv", index=False)
    tails.to_csv(OUT / "p2_tails.csv", index=False)

    p1 = pd.read_csv(OUT / "p1_summary_A_B.csv")
    b_row = p1[p1["config"].eq("B")].iloc[0]
    cuts = [f"top_{n}_capture" for n in (10, 25, 50, 100)]
    b_capture = float(np.mean([b_row[c] for c in cuts]))
    hurdle_capture = float(np.mean([folds[c].mean() for c in cuts]))
    agg = tails.groupby("head").agg(
        brier=("brier", "mean"),
        p90_gap=("p90_coverage", lambda s: float(np.mean(np.abs(s - 0.90)))),
    )
    verdict = {
        "brier_hurdle": float(agg.loc["hurdle", "brier"]),
        "brier_plain": float(agg.loc["plain_nb", "brier"]),
        "p90_gap_hurdle": float(agg.loc["hurdle", "p90_gap"]),
        "p90_gap_plain": float(agg.loc["plain_nb", "p90_gap"]),
        "capture_hurdle": hurdle_capture,
        "capture_B": b_capture,
        "mae_hurdle": float(folds["mae"].mean()),
        "mae_B": float(b_row["mae"]),
    }
    verdict["admit"] = bool(
        verdict["brier_hurdle"] < verdict["brier_plain"]
        and verdict["p90_gap_hurdle"] < verdict["p90_gap_plain"]
        and hurdle_capture >= b_capture - CAPTURE_TOLERANCE
    )
    (OUT / "p2_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
