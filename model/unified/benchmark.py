"""Frozen cross-competition benchmark for the first unified artifacts."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model.ncr_rank_eval import CUTOFFS, ROUNDS, load_actuals, tie_aware_hits

from .data import ROOT
from .features import build_pit_features
from .gbdt import UniversalGBDT
from .neural import UniversalNeuralModel
from .scoring import scorer_for

DATA = ROOT / "data"
OUT = DATA / "unified"
ARTIFACTS = {
    "Unified GBDT": ("gbdt", OUT / "models/unified_gbdt_2026-07-04.pkl"),
    "Unified neural": ("neural", OUT / "models/unified_neural_2026-07-04.pt"),
}


def _load_models(wanted: str):
    for name, (engine, path) in ARTIFACTS.items():
        if engine == wanted:
            return {name: UniversalGBDT.load(path) if engine == "gbdt" else UniversalNeuralModel.load(path)}
    raise ValueError(wanted)


def _expected(predictions, competition: str) -> np.ndarray:
    scorer = scorer_for(competition)
    return np.array([
        scorer.score_prediction(pred, n=1200, seed=1009 + i).mean
        for i, pred in enumerate(predictions)
    ])


def _rank_rows(cohort: pd.DataFrame, actual_pool: pd.DataFrame, model: str, *, gw=None):
    rows = []
    rho = float(spearmanr(cohort[model], cohort["actual"]).statistic)
    oracle = actual_pool.sort_values(["actual", "id"], ascending=[False, True])
    for n in CUTOFFS:
        predicted_ids = set(cohort.nlargest(n, [model, "id"])["id"].astype(int))
        hits, _ = tie_aware_hits(predicted_ids, actual_pool, n)
        captured = float(actual_pool.loc[actual_pool.id.isin(predicted_ids), "actual"].sum())
        rows.append({
            "competition": "NCR", "gw": gw, "model": model, "n": n,
            "overlap": hits / n, "points_captured": captured / float(oracle.head(n).actual.sum()),
            "spearman": rho,
            "points_mae": float(np.mean(np.abs(cohort[model] - cohort["actual"]))),
            "cohort": len(cohort),
        })
    return rows


def benchmark_ncr(features: pd.DataFrame, models) -> pd.DataFrame:
    crosswalk = pd.read_csv(DATA / "ncr/ncr_player_crosswalk.csv")
    crosswalk = crosswalk.dropna(subset=["api_player_id"])
    crosswalk["player_id"] = crosswalk.api_player_id.astype(int).astype(str)
    crosswalk["id"] = crosswalk.fantasy_id.astype(int)
    rows = []
    for gw, files in ROUNDS.items():
        fixture_date = pd.to_datetime(pd.read_csv(DATA / "ncr/ncr_fixtures.csv")
                                      .query("gameday == @gw")["game_date"]).dt.normalize().iloc[0]
        test = features[(features.date.dt.normalize() == fixture_date)
                        & features.source.str.startswith("ncr_international")].copy()
        test["player_id"] = test["player_id"].astype(str)
        if len(test) != 276:
            raise AssertionError(f"GW{gw}: expected 276 NCR rows, got {len(test)}")
        scored = crosswalk[["player_id", "id"]].merge(
            test[["player_id"]].reset_index(drop=True), on="player_id", validate="many_to_one"
        )
        # Preserve prediction order through the explicit API id join.
        for name, model in models.items():
            pred = model.predict_frame(test)
            by_id = {str(p.player_id): score for p, score in zip(pred, _expected(pred, "ncr"))}
            scored[name] = scored.player_id.map(by_id)
        actual_pool = load_actuals(DATA / "ncr" / files["actual"])
        cohort = scored.merge(actual_pool[["id", "actual"]], on="id", validate="one_to_one")
        for name in models:
            rows.extend(_rank_rows(cohort.dropna(subset=[name]), actual_pool, name, gw=gw))
        # Reuse the already-audited incumbent rows for the same official outcome.
        incumbent = pd.read_csv(DATA / "ncr/ncr_rank_evaluation.csv")
        incumbent_scores = actual_pool[["id", "actual"]].copy()
        for label, key in (("NCR", "ncr"), ("Champion", "champion")):
            values = pd.read_csv(DATA / "ncr" / files[key])[["id", "starter_exp"]]
            incumbent_scores = incumbent_scores.merge(
                values.rename(columns={"starter_exp": label}), on="id", validate="one_to_one"
            )
        incumbent_mae = {
            label: float((incumbent_scores[label] - incumbent_scores["actual"]).abs().mean())
            for label in ("NCR", "Champion")
        }
        for row in incumbent[incumbent.gw.eq(gw)].itertuples(index=False):
            rows.append({
                "competition": "NCR", "gw": gw, "model": row.model, "n": int(row.n),
                "overlap": row.overlap, "points_captured": row.points_captured,
                "spearman": row.spearman_rho, "points_mae": incumbent_mae[row.model],
                "cohort": len(incumbent_scores),
            })
    return pd.DataFrame(rows)


def _six_metrics(scores: pd.DataFrame, model: str) -> list[dict]:
    scores = scores.dropna(subset=[model, "actual"]).copy()
    rows = []
    rho = float(spearmanr(scores[model], scores.actual).statistic)
    for n in CUTOFFS:
        n = min(n, len(scores))
        selected = scores.nlargest(n, model)
        oracle = scores.nlargest(n, "actual")
        cutoff = oracle.actual.iloc[-1]
        strict = set(scores.index[scores.actual > cutoff])
        tied = set(scores.index[scores.actual == cutoff])
        weight = (n - len(strict)) / len(tied)
        picked = set(selected.index)
        hits = len(picked & strict) + weight * len(picked & tied)
        rows.append({
            "competition": "Six Nations", "gw": "2026", "model": model, "n": n,
            "overlap": hits / n,
            "points_captured": float(selected.actual.sum() / oracle.actual.sum()),
            "spearman": rho, "points_mae": float(np.mean(np.abs(scores[model] - scores.actual))),
            "cohort": len(scores),
        })
    return rows


def benchmark_six_nations(features: pd.DataFrame, models) -> pd.DataFrame:
    test = features[(pd.to_numeric(features.competition_id, errors="coerce") == 1266)
                    & (features.season == 2026)].copy()
    test["fixture_id"] = test["fixture_id"].astype(str)
    test["player_id"] = test["player_id"].astype(str)
    incumbent = pd.read_csv(DATA / "model_predictions_2026.csv")
    incumbent["fixture_id"] = incumbent.fixture_id.astype(str)
    incumbent["player_id"] = incumbent.player_id.astype(str)
    scores = test[["fixture_id", "player_id"]].reset_index(drop=True).merge(
        incumbent[["fixture_id", "player_id", "official_pts", "target_pts_hat"]],
        on=["fixture_id", "player_id"], validate="one_to_one",
    ).rename(columns={"official_pts": "actual", "target_pts_hat": "6N champion"})
    for name, model in models.items():
        predictions = model.predict_frame(test)
        by_key = {(str(p.fixture_id), str(p.player_id)): s
                  for p, s in zip(predictions, _expected(predictions, "six_nations"))}
        scores[name] = [by_key[(f, p)] for f, p in zip(scores.fixture_id, scores.player_id)]
    rows = []
    for name in [*models, "6N champion"]:
        rows.extend(_six_metrics(scores, name))
    return pd.DataFrame(rows)


def render(results: pd.DataFrame) -> str:
    mean = (results.groupby(["competition", "model", "n"], as_index=False)
            .agg(overlap=("overlap", "mean"), capture=("points_captured", "mean"),
                 spearman=("spearman", "mean"), mae=("points_mae", "mean")))
    lines = [
        "# Unified rugby model — first cross-competition benchmark", "",
        "All unified forecasts come from the same raw-event model. NCR and Six Nations points "
        "are produced afterward by deterministic rule adapters.", "",
    ]
    for competition in ("NCR", "Six Nations"):
        sub = mean[mean.competition.eq(competition)]
        lines.extend([f"## {competition}", "", "| Model | MAE | Spearman | Top 10 capture | Top 25 capture | Top 50 capture | Top 100 capture |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for model in sub.model.unique():
            m = sub[sub.model.eq(model)].set_index("n")
            metric_rows = results[(results.competition.eq(competition))
                                  & (results.model.eq(model))].drop_duplicates(["gw", "model"])
            valid_mae = metric_rows.dropna(subset=["points_mae", "cohort"])
            mae = (float(np.average(valid_mae.points_mae, weights=valid_mae.cohort))
                   if len(valid_mae) else np.nan)
            mae_s = "—" if np.isnan(mae) else f"{mae:.2f}"
            lines.append(f"| {model} | {mae_s} | {m.spearman.mean():.3f} | " + " | ".join(
                f"{100*m.loc[n, 'capture']:.1f}%" if n in m.index else "—" for n in CUTOFFS
            ) + " |")
        lines.append("")
        if competition == "NCR":
            lines.extend([
                "### MAE by completed round", "",
                "MAE is calculated against official platform points. NCR and champion share one "
                "common cohort; unified coverage is shown separately.", "",
                "| Model | GW1 MAE | GW2 MAE | Two-round weighted MAE | Player-rounds |",
                "|---|---:|---:|---:|---:|",
            ])
            for model in ("NCR", "Champion", "Unified GBDT", "Unified neural"):
                mr = results[(results.competition.eq("NCR")) & results.model.eq(model)] \
                    .drop_duplicates(["gw", "model"]).sort_values("gw")
                weighted = float(np.average(mr.points_mae, weights=mr.cohort))
                by_gw = mr.set_index("gw").points_mae
                lines.append(
                    f"| {model} | {by_gw.loc[1]:.2f} | {by_gw.loc[2]:.2f} | "
                    f"**{weighted:.2f}** | {int(mr.cohort.sum())} |"
                )
            lines.extend([
                "",
                "The unified models cover fewer player-rounds because their API-to-fantasy "
                "crosswalk is incomplete. Their MAEs are therefore informative but not a strict "
                "like-for-like win or loss against the 532-row incumbent cohort.", "",
            ])
    neural = mean[mean.model.eq("Unified neural")]
    reasons = []
    for competition, incumbent in (("NCR", "NCR"), ("Six Nations", "6N champion")):
        c = neural[neural.competition.eq(competition)].set_index("n")
        b = mean[(mean.competition.eq(competition)) & (mean.model.eq(incumbent))].set_index("n")
        for n in CUTOFFS:
            if n in c.index and n in b.index and c.loc[n, "capture"] < b.loc[n, "capture"] - .02:
                reasons.append(f"{competition} top-{n} capture is more than 2pp below {incumbent}")
    lines.extend(["## Promotion decision", ""])
    if reasons:
        lines.append("**Do not promote yet.** The unified neural model fails the frozen non-inferiority gate:")
        lines.extend(["", *[f"- {reason}" for reason in reasons]])
    else:
        lines.append("**Passes the rank non-inferiority gate.** Continue the three-round NCR shadow before routing production picks.")
    lines.extend(["", "The GBDT remains a must-beat universal control; it is not blended into the neural output.", ""])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("gbdt", "neural"))
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.render:
        paths = [OUT / "benchmark_gbdt.csv", OUT / "benchmark_neural.csv"]
        results = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        results = results.drop_duplicates(["competition", "gw", "model", "n"])
        results.to_csv(OUT / "benchmark.csv", index=False)
        report = render(results)
        (OUT / "benchmark.md").write_text(report)
        print(report)
        return
    if not args.engine:
        ap.error("--engine is required unless --render is used")
    raw = pd.read_csv(OUT / "player_match.csv", low_memory=False, parse_dates=["date"])
    features = build_pit_features(raw)
    models = _load_models(args.engine)
    results = pd.concat([benchmark_ncr(features, models), benchmark_six_nations(features, models)], ignore_index=True)
    path = OUT / f"benchmark_{args.engine}.csv"
    results.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
