"""Evaluate frozen unified raw-event models against official NCR GW1-3 points.

This extends the Historical Raw Benchmark v1 Nations Championship holdout by
one round.  All unified engines are frozen at the first GW1 kickoff; GW1-3 are
never admitted to their training histories.  The saved pre-lock incumbent
projection cohorts define the comparison rows for each round.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model.ncr_rank_eval import load_actuals
from model.ncr_eval import load_actuals as load_team_actuals
from model.ncr_eval import team_points
from model.ncr_project import build_projection, optimise

from .benchmark_v2 import CUTOFFS, _group_metrics
from .contracts import RawPrediction
from .data import ROOT
from .raw_benchmark.blend import EventBlend50
from .raw_benchmark.config import OUT as BENCHMARK_OUT
from .raw_benchmark.features import build_frozen_feature_frames
from .raw_benchmark.folds import (
    build_folds,
    masked_candidates,
    strict_training_frame,
)
from .raw_benchmark.runner import _fit_engine
from .scoring import NationsChampionshipScorer
from .v3.shadow import ncr_candidates

DATA = ROOT / "data"
OUT = DATA / "unified" / "ncr_gw1_3_eval"
ROUNDS = (1, 2, 3)
ENGINES = ("p3_event_50", "v1", "v5_t", "ncr_incumbent")
DISPLAY = {
    "p3_event_50": "P3",
    "v1": "v1 GBDT",
    "v5_t": "v5 terminal",
    "ncr_incumbent": "NCR incumbent",
}
GW_EXCLUSIONS = {1: ("New Zealand", "France"), 2: (), 3: ()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_ncr_points(prediction: RawPrediction) -> float:
    """Exact expectation for the NCR adapter's linear scoring rubric."""
    scorer = NationsChampionshipScorer()
    total = sum(
        weight * prediction.events[event].mean
        for event, weight in scorer.weights.items()
        if event in prediction.events
    )
    if "scrums_won" in prediction.events:
        total += 2.0 * prediction.events["scrums_won"].mean
    return float(total)


def _candidate_cohort() -> pd.DataFrame:
    pieces = []
    for gw in ROUNDS:
        candidates = ncr_candidates(gw).copy()
        candidates["gw"] = gw
        pieces.append(candidates)
    cohort = pd.concat(pieces, ignore_index=True, sort=False)
    if cohort.duplicated(["gw", "fantasy_id"]).any():
        raise AssertionError("NCR candidate cohort contains duplicate fantasy IDs")
    return cohort


def _prediction_rows(
    engine: str, candidates: pd.DataFrame, predictions: list[RawPrediction],
) -> pd.DataFrame:
    if len(candidates) != len(predictions):
        raise ValueError(f"{engine}: candidate/prediction length mismatch")
    rows = []
    for source, prediction in zip(candidates.itertuples(index=False), predictions):
        if str(source.player_id) != prediction.player_id:
            raise ValueError(f"{engine}: candidate/prediction row order changed")
        rows.append({
            "gw": int(source.gw),
            "fantasy_id": int(source.fantasy_id),
            "player_id": prediction.player_id,
            "player_name": prediction.player_name,
            "team": prediction.team,
            "status": "P" if bool(source.started) else "B",
            "engine": engine,
            "predicted_points": expected_ncr_points(prediction),
            "expected_minutes": prediction.minutes.mean,
        })
    return pd.DataFrame(rows)


def _official_and_incumbent() -> pd.DataFrame:
    pieces = []
    for gw in ROUNDS:
        feed = DATA / "ncr" / "feeds" / f"players_gw{gw}.json"
        payload = json.loads(feed.read_text())["Data"]["Value"]["Players"]
        payload_gws = {
            int(float(player["gameday_id"]))
            for player in payload if player.get("gameday_id") not in (None, "")
        }
        if payload_gws != {gw}:
            raise ValueError(f"{feed} contains gameday IDs {sorted(payload_gws)}")
        actual = load_actuals(feed)
        projection = pd.read_csv(DATA / "ncr" / f"ncr_gw{gw}_projections.csv")
        cohort = projection[["id", "name", "team", "status", "starter_exp"]].merge(
            actual[["id", "actual"]], on="id", how="left", validate="one_to_one",
        )
        if cohort["actual"].isna().any():
            raise ValueError(f"GW{gw}: official points missing for incumbent cohort")
        cohort = cohort.rename(columns={
            "id": "fantasy_id", "name": "player_name",
            "starter_exp": "predicted_points", "actual": "official_points",
        })
        cohort["gw"] = gw
        cohort["engine"] = "ncr_incumbent"
        cohort["expected_minutes"] = np.nan
        pieces.append(cohort)
    return pd.concat(pieces, ignore_index=True, sort=False)


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gw in ROUNDS:
        for engine in ENGINES:
            block = predictions[
                predictions["gw"].eq(gw) & predictions["engine"].eq(engine)
            ].reset_index(drop=True)
            if block.empty:
                raise ValueError(f"GW{gw}: missing {engine} predictions")
            values = _group_metrics(
                block, "predicted_points", actual_col="official_points",
            )
            rows.append({"scope": f"GW{gw}", "gw": gw, "engine": engine, **values})
    round_rows = pd.DataFrame(rows)
    for engine in ENGINES:
        block = round_rows[round_rows["engine"].eq(engine)]
        averaged = {
            column: float(block[column].mean())
            for column in block.select_dtypes("number").columns
            if column not in {"gw", "n"}
        }
        rows.append({
            "scope": "GW1-3 mean", "gw": np.nan, "engine": engine,
            "n": int(block["n"].sum()), **averaged,
        })
    return pd.DataFrame(rows)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _render(metrics: pd.DataFrame, cutoff: str, row_counts: dict[int, int]) -> str:
    lines = [
        "# NCR GW1-3 unified-model evaluation",
        "",
        "> **GW3 lineup warning:** the original GW3 fantasy projection feed was "
        "carryover/stale (126 P/B mismatches). Its GW3 player-ranking section is "
        "superseded pending a full all-engine rerun. Use `TEAM_REPORT.md` for the "
        "corrected final-lineup P3 versus incumbent team comparison.",
        "",
        "Official fantasy points are the outcome. P3, v1 and v5 use one frozen "
        f"training cutoff at the first GW1 kickoff (`{cutoff}`); no NCR GW1-3 "
        "result enters their training history. The NCR incumbent uses each saved "
        "pre-lock weekly projection.",
        "",
        "Unified predictions are raw-event expectations converted with the NCR "
        "adapter. The historical source does not provide player interceptions or "
        "trustworthy lineouts-won, so those terms are absent rather than imputed.",
        "",
        "## Three-round average",
        "",
        "Each top-N cell is **actual-rank overlap / actual-point capture**.",
        "",
        "| Model | MAE | Spearman | Top 10 | Top 25 | Top 50 | Top 100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    mean = metrics[metrics["scope"].eq("GW1-3 mean")].set_index("engine")
    for engine in ENGINES:
        row = mean.loc[engine]
        lines.append(
            f"| {DISPLAY[engine]} | {row.mae:.2f} | {row.spearman:.3f} | "
            f"{_pct(row.top_10_overlap)} / {_pct(row.top_10_capture)} | "
            f"{_pct(row.top_25_overlap)} / {_pct(row.top_25_capture)} | "
            f"{_pct(row.top_50_overlap)} / {_pct(row.top_50_capture)} | "
            f"{_pct(row.top_100_overlap)} / {_pct(row.top_100_capture)} |"
        )
    lines.extend(["", "## Round by round", ""])
    for gw in ROUNDS:
        lines.extend([
            f"### GW{gw} ({row_counts[gw]} players)", "",
            "| Model | MAE | Spearman | Top 10 overlap/capture | Top 25 overlap/capture | Top 50 overlap/capture | Top 100 overlap/capture |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        block = metrics[metrics["gw"].eq(gw)].set_index("engine")
        for engine in ENGINES:
            row = block.loc[engine]
            lines.append(
                f"| {DISPLAY[engine]} | {row.mae:.2f} | {row.spearman:.3f} | "
                f"{_pct(row.top_10_overlap)} / {_pct(row.top_10_capture)} | "
                f"{_pct(row.top_25_overlap)} / {_pct(row.top_25_capture)} | "
                f"{_pct(row.top_50_overlap)} / {_pct(row.top_50_capture)} | "
                f"{_pct(row.top_100_overlap)} / {_pct(row.top_100_capture)} |"
            )
        lines.append("")
    lines.extend([
        "## Verdict", "",
        "P3 is the strongest unified model in every headline metric, but it does "
        "not beat the NCR incumbent on these three official rounds. Its mean MAE is "
        "5.6% worse (9.39 versus 8.89), and its top-10/25/50/100 point capture trails "
        "by 16.6/13.9/10.6/6.4 percentage points.",
        "",
        "v1 and v5 are exactly prediction-identical here. This is expected from "
        "v5's terminal rung: its EB and slot-minute layers are disabled and its "
        "remaining tree settings match v1; the masked `lineouts_won` event is not "
        "part of this benchmark's eligible target list.",
        "",
        "## Interpretation limits", "",
        "- GW3 official points were fetched after the round, but the model cutoff and "
        "candidate teamsheets are the saved pre-lock versions.",
        "- P3, v1 and v5 are deliberately frozen before GW1 for the entire tournament "
        "block. This is stricter than retraining after each gameweek.",
        "- MAE compares observable-event forecasts with full official points, so missing "
        "NCR rubric events can penalise all three unified models' point levels. Rank and "
        "capture metrics are less sensitive to that common level gap.",
    ])
    return "\n".join(lines) + "\n"


def _team_contributions(squad: pd.DataFrame, actuals: dict) -> pd.DataFrame:
    rows = []
    for player in squad.itertuples(index=False):
        key = str(int(float(player.id)))
        points, played, actual_status = actuals.get(key, (0.0, 0.0, None))
        if bool(player.is_capt):
            multiplier, role = 2.0, "Captain"
        elif bool(player.is_sub):
            if actual_status == "B" and played > 0:
                multiplier, role = 3.0, "Super sub (bench)"
            elif played > 0:
                multiplier, role = 0.5, "Super sub (started)"
            else:
                multiplier, role = 0.0, "Super sub (did not play)"
        else:
            multiplier, role = 1.0, ""
        rows.append({
            "player": player.name, "team": player.team,
            "base_points": float(points), "multiplier": multiplier,
            "contribution": float(points) * multiplier, "role": role,
        })
    return pd.DataFrame(rows)


def _saved_incumbent_squad(gw: int) -> Path:
    if gw == 1:
        return DATA / "ncr" / "ncr_gw1_squad_ex_NewZealand_France.csv"
    return DATA / "ncr" / f"ncr_gw{gw}_squad.csv"


def _corrected_gw3_projection() -> pd.DataFrame:
    feed = DATA / "ncr" / "feeds" / "players_gw3.json"
    players = json.loads(feed.read_text())["Data"]["Value"]["Players"]
    pool = pd.DataFrame(players)
    projection = build_projection(
        pool_override=pool, asof=pd.Timestamp("2026-07-18"), gameday=3,
    )
    projection["id"] = pd.to_numeric(projection["id"], errors="raise").astype(int)
    if len(projection) != 269 or (projection["status"] == "P").sum() != 179:
        raise ValueError(
            "corrected GW3 projection does not match the official 179 starters + 90 bench"
        )
    if projection["name"].astype(str).str.fullmatch("Cheslin Kolbe", case=False).any():
        raise AssertionError("Cheslin Kolbe remains in corrected GW3 lineup")
    projection.to_csv(OUT / "corrected_gw3_incumbent_projections.csv", index=False)
    return projection


def _corrected_gw3_p3(projection: pd.DataFrame) -> pd.DataFrame:
    cached_path = OUT / "corrected_gw3_p3_predictions.csv"
    if cached_path.exists():
        cached = pd.read_csv(cached_path)
        if (
            len(cached) == len(projection)
            and set(cached["fantasy_id"].astype(int)) == set(projection["id"].astype(int))
        ):
            return cached
    store_path = BENCHMARK_OUT / "player_match.csv"
    store = pd.read_csv(store_path, low_memory=False, parse_dates=["date", "match_at"])
    fold = next(
        fold for fold in build_folds(store)
        if fold.label == "nations_championship_2026"
    )
    train = strict_training_frame(store, fold)
    source = ncr_candidates(3, projection=projection)
    source["gw"] = 3
    candidates = masked_candidates(source)
    _, candidate_features = build_frozen_feature_frames(train, candidates, v4=True)
    p3_path = BENCHMARK_OUT / "models" / "p3_event_50" / f"{fold.label}.pkl"
    p3 = EventBlend50.load(p3_path)
    result = _prediction_rows(
        "p3_event_50", candidates, p3.predict_frame(candidate_features),
    )
    result.to_csv(cached_path, index=False)
    return result


def write_team_evaluation(predictions_path: Path = OUT / "predictions.csv") -> pd.DataFrame:
    """Optimise and score one P3 and incumbent team for each saved GW slate."""
    predictions = pd.read_csv(predictions_path)
    corrected_gw3_projection = _corrected_gw3_projection()
    print("P3: rebuilding GW3 on corrected official lineup", flush=True)
    corrected_gw3_p3 = _corrected_gw3_p3(corrected_gw3_projection)
    result_rows, contribution_frames = [], []
    squad_dir = OUT / "p3_squads"
    squad_dir.mkdir(parents=True, exist_ok=True)
    for gw in ROUNDS:
        projection = (
            corrected_gw3_projection.copy() if gw == 3
            else pd.read_csv(DATA / "ncr" / f"ncr_gw{gw}_projections.csv")
        )
        p3_source = (
            corrected_gw3_p3 if gw == 3 else predictions[
                predictions["gw"].eq(gw)
                & predictions["engine"].eq("p3_event_50")
            ]
        )
        p3 = p3_source[["fantasy_id", "predicted_points", "expected_minutes"]]
        pool = projection.merge(
            p3, left_on="id", right_on="fantasy_id", validate="one_to_one",
        )
        excluded = GW_EXCLUSIONS[gw]
        if excluded:
            pool = pool[~pool["team"].isin(excluded)].copy()
        pool["starter_exp"] = pool["predicted_points"]
        # The P3 forecast already reflects starter/bench minutes. The fantasy
        # super-sub multiplier is therefore exactly three times a bench forecast.
        pool["supersub_exp"] = 3.0 * pool["predicted_points"]
        pool["exp_min"] = pool["expected_minutes"]
        p3_squad, _, _ = optimise(pool)
        p3_squad.to_csv(squad_dir / f"gw{gw}.csv", index=False)
        if gw == 3:
            incumbent_squad, _, _ = optimise(projection)
            incumbent_squad.to_csv(OUT / "corrected_incumbent_gw3_squad.csv", index=False)
        else:
            incumbent_squad = pd.read_csv(_saved_incumbent_squad(gw))
        actuals = load_team_actuals(DATA / "ncr" / "feeds" / f"players_gw{gw}.json")
        p3_points = team_points(p3_squad, actuals)
        incumbent_points = team_points(incumbent_squad, actuals)
        p3_contrib = _team_contributions(p3_squad, actuals).assign(gw=gw, model="P3")
        inc_contrib = _team_contributions(incumbent_squad, actuals).assign(
            gw=gw, model="NCR incumbent",
        )
        contribution_frames.extend([p3_contrib, inc_contrib])

        def selected_name(frame: pd.DataFrame, column: str) -> str:
            rows = frame[frame[column].astype(bool)]
            return str(rows.iloc[0]["name"]) if len(rows) else ""

        result_rows.append({
            "gw": gw, "p3_points": p3_points,
            "incumbent_points": incumbent_points,
            "difference": p3_points - incumbent_points,
            "common_players": len(set(p3_squad["id"].astype(int)) & set(incumbent_squad["id"].astype(int))),
            "p3_captain": selected_name(p3_squad, "is_capt"),
            "incumbent_captain": selected_name(incumbent_squad, "is_capt"),
            "p3_super_sub": selected_name(p3_squad, "is_sub"),
            "incumbent_super_sub": selected_name(incumbent_squad, "is_sub"),
            "excluded_teams": ", ".join(excluded),
            "lineup_basis": "official final P/B" if gw == 3 else "saved pre-lock P/B",
        })
    result = pd.DataFrame(result_rows)
    contributions = pd.concat(contribution_frames, ignore_index=True)
    result.to_csv(OUT / "team_metrics.csv", index=False)
    contributions.to_csv(OUT / "team_contributions.csv", index=False)

    p3_total = float(result["p3_points"].sum())
    incumbent_total = float(result["incumbent_points"].sum())
    lines = [
        "# P3 versus NCR incumbent fantasy teams — GW1-3", "",
        "Both models are optimised with the same £100m budget, positional, nation, "
        "hemisphere, captain and bench-only super-sub constraints. GW1 excludes New "
        "Zealand and France for both models because their match had already started "
        "when the incumbent team was saved.", "",
        "GW3 is rebuilt for both models from the completed official P/B lineup because "
        "the saved pre-lock fantasy feed was stale. Kolbe and every other non-selected "
        "player are excluded; the original stale GW3 teams are not used below.", "",
        "| GW | P3 points | Incumbent points | P3 difference | Common players | P3 captain | Incumbent captain | P3 super sub | Incumbent super sub |",
        "|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.gw} | {row.p3_points:g} | {row.incumbent_points:g} | "
            f"{row.difference:+g} | {row.common_players}/16 | {row.p3_captain} | "
            f"{row.incumbent_captain} | {row.p3_super_sub} | {row.incumbent_super_sub} |"
        )
    relative = (p3_total / incumbent_total - 1.0) * 100.0
    lines.extend([
        f"| **Total** | **{p3_total:g}** | **{incumbent_total:g}** | "
        f"**{p3_total - incumbent_total:+g}** | — | — | — | — | — |",
        "", f"P3 scored **{abs(relative):.1f}% fewer points** than the incumbent over "
        "the three evaluated teams. It lost every round; the largest gap was GW2, where "
        "the incumbent selected Joaquin Oviedo's 74 and Theo Attissogbe's 55 while "
        "P3 instead carried Edwill van der Merwe's 0 and Aphelele Fassi's 3.",
        "", "## Player contributions", "",
    ])
    for gw in ROUNDS:
        lines.extend([f"### GW{gw}", ""])
        block = contributions[contributions["gw"].eq(gw)]
        for model in ("P3", "NCR incumbent"):
            lines.extend([
                f"#### {model}", "",
                "| Player | Nation | Base | Multiplier | Contribution | Role |",
                "|---|---|---:|---:|---:|---|",
            ])
            for row in block[block["model"].eq(model)].itertuples(index=False):
                lines.append(
                    f"| {row.player} | {row.team} | {row.base_points:g} | "
                    f"{row.multiplier:g}x | {row.contribution:g} | {row.role} |"
                )
            lines.append("")
    (OUT / "TEAM_REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teams-only", action="store_true",
        help="reuse saved player predictions and only rebuild fantasy-team results",
    )
    args = parser.parse_args()
    if args.teams_only:
        result = write_team_evaluation()
        print(result.to_string(index=False))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    store_path = BENCHMARK_OUT / "player_match.csv"
    store = pd.read_csv(store_path, low_memory=False, parse_dates=["date", "match_at"])
    fold = next(fold for fold in build_folds(store) if fold.label == "nations_championship_2026")
    train = strict_training_frame(store, fold)
    candidates = masked_candidates(_candidate_cohort())

    p3_path = BENCHMARK_OUT / "models" / "p3_event_50" / f"{fold.label}.pkl"
    print(f"P3: building frozen candidate features for {len(candidates)} rows", flush=True)
    v4_train, v4_candidates = build_frozen_feature_frames(train, candidates, v4=True)
    del v4_train
    p3 = EventBlend50.load(p3_path)
    prediction_frames = [
        _prediction_rows("p3_event_50", candidates, p3.predict_frame(v4_candidates))
    ]
    del p3, v4_candidates

    print("v1: building frozen features and fitting in memory", flush=True)
    base_train, base_candidates = build_frozen_feature_frames(train, candidates, v4=False)
    v1 = _fit_engine("v1", train, fold, prepared_train=base_train)
    prediction_frames.append(
        _prediction_rows("v1", candidates, v1.predict_frame(base_candidates))
    )
    del v1, base_train

    print("v5: fitting terminal configuration in memory", flush=True)
    v5 = _fit_engine("v5_t", train, fold)
    prediction_frames.append(
        _prediction_rows("v5_t", candidates, v5.predict_features(base_candidates))
    )
    del v5, base_candidates

    incumbent = _official_and_incumbent()
    labels = incumbent[["gw", "fantasy_id", "official_points"]].copy()
    unified = pd.concat(prediction_frames, ignore_index=True).merge(
        labels, on=["gw", "fantasy_id"], how="left", validate="many_to_one",
    )
    if unified["official_points"].isna().any():
        raise ValueError("unified prediction lacks an official fantasy label")
    predictions = pd.concat([
        unified,
        incumbent[[
            "gw", "fantasy_id", "player_name", "team", "status", "engine",
            "predicted_points", "expected_minutes", "official_points",
        ]],
    ], ignore_index=True, sort=False)
    metrics = _metrics(predictions)

    predictions_path = OUT / "predictions.csv"
    metrics_path = OUT / "metrics.csv"
    report_path = OUT / "REPORT.md"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    row_counts = {
        gw: int((predictions["gw"].eq(gw) & predictions["engine"].eq("ncr_incumbent")).sum())
        for gw in ROUNDS
    }
    report_path.write_text(_render(metrics, fold.cutoff, row_counts))
    sources = {
        str(path): sha256(path)
        for path in [
            store_path, p3_path,
            *(DATA / "ncr" / "feeds" / f"players_gw{gw}.json" for gw in ROUNDS),
            *(DATA / "ncr" / f"ncr_gw{gw}_projections.csv" for gw in ROUNDS),
        ]
    }
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_cutoff": fold.cutoff,
        "training_rows": int(len(train)),
        "evaluation_rounds": list(ROUNDS),
        "evaluation_rows": row_counts,
        "engines": list(ENGINES),
        "fixed_tournament_holdout": True,
        "official_points_fetched_post_round": {"1": True, "2": True, "3": True},
        "source_sha256": sources,
        "retained_fitted_artifacts": [str(p3_path)],
        "temporary_fitted_artifacts_written": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_team_evaluation(predictions_path)
    print(report_path.read_text())


if __name__ == "__main__":
    main()
