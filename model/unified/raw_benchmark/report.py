"""Aggregate historical raw metrics and apply the frozen promotion ladder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ENGINE_ORDER, EXTENDED_EVENTS, OUT, TOP_NS


def _write_same_or_new(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"immutable report differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float", "float64"]).columns:
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    headers = [str(column) for column in shown.columns]
    rows = [[str(value) for value in row] for row in shown.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _stable_scores(events: pd.DataFrame, cohort: str = "all") -> pd.DataFrame:
    selected = events[
        events["cohort"].eq(cohort)
        & events["tier"].isin(["stable", "minutes"])
    ]
    return (
        selected.groupby(["engine", "fold", "tournament", "calendar_year"], as_index=False)
        ["relative_loss"].mean()
        .rename(columns={"relative_loss": "stable_score"})
    )


def _mean_capture(frame: pd.DataFrame) -> pd.Series:
    columns = [f"top_{n}_capture" for n in TOP_NS if f"top_{n}_capture" in frame]
    return frame[columns].mean(axis=1, skipna=True)


def _extended_fixture_support(extended: pd.DataFrame) -> pd.Series:
    """Count model-independent supported fixtures without engine multiplication."""
    return extended[extended["engine"].eq("v1")].groupby("target")["fixtures"].sum()


def _bootstrap_difference(differences: np.ndarray, seed: int = 17) -> dict[str, float]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return {"n_clusters": 0, "mean": np.nan, "p05": np.nan, "p95": np.nan}
    rng = np.random.default_rng(seed)
    samples = np.empty(2000)
    for index in range(len(samples)):
        chosen = rng.integers(0, len(differences), len(differences))
        samples[index] = float(np.mean(differences[chosen]))
    return {
        "n_clusters": int(len(differences)), "mean": float(np.mean(differences)),
        "p05": float(np.quantile(samples, 0.05)), "p95": float(np.quantile(samples, 0.95)),
    }


def _bootstrap_tables(scores: pd.DataFrame, rankings: pd.DataFrame) -> dict:
    result: dict[str, dict] = {}
    score_pivot = scores.pivot_table(index="fold", columns="engine", values="stable_score")
    if "v1" in score_pivot:
        for engine in score_pivot.columns:
            if engine == "v1":
                continue
            paired = score_pivot[[engine, "v1"]].dropna()
            result.setdefault(engine, {})["stable_score_vs_v1_by_fold"] = _bootstrap_difference(
                paired[engine].to_numpy(float) - paired["v1"].to_numpy(float)
            )
    stable_rank = rankings[rankings["tier"].eq("stable")].copy()
    stable_rank["mean_capture"] = _mean_capture(stable_rank)
    for rubric in sorted(stable_rank["rubric"].dropna().unique()):
        block = stable_rank[stable_rank["rubric"].eq(rubric)]
        for metric in ("mae", "mean_capture"):
            pivot = block.pivot_table(index="slate_id", columns="engine", values=metric)
            if "v1" not in pivot:
                continue
            for engine in pivot.columns:
                if engine == "v1":
                    continue
                paired = pivot[[engine, "v1"]].dropna()
                result.setdefault(engine, {})[f"{rubric}_{metric}_vs_v1_by_slate"] = (
                    _bootstrap_difference(
                        paired[engine].to_numpy(float) - paired["v1"].to_numpy(float)
                    )
                )
    return result


def build_decision(events: pd.DataFrame, rankings: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    scores = _stable_scores(events)
    overall = scores.groupby("engine")["stable_score"].mean()
    if "v1" not in overall:
        raise ValueError("raw benchmark decision requires v1")
    north = _stable_scores(events, "north").groupby("engine")["stable_score"].mean()
    south = _stable_scores(events, "south").groupby("engine")["stable_score"].mean()
    tournament = scores.groupby(["engine", "tournament"])["stable_score"].mean().unstack(0)

    raw_eligible: dict[str, bool] = {}
    reasons: dict[str, list[str]] = {}
    for engine in overall.index:
        engine_reasons = []
        if engine != "v1" and float(overall[engine]) >= float(overall["v1"]):
            engine_reasons.append("competition-balanced stable raw score did not improve over v1")
        for hemisphere, table in (("north", north), ("south", south)):
            if engine in table and "v1" in table and table[engine] > table["v1"] * 1.02:
                engine_reasons.append(f"{hemisphere} stable raw score regressed by more than 2%")
        if engine in tournament and "v1" in tournament:
            regressed = tournament.index[tournament[engine] > tournament["v1"] * 1.02].tolist()
            if regressed:
                engine_reasons.append("tournament-family stable regression >2%: " + ", ".join(regressed))
        raw_eligible[engine] = engine == "v1" or not engine_reasons
        reasons[engine] = engine_reasons

    stable_rank = rankings[rankings["tier"].eq("stable")].copy()
    stable_rank["mean_capture"] = _mean_capture(stable_rank)
    rank_agg = stable_rank.groupby(["engine", "rubric"], as_index=False).agg(
        mae=("mae", "mean"), mean_capture=("mean_capture", "mean"), spearman=("spearman", "mean"),
    )
    eligible_engines = [engine for engine, passed in raw_eligible.items() if passed]
    for rubric in ("six_nations", "ncr"):
        block = rank_agg[rank_agg["rubric"].eq(rubric) & rank_agg["engine"].isin(eligible_engines)]
        if block.empty:
            continue
        best_mae = float(block["mae"].min())
        best_capture = float(block["mean_capture"].max())
        for row in block.itertuples(index=False):
            if row.engine == "v1":
                continue
            if row.mae > best_mae * 1.02:
                reasons[row.engine].append(f"{rubric} reconstructed MAE regressed by more than 2%")
            if row.mean_capture < best_capture - 0.02:
                reasons[row.engine].append(f"{rubric} mean capture regressed by more than 2pp")

    extended = events[(events["tier"].eq("extended")) & events["cohort"].eq("all")]
    # Availability is model-independent. Count each held-out fixture once via
    # the v1 cohort; summing all engines would multiply support by the number
    # of candidates and promote sparse extensions into hard gates too early.
    fixture_support = _extended_fixture_support(extended)
    extended_targets = fixture_support[fixture_support.ge(100)].index
    ext_loss = extended[extended["target"].isin(extended_targets)].groupby(
        ["engine", "target"]
    )["loss"].mean().unstack(0)
    if "v1" in ext_loss:
        for engine in ext_loss.columns:
            if engine == "v1":
                continue
            bad = ext_loss.index[ext_loss[engine] > ext_loss["v1"] * 1.05].tolist()
            if bad:
                reasons[engine].append("extended-event loss regression >5%: " + ", ".join(bad))

    passed = {
        engine: engine == "v1" or (raw_eligible.get(engine, False) and not reasons[engine])
        for engine in overall.index
    }
    candidates = [engine for engine, ok in passed.items() if ok and engine != "v1"]
    selected = None
    if candidates:
        best_score = min(float(overall[engine]) for engine in candidates)
        tied = [engine for engine in candidates if float(overall[engine]) <= best_score * 1.005]
        simplicity = {engine: index for index, engine in enumerate(ENGINE_ORDER)}
        non_blends = [engine for engine in tied if engine != "p3_event_50"]
        selected = min(non_blends or tied, key=lambda engine: simplicity.get(engine, 999))

    summary = pd.DataFrame({
        "engine": overall.index,
        "stable_score": [float(overall[engine]) for engine in overall.index],
        "improvement_vs_v1": [
            1.0 - float(overall[engine]) / float(overall["v1"]) for engine in overall.index
        ],
        "passed": [passed[engine] for engine in overall.index],
        "reasons": ["; ".join(reasons[engine]) for engine in overall.index],
    }).sort_values("stable_score")
    decision = {
        "schema_version": 1,
        "historical_primary_gate": True,
        "selected": selected,
        "v1_stable_score": float(overall["v1"]),
        "passed": passed,
        "reasons": reasons,
        "extended_gated_targets": list(extended_targets),
        "prospective_ncr_status": "confirmatory_veto_pending_or_separate",
        "prospective_veto": {"mae_regression": 0.05, "mean_capture_pp_regression": 0.05},
        "paired_bootstrap": _bootstrap_tables(scores, rankings),
    }
    return decision, summary


def render_report(
    output_dir: Path = OUT, events_path: Path | None = None,
    rankings_path: Path | None = None,
) -> dict:
    events_path = events_path or output_dir / "event_metrics.csv"
    rankings_path = rankings_path or output_dir / "ranking_metrics.csv"
    events = pd.read_csv(events_path)
    rankings = pd.read_csv(rankings_path)
    decision, summary = build_decision(events, rankings)

    stable = _stable_scores(events).groupby("engine", as_index=False)["stable_score"].mean()
    stable = stable.sort_values("stable_score")
    rank = rankings[rankings["tier"].eq("stable")].copy()
    rank["mean_capture"] = _mean_capture(rank)
    rank_agg = rank.groupby(["engine", "rubric"], as_index=False).agg(
        mae=("mae", "mean"), spearman=("spearman", "mean"),
        mean_capture=("mean_capture", "mean"), slates=("slate_id", "nunique"),
    )
    catalog = pd.read_csv(output_dir / "event_catalog.csv")
    lines = [
        "# Historical Raw Rugby Benchmark v1", "",
        "All figures are retrospective, point-in-time tournament holdouts using oracle teamsheets.",
        "Fantasy values are reconstructed/observable, never official points.", "",
        "## Stable raw score", "", _markdown(stable), "",
        "## Reconstructed stable rubrics", "", _markdown(rank_agg), "",
        "## Event coverage", "", _markdown(catalog, digits=3), "",
        "## Historical selection", "", _markdown(summary), "",
        f"Selected candidate: **{decision['selected'] or 'none'}**", "",
        "NCR GW4–7 remains a confirmatory severe-failure veto, not the primary gate.",
    ]
    report = "\n".join(lines) + "\n"
    _write_same_or_new(output_dir / "REPORT.md", report)
    _write_same_or_new(
        output_dir / "decision.json", json.dumps(decision, indent=2, sort_keys=True) + "\n",
    )
    _write_same_or_new(output_dir / "selection_summary.csv", summary.to_csv(index=False))
    return decision
