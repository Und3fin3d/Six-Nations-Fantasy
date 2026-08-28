"""Separate report for post-freeze raw benchmark challengers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import CHALLENGER_ENGINE_ORDER, OUT
from .report import (
    _markdown, _mean_capture, _stable_scores, _write_same_or_new, build_decision,
)


INELIGIBLE_MODELS = {
    "v2_rank_stack": (
        "fantasy-points-only: learns rank and official-points heads over v1 raw forecasts; "
        "it does not emit independent raw-event predictions"
    ),
    "six_nations_champion": (
        "specialist/partial contract: promoted output includes latent and points-specialist "
        "heads, its native store covers only Six Nations 2023-26, and its internal component "
        "set omits stable benchmark events"
    ),
}


def _read_engine_metrics(output_dir: Path, kind: str, engine: str) -> pd.DataFrame:
    paths = sorted((output_dir / "metrics" / kind / engine).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no {kind} metrics found for challenger {engine}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False)


def render_challenger_report(output_dir: Path = OUT) -> dict:
    base_events = pd.read_csv(output_dir / "event_metrics.csv")
    base_rankings = pd.read_csv(output_dir / "ranking_metrics.csv")
    event_parts = [base_events]
    ranking_parts = [base_rankings]
    for engine in CHALLENGER_ENGINE_ORDER:
        event_parts.append(_read_engine_metrics(output_dir, "events", engine))
        ranking_parts.append(_read_engine_metrics(output_dir, "rankings", engine))
    events = pd.concat(event_parts, ignore_index=True, sort=False)
    rankings = pd.concat(ranking_parts, ignore_index=True, sort=False)
    decision, summary = build_decision(events, rankings)

    stable = _stable_scores(events).groupby("engine", as_index=False)["stable_score"].mean()
    stable = stable.sort_values("stable_score")
    stable_year = _stable_scores(events).groupby(
        ["calendar_year", "engine"], as_index=False,
    )["stable_score"].mean()
    stable_year = stable_year[
        stable_year["engine"].isin(["p3_event_50", "v1", *CHALLENGER_ENGINE_ORDER])
    ].pivot(index="calendar_year", columns="engine", values="stable_score").reset_index()
    rank = rankings[rankings["tier"].eq("stable")].copy()
    rank["mean_capture"] = _mean_capture(rank)
    rank_summary = rank.groupby(["engine", "rubric"], as_index=False).agg(
        mae=("mae", "mean"), spearman=("spearman", "mean"),
        mean_capture=("mean_capture", "mean"), slates=("slate_id", "nunique"),
    )
    champion_path = output_dir / "champion_partial_comparison.csv"
    champion_partial = pd.read_csv(champion_path) if champion_path.exists() else pd.DataFrame()
    champion_event_path = output_dir / "champion_partial_by_event.csv"
    champion_events = (
        pd.read_csv(champion_event_path) if champion_event_path.exists() else pd.DataFrame()
    )
    compatibility = pd.DataFrame([
        {"model": engine, "raw_contract": "eligible", "result": "evaluated on all 21 folds"}
        for engine in CHALLENGER_ENGINE_ORDER
    ] + [
        {"model": model, "raw_contract": "ineligible", "result": reason}
        for model, reason in INELIGIBLE_MODELS.items()
    ])
    lines = [
        "# Historical Raw Rugby Benchmark v1 — challengers", "",
        "The original v1 benchmark artifacts remain frozen. Raw-eligible challengers are "
        "trained on the same 21 cutoff manifests and compared with the frozen core engines.", "",
        "## Raw-contract compatibility", "", _markdown(compatibility), "",
        "## Stable raw score", "", _markdown(stable), "",
        "## Stable raw score by year", "", _markdown(stable_year), "",
        "## Reconstructed stable rubrics", "", _markdown(rank_summary), "",
        "## Six Nations champion partial component diagnostic", "",
        "This is not promotion-eligible: it covers only Six Nations 2025-26 and "
        "12 of the 23 stable events plus minutes.", "",
        _markdown(champion_partial) if not champion_partial.empty else "Not run.", "",
        "### Component breakdown", "",
        _markdown(champion_events) if not champion_events.empty else "Not run.", "",
        "## Gate result", "", _markdown(summary), "",
        f"Selected model after adding challengers: **{decision['selected'] or 'none'}**", "",
    ]
    _write_same_or_new(output_dir / "CHALLENGERS.md", "\n".join(lines))
    payload = {
        "schema_version": 1,
        "challengers": list(CHALLENGER_ENGINE_ORDER),
        "ineligible_models": INELIGIBLE_MODELS,
        "champion_partial_comparison": (
            champion_partial.to_dict("records") if not champion_partial.empty else []
        ),
        "champion_partial_by_event": (
            champion_events.to_dict("records") if not champion_events.empty else []
        ),
        "decision": decision,
    }
    _write_same_or_new(
        output_dir / "challenger_decision.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload
