"""Audit scorer reconstruction, event coverage, appearance labels, and context."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import ROOT
from ..labels import build_fantasy_labels
from ..scoring import scorer_for
from .cohorts import match_labels_to_store
from .harness import OUT, load_timed_store
from .metrics import observable_events, observable_points_actual

DATA = ROOT / "data"


def appearance_eligibility(store: pd.DataFrame) -> dict:
    """Decide whether zero-minute rows are selected-but-unused observations."""
    zero = store[pd.to_numeric(store["minutes"], errors="coerce").eq(0)].copy()
    jersey_valid = pd.to_numeric(zero["jersey"], errors="coerce").between(1, 23)
    started_share = float(zero["started"].astype(bool).mean()) if len(zero) else 0.0
    appearance_enabled = bool(
        len(zero) >= 1000 and jersey_valid.mean() >= 0.90 and started_share <= 0.05
    )
    return {
        "zero_minute_rows": int(len(zero)),
        "zero_minute_jersey_1_23_share": float(jersey_valid.mean()) if len(zero) else 0.0,
        "zero_minute_started_share": started_share,
        "appearance_head_enabled": appearance_enabled,
    }


def _reconstructed_points(frame: pd.DataFrame, competition: str) -> np.ndarray:
    scorer = scorer_for(competition)
    rows = []
    for row in frame.itertuples(index=False):
        events = {}
        for event in set(getattr(scorer, "weights", {})) | {"scrums_won", "metres"}:
            available = getattr(row, f"available__{event}", False)
            value = getattr(row, event, np.nan)
            if bool(available) and pd.notna(value):
                events[event] = np.array([float(value)])
        rows.append(float(scorer.score_samples(events, is_forward=bool(row.is_forward))[0]))
    return np.asarray(rows)


def run_audit(store_path: Path = DATA / "unified" / "player_match.csv",
              output_dir: Path = OUT / "audit") -> dict:
    store = load_timed_store(store_path)
    labels = build_fantasy_labels()
    joined = match_labels_to_store(labels, store)
    matched = joined[joined["store_matched"]].copy()
    reconstruction_rows = []
    coverage_rows = []
    correlation_rows = []
    row_coverage_rows = []
    for competition, block in matched.groupby("competition_label"):
        block = block.copy()
        block["reconstructed_pts"] = _reconstructed_points(block, competition)
        block["reconstruction_error"] = block["reconstructed_pts"] - block["official_pts"]
        allowed = observable_events(block)
        block["observable_pts"] = observable_points_actual(block, competition, allowed)
        for group_id, group in block.groupby("group_id"):
            reconstruction_rows.append({
                "competition": competition, "group_id": group_id, "n": len(group),
                "mae": float(np.mean(np.abs(group["reconstruction_error"]))),
                "bias": float(group["reconstruction_error"].mean()),
                "spearman": float(group["reconstructed_pts"].corr(
                    group["official_pts"], method="spearman")),
            })
            correlation_rows.append({
                "competition": competition, "group_id": group_id, "n": len(group),
                "observable_events": "|".join(allowed),
                "observable_official_spearman": float(group["observable_pts"].corr(
                    group["official_pts"], method="spearman")),
            })
        scorer = scorer_for(competition)
        for event in sorted(set(scorer.weights) | {"scrums_won", "metres"}):
            col = f"available__{event}"
            coverage_rows.append({
                "competition": competition, "event": event, "n": len(block),
                "coverage": float(block.get(col, pd.Series(False, index=block.index))
                                  .fillna(False).astype(bool).mean()),
                "observable": event in allowed,
            })
        expected = sorted(set(scorer.weights) | {"scrums_won", "metres"})
        for row in block.itertuples(index=False):
            missing = [
                event for event in expected
                if not bool(getattr(row, f"available__{event}", False))
            ]
            row_coverage_rows.append({
                "competition": competition, "group_id": row.group_id,
                "key_fixture": row.key_fixture, "key_player": row.key_player,
                "player_name": row.player_name_label, "store_matched": True,
                "expected_scoring_events": len(expected),
                "available_scoring_events": len(expected) - len(missing),
                "row_coverage": (len(expected) - len(missing)) / len(expected),
                "missing_scoring_events": "|".join(missing),
            })
    for row in joined[~joined["store_matched"]].itertuples(index=False):
        scorer = scorer_for(row.competition_label)
        expected = sorted(set(scorer.weights) | {"scrums_won", "metres"})
        row_coverage_rows.append({
            "competition": row.competition_label, "group_id": row.group_id,
            "key_fixture": row.key_fixture, "key_player": row.key_player,
            "player_name": row.player_name_label, "store_matched": False,
            "expected_scoring_events": len(expected), "available_scoring_events": 0,
            "row_coverage": 0.0, "missing_scoring_events": "|".join(expected),
        })

    zero_summary = appearance_eligibility(store)

    weather = DATA / "external_fixture_weather.csv"
    context = {
        "wr": {"pit_safe": True, "reason": "snapshot_date is joined backward at fixture date"},
        "roles": {"pit_safe": True, "reason": "builder uses shifted role/form features and named team sheet"},
        "style": {"pit_safe": True, "reason": "builder filters team history strictly before fixture date"},
        "weather": {
            "pit_safe": False,
            "reason": "current file uses archived observed daily weather, not a pre-lock forecast",
            "rows": int(len(pd.read_csv(weather))) if weather.exists() else 0,
        },
    }
    reconstruction = pd.DataFrame(reconstruction_rows)
    correlations = pd.DataFrame(correlation_rows)
    coverage = pd.DataFrame(coverage_rows)
    row_coverage = pd.DataFrame(row_coverage_rows)
    correlation_min = float(correlations["observable_official_spearman"].min())
    payload = {
        "schema_version": 1,
        "official_label_rows": int(len(labels)),
        "matched_label_rows": int(len(matched)),
        "match_rate": float(len(matched) / len(labels)),
        "reconstruction_weighted_mae": float(np.average(
            reconstruction["mae"], weights=reconstruction["n"])),
        "observable_official_spearman_min": correlation_min,
        "observable_points_enabled": bool(correlation_min >= 0.85),
        "zero_minutes": zero_summary,
        "context": context,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = (
        store[[
            "fixture_id", "date", "team", "opponent", "match_at", "lock_at",
            "timestamp_precision", "ncr_gameday", "ncr_fantasy_match_id",
        ]]
        .drop_duplicates()
        .sort_values(["match_at", "fixture_id", "team"])
    )
    timestamps.to_csv(OUT / "match_timestamps.csv", index=False)
    reconstruction.to_csv(output_dir / "scorer_reconstruction.csv", index=False)
    correlations.to_csv(output_dir / "observable_points_correlation.csv", index=False)
    coverage.to_csv(output_dir / "scoring_event_coverage.csv", index=False)
    row_coverage.to_csv(output_dir / "label_row_coverage.csv", index=False)
    (output_dir / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Unified v3 data and scorer audit", "",
        f"- Official label rows: {len(labels):,}; matched to store: {len(matched):,} "
        f"({len(matched)/len(labels):.1%}).",
        f"- Reconstructed-vs-official weighted MAE: {payload['reconstruction_weighted_mae']:.2f}.",
        f"- Minimum within-round observable-points Spearman: {correlation_min:.3f}; "
        f"observable-points tuning: **{'enabled' if payload['observable_points_enabled'] else 'disabled'}**.",
        f"- Zero-minute rows: {zero_summary['zero_minute_rows']:,}; jersey 1–23 share: "
        f"{zero_summary['zero_minute_jersey_1_23_share']:.1%}; appearance head: "
        f"**{'enabled' if zero_summary['appearance_head_enabled'] else 'disabled'}**.",
        "- Weather is excluded by default because the existing data is archived observed weather.",
        "", "Official points remain the final evaluation outcome; reconstructed and observable "
        "points are development diagnostics only.", "",
    ]
    (output_dir / "audit.md").write_text("\n".join(lines))
    return payload
