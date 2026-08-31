from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SYNTHESIS = ROOT / "data" / "unified" / "research_synthesis"


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_research_synthesis_matches_canonical_pr6_outputs():
    synthesis = _json(SYNTHESIS / "summary.json")
    hillclimb = _json(ROOT / "data" / "unified" / "p3_hillclimb" / "summary.json")
    checkpoint = _json(ROOT / "data" / "unified" / "p3_checkpoint" / "summary.json")

    raw_all = next(row for row in hillclimb["raw"] if row["split"] == "all")
    weighted_total = next(
        row["team_points"]
        for row in hillclimb["official_ncr_team_metrics"]
        if row["gw"] == 0 and row["engine"] == "p3_event_weighted"
    )
    incumbent_total = next(
        row["team_points"]
        for row in hillclimb["official_ncr_team_metrics"]
        if row["gw"] == 0 and row["engine"] == "ncr_incumbent"
    )
    canonical = synthesis["canonical_pr6"]

    assert canonical["baseline_raw_score"] == pytest.approx(raw_all["baseline_score"])
    assert canonical["weighted_raw_score"] == pytest.approx(raw_all["candidate_score"])
    assert canonical["weighted_ncr_gw1_3_team_points"] == weighted_total
    assert canonical["ncr_incumbent_gw1_3_team_points"] == incumbent_total
    assert canonical["retrospective_checkpoint_team_points"] == checkpoint["official"]["total"]
    assert canonical["official_ncr_used_for_hillclimb_search"] == hillclimb[
        "official_ncr_used_for_search"
    ]
    assert canonical["checkpoint_selection_feedback_competition_specific"] == (
        checkpoint["retrospective_oracle"]
        and checkpoint["raw"]["uses_2026_raw_feedback"]
    )
    assert not canonical["active_model_changed"]


def test_research_synthesis_keeps_superseded_sources_as_findings_only():
    synthesis = _json(SYNTHESIS / "summary.json")
    sources = {source["pr"]: source for source in synthesis["sources"]}

    assert set(sources) == {3, 4, 5}
    assert all(source["disposition"] == "findings_only" for source in sources.values())
    assert not synthesis["retained_findings"]["empirical_engine"][
        "any_candidate_passed_original_v1_gate"
    ]
    assert synthesis["retained_findings"]["p3_event_weights"][
        "pr4_used_2026_for_candidate_family_selection"
    ]
    assert synthesis["retained_findings"]["six_nations_champion"][
        "evaluation_folds"
    ] == 2


def test_research_synthesis_weight_table_matches_pr6_config():
    synthesis = _json(SYNTHESIS / "summary.json")
    config = _json(ROOT / "data" / "unified" / "p3_hillclimb" / "config.json")
    with (SYNTHESIS / "p3_event_weight_comparison.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 24
    for row in rows:
        expected = config["event_weights_v4"].get(
            row["target"], config["default_weight_v4"]
        )
        assert float(row["pr6_weight_v4"]) == pytest.approx(expected)
        assert float(row["difference_pr4_minus_pr6"]) == pytest.approx(
            float(row["pr4_weight_v4"]) - expected
        )

    pr4 = [float(row["pr4_weight_v4"]) for row in rows]
    pr6 = [float(row["pr6_weight_v4"]) for row in rows]
    pr4_mean = sum(pr4) / len(pr4)
    pr6_mean = sum(pr6) / len(pr6)
    covariance = sum(
        (left - pr4_mean) * (right - pr6_mean)
        for left, right in zip(pr4, pr6)
    )
    correlation = covariance / math.sqrt(
        sum((value - pr4_mean) ** 2 for value in pr4)
        * sum((value - pr6_mean) ** 2 for value in pr6)
    )
    mean_absolute_difference = sum(
        abs(left - right) for left, right in zip(pr4, pr6)
    ) / len(pr4)
    finding = synthesis["retained_findings"]["p3_event_weights"]
    assert finding["weight_correlation_with_pr6"] == pytest.approx(correlation)
    assert finding["weight_mean_absolute_difference"] == pytest.approx(
        mean_absolute_difference
    )


def test_research_synthesis_preserves_negative_gate_results():
    with (SYNTHESIS / "empirical_candidates.csv").open(newline="") as handle:
        empirical = list(csv.DictReader(handle))
    with (SYNTHESIS / "six_nations_champion.csv").open(newline="") as handle:
        champion = list(csv.DictReader(handle))

    assert len(empirical) == 6
    assert {row["passes_original_v1_gate"] for row in empirical} == {"false"}
    champion_final = next(
        row for row in champion
        if row["model"] == "champion_final" and row["scope"] == "full_24"
    )
    p3 = next(row for row in champion if row["model"] == "p3_event_50")
    assert float(champion_final["stable_score"]) > float(p3["stable_score"])


def test_research_synthesis_manifest_matches_files_and_sources():
    manifest = _json(SYNTHESIS / "manifest.json")
    summary = _json(SYNTHESIS / "summary.json")

    for name, expected in manifest["files_sha256"].items():
        actual = hashlib.sha256((SYNTHESIS / name).read_bytes()).hexdigest()
        assert actual == expected
    assert manifest["source_heads"] == {
        str(source["pr"]): source["head"] for source in summary["sources"]
    } | {"6": summary["canonical_pr6"]["head_before_synthesis"]}
