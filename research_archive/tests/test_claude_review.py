from __future__ import annotations

import json
import unittest

from model.claude_review import (
    _extract_opinion,
    _compact_trial,
    _validate_opinion,
    build_evidence_packet,
    evidence_fingerprint,
)


OPINION = {
    "verdict": "concern",
    "recommendation": "manual_review",
    "confidence": "medium",
    "summary": "The gain comes from one round.",
    "supporting_evidence": ["MAE is unchanged."],
    "concerns": ["Only one round changes."],
    "reversal_conditions": ["A new season reproduces the gain."],
    "next_experiments": [
        {
            "priority": 1,
            "name": "bench_minutes",
            "hypothesis": "Direct minutes evidence improves supersub value.",
            "minimum_test": "Forward-chain by round.",
        }
    ],
}


class ClaudeReviewTests(unittest.TestCase):
    def test_missing_stability_is_unknown_not_false(self) -> None:
        self.assertIsNone(_compact_trial({"name": "old"})["stability_passed"])

    def test_packet_is_bounded_and_excludes_sealed_data(self) -> None:
        trials = [
            {
                "trial": 0,
                "name": "incumbent",
                "accepted": True,
                "value_team": 0.70,
                "mae": 7.4,
                "by_pos": {"Back-row": {"mae": 9.1}},
                "selection_diagnostics": {"large": "payload"},
            },
            {
                "trial": 1,
                "name": "candidate",
                "accepted": True,
                "value_team": 0.71,
                "mae": 7.4,
                "round_team": [0.7] * 5,
            },
        ]
        packet = build_evidence_packet(
            trials,
            {"name": "candidate", "captain_market_weight": 0.0},
            [{"candidate": "candidate", "passed": True}],
            {
                "tested_candidate_names": ["captain_mean_only"],
                "previously_accepted_names": [],
                "recent_results": [],
            },
            {"data_provenance": {"future_timestamps": 0}},
        )
        self.assertFalse(packet["sealed_season_included"])
        self.assertEqual(packet["selected_best"]["name"], "candidate")
        self.assertNotIn("by_pos", packet["trials"][0])
        self.assertNotIn("selection_diagnostics", packet["trials"][0])
        self.assertEqual(
            packet["prior_research_summary"]["tested_candidate_names"][0],
            "captain_mean_only",
        )
        self.assertEqual(
            packet["research_context"]["data_provenance"]["future_timestamps"], 0
        )

    def test_fingerprint_is_stable_across_key_order(self) -> None:
        self.assertEqual(
            evidence_fingerprint({"a": 1, "b": 2}),
            evidence_fingerprint({"b": 2, "a": 1}),
        )

    def test_extracts_claude_json_envelope(self) -> None:
        wrapped = json.dumps({"result": json.dumps(OPINION)})
        parsed = _extract_opinion(wrapped)
        _validate_opinion(parsed)
        self.assertEqual(parsed["recommendation"], "manual_review")

    def test_extracts_structured_output(self) -> None:
        parsed = _extract_opinion(json.dumps({"structured_output": OPINION}))
        _validate_opinion(parsed)
        self.assertEqual(parsed["verdict"], "concern")


if __name__ == "__main__":
    unittest.main()
