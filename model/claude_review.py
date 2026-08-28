#!/usr/bin/env python3
"""Evidence-only Claude Code second opinion for autoresearch batches.

The reviewer is deliberately advisory. It may challenge the interpretation or
suggest the next experiment, but it cannot alter acceptance, promotion, or
prediction artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
LATEST_REVIEW = RESEARCH / "claude_review.json"
REVIEW_HISTORY = RESEARCH / "claude_review_history.jsonl"
REVIEW_MARKDOWN = RESEARCH / "CLAUDE_SECOND_OPINION_LATEST.md"
REVIEW_CONTEXT = RESEARCH / "claude_review_context.json"

KEY_METRICS = (
    "value_team",
    "value_xv",
    "mae",
    "bench_mae",
    "top15",
    "top30",
    "capt_top1",
    "capt_top3",
    "capt_top5",
    "spearman_sel",
)

OPINION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["agree", "concern", "insufficient_evidence"],
        },
        "recommendation": {
            "type": "string",
            "enum": [
                "keep_incumbent",
                "advance_candidate",
                "do_not_advance",
                "manual_review",
            ],
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "summary": {"type": "string"},
        "supporting_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "concerns": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "reversal_conditions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "next_experiments": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "priority": {"type": "integer", "minimum": 1, "maximum": 3},
                    "name": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "minimum_test": {"type": "string"},
                },
                "required": ["priority", "name", "hypothesis", "minimum_test"],
            },
        },
    },
    "required": [
        "verdict",
        "recommendation",
        "confidence",
        "summary",
        "supporting_evidence",
        "concerns",
        "reversal_conditions",
        "next_experiments",
    ],
}


def _compact_trial(trial: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "trial": trial.get("trial"),
        "name": trial.get("name"),
        "note": trial.get("note"),
        "accepted": trial.get("accepted"),
        "stability_passed": trial.get("stability_passed"),
        "stability_reason": trial.get("stability_reason"),
        "reason": trial.get("reason"),
    }
    compact.update({key: trial.get(key) for key in KEY_METRICS if key in trial})
    for key in ("round_team", "round_xv", "round_mae", "round_bias"):
        if key in trial:
            compact[key] = trial[key]
    return compact


def build_evidence_packet(
    trials: list[dict[str, Any]],
    best_config: dict[str, Any],
    stability_reports: list[dict[str, Any]] | None = None,
    prior_research: dict[str, Any] | None = None,
    research_context: dict[str, Any] | None = None,
    *,
    dev_season: int = 2025,
) -> dict[str, Any]:
    """Build the bounded, auditable evidence shown to the reviewer."""
    incumbent = _compact_trial(trials[0]) if trials else {}
    best_name = best_config.get("name")
    best_trial = next(
        (_compact_trial(t) for t in reversed(trials) if t.get("name") == best_name),
        incumbent,
    )
    return {
        "schema_version": 1,
        "review_role": "reasoning second opinion, not implementation verification",
        "decision_authority": "advisory_only",
        "dev_season": dev_season,
        "sealed_season_included": False,
        "acceptance_policy": {
            "value_team_min_gain": 0.003,
            "value_leg_max_mae_regression": 0.02,
            "mae_min_gain": 0.02,
            "mae_leg_max_value_team_regression": 0.003,
            "round_robustness": "gain normally required in at least 3 of 5 rounds",
            "stability_required": True,
        },
        "incumbent": incumbent,
        "selected_best": best_trial,
        "best_config": best_config,
        "trials": [_compact_trial(trial) for trial in trials],
        "stability_reports": stability_reports or [],
        "prior_research_summary": prior_research or {},
        "research_context": research_context or {},
        "sample_warning": (
            "Only five dev rounds are available. Treat one-round changes and "
            "small metric deltas as weak evidence."
        ),
    }


def evidence_fingerprint(packet: dict[str, Any]) -> str:
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _claude_binary(explicit: str | None = None) -> str | None:
    if explicit:
        path = Path(explicit).expanduser()
        return str(path) if path.exists() else None
    found = shutil.which("claude")
    if found:
        return found
    fallback = Path.home() / ".local/bin/claude"
    return str(fallback) if fallback.exists() else None


def _prompt(packet: dict[str, Any]) -> str:
    return (
        "Act as an independent ML research reviewer. This is a reasoning-only "
        "second opinion, not code verification. Use only the evidence packet "
        "below; do not inspect files, run tools, or assume unreported results. "
        "Challenge leakage, overfitting, unstable slices, post-hoc thresholds, "
        "and conclusions unsupported by five rounds. The numerical harness "
        "retains decision authority: your recommendation is advisory. Return "
        "only JSON matching the supplied schema. Rank at most three concrete "
        "next experiments by expected information gain, preferring direct new "
        "signal over another parameter sweep. Consult prior_research_summary and "
        "do not recommend a previously tested family unless you identify a "
        "materially different mechanism or validation design. A matching name "
        "in tested_candidate_names is proof that mechanism was attempted; never "
        "describe it as untested.\n\nEVIDENCE_PACKET:\n"
        + json.dumps(packet, indent=2, sort_keys=True)
    )


def _prior_trial_summary(
    current_names: set[str],
    *,
    max_names: int = 500,
    max_recent_results: int = 20,
) -> dict[str, Any]:
    """Summarise earlier trials so the reviewer does not recycle failed ideas."""
    history_path = RESEARCH / "history.jsonl"
    if not history_path.exists():
        return {}
    by_name: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(history_path.read_text().splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for trial in record.get("trials", []):
            name = trial.get("name")
            if not name or name in current_names:
                continue
            previous = by_name.get(name, {})
            by_name[name] = {
                "name": name,
                "times_tested": int(previous.get("times_tested", 0)) + 1,
                "ever_accepted": bool(previous.get("ever_accepted")) or bool(
                    trial.get("accepted")
                ),
                "last_accepted": trial.get("accepted"),
                "last_reason": trial.get("reason"),
                "last_value_team": trial.get("value_team"),
                "last_mae": trial.get("mae"),
                "_last_seen": line_number,
            }
    recent = sorted(
        by_name.values(), key=lambda item: item["_last_seen"], reverse=True
    )[:max_names]
    accepted_names = [item["name"] for item in recent if item["ever_accepted"]]
    recent_results = []
    for item in recent[:max_recent_results]:
        reason = item.get("last_reason")
        recent_results.append(
            {
                "name": item["name"],
                "last_accepted": item["last_accepted"],
                "last_reason": str(reason)[:300] if reason is not None else None,
            }
        )
    for item in recent:
        item.pop("_last_seen", None)
    return {
        "tested_candidate_names": [item["name"] for item in recent],
        "previously_accepted_names": accepted_names,
        "recent_results": recent_results,
    }


def _load_review_context() -> dict[str, Any]:
    if not REVIEW_CONTEXT.exists():
        return {}
    try:
        context = json.loads(REVIEW_CONTEXT.read_text())
    except json.JSONDecodeError:
        return {"context_error": "claude_review_context.json is invalid JSON"}
    return context if isinstance(context, dict) else {
        "context_error": "claude_review_context.json must contain an object"
    }


def _extract_opinion(stdout: str) -> dict[str, Any]:
    """Handle Claude's JSON envelope and plain/schema JSON output forms."""
    raw: Any
    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError:
        raw = stdout

    if isinstance(raw, dict):
        if isinstance(raw.get("structured_output"), dict):
            return raw["structured_output"]
        result = raw.get("result")
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            raw = result
        elif all(key in raw for key in OPINION_SCHEMA["required"]):
            return raw

    text = str(raw).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    opinion = json.loads(text)
    if not isinstance(opinion, dict):
        raise ValueError("Claude review was not a JSON object")
    return opinion


def _validate_opinion(opinion: dict[str, Any]) -> None:
    missing = set(OPINION_SCHEMA["required"]) - set(opinion)
    if missing:
        raise ValueError(f"Claude review missing fields: {sorted(missing)}")
    if opinion["verdict"] not in {"agree", "concern", "insufficient_evidence"}:
        raise ValueError(f"invalid verdict: {opinion['verdict']}")
    if opinion["recommendation"] not in {
        "keep_incumbent",
        "advance_candidate",
        "do_not_advance",
        "manual_review",
    }:
        raise ValueError(f"invalid recommendation: {opinion['recommendation']}")
    if opinion["confidence"] not in {"low", "medium", "high"}:
        raise ValueError(f"invalid confidence: {opinion['confidence']}")


def _render_markdown(payload: dict[str, Any]) -> str:
    opinion = payload.get("opinion") or {}
    lines = [
        "# Claude Code second opinion - latest",
        "",
        f"- Status: `{payload['status']}`",
        f"- Created: `{payload['created_at']}`",
        f"- Evidence fingerprint: `{payload['evidence_fingerprint']}`",
        "- Authority: advisory only; numerical acceptance remains authoritative.",
    ]
    if payload["status"] != "completed":
        lines.extend(["", "## Failure", "", str(payload.get("error", "unknown"))])
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Verdict: `{opinion['verdict']}`",
            f"- Recommendation: `{opinion['recommendation']}`",
            f"- Confidence: `{opinion['confidence']}`",
            "",
            opinion["summary"],
        ]
    )
    for heading, key in (
        ("Supporting evidence", "supporting_evidence"),
        ("Concerns", "concerns"),
        ("What would reverse it", "reversal_conditions"),
    ):
        lines.extend(["", f"## {heading}", ""])
        values = opinion.get(key) or ["None reported."]
        lines.extend(f"- {value}" for value in values)
    lines.extend(["", "## Next experiments", ""])
    for item in opinion.get("next_experiments", []):
        lines.append(
            f"{item['priority']}. **{item['name']}** - {item['hypothesis']} "
            f"Minimum test: {item['minimum_test']}"
        )
    return "\n".join(lines) + "\n"


def _persist(payload: dict[str, Any]) -> None:
    RESEARCH.mkdir(exist_ok=True)
    LATEST_REVIEW.write_text(json.dumps(payload, indent=2))
    REVIEW_MARKDOWN.write_text(_render_markdown(payload))
    with REVIEW_HISTORY.open("a") as history:
        history.write(json.dumps(payload) + "\n")


def _annotate_ledger(payload: dict[str, Any]) -> None:
    ledger = RESEARCH / "LEDGER.md"
    if not ledger.exists():
        return
    marker = "\n## Claude second opinion\n"
    body = ledger.read_text()
    body = body.split(marker, 1)[0].rstrip()
    if payload["status"] == "completed":
        opinion = payload["opinion"]
        annotation = (
            f"{marker}\n"
            f"`{opinion['verdict']}` / `{opinion['recommendation']}` "
            f"({opinion['confidence']} confidence): {opinion['summary']}\n\n"
            "Advisory only. See `research/CLAUDE_SECOND_OPINION_LATEST.md`."
        )
    else:
        annotation = (
            f"{marker}\nReview unavailable: {payload.get('error', 'unknown error')}. "
            "The numerical batch completed normally."
        )
    ledger.write_text(body + annotation + "\n")


def review_research_batch(
    trials: list[dict[str, Any]],
    best_config: dict[str, Any],
    stability_reports: list[dict[str, Any]] | None = None,
    *,
    dev_season: int = 2025,
    timeout_seconds: int = 300,
    claude_bin: str | None = None,
    model: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run or reuse the Claude review for a completed numerical batch."""
    packet = build_evidence_packet(
        trials,
        best_config,
        stability_reports,
        _prior_trial_summary({str(t.get("name")) for t in trials}),
        _load_review_context(),
        dev_season=dev_season,
    )
    fingerprint = evidence_fingerprint(packet)
    if not force and LATEST_REVIEW.exists():
        try:
            prior = json.loads(LATEST_REVIEW.read_text())
            if (
                prior.get("status") == "completed"
                and prior.get("evidence_fingerprint") == fingerprint
            ):
                prior["reused"] = True
                _annotate_ledger(prior)
                return prior
        except (OSError, json.JSONDecodeError):
            pass

    binary = _claude_binary(claude_bin)
    created_at = datetime.now(timezone.utc).isoformat()
    base_payload = {
        "schema_version": 1,
        "created_at": created_at,
        "status": "failed",
        "advisory_only": True,
        "evidence_fingerprint": fingerprint,
        "evidence": packet,
        "reviewer": {
            "tool": "Claude Code",
            "binary": binary,
            "model": model or os.environ.get("CLAUDE_REVIEW_MODEL"),
        },
    }
    if not binary:
        payload = {**base_payload, "status": "unavailable", "error": "claude binary not found"}
        _persist(payload)
        _annotate_ledger(payload)
        return payload

    command = [
        binary,
        "-p",
        _prompt(packet),
        "--effort",
        "high",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(OPINION_SCHEMA, separators=(",", ":")),
        "--disallowedTools",
        "Bash,Read,Edit,Write,WebFetch,WebSearch,Grep,Glob",
    ]
    selected_model = model or os.environ.get("CLAUDE_REVIEW_MODEL")
    if selected_model:
        command.extend(["--model", selected_model])
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Claude exited {completed.returncode}: {detail[-1000:]}")
        opinion = _extract_opinion(completed.stdout)
        _validate_opinion(opinion)
        payload = {**base_payload, "status": "completed", "opinion": opinion}
    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as exc:
        payload = {**base_payload, "status": "failed", "error": str(exc)}

    _persist(payload)
    _annotate_ledger(payload)
    return payload


def review_saved_batch(
    *,
    config_path: Path | None = None,
    timeout_seconds: int = 300,
    claude_bin: str | None = None,
    model: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    ledger_path = RESEARCH / "ledger.json"
    config_path = config_path or RESEARCH / "best_config.json"
    if not ledger_path.exists() or not config_path.exists():
        raise FileNotFoundError("research ledger and requested config are required")
    trials = json.loads(ledger_path.read_text())
    best_config = json.loads(config_path.read_text())
    stability_path = RESEARCH / "stability_report.json"
    stability = json.loads(stability_path.read_text()) if stability_path.exists() else {}
    return review_research_batch(
        trials,
        best_config,
        stability.get("reports", []),
        timeout_seconds=timeout_seconds,
        claude_bin=claude_bin,
        model=model,
        force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="ignore a matching cached review")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--claude-bin")
    parser.add_argument("--model")
    parser.add_argument(
        "--print-evidence",
        action="store_true",
        help="print the bounded packet without calling Claude",
    )
    args = parser.parse_args()
    if args.print_evidence:
        trials = json.loads((RESEARCH / "ledger.json").read_text())
        config = json.loads((RESEARCH / "best_config.json").read_text())
        stability_path = RESEARCH / "stability_report.json"
        stability = json.loads(stability_path.read_text()) if stability_path.exists() else {}
        print(
            json.dumps(
                build_evidence_packet(
                    trials,
                    config,
                    stability.get("reports", []),
                    _prior_trial_summary({str(t.get("name")) for t in trials}),
                    _load_review_context(),
                ),
                indent=2,
            )
        )
        return
    result = review_saved_batch(
        timeout_seconds=args.timeout,
        claude_bin=args.claude_bin,
        model=args.model,
        force=args.force,
    )
    opinion = result.get("opinion") or {}
    print(
        f"Claude review: {result['status']} "
        f"{opinion.get('verdict', '')} {opinion.get('recommendation', '')}".rstrip()
    )


if __name__ == "__main__":
    main()
