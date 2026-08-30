"""Build the hill-climb ledger from the per-trial metric bundles.

Applies the precommitted acceptance rule to every trial in run order and emits
both a JSON ledger and LEDGER.md. A candidate replaces the incumbent iff ALL of:

  (a) competition-balanced stable_score improves on the incumbent;
  (b) the paired-by-fold bootstrap difference vs the incumbent excludes 0;
  (c) neither the north nor the south cohort regresses by >2% vs v1;
  (d) no tournament-family stable regression >2% vs v1;
  (e) no extended-event loss regression >5% vs v1.

Ties and sub-threshold trials are rejected, biasing to the simpler incumbent.

Usage: python -m tools.ledger
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "unified" / "raw_benchmark" / "allrugby_empirical"
TRIALS = WORK / "trials"
FROZEN_BASELINE = 0.939031
FROZEN_V1 = 0.907918

# Run order of the hill-climb. Each entry is a small delta off the incumbent
# standing at the time it was proposed.
ORDER = [
    ("base", "control: reproduce the frozen empirical_event contract"),
    ("t1_eb_off",
     "Counts are rate*minutes/80, so the small-sample per-player minutes table "
     "multiplies its noise into every event; shrink it toward the position/started "
     "mean by empirical Bayes. Also drop the opponent-strength multiplier, which "
     "saturates at its clips for any lopsided fixture."),
    ("t2_eb_off_ebk",
     "Replace the global K=220 shrinkage of player history toward the position "
     "prior with a per-event empirical-Bayes K fitted from the training fold's "
     "within/between player rate variance."),
    ("t3_eb_off_ebk_head",
     "The minutes target is MAE-scored (median-optimal) but also scales counts "
     "(mean-optimal). Fit a separate median-valued minutes head, leaving the "
     "count scaler on E[minutes]."),
]


def _bootstrap(differences: np.ndarray, seed: int = 17) -> dict:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return {"n": 0, "mean": float("nan"), "p05": float("nan"), "p95": float("nan")}
    rng = np.random.default_rng(seed)
    samples = np.array([
        float(np.mean(differences[rng.integers(0, len(differences), len(differences))]))
        for _ in range(2000)
    ])
    return {
        "n": int(len(differences)), "mean": float(np.mean(differences)),
        "p05": float(np.quantile(samples, 0.05)), "p95": float(np.quantile(samples, 0.95)),
    }


def _folds(bundle: dict, engine: str) -> dict:
    return {fold: value for fold, value in bundle["by_fold"][engine].items()}


def evaluate(bundle: dict, incumbent: dict | None) -> tuple[bool, list[str], dict]:
    """Apply the precommitted rule; returns (accepted, reasons, evidence)."""
    reasons: list[str] = []
    candidate = bundle["stable_score"]["empirical_event"]
    evidence: dict = {"stable_score": candidate}

    if incumbent is not None:
        previous = incumbent["stable_score"]["empirical_event"]
        evidence["incumbent_stable_score"] = previous
        if not candidate < previous:
            reasons.append(
                f"(a) stable_score {candidate:.6f} did not improve on incumbent {previous:.6f}"
            )
        shared = sorted(
            set(_folds(bundle, "empirical_event")) & set(_folds(incumbent, "empirical_event"))
        )
        paired = np.array([
            _folds(bundle, "empirical_event")[fold]
            - _folds(incumbent, "empirical_event")[fold]
            for fold in shared
        ])
        boot = _bootstrap(paired)
        evidence["bootstrap_vs_incumbent"] = boot
        if not (boot["p05"] < 0 and boot["p95"] < 0):
            reasons.append(
                f"(b) paired bootstrap vs incumbent does not exclude 0 "
                f"[{boot['p05']:.5f}, {boot['p95']:.5f}]"
            )

    for cohort in ("north", "south"):
        values = bundle[cohort]
        if values["empirical_event"] > values["v1"] * 1.02:
            reasons.append(
                f"(c) {cohort} regressed >2% vs v1 "
                f"({values['empirical_event']:.4f} vs {values['v1']:.4f})"
            )
    evidence["north"] = bundle["north"]
    evidence["south"] = bundle["south"]

    tournaments = bundle["by_tournament"]
    regressed = [
        name for name, value in tournaments["empirical_event"].items()
        if value > tournaments["v1"].get(name, np.inf) * 1.02
    ]
    if regressed:
        reasons.append("(d) tournament-family regression >2%: " + ", ".join(sorted(regressed)))

    extended = bundle["extended_loss"]
    bad = [
        name for name, value in extended.get("empirical_event", {}).items()
        if np.isfinite(value) and value > extended["v1"].get(name, np.inf) * 1.05
    ]
    if bad:
        reasons.append("(e) extended-event loss regression >5%: " + ", ".join(sorted(bad)))

    return (not reasons), reasons, evidence


def main() -> None:
    entries, incumbent, incumbent_name = [], None, None
    for name, hypothesis in ORDER:
        path = TRIALS / name / "bundle.json"
        if not path.exists():
            continue
        bundle = json.loads(path.read_text())
        if name == "base":
            accepted, reasons, evidence = True, [], {
                "stable_score": bundle["stable_score"]["empirical_event"],
                "north": bundle["north"], "south": bundle["south"],
            }
            note = "control, not a candidate"
        else:
            accepted, reasons, evidence = evaluate(bundle, incumbent)
            note = "ACCEPTED" if accepted else "REJECTED"
        entries.append({
            "trial": name, "hypothesis": hypothesis, "config": bundle["config"],
            "verdict": note, "accepted": bool(accepted), "reasons": reasons,
            "evidence": evidence, "rubrics": bundle["rubrics"],
            "gate_vs_v1_passed": bundle["gate_passed"],
            "gate_vs_v1_reasons": bundle["gate_reasons"],
            "by_tournament": bundle["by_tournament"],
            "by_target_relative_loss": bundle["by_target_relative_loss"],
        })
        if accepted:
            incumbent, incumbent_name = bundle, name

    ledger = {
        "frozen_empirical_event": FROZEN_BASELINE,
        "frozen_v1": FROZEN_V1,
        "final_incumbent": incumbent_name,
        "final_stable_score": (
            incumbent["stable_score"]["empirical_event"] if incumbent else None
        ),
        "trials": entries,
    }
    TRIALS.mkdir(parents=True, exist_ok=True)
    (WORK / "ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    lines = [
        "# All-rugby hill-climb ledger: NCR empirical engine", "",
        f"Frozen baseline `empirical_event` = **{FROZEN_BASELINE:.6f}**, "
        f"frozen `v1` gate = **{FROZEN_V1:.6f}** (lower is better).", "",
        "Acceptance rule (precommitted, all must hold): (a) stable_score improves on "
        "the incumbent; (b) paired-by-fold bootstrap vs incumbent excludes 0; "
        "(c) no north/south regression >2% vs v1; (d) no tournament-family "
        "regression >2% vs v1; (e) no extended-event loss regression >5% vs v1.", "",
        "| trial | stable_score | verdict |", "| --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['trial']}` | {entry['evidence']['stable_score']:.6f} | {entry['verdict']} |"
        )
    lines += ["", "## Trials", ""]
    for entry in entries:
        lines += [
            f"### `{entry['trial']}` — {entry['verdict']}", "",
            f"**Hypothesis.** {entry['hypothesis']}", "",
            f"**stable_score** {entry['evidence']['stable_score']:.6f}",
            "",
        ]
        boot = entry["evidence"].get("bootstrap_vs_incumbent")
        if boot:
            lines += [
                f"Paired bootstrap vs incumbent: mean {boot['mean']:+.5f}, "
                f"90% CI [{boot['p05']:+.5f}, {boot['p95']:+.5f}] over {boot['n']} folds.", "",
            ]
        if entry["reasons"]:
            lines += ["Rejected because:", ""]
            lines += [f"- {reason}" for reason in entry["reasons"]]
            lines.append("")
        rubrics = ", ".join(
            f"{row['rubric']} MAE {row['mae']:.4f} / spearman {row['spearman']:.4f} "
            f"/ capture {row['mean_capture']:.4f}"
            for row in entry["rubrics"] if row["engine"] == "empirical_event"
        )
        lines += [f"Reconstructed rubrics: {rubrics}", ""]
        lines += [
            "Historical gate vs v1: "
            + ("PASS" if entry["gate_vs_v1_passed"] else "FAIL — "
               + "; ".join(entry["gate_vs_v1_reasons"])),
            "",
        ]
    (WORK / "LEDGER.md").write_text("\n".join(lines) + "\n")
    print(f"final incumbent: {incumbent_name} = {ledger['final_stable_score']}")


if __name__ == "__main__":
    main()
