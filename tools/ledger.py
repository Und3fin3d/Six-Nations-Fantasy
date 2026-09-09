"""Build the hill-climb ledger from the per-trial metric bundles.

Applies the precommitted acceptance rule to every trial in run order and emits
both a JSON ledger and LEDGER.md. A candidate replaces the incumbent iff ALL of:

  (a) competition-balanced stable_score improves on the incumbent;
  (b) the paired-by-fold bootstrap difference vs the incumbent excludes 0;
  (c) neither the north nor the south cohort regresses by >2%;
  (d) no tournament-family stable regression >2%;
  (e) no extended-event loss regression >5%.

All five are evaluated against the INCUMBENT, with the thresholds mirroring the
frozen promotion gate's magnitudes. They have to be: the starting incumbent (the
frozen empirical_event) already fails (c)-(e) against v1, so scoring a hill-climb
step against v1 would make the rule vacuous -- nothing could ever be accepted.
Read as incumbent-relative, (c)-(e) do the job they exist to do, which is to stop
a step that buys a better competition-balanced mean by wrecking a hemisphere, a
tournament family, or a low-coverage event.

Whether the final config would pass the frozen promotion gate *against v1* is a
separate question, reported per trial as `gate_vs_v1`.

(e) is restricted to the extended targets the frozen gate actually gates -- those
with >=100 supporting fixtures, read from the immutable v1 decision -- because
availability is model-independent and sparse extensions are not hard gates.

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
    ("t4_no_signal_max",
     "Inspecting T2's fitted constants showed red_cards, drop_goals_converted, "
     "drop_goal_missed and potm all falling back to the global K=220 because "
     "their estimated between-player variance was <= 0 -- precisely the worst "
     "remaining targets. A non-positive between-player variance means the data "
     "detect no player-level signal, so the empirical-Bayes answer is full "
     "shrinkage to the position prior, not moderate shrinkage."),
    ("t5_prior_recency",
     "The two gated extended events (tackle_turnover, tackle_try_saver) are "
     "only recorded from 2021 but are primed from a position prior pooled over "
     "all history, while player profiles are already recency-weighted. Weight "
     "the position prior by the same half-life."),
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


def gated_extended_targets() -> list[str]:
    """Extended targets the frozen gate actually gates (>=100 supporting fixtures)."""
    frozen = json.loads(
        (ROOT / "data" / "unified" / "raw_benchmark" / "v1" / "decision.json").read_text()
    )
    return list(frozen["extended_gated_targets"])


def evaluate(bundle: dict, incumbent: dict) -> tuple[bool, list[str], dict]:
    """Apply the precommitted rule against the incumbent."""
    reasons: list[str] = []
    candidate = bundle["stable_score"]["empirical_event"]
    previous = incumbent["stable_score"]["empirical_event"]
    evidence: dict = {"stable_score": candidate, "incumbent_stable_score": previous}

    if not candidate < previous:
        reasons.append(
            f"(a) stable_score {candidate:.6f} did not improve on incumbent {previous:.6f}"
        )

    shared = sorted(
        set(_folds(bundle, "empirical_event")) & set(_folds(incumbent, "empirical_event"))
    )
    paired = np.array([
        _folds(bundle, "empirical_event")[fold] - _folds(incumbent, "empirical_event")[fold]
        for fold in shared
    ])
    boot = _bootstrap(paired)
    evidence["bootstrap_vs_incumbent"] = boot
    if not (boot["p05"] < 0 and boot["p95"] < 0):
        reasons.append(
            f"(b) paired bootstrap vs incumbent does not exclude 0 "
            f"[{boot['p05']:+.5f}, {boot['p95']:+.5f}]"
        )

    for cohort in ("north", "south"):
        new, old = bundle[cohort]["empirical_event"], incumbent[cohort]["empirical_event"]
        if new > old * 1.02:
            reasons.append(f"(c) {cohort} regressed >2% vs incumbent ({new:.4f} vs {old:.4f})")
    evidence["north"] = bundle["north"]
    evidence["south"] = bundle["south"]

    new_t, old_t = bundle["by_tournament"], incumbent["by_tournament"]
    regressed = [
        name for name, value in new_t["empirical_event"].items()
        if value > old_t["empirical_event"].get(name, np.inf) * 1.02
    ]
    if regressed:
        reasons.append(
            "(d) tournament-family regression >2% vs incumbent: " + ", ".join(sorted(regressed))
        )

    gated = gated_extended_targets()
    new_x, old_x = bundle["extended_loss"], incumbent["extended_loss"]
    bad = [
        name for name in gated
        if np.isfinite(new_x["empirical_event"].get(name, np.nan))
        and new_x["empirical_event"][name] > old_x["empirical_event"].get(name, np.inf) * 1.05
    ]
    if bad:
        reasons.append(
            "(e) extended-event loss regression >5% vs incumbent: " + ", ".join(sorted(bad))
        )
    evidence["gated_extended"] = {
        name: {
            "candidate": new_x["empirical_event"].get(name),
            "incumbent": old_x["empirical_event"].get(name),
            "v1": new_x["v1"].get(name),
        }
        for name in gated
    }
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
        gated = entry["evidence"].get("gated_extended", {})
        if gated:
            lines += ["Gated extended-event loss (candidate / incumbent / v1):", ""]
            for name, values in sorted(gated.items()):
                lines.append(
                    f"- `{name}`: {values['candidate']:.4f} / "
                    f"{values['incumbent']:.4f} / {values['v1']:.4f}"
                )
            lines.append("")
        lines += [
            "Frozen promotion gate vs v1 (reported, not the acceptance rule): "
            + ("PASS" if entry["gate_vs_v1_passed"] else "FAIL — "
               + "; ".join(entry["gate_vs_v1_reasons"])),
            "",
        ]
    (WORK / "LEDGER.md").write_text("\n".join(lines) + "\n")
    print(f"final incumbent: {incumbent_name} = {ledger['final_stable_score']}")


if __name__ == "__main__":
    main()
