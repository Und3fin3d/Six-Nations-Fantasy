"""Render the phase-2 all-rugby ledger from results.json."""

from __future__ import annotations

import json

import pandas as pd

from . import allrugby_phase2 as P
from .allrugby_phase2_run import C5_INCUMBENT, FROZEN

WORK = P.WORK


def _row(result: dict) -> dict:
    temporal = result.get("temporal") or {}
    return {
        "candidate": result["candidate"],
        "components": " + ".join(result["components"]),
        "stable_score": round(result["stable_score"], 6),
        "vs_C5": round(result["delta_vs_incumbent"], 6),
        "vs_frozen": round(result["delta_vs_frozen"], 6),
        "boot_p05": round(result["bootstrap"]["p05"], 6),
        "boot_p95": round(result["bootstrap"]["p95"], 6),
        "temporal_delta": round(temporal.get("delta", float("nan")), 6),
        "accepted": result["accepted"],
    }


def render() -> str:
    results = json.loads((WORK / "results.json").read_text())
    control = json.loads((WORK / "control.json").read_text())
    table = pd.DataFrame([_row(result) for result in results])
    lines = [
        "# All-rugby hill-climb ledger — P3 phase 2",
        "",
        "Phase 1 held the two components fixed and searched the blend rule; that family",
        "terminated at C5 (shrunk per-target weight, cross-fitted **0.879366**). Phase 2",
        "moves the search onto the components and onto axes orthogonal to the blend",
        "weight. Same metric, same folds, same precommitted acceptance rule, with the",
        "incumbent raised to C5. Seed 17. NCR GW4–7 was never read.",
        "",
        "## Control",
        "",
        f"- Frozen `p3_event_50` reconstructed through the generalised code: "
        f"**{control['frozen_reconstructed']:.10f}** vs {FROZEN:.10f} "
        f"(|Δ| {control['frozen_abs_difference']:.2e})",
        f"- Phase-1 C5 reconstructed: **{control['c5_reconstructed']:.10f}** vs "
        f"{C5_INCUMBENT:.10f} (|Δ| {control['c5_abs_difference']:.2e})",
        f"- Folds: {control['n_folds']}",
        "",
        "The two-component pair grid is a special case of the simplex, so reproducing",
        "both phase-1 numbers through the generalised code is the control for it.",
        "",
        "## Acceptance rule (unchanged from phase 1, incumbent raised to C5)",
        "",
        "- **(a)** stable_score improves on the incumbent",
        "- **(b)** paired-by-fold bootstrap difference excludes 0",
        "- **(c)** neither hemisphere regresses by more than 2%",
        "- **(d)** no tournament-family stable regression greater than 2%",
        "- **(e)** no extended-event loss regression greater than 5% vs frozen P3",
        "- **(f)** every fitted parameter is cross-fitted leave-one-fold-out and checked",
        "  again on a strict temporal split (fit 10 earliest folds, evaluate 11 latest)",
        "",
        "## Trials",
        "",
        table.to_markdown(index=False),
        "",
    ]
    for result in results:
        lines += [
            f"### `{result['candidate']}` — {'ACCEPTED' if result['accepted'] else 'REJECTED'}",
            "",
            f"**Hypothesis.** {result['hypothesis']}",
            "",
            f"Components: `{'`, `'.join(result['components'])}`"
            + (f"; row groups: `{'`, `'.join(result['groups'])}`" if result.get("groups") else "")
            + f"; {result['n_points']} weight points"
            + (f"; shrinkage λ {result['hyperparameter']}" if result.get("hyperparameter") is not None else ""),
            "",
            f"Cross-fitted stable score **{result['stable_score']:.6f}** "
            f"({result['delta_vs_incumbent']:+.6f} vs C5, "
            f"{result['delta_vs_frozen']:+.6f} vs frozen P3). "
            f"Paired bootstrap vs C5: mean {result['bootstrap']['mean']:+.6f}, "
            f"90% CI [{result['bootstrap']['p05']:+.6f}, {result['bootstrap']['p95']:+.6f}].",
            "",
        ]
        if result.get("temporal"):
            temporal = result["temporal"]
            lines += [
                f"Temporal split: {temporal['candidate']:.6f} vs {temporal['baseline']:.6f} "
                f"(Δ {temporal['delta']:+.6f}, CI [{temporal['bootstrap']['p05']:+.6f}, "
                f"{temporal['bootstrap']['p95']:+.6f}]).",
                "",
            ]
        if result["reasons"]:
            lines += ["Rejected because:", ""]
            lines += [f"- {reason}" for reason in result["reasons"]]
            lines += [""]
        families = pd.DataFrame(result["families"])
        if not families.empty:
            families = families.sort_values("pct")
            families["pct"] = families["pct"].round(4)
            lines += ["Per tournament family (candidate vs incumbent, % change):", "",
                      families.to_markdown(index=False), ""]

    accepted = [result for result in results if result["accepted"]]
    if accepted:
        best = min(accepted, key=lambda result: result["stable_score"])
        lines += _incumbent_section(best)
    return "\n".join(lines) + "\n"


def _incumbent_section(best: dict) -> list[str]:
    lines = [
        f"## Phase-2 incumbent: `{best['candidate']}`",
        "",
        f"- Cross-fitted stable score **{best['stable_score']:.6f}** vs C5 "
        f"{C5_INCUMBENT:.6f} and frozen P3 {FROZEN:.6f}",
        f"- Improvement **{100.0 * -best['delta_vs_frozen'] / FROZEN:.3f}%** on the frozen "
        f"blend, **{100.0 * -best['delta_vs_incumbent'] / C5_INCUMBENT:.3f}%** on C5",
        f"- Paired-by-fold bootstrap vs C5: mean {best['bootstrap']['mean']:+.6f}, "
        f"90% CI [{best['bootstrap']['p05']:+.6f}, {best['bootstrap']['p95']:+.6f}]",
        f"- Temporal split: Δ {best['temporal']['delta']:+.6f}, CI "
        f"[{best['temporal']['bootstrap']['p05']:+.6f}, "
        f"{best['temporal']['bootstrap']['p95']:+.6f}]",
        "",
    ]
    weights = pd.DataFrame([
        {"target": target, **dict(zip(best["components"], [round(v, 4) for v in vector]))}
        for target, vector in best["deployed_weights"].items()
    ])
    if best.get("deployed_scales"):
        weights["scale"] = weights["target"].map(
            {k: round(v, 4) for k, v in best["deployed_scales"].items()}
        )
    sort_key = "naive" if "naive" in weights else best["components"][0]
    lines += [
        "Deployed weights (all-folds fit; the headline number is cross-fitted):", "",
        weights.sort_values(sort_key).to_markdown(index=False), "",
    ]
    return lines


if __name__ == "__main__":
    (WORK / "LEDGER.md").write_text(render())
    print(f"wrote {WORK / 'LEDGER.md'}")
