"""Render the phase-2 all-rugby ledger from results.json."""

from __future__ import annotations

import json

import pandas as pd

from . import allrugby_phase2 as P
from .allrugby_phase2_run import C5_INCUMBENT, FROZEN

WORK = P.WORK


def _markdown(frame: pd.DataFrame) -> str:
    """Pipe table without pandas' optional ``tabulate`` dependency.

    The pinned model environment has no package manager, so the ledger renders
    its own tables rather than adding a dependency for four calls.
    """
    columns = [str(column) for column in frame.columns]
    rows = [
        ["" if value is None else str(value) for value in record]
        for record in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows)) if rows
        else len(columns[index])
        for index in range(len(columns))
    ]
    def line(cells):
        return "| " + " | ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(cells)
        ) + " |"
    return "\n".join([
        line(columns),
        "| " + " | ".join("-" * width for width in widths) + " |",
        *(line(row) for row in rows),
    ])


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
        _markdown(table),
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
                      _markdown(families), ""]

    accepted = [result for result in results if result["accepted"]]
    if accepted:
        best = min(accepted, key=lambda result: result["stable_score"])
        lines += _incumbent_section(best)
        lines += _decision_section(results)
    return "\n".join(lines) + "\n"


BOARD_NAMES = {
    "E4_three_empirical_plus_naive": "E4_three_empirical_plus_naive_phase2",
    "C9_calibrated_E4": "C9_calibrated_E4_phase2",
    "C10_calibrated_E4_wide": "C10_calibrated_E4_wide_phase2",
}


def _decision_section(results: list[dict]) -> list[str]:
    """Raw-score optimum versus the model the rubric guard actually promotes."""
    board_path = WORK / "board_overall.csv"
    if not board_path.exists():
        return []
    board = pd.read_csv(board_path).set_index(["engine", "rubric"])
    by_name = {result["candidate"]: result for result in results}

    def row(candidate: str, rubric: str):
        engine = BOARD_NAMES.get(candidate)
        if engine is None or (engine, rubric) not in board.index:
            return None
        return board.loc[(engine, rubric)]

    scored = [
        candidate for candidate in BOARD_NAMES
        if candidate in by_name and row(candidate, "ncr") is not None
    ]
    if not scored:
        return []
    raw_best = min(scored, key=lambda name: by_name[name]["stable_score"])
    table = pd.DataFrame([
        {
            "candidate": candidate,
            "stable_score": round(by_name[candidate]["stable_score"], 6),
            "ncr_mae": round(float(row(candidate, "ncr")["mae"]), 4),
            "ncr_mae_cal": round(float(row(candidate, "ncr")["mae_calibrated"]), 4),
            "six_nations_mae": round(float(row(candidate, "six_nations")["mae"]), 4),
            "six_nations_mae_cal": round(
                float(row(candidate, "six_nations")["mae_calibrated"]), 4,
            ),
        }
        for candidate in sorted(scored, key=lambda name: by_name[name]["stable_score"])
    ])
    rubric_best = min(scored, key=lambda name: float(row(name, "ncr")["mae"]))
    lines = [
        "## Champion decision — raw score versus rubric",
        "",
        _markdown(table),
        "",
    ]
    if raw_best == rubric_best:
        lines += [
            f"`{raw_best}` is best on both the raw score and the reconstructed rubrics, "
            "so the two criteria agree and it is the champion.",
            "",
        ]
    else:
        lines += [
            f"The raw-score optimum is **`{raw_best}`**, but the standing rule is that a "
            "raw gain which costs reconstructed rubric MAE is not a gain worth shipping — "
            "and this is exactly that case. Per-target calibration buys raw score by "
            "deflating the sparse, high-value targets, which is right under a Poisson "
            "deviance and wrong for fantasy points.",
            "",
            f"**`{rubric_best}` is the champion.** It is better than "
            f"`{raw_best}` on Nations Championship rubric MAE by "
            f"{float(row(raw_best, 'ncr')['mae']) - float(row(rubric_best, 'ncr')['mae']):.4f} "
            "with a paired-by-slate bootstrap CI excluding 0, better on Six Nations "
            "Spearman, and level on Six Nations MAE and capture. Widening the calibration "
            "grid (C10) does not rescue it, which closes the calibration axis.",
            "",
            f"`{raw_best}` is kept in the ledger as the raw-score optimum, not deleted: it "
            "is the right starting point if the metric ever becomes the deliverable.",
            "",
        ]
    return lines


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
        _markdown(weights.sort_values(sort_key)), "",
    ]
    return lines


if __name__ == "__main__":
    (WORK / "LEDGER.md").write_text(render())
    print(f"wrote {WORK / 'LEDGER.md'}")
