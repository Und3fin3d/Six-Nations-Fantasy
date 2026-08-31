"""Render the per-competition scoreboard as BOARD.md.

The raw score says which model fits raw events best. This says which model
would actually be the one to use in each competition, under that competition's
own scoring rubric, on identical folds and slates.
"""

from __future__ import annotations

import pandas as pd

from . import allrugby_phase2 as P
from . import allrugby_phase2_rubric as R
from .allrugby_phase2_ledger import _markdown

WORK = P.WORK
SHORT = {
    "p3_event_50_frozen": "P3 frozen",
    "C5_phase1": "C5 (phase 1)",
    "E4_three_empirical_plus_naive_phase2": "E4 (phase 2)",
    "C9_calibrated_E4_phase2": "C9 (phase 2)",
    "C10_calibrated_E4_wide_phase2": "C10 (phase 2)",
    "empirical_only": "empirical (NCR incumbent engine)",
    "empirical_empirical_t3_eb_off_ebk_head_only": "empirical t3",
    "v4_only": "v4 GBDT",
    "naive_comparator": "stratum mean",
}
ORDER = list(SHORT)


def _pivot(frame: pd.DataFrame, rubric: str, metric: str, index: str) -> pd.DataFrame:
    block = frame[frame["rubric"].eq(rubric)].copy()
    block["engine"] = block["engine"].map(SHORT).fillna(block["engine"])
    table = block.pivot_table(index=index, columns="engine", values=metric)
    columns = [SHORT[name] for name in ORDER if SHORT[name] in table.columns]
    table = table[columns]
    best = table.idxmin(axis=1) if metric.startswith("mae") else table.idxmax(axis=1)
    table = table.round(4).astype(str)
    table["best"] = best
    return table.reset_index()


def render() -> str:
    overall = pd.read_csv(WORK / "board_overall.csv")
    by_tournament = pd.read_csv(WORK / "board_by_tournament.csv")
    rankings = pd.read_csv(WORK / "board_rankings.csv")
    lines = [
        "# Per-competition scoreboard — is one agnostic model the best everywhere?",
        "",
        "Every engine below is scored on the same 21 exact-kickoff tournament holdouts and",
        "the same 98 slates, under both competition rubrics. `empirical` is the engine the",
        "NCR incumbent is built on, so the agnostic-vs-competition-specific comparison is",
        "`E4` against it. The Six Nations champion is absent because it only has two folds",
        "of 13-target coverage; that comparison lives in `allrugby_sixnations/LEDGER.md`.",
        "",
        "`mae` is raw predicted-vs-actual fantasy points; `mae_calibrated` applies a",
        "leave-one-fold-out affine recalibration, so neither is in-sample.",
        "",
        "## Pooled over all 98 slates",
        "",
    ]
    block = overall.copy()
    block["engine"] = block["engine"].map(SHORT).fillna(block["engine"])
    block = block.set_index("engine").loc[
        [SHORT[name] for name in ORDER if SHORT[name] in set(block["engine"])]
    ].reset_index() if False else block
    for rubric in ("six_nations", "ncr"):
        subset = block[block["rubric"].eq(rubric)][
            ["engine", "mae", "mae_calibrated", "spearman", "mean_capture", "slates"]
        ].copy()
        subset = subset.sort_values("mae").round(4)
        lines += [f"### `{rubric}` rubric", "", _markdown(subset), ""]
    lines += ["## Per tournament family", ""]
    for rubric in ("six_nations", "ncr"):
        for metric in ("mae", "spearman", "mean_capture"):
            lines += [
                f"### `{rubric}` rubric — {metric}", "",
                _markdown(_pivot(by_tournament, rubric, metric, "tournament")), "",
            ]
    lines += ["## Paired-by-slate bootstraps (candidate minus baseline, 90% CI)", ""]
    rows = []
    candidate = "E4_three_empirical_plus_naive_phase2"
    for baseline in (
        "empirical_only", "C5_phase1", "p3_event_50_frozen",
        "empirical_empirical_t3_eb_off_ebk_head_only", "C9_calibrated_E4_phase2",
    ):
        for metric in ("mae", "spearman", "mean_capture"):
            for rubric, stats in sorted(
                R.paired_by_slate(rankings, candidate, baseline, metric).items()
            ):
                excludes = (stats["p05"] < 0 and stats["p95"] < 0) or (
                    stats["p05"] > 0 and stats["p95"] > 0
                )
                rows.append({
                    "baseline": SHORT.get(baseline, baseline), "metric": metric,
                    "rubric": rubric, "mean": round(stats["mean"], 5),
                    "p05": round(stats["p05"], 5), "p95": round(stats["p95"], 5),
                    "excludes_0": excludes,
                })
    lines += [_markdown(pd.DataFrame(rows)), ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    (WORK / "BOARD.md").write_text(render())
    print(f"wrote {WORK / 'BOARD.md'}")
