"""Emit the all-rugby hill-climb ledger (LEDGER.md + ledger.json).

Every number is recomputed from the committed trial artifacts under
``data/unified/raw_benchmark/allrugby_sixnations/trials/`` so the ledger can
never drift from the runs it describes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .allrugby_champion import SHARED_STABLE_TARGETS
from .allrugby_runner import FROZEN, FROZEN_BASELINE_SCORE, OUT, decide, load_trial, summarise

#: The greedy chain actually walked: (step, incumbent, [candidates], accepted).
CHAIN: list[tuple[int, str, list[str], str | None]] = [
    (1, "t0_baseline",
     ["c1_calib_global", "c2_calib_global_sparse_only", "c3_calib_position",
      "c4_revive_dead_heads"], "c1_calib_global"),
    (2, "c1_calib_global",
     ["c6_revive_red_only", "c7_revive_drop_only", "c8_revive_both",
      "c9_calib_position_plus_revive"], "c6_revive_red_only"),
    (3, "c6_revive_red_only",
     ["c11_coverage", "c12_coverage_shrink120", "c13_coverage_shrink480",
      "c14_minutes_blend025", "c15_minutes_blend050", "c16_minutes_blend075",
      "c17_minutes_blend100"], "c13_coverage_shrink480"),
    (4, "c13_coverage_shrink480",
     ["c18_cov_min050", "c19_cov_min100", "c20_cov_min075",
      "c21_cov_position_calib", "c22_cov_revive_both", "c23_cov_shrink960",
      "c24_cov_calib_shrink5"], "c19_cov_min100"),
    (5, "c19_cov_min100",
     ["c25_min100_shrink960", "c26_min100_shrink240", "c27_min100_nocalib",
      "c28_min100_revive_both"], "c25_min100_shrink960"),
    (6, "c25_min100_shrink960",
     ["c29_shrink1920", "c30_shrink3840"], None),
]

HYPOTHESIS = {
    "t0_baseline": "Control: reproduce the frozen champion partial score exactly.",
    "c1_calib_global": "Experiment 18's volume deflation is load-bearing for XV selection but "
                       "hurts point-accurate raw prediction; rescale each head to the "
                       "strictly-prior observed level.",
    "c2_calib_global_sparse_only": "Restrict the deflation correction to the sparse-count heads.",
    "c3_calib_position": "Deflation is position-specific, so calibrate per position.",
    "c4_revive_dead_heads": "drop_goals_converted and red_cards heads emit identically zero; "
                            "replace them with a shrunk per-minute rate x minutes_hat.",
    "c5_revive_plus_calib": "Exploratory: dead-head revival stacked on global calibration.",
    "c6_revive_red_only": "Revive only red_cards, whose occurrences are present in both folds.",
    "c7_revive_drop_only": "Revive only drop_goals_converted, absent from the 2025 fold.",
    "c8_revive_both": "Revive both dead heads.",
    "c9_calib_position_plus_revive": "Position calibration stacked on full revival.",
    "c10_extra_heads_coverage": "Exploratory: add heads for the 11 unmodelled stable events.",
    "c11_coverage": "Extend coverage 13 -> 24 targets with per-minute rate heads fitted on the "
                    "all-rugby strictly-prior training frame.",
    "c12_coverage_shrink120": "Lighter pooling for the added heads.",
    "c13_coverage_shrink480": "Heavier pooling for the added heads.",
    "c14_minutes_blend025": "The champion's minutes model was fitted on 6N rotation; blend it "
                            "toward a strictly-prior stratum mean.",
    "c15_minutes_blend050": "Half-weight the champion minutes head.",
    "c16_minutes_blend075": "Quarter-weight the champion minutes head.",
    "c17_minutes_blend100": "Discard the champion minutes head entirely.",
    "c18_cov_min050": "Coverage plus half-weighted champion minutes.",
    "c19_cov_min100": "Coverage plus fully replaced minutes.",
    "c20_cov_min075": "Coverage plus quarter-weighted champion minutes.",
    "c21_cov_position_calib": "Coverage plus per-position calibration.",
    "c22_cov_revive_both": "Coverage plus both dead heads revived.",
    "c23_cov_shrink960": "Coverage with heavier pooling still.",
    "c24_cov_calib_shrink5": "Stronger shrinkage on the calibration ratio itself.",
    "c25_min100_shrink960": "Best minutes setting with heavier head pooling.",
    "c26_min100_shrink240": "Best minutes setting with lighter head pooling.",
    "c27_min100_nocalib": "Ablation: is the deflation correction still carrying weight?",
    "c28_min100_revive_both": "Best config plus drop-goal revival.",
    "c29_shrink1920": "Push pooling further.",
    "c30_shrink3840": "Push pooling further still.",
}


def _reference_scores(targets: tuple[str, ...]) -> dict[str, float]:
    base = pd.read_csv(FROZEN / "event_metrics.csv")
    folds = {"six_nations_2025", "six_nations_2026"}
    block = base[base["fold"].isin(folds) & base["cohort"].eq("all") & base["target"].isin(targets)]
    return {
        engine: float(
            block[block["engine"].eq(engine)].groupby("fold")["relative_loss"].mean().mean()
        )
        for engine in ("v1", "p3_event_50")
    }


def build() -> dict:
    trials: dict[str, dict] = {}
    for name in sorted(HYPOTHESIS):
        directory = OUT / "trials" / name
        if not (directory / "event_metrics.csv").exists():
            continue
        events, rankings = load_trial(name)
        summary = summarise(events, rankings)
        summary["config"] = json.loads((directory / "config.json").read_text())
        summary["hypothesis"] = HYPOTHESIS[name]
        trials[name] = summary

    steps = []
    for index, incumbent, candidates, accepted in CHAIN:
        incumbent_events, incumbent_rankings = load_trial(incumbent)
        incumbent_summary = summarise(incumbent_events, incumbent_rankings)
        rows = []
        for candidate in candidates:
            if candidate not in trials:
                continue
            candidate_events, candidate_rankings = load_trial(candidate)
            verdict = decide(
                summarise(candidate_events, candidate_rankings), incumbent_summary,
                candidate_events, incumbent_events,
            )
            rows.append({"candidate": candidate, **verdict})
        steps.append({
            "step": index, "incumbent": incumbent, "accepted": accepted, "trials": rows,
        })

    final = "c25_min100_shrink960"
    final_summary = trials[final]
    like_for_like = _reference_scores(SHARED_STABLE_TARGETS)
    full = _reference_scores(tuple(final_summary["covered_targets"]))
    gap_before = FROZEN_BASELINE_SCORE - like_for_like["v1"]
    gap_after = final_summary["partial_stable_score"] - like_for_like["v1"]
    return {
        "schema_version": 1,
        "seed": 17,
        "frozen_baseline_partial_stable_score": FROZEN_BASELINE_SCORE,
        "control_reproduced": trials["t0_baseline"]["partial_stable_score"],
        "final_incumbent": final,
        "final_summary": final_summary,
        "reference_like_for_like_13": like_for_like,
        "reference_full_24": full,
        "gap_to_v1_before": gap_before,
        "gap_to_v1_after": gap_after,
        "gap_closed_fraction": 1.0 - gap_after / gap_before,
        "chain": steps,
        "trials": trials,
    }


def _table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def render(payload: dict) -> str:
    final = payload["final_summary"]
    lines = [
        "# All-rugby hill-climb — Six Nations champion components", "",
        "Objective: minimise the competition-balanced **stable raw score** of the",
        "Historical Raw Benchmark v1 (lower is better; 1.0 = no better than the",
        "training-only position x started x competition_level naive comparator).",
        "", "Seed 17. The frozen `v1` ledger is untouched; every run here writes to",
        "`data/unified/raw_benchmark/allrugby_sixnations/`.", "",
        "## Control", "",
        f"Frozen champion partial stable score: **{payload['frozen_baseline_partial_stable_score']:.6f}**",
        f"Reproduced by this harness: **{payload['control_reproduced']:.6f}** "
        f"(delta {payload['control_reproduced'] - payload['frozen_baseline_partial_stable_score']:+.1e})",
        "", "## Result", "",
        f"Final incumbent: **`{payload['final_incumbent']}`**", "",
        _table([
            ["like-for-like (13 targets)",
             f"{payload['frozen_baseline_partial_stable_score']:.6f}",
             f"{final['partial_stable_score']:.6f}",
             f"{payload['reference_like_for_like_13']['v1']:.6f}",
             f"{payload['reference_like_for_like_13']['p3_event_50']:.6f}"],
            ["full (24 targets)", "n/a (13-target coverage)",
             f"{final['full_stable_score']:.6f}",
             f"{payload['reference_full_24']['v1']:.6f}",
             f"{payload['reference_full_24']['p3_event_50']:.6f}"],
            ["coverage (stable + minutes)", "13", str(final["coverage"]), "24", "24"],
        ], ["basis", "champion frozen", "champion final", "v1", "p3_event_50"]),
        "",
        f"Gap to v1 on the like-for-like basis: {payload['gap_to_v1_before']:.6f} -> "
        f"{payload['gap_to_v1_after']:.6f} (**{payload['gap_closed_fraction'] * 100:.1f}% closed**).",
        "", "## Reconstructed-rubric MAE (stable tier)", "",
    ]
    rows = []
    for name in ("t0_baseline", payload["final_incumbent"]):
        for rubric, value in sorted(payload["trials"][name]["rubric_mae"].items()):
            rows.append([
                name, rubric, f"{value['mae']:.6f}", f"{value['spearman']:.4f}",
                f"{value['mean_capture']:.4f}", str(value["slates"]),
            ])
    lines += [
        _table(rows, ["trial", "rubric", "mae", "spearman", "mean_capture", "slates"]), "",
        "The frozen champion produces **no NCR rubric rows at all**: it lacks heads for",
        "NCR-scored events, so every row fails the rubric's completeness mask. Extending",
        "coverage makes the champion NCR-scorable for the first time.", "",
        "## Per-target relative loss (final)", "",
    ]
    per_target = final["per_target"]
    lines.append(_table(
        [[t, f"{per_target[t]:.4f}"] for t in sorted(per_target, key=lambda k: -per_target[k])],
        ["target", "relative_loss"],
    ))
    lines += [
        "", "## Per-fold / cohort / tournament", "",
        _table([[k, f"{v:.6f}"] for k, v in sorted(final["by_fold_full"].items())],
               ["fold", "full stable score"]),
        "",
        f"north cohort: {final['cohorts']['north']:.6f}; south cohort: not evaluable "
        "(both folds are Six Nations, so the north/south and tournament-family guards "
        "cannot bind on this fold set).", "",
        "## Trial ledger", "",
    ]
    for step in payload["chain"]:
        lines += [f"### Step {step['step']} — incumbent `{step['incumbent']}`", ""]
        rows = []
        for trial in step["trials"]:
            summary = payload["trials"][trial["candidate"]]
            bootstrap = trial["bootstrap_by_fold"]
            rows.append([
                trial["candidate"],
                f"{trial['like_for_like_candidate']:.6f}",
                f"{summary['full_stable_score']:.6f}",
                str(summary["coverage"]),
                f"{bootstrap['p05']:+.5f}/{bootstrap['p95']:+.5f}",
                "ACCEPT" if trial["accepted"] else "REJECT",
                "; ".join(trial["reasons"]) or "-",
            ])
        lines += [_table(rows, [
            "candidate", "l4l score", "full score", "cov", "bootstrap p05/p95",
            "verdict", "reasons",
        ]), ""]
        if step["accepted"]:
            lines += [f"Accepted: **`{step['accepted']}`**", ""]
        else:
            lines += ["No candidate accepted — hill climb converged.", ""]
    lines += ["## Hypotheses", ""]
    lines.append(_table(
        [[name, payload["trials"][name]["hypothesis"]] for name in sorted(payload["trials"])],
        ["trial", "hypothesis"],
    ))
    return "\n".join(lines) + "\n"


def write(output_dir: Path = OUT) -> dict:
    payload = build()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ledger.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output_dir / "LEDGER.md").write_text(render(payload))
    return payload


if __name__ == "__main__":
    written = write()
    print(f"final={written['final_incumbent']} "
          f"full={written['final_summary']['full_stable_score']:.6f} "
          f"gap_closed={written['gap_closed_fraction'] * 100:.1f}%")
