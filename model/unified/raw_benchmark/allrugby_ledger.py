"""Render LEDGER.md from the all-rugby trial ledger, weights and rubric outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .allrugby import WORK


def _md(frame: pd.DataFrame, digits: int = 4) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include=["float", "float64"]).columns:
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    headers = [str(column) for column in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def render() -> str:
    ledger = json.load(open(WORK / "ledger.json"))
    weights = json.load(open(WORK / "c5_weights.json"))
    rubric = pd.read_csv(WORK / "rubric_summary.csv")
    rubric_boot = json.load(open(WORK / "rubric_bootstrap.json"))
    diagnostic = pd.read_csv(WORK / "diagnostic_per_target.csv")
    c5 = next(t for t in ledger["trials"] if t["name"] == "C5_shrunk_per_target_weight")
    per_fold = pd.DataFrame(c5["per_fold"])
    per_fold["delta"] = per_fold.stable_cand - per_fold.stable_base
    frozen = ledger["frozen_baseline"]

    out: list[str] = []
    add = out.append
    add("# All-rugby hill-climb ledger — P3 unified event model\n")
    add("Branch `agents/allrugby-p3`. Objective: minimise the competition-balanced stable")
    add("raw score of Historical Raw Benchmark v1, pooled over **all** international rugby")
    add("from 2022-01-01 (21 exact-kickoff tournament holdout folds, both hemispheres).")
    add("Lower is better. Per-competition rubric scores are reported, never fitted.\n")
    add(f"Seed {ledger['seed']}. Weight grid step {ledger['weight_grid_step']}. "
        "NCR GW4–7 was never read.\n")

    control = ledger["control"]
    add("## Control\n")
    add(f"- Reconstructed frozen `p3_event_50` stable score: **{control['reconstructed_stable_score']:.10f}**")
    add(f"- Frozen ledger value: **{control['frozen_stable_score']:.10f}**")
    add(f"- Absolute difference `{control['abs_difference']:.1e}`; worst per-fold difference "
        f"`{control['max_abs_per_fold_difference_vs_ledger']:.1e}`")
    add(f"- Reproduced: **{control['reproduced']}** across {control['n_folds']} folds\n")
    add("The frozen `p3_event_50` artifacts are self-contained, so the empirical and v4")
    add("component predictions were recovered from them directly and re-blended offline —")
    add("no engine was refitted. Re-blending at w=0.5 reproduces the frozen predictions and")
    add("the frozen per-fold metrics exactly. The frozen v1 ledger was never written to.\n")
    add("Two further control matches: the reconstructed rubric aggregates reproduce the")
    add("ledger's `p3_event_50` figures exactly (ncr MAE 6.1727 / spearman 0.5367 / capture")
    add("0.7187; six_nations 6.2767 / 0.6287 / 0.7689, 98 slates each), and the degenerate")
    add("`low_history` relative loss (848.68, from near-zero naive kicking losses) reproduces")
    add("the frozen ledger's value too. That cohort gates nothing.\n")

    add("## Precommitted acceptance rule\n")
    for key, value in sorted(ledger["acceptance_rule"].items()):
        add(f"- **({key})** {value}")
    add("\nTies and sub-threshold trials are rejected and logged.\n")

    add("## Trials\n")
    rows = []
    for trial in ledger["trials"]:
        boot = trial["bootstrap_paired_by_fold"]
        temporal = trial.get("temporal_holdout")
        params = trial.get("n_fitted_parameters")
        rows.append({
            "candidate": trial["name"],
            "params": params if isinstance(params, int) else 0,
            "stable_score": trial["stable_score"],
            "delta_vs_frozen": trial["stable_score"] - frozen,
            "boot_p05": boot["p05"], "boot_p95": boot["p95"],
            "temporal_delta": temporal["held_out_delta"] if temporal else np.nan,
            "accepted": trial["accepted"],
        })
    add(_md(pd.DataFrame(rows), 6))
    add("")
    add(f"Frozen baseline **{frozen:.6f}**. `stable_score` for fitted candidates is")
    add("leave-one-fold-out cross-fitted: the weights applied to a fold were fitted without")
    add("it. `temporal_delta` is a second, stricter check — fit on the 10 earliest folds,")
    add("evaluate on the 11 latest.\n")

    add("### Rejected, and why\n")
    for trial in ledger["trials"]:
        if trial["accepted"]:
            continue
        add(f"- **{trial['name']}** — {trial['hypothesis']}")
        for reason in trial["reasons"]:
            add(f"  - REJECTED: {reason}")
        add(f"  - In-sample reference {trial.get('in_sample_reference', float('nan')):.6f}; "
            f"cross-fitted {trial['stable_score']:.6f}.")
    add("")
    add("Three independent negatives, all pointing the same way — **the blend weight does")
    add("not want to move globally, by event density, or by player history; it wants to move")
    add("by event type**:\n")
    add("- **C1** is the sharpest. The global weight curve bottoms at **w=0.52** (0.886433)")
    add("  against 0.886470 at w=0.50 — a gain of 0.000037. The 0.5 chosen on 6N-2025 LORO")
    add("  was already sitting at the all-rugby optimum, and *refitting it costs* accuracy")
    add("  (cross-fitted 0.886763) because the refit adds estimation variance to a flat curve.")
    add("- **C4** fails because density is the wrong axis: `metres` is dense (0.83) yet wants")
    add("  w≈0.09, while `red_cards` is ultra-sparse (0.003) yet wants w≈0.91.")
    add("- **C7** is the cleanest null of all: given the freedom to vary the weight with")
    add("  player career history, the all-folds fit chooses a=0.5, b=0 — exactly the constant")
    add("  frozen weight, reproducing 0.886470 in-sample to the digit. History depth carries")
    add("  no usable signal for this blend.\n")

    add("### Accepted\n")
    for trial in ledger["trials"]:
        if trial["accepted"]:
            add(f"- **{trial['name']}** — cross-fitted {trial['stable_score']:.6f} "
                f"({100 * (1 - trial['stable_score'] / frozen):+.3f}% vs frozen). "
                f"{trial['hypothesis']}")
    add("")
    add("All accepted candidates pass every gate against the frozen baseline. They are")
    add("alternative parameterisations of the same one-dimensional blend family rather than")
    add("composable deltas, so the hill-climb terminates at the best of them:")
    add("**C5, the shrunk per-target weight**, at **0.879366**.\n")

    add("## Incumbent: C5 — shrunk per-target blend weight\n")
    boot = c5["bootstrap_paired_by_fold"]
    temporal = c5["temporal_holdout"]
    tboot = temporal["held_out_bootstrap"]
    add(f"- Cross-fitted stable score **{c5['stable_score']:.6f}** vs frozen {frozen:.6f}")
    add(f"- Improvement **{100 * c5['improvement_vs_frozen']:.3f}%**")
    add(f"- Paired-by-fold bootstrap: mean {boot['mean']:.6f}, 90% CI "
        f"[{boot['p05']:.6f}, {boot['p95']:.6f}] — excludes 0")
    add(f"- Temporal holdout (fit 10 earliest folds, evaluate 11 latest): "
        f"{temporal['held_out_stable']:.6f} vs {temporal['held_out_baseline']:.6f}, delta "
        f"{temporal['held_out_delta']:.6f}, CI [{tboot['p05']:.6f}, {tboot['p95']:.6f}] — excludes 0")
    add(f"- Folds improved: {(per_fold.delta < 0).sum()}/{len(per_fold)}")
    add(f"- Shrinkage λ by inner leave-one-fold-out: "
        f"{sorted(set(f['hyperparameter'] for f in c5['cross_fits']))} across outer folds; "
        f"{weights['deployed_shrinkage']} when fitted on all folds\n")

    add("### In-sample vs held-out statement\n")
    add("C5 fits 24 per-target weights plus one shrinkage strength (25 parameters).")
    add("**Nothing in the headline number is in-sample.** Every fold is scored with weights")
    add("fitted on the other 20 folds, and the shrinkage strength is chosen by a further")
    add("leave-one-fold-out pass *inside* those 20 — the held-out fold informs neither. For")
    add(f"reference the fully in-sample score is {c5['in_sample_reference']:.6f}; the honest")
    add(f"cross-fitted score is {c5['stable_score']:.6f}. The gap")
    add(f"({c5['stable_score'] - c5['in_sample_reference']:+.6f}) is the overfitting that")
    add("cross-fitting removes and that a naive all-folds fit would have banked as a fake")
    add("win. The independent temporal split, where no held-out fold is even contemporaneous")
    add("with the fitting set, confirms the effect survives.\n")

    add("### Fitted weights (v4 share; 1 − w is the empirical share)\n")
    deployed = weights["deployed_all_folds"]
    frame = pd.DataFrame({"target": list(deployed), "weight_v4": list(deployed.values())})
    frame = frame.merge(
        diagnostic[["target", "density", "empirical", "v4"]], on="target", how="left",
    ).sort_values("weight_v4")
    add(_md(frame))
    add("")
    add("The structure is not noise — it is a clean split by *event type*:\n")
    add("- **Attacking / scoring events go to the empirical prior** (w≈0.09–0.27): metres,")
    add("  tries, try assists, conversions, clean breaks, offloads, defenders beaten.")
    add("- **Volume / workrate and discipline events go to the GBDT** (w≈0.55–0.66): passes,")
    add("  runs, tackles, rucks, bad passes, place-kicking attempts, cards.")
    add("- **Neutral, near the frozen 0.5**: minutes, missed tackles, turnovers and penalties")
    add("  conceded.\n")
    add("The v4 GBDT models minutes- and role-driven volume well but over-smooths rare")
    add("attacking upside, where a player's own empirical history carries more signal. The")
    add("two ultra-sparse outliers (`drop_goals_converted` 0.84, `red_cards` 0.91) rest on")
    add("very few events; shrinkage pulls them in and they move the mean-of-24 by little.\n")

    add("## Breakdown\n\n### Per fold\n")
    show = per_fold[["fold", "tournament", "hemisphere", "stable_base", "stable_cand", "delta"]].copy()
    show["pct"] = 100 * (per_fold.stable_cand / per_fold.stable_base - 1)
    add(_md(show))
    add("\n### Per tournament family\n")
    tournaments = pd.DataFrame([
        {"tournament": key, "frozen": value["baseline"], "C5": value["candidate"],
         "pct": 100 * (value["candidate"] / value["baseline"] - 1)}
        for key, value in c5["per_tournament"].items()
    ]).sort_values("pct")
    add(_md(tournaments))
    add("\n### Hemisphere\n")
    add(_md(pd.DataFrame([
        {"cohort": "north", "frozen": c5["north_base"], "C5": c5["north"],
         "pct": 100 * (c5["north"] / c5["north_base"] - 1)},
        {"cohort": "south", "frozen": c5["south_base"], "C5": c5["south"],
         "pct": 100 * (c5["south"] / c5["south_base"] - 1)},
    ])))
    add("")
    add("Every tournament family improves and both hemispheres improve — no gate (c) or (d)")
    add("regression anywhere. The gains are largest exactly where P3's edge over v1 was")
    add("thinnest: the south (−1.36% vs −0.58% north) and the Rugby Championship (−2.06%).")
    add("The 6 regressing folds are concentrated in 2022–early 2024, where the training")
    add("history behind both components is thinnest.\n")

    add("## Reconstructed rubrics — always reported\n")
    add(_md(rubric[["engine", "rubric", "mae", "mae_calibrated", "spearman",
                    "mean_capture", "slates"]]))
    add("")
    add("`C5_cross_fitted` is the honest out-of-sample rule (weights fitted without the fold")
    add("being scored); `C5_deployed` is the all-folds fit that would actually ship.")
    add("`mae_calibrated` applies a leave-one-fold-out affine recalibration, so it is not")
    add("in-sample either.\n")
    add("Paired-by-slate bootstrap, C5 cross-fitted minus frozen P3 (98 slates):\n")
    rows = []
    for metric in ("mae", "mean_capture", "spearman"):
        for rubric_name in ("ncr", "six_nations"):
            value = rubric_boot[metric][rubric_name]
            rows.append({
                "metric": metric, "rubric": rubric_name, "mean": value["mean"],
                "p05": value["p05"], "p95": value["p95"],
                "excludes_0": (value["p05"] < 0 and value["p95"] < 0)
                              or (value["p05"] > 0 and value["p95"] > 0),
            })
    add(_md(pd.DataFrame(rows), 6))
    add("")
    add("**The raw-score gain does not cost rubric MAE — it improves it.** NCR MAE falls")
    add("6.1727 → 6.0669 and Six Nations MAE 6.2767 → 6.1771, both with bootstrap CIs")
    add("excluding 0. Capture and Spearman improve on both rubrics; significantly on NCR,")
    add("directionally but not significantly on Six Nations.\n")
    add("This resolves the tension flagged at the outset. `empirical_event` had better rubric")
    add("MAE (ncr 6.0608, capture 0.7279) than P3 despite a far worse raw score (0.939),")
    add("because the rubrics are dominated by tries, assists, conversions and metres —")
    add("precisely the events where the empirical prior beats the GBDT. The frozen global 0.5")
    add("was mis-weighting those rare high-value events. Routing them to the empirical")
    add("component while leaving volume events with the GBDT captures nearly all of the")
    add("empirical engine's rubric quality (ncr MAE 6.0669 vs 6.0608, capture 0.7278 vs")
    add("0.7279) at a raw score of 0.879 instead of 0.939. The component-aware blend does get")
    add("both.\n")
    add("One honest caveat: calibrated MAE narrows the gap. The frozen P3 carried a systematic")
    add("scale bias that affine recalibration corrects (ncr 6.1727 → 6.1305), whereas C5 is")
    add("already well calibrated so recalibration does not help it (6.0669 → 6.0677).")
    add("Comparing calibrated to calibrated, C5 still wins — ncr 6.0677 vs 6.1305,")
    add("six_nations 6.1936 vs 6.2478 — but by roughly 0.06 rather than 0.10. Part of the")
    add("raw-MAE gain is calibration that a recalibrated deployment would already capture.\n")

    add("## Verification\n")
    add("- The precomputed loss table is a shortcut, so it was checked against the unmodified")
    add("  `event_metrics` pipeline on all 21 folds: **max absolute per-fold difference 0.0**")
    add("  for both the frozen rule and C5. Pipeline stable score for C5 is 0.8793658777,")
    add("  identical to the table.")
    add("- Gate (e): extended-event predictions under C5 are **bit-identical** to frozen P3")
    add("  (max absolute difference 0.0 across all folds and all 8 extended events), because")
    add("  every candidate leaves extended events at w=0.5. No extended-event loss can regress.")
    add("- Fold hygiene is inherited unchanged from `folds.py`: exact-kickoff cutoffs, training")
    add("  rows strictly before, evaluation fixtures excluded from training, targets masked in")
    add("  candidate features.")
    add("- Determinism: SEED=17 throughout; reruns reproduce.\n")

    add("## Reproduce\n")
    add("```bash")
    add("python -m model.unified.raw_benchmark.allrugby_run      # control, diagnostics, trials")
    add("python -m model.unified.raw_benchmark.allrugby_history  # C7 (row-varying weight)")
    add("python model/unified/raw_benchmark/allrugby_verify.py   # pipeline-vs-table + gate (e)")
    add("python -m model.unified.raw_benchmark.allrugby_ledger   # render this file")
    add("```")
    add("Component predictions, the fold cache and the loss table are derived artifacts,")
    add("rebuilt on demand from the frozen P3 blends and excluded from git.\n")
    return "\n".join(out) + "\n"


def main() -> None:
    (WORK / "LEDGER.md").write_text(render())
    print(f"wrote {WORK / 'LEDGER.md'}")


if __name__ == "__main__":
    main()
