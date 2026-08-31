"""Driver for phase 2 of the all-rugby P3 hill-climb.

Phase 1 (``allrugby_p3/LEDGER.md``) exhausted the blend-rule family over a fixed
component pair and stopped at C5 -- the shrunk per-target weight, cross-fitted
0.879366. Every other accepted phase-1 candidate was an alternative
parameterisation of the same one-dimensional family, so phase 2 moves the search
onto the components themselves and onto axes orthogonal to the blend weight.

The acceptance rule is phase 1's, with the incumbent raised to C5:

  (a) competition-balanced stable_score improves on the incumbent
  (b) paired-by-fold bootstrap difference excludes 0
  (c) neither hemisphere regresses by more than 2%
  (d) no tournament-family stable regression greater than 2%
  (e) no extended-event loss regression greater than 5% vs frozen P3
  (f) every fitted parameter is cross-fitted leave-one-fold-out and additionally
      checked on a strict temporal split

Standing project requirement: report the reconstructed rubric MAE for both
rubrics alongside every raw score. NCR GW4-7 is never read.
"""

from __future__ import annotations

import json
import time
from functools import partial

import numpy as np
import pandas as pd

from . import allrugby as A
from . import allrugby_phase2 as P
from .config import EXTENDED_EVENTS, SEED

WORK = P.WORK
FROZEN = 0.8864704364786675
C5_INCUMBENT = 0.8793658777
BASE_PAIR = ("v4", "empirical")
T3 = "empirical_t3_eb_off_ebk_head"
T4 = "empirical_t4_no_signal_max"
TEMPORAL_SPLIT = 10


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def _tournament_frame(table, base: np.ndarray, candidate: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({
        "fold": table.folds, "tournament": table.tournaments,
        "base": base, "candidate": candidate,
    })
    grouped = frame.groupby("tournament", as_index=False)[["base", "candidate"]].mean()
    grouped["pct"] = 100.0 * (grouped["candidate"] - grouped["base"]) / grouped["base"]
    return grouped


def evaluate_gates(table, base: dict[str, np.ndarray], candidate: dict[str, np.ndarray]):
    """(a)-(d) against the incumbent's per-fold scores, per cohort."""
    reasons = []
    score, incumbent = float(np.mean(candidate["all"])), float(np.mean(base["all"]))
    if not score < incumbent:
        reasons.append(f"(a) stable_score {score:.6f} did not improve on {incumbent:.6f}")
    boot = P.bootstrap(candidate["all"] - base["all"])
    if not (boot["p05"] < 0 and boot["p95"] < 0):
        reasons.append(f"(b) bootstrap CI [{boot['p05']:.6f}, {boot['p95']:.6f}] includes 0")
    cohort_pct = {}
    for cohort in ("north", "south"):
        before = float(np.nanmean(base[cohort]))
        after = float(np.nanmean(candidate[cohort]))
        cohort_pct[cohort] = 100.0 * (after - before) / before
        if cohort_pct[cohort] > 2.0:
            reasons.append(f"(c) {cohort} regressed {cohort_pct[cohort]:.2f}%")
    families = _tournament_frame(table, base["all"], candidate["all"])
    bad = families[families["pct"] > 2.0]["tournament"].tolist()
    if bad:
        reasons.append(f"(d) tournament-family regression >2%: {', '.join(bad)}")
    return reasons, {
        "stable_score": score, "incumbent_score": incumbent,
        "delta_vs_incumbent": score - incumbent,
        "delta_vs_frozen": score - FROZEN,
        "bootstrap": boot, "cohort_pct": cohort_pct,
        "families": families.to_dict("records"),
    }


def extended_check(caches, components, weights: np.ndarray, frozen_extended: dict) -> tuple[list[str], dict]:
    """(e) extended-event loss must not regress more than 5% vs frozen P3."""
    reasons, values = [], {}
    for event in EXTENDED_EVENTS:
        scores = [
            P.target_relative_loss(cache, components, event, weights) for cache in caches
        ]
        scores = [value for value in scores if np.isfinite(value)]
        if not scores:
            continue
        values[event] = float(np.mean(scores))
        reference = frozen_extended.get(event)
        if reference and values[event] > reference * 1.05:
            reasons.append(
                f"(e) {event} extended loss {values[event]:.4f} regressed >5% vs {reference:.4f}"
            )
    return reasons, values


# --------------------------------------------------------------------------
# candidate runner
# --------------------------------------------------------------------------

def per_cohort(table, per_fold_indices) -> dict[str, np.ndarray]:
    return {
        cohort: np.array([
            P.fold_score(table, index, per_fold_indices[index], cohort=cohort)
            for index in range(len(table.folds))
        ])
        for cohort in P.COHORTS
    }


def fixed_per_cohort(table, indices) -> dict[str, np.ndarray]:
    return per_cohort(table, [indices] * len(table.folds))


def temporal_check(table, fitter, baseline_per_fold_indices) -> dict:
    train = np.arange(TEMPORAL_SPLIT)
    held = np.arange(TEMPORAL_SPLIT, len(table.folds))
    fitted = fitter(table, train)
    indices = fitted[0] if isinstance(fitted, tuple) else fitted
    candidate = np.array([P.fold_score(table, index, indices) for index in held])
    base = np.array([
        P.fold_score(table, index, baseline_per_fold_indices[index]) for index in held
    ])
    return {
        "candidate": float(np.mean(candidate)), "baseline": float(np.mean(base)),
        "delta": float(np.mean(candidate - base)),
        "bootstrap": P.bootstrap(candidate - base),
    }


def run_candidate(
    name: str, hypothesis: str, components: tuple[str, ...], points: np.ndarray,
    fitter, *, caches, incumbent: dict[str, np.ndarray], incumbent_indices,
    frozen_extended: dict, table=None,
) -> dict:
    started = time.time()
    table = table if table is not None else P.build_multi_table(caches, components, points)
    fitted = fitter(table, np.arange(len(table.folds)))
    deployed, hyperparameter = fitted if isinstance(fitted, tuple) else (fitted, None)
    per_fold_indices, fits = P.cross_fit(table, fitter)
    candidate = per_cohort(table, per_fold_indices)
    reasons, summary = evaluate_gates(table, incumbent, candidate)
    extended_reasons, extended_values = extended_check(
        caches, components, _extended_weights(components), frozen_extended,
    )
    reasons = reasons + extended_reasons
    temporal = temporal_check(table, fitter, incumbent_indices)
    if not (temporal["delta"] < 0):
        reasons.append(f"(f) temporal split delta {temporal['delta']:+.6f} is not an improvement")
    in_sample = float(np.mean([
        P.fold_score(table, index, deployed) for index in range(len(table.folds))
    ]))
    return {
        "candidate": name, "hypothesis": hypothesis, "components": list(components),
        "n_points": int(len(points)), "hyperparameter": hyperparameter,
        "accepted": not reasons, "reasons": reasons,
        "in_sample_score": in_sample, "temporal": temporal,
        "extended": extended_values,
        "deployed_weights": {
            target: table.points[deployed[position]].tolist()
            for position, target in enumerate(table.targets)
        },
        "per_fold": {
            "folds": list(table.folds),
            "candidate": candidate["all"].tolist(),
            "incumbent": incumbent["all"].tolist(),
        },
        "fits": fits,
        "seconds": round(time.time() - started, 1),
        **summary,
    }


def _extended_weights(components) -> np.ndarray:
    """Extended events are not fitted: they stay at the frozen equal split
    between the GBDT and the empirical family, with any naive share at zero."""
    weights = np.zeros(len(components))
    empirical_like = [
        index for index, component in enumerate(components)
        if component.startswith("empirical")
    ]
    v4_index = components.index("v4") if "v4" in components else None
    if v4_index is not None:
        weights[v4_index] = 0.5
    for index in empirical_like:
        weights[index] = 0.5 / len(empirical_like)
    if weights.sum() <= 0:
        weights[:] = 1.0 / len(components)
    return weights / weights.sum()


# --------------------------------------------------------------------------
# control and queue
# --------------------------------------------------------------------------

def build_incumbent(caches):
    """Frozen P3 and the phase-1 C5 incumbent, reconstructed here."""
    table = P.build_multi_table(caches, BASE_PAIR, P.pair_grid())
    half = table.index_of([0.5, 0.5])
    frozen_indices = np.full(len(table.targets), half, dtype=int)
    frozen = fixed_per_cohort(table, frozen_indices)
    c5_indices, c5_fits = P.cross_fit(table, P.fit_shrunk_per_target)
    c5 = per_cohort(table, c5_indices)
    control = {
        "frozen_reconstructed": float(np.mean(frozen["all"])),
        "frozen_expected": FROZEN,
        "frozen_abs_difference": abs(float(np.mean(frozen["all"])) - FROZEN),
        "c5_reconstructed": float(np.mean(c5["all"])),
        "c5_expected": C5_INCUMBENT,
        "c5_abs_difference": abs(float(np.mean(c5["all"])) - C5_INCUMBENT),
        "n_folds": len(table.folds),
        "frozen_per_fold": frozen["all"].tolist(),
        "c5_per_fold": c5["all"].tolist(),
        "c5_fits": c5_fits,
    }
    frozen_extended = extended_check(
        caches, BASE_PAIR, np.array([0.5, 0.5]), {},
    )[1]
    return table, frozen, frozen_indices, c5, c5_indices, c5_fits, control, frozen_extended


def standalone_scores(union, components: tuple[str, ...]) -> dict[str, float]:
    """Each component scored alone, as a control against its own ledger.

    The frozen empirical engine must land on the v1 ledger's 0.939031 and the
    re-fitted t3 variant on the empirical hill-climb's 0.904538; if they do not,
    the components were not regenerated faithfully and nothing downstream is
    trustworthy.
    """
    scores = {}
    for component in components:
        caches = P.project_cache(union, (component,))
        table = P.build_multi_table(
            caches, (component,), np.ones((1, 1)), persist=False,
        )
        indices = np.zeros(len(table.targets), dtype=int)
        scores[component] = float(np.mean(P.fixed_scores(table, indices)))
    return scores


def queue(caches, table, frozen_extended):
    """Ordered candidate queue. Each entry: (name, hypothesis, components, points, fitter)."""
    shrunk = P.fit_shrunk_per_target
    return [
        (
            "E1_swap_t3_global",
            "The empirical component was hill-climbed separately to 0.904538 standalone "
            "(t3: EB-shrunk minutes, per-event EB shrinkage, median minutes head, no "
            "opponent multiplier). P3's blend was fitted against the OLD component, so "
            "simply substituting the better one at the frozen 0.5 should already move "
            "the blend.",
            ("v4", T3), P.pair_grid(),
            lambda table, folds: np.full(
                len(table.targets), table.index_of([0.5, 0.5]), dtype=int,
            ),
        ),
        (
            "E2_swap_t3_shrunk",
            "Re-fit C5's shrunk per-target weight against the improved empirical "
            "component. If the component swap and the per-target routing are "
            "complementary the two gains should compose.",
            ("v4", T3), P.pair_grid(), shrunk,
        ),
        (
            "E3_three_empirical",
            "Keep BOTH empirical components. The frozen one and t3 differ mainly in "
            "minutes propagation, so a per-target simplex can take the frozen "
            "component where its noisier minutes happen to help and t3 elsewhere.",
            ("v4", "empirical", T3), P.simplex_grid(3, 0.025), shrunk,
        ),
        (
            "N1_naive_shrinkage",
            "Half the targets sit at or above 1.0 relative loss for BOTH components -- "
            "neither engine beats the training-only position x started stratum mean "
            "there. Adding that comparator as a third component lets the per-target fit "
            "shrink toward it exactly where the engines have no edge. Guard: this cannot "
            "be judged on the raw score alone, because the comparator is the metric's "
            "denominator and cannot rank players; the rubric report decides.",
            ("v4", "empirical", P.NAIVE), P.simplex_grid(3, 0.025), shrunk,
        ),
        (
            "N3_naive_capped",
            "N1 with the comparator share capped at 0.5 per target. A target driven "
            "fully to the stratum mean has no within-stratum ranking left, however good "
            "its raw loss looks; the cap keeps at least half the mass on a real engine. "
            "If N3 recovers most of N1's raw gain then the gain was shrinkage, not the "
            "metric's denominator being handed the answer.",
            ("v4", "empirical", P.NAIVE),
            P.cap_component(P.simplex_grid(3, 0.025), 2, 0.5), shrunk,
        ),
        (
            "N2_naive_plus_t3",
            "N1's shrinkage and E2's better component are orthogonal; fit them together.",
            ("v4", T3, P.NAIVE), P.simplex_grid(3, 0.025), shrunk,
        ),
    ]


def group_queue():
    """Row-group candidates. Groups are row attributes, never the tournament."""
    return [
        (
            "G1_started_weight",
            "The two engines disagree most about bench players: the empirical engine "
            "scales counts by a per-player minutes table while the GBDT models minutes "
            "from exposure features. Splitting the blend weight by starter/bench is one "
            "row attribute the phase-1 search never tried, and it stays "
            "competition-independent. Every benchmark loss is a mean of per-row terms, "
            "so group weights are additively separable and fitted exactly.",
            ("v4", "empirical"), P.pair_grid(), P.started_groups, P.STARTED_NAMES,
        ),
        (
            "G2_position_weight",
            "Event-type routing was phase 1's finding; position is the other structural "
            "axis. A hooker's rucks and a winger's metres are produced by different "
            "mechanisms, so the component that models them best may differ by position "
            "as well as by event.",
            ("v4", "empirical"), P.pair_grid(), P.position_groups, P.POSITION_NAMES,
        ),
    ]


UNION = ("v4", "empirical", T3, T4, P.NAIVE)


def main(names: tuple[str, ...] | None = None) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    union = P.build_multi_cache(UNION)
    caches = P.project_cache(union, BASE_PAIR)
    (table, frozen, frozen_indices, c5, c5_indices, c5_fits,
     control, frozen_extended) = build_incumbent(caches)
    control["standalone"] = standalone_scores(union, UNION)
    (WORK / "control.json").write_text(json.dumps(control, indent=2) + "\n")
    print(json.dumps(
        {key: value for key, value in control.items()
         if not key.endswith(("per_fold", "fits"))},
        indent=2,
    ), flush=True)

    results = []
    if (WORK / "results.json").exists():
        results = json.loads((WORK / "results.json").read_text())
    done = {entry["candidate"] for entry in results}

    def record(result: dict) -> None:
        results[:] = [entry for entry in results if entry["candidate"] != result["candidate"]]
        results.append(result)
        print(json.dumps({
            k: result[k] for k in
            ("stable_score", "delta_vs_incumbent", "delta_vs_frozen", "accepted", "reasons")
        }, indent=2), flush=True)
        (WORK / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    for name, hypothesis, components, points, fitter in queue(caches, table, frozen_extended):
        if (names and name not in names) or (not names and name in done):
            continue
        print(f"\n=== {name} ({', '.join(components)}, {len(points)} points)", flush=True)
        record(run_candidate(
            name, hypothesis, components, points, fitter,
            caches=P.project_cache(union, components), incumbent=c5,
            incumbent_indices=c5_indices, frozen_extended=frozen_extended,
        ))

    for name, hypothesis, components, points, group_fn, group_names in group_queue():
        if (names and name not in names) or (not names and name in done):
            continue
        print(f"\n=== {name} ({', '.join(components)}, {len(group_names)} groups)", flush=True)
        record(run_group_candidate(
            name, hypothesis, components, points, group_fn, group_names,
            caches=P.project_cache(union, components), incumbent=c5,
            frozen_extended=frozen_extended,
        ))


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or None)


# --------------------------------------------------------------------------
# row-group candidates
# --------------------------------------------------------------------------

def group_per_cohort(table, per_fold_indices) -> dict[str, np.ndarray]:
    """Per-fold scores under a GroupTable, one index array per fold."""
    scores = np.array([
        float(table.relative(per_fold_indices[index])[index])
        for index in range(len(table.folds))
    ])
    return {"all": scores}


def run_group_candidate(
    name: str, hypothesis: str, components: tuple[str, ...], points: np.ndarray,
    group_fn, group_names: tuple[str, ...], *, caches, incumbent, frozen_extended,
) -> dict:
    started = time.time()
    table = P.build_group_table(caches, components, points, group_fn, group_names)
    deployed, hyperparameter = P.fit_group_shrunk(table, np.arange(len(table.folds)))
    per_fold_indices, fits = [], []
    for index in range(len(table.folds)):
        train = np.delete(np.arange(len(table.folds)), index)
        indices, chosen = P.fit_group_shrunk(table, train)
        per_fold_indices.append(indices)
        fits.append({"held_out_fold": table.folds[index], "hyperparameter": chosen})
    candidate_all = np.array([
        float(table.relative(per_fold_indices[index])[index])
        for index in range(len(table.folds))
    ])
    reasons = []
    score, base = float(np.mean(candidate_all)), float(np.mean(incumbent["all"]))
    if not score < base:
        reasons.append(f"(a) stable_score {score:.6f} did not improve on {base:.6f}")
    boot = P.bootstrap(candidate_all - incumbent["all"])
    if not (boot["p05"] < 0 and boot["p95"] < 0):
        reasons.append(f"(b) bootstrap CI [{boot['p05']:.6f}, {boot['p95']:.6f}] includes 0")
    families = _tournament_frame(table, incumbent["all"], candidate_all)
    bad = families[families["pct"] > 2.0]["tournament"].tolist()
    if bad:
        reasons.append(f"(d) tournament-family regression >2%: {', '.join(bad)}")
    cohort_pct = {}
    for cohort in ("north", "south"):
        after = np.array([
            P.group_fold_score(
                caches[index], components, table.targets, table.points,
                per_fold_indices[index], group_fn(caches[index]), cohort=cohort,
            )
            for index in range(len(table.folds))
        ])
        before = np.nanmean(incumbent[cohort])
        cohort_pct[cohort] = 100.0 * (float(np.nanmean(after)) - before) / before
        if cohort_pct[cohort] > 2.0:
            reasons.append(f"(c) {cohort} regressed {cohort_pct[cohort]:.2f}%")
    extended_reasons, extended_values = extended_check(
        caches, components, _extended_weights(components), frozen_extended,
    )
    reasons += extended_reasons
    train = np.arange(TEMPORAL_SPLIT)
    held = np.arange(TEMPORAL_SPLIT, len(table.folds))
    temporal_indices, _ = P.fit_group_shrunk(table, train)
    temporal_scores = table.relative(temporal_indices)[held]
    temporal_base = incumbent["all"][held]
    temporal = {
        "candidate": float(np.mean(temporal_scores)),
        "baseline": float(np.mean(temporal_base)),
        "delta": float(np.mean(temporal_scores - temporal_base)),
        "bootstrap": P.bootstrap(temporal_scores - temporal_base),
    }
    if not (temporal["delta"] < 0):
        reasons.append(f"(f) temporal split delta {temporal['delta']:+.6f} is not an improvement")
    return {
        "candidate": name, "hypothesis": hypothesis, "components": list(components),
        "groups": list(group_names), "n_points": int(len(points)),
        "hyperparameter": hyperparameter, "accepted": not reasons, "reasons": reasons,
        "stable_score": score, "incumbent_score": base,
        "delta_vs_incumbent": score - base, "delta_vs_frozen": score - FROZEN,
        "bootstrap": boot, "families": families.to_dict("records"),
        "cohort_pct": cohort_pct,
        "in_sample_score": float(np.mean(table.relative(deployed))),
        "temporal": temporal, "extended": extended_values,
        "deployed_weights": {
            target: {
                group: table.points[deployed[target_index, group_index]].tolist()
                for group_index, group in enumerate(group_names)
            }
            for target_index, target in enumerate(table.targets)
        },
        "per_fold": {
            "folds": list(table.folds), "candidate": candidate_all.tolist(),
            "incumbent": incumbent["all"].tolist(),
        },
        "fits": fits, "seconds": round(time.time() - started, 1),
    }


# --------------------------------------------------------------------------
# stage-2 calibration on top of an accepted candidate
# --------------------------------------------------------------------------

def run_calibration_candidate(
    name: str, hypothesis: str, components: tuple[str, ...], points: np.ndarray,
    fitter, *, caches, incumbent, incumbent_indices, frozen_extended,
) -> dict:
    """Per-target multiplicative scale fitted on top of the blend weights.

    The blend mixes two calibrated-ish engines, but the mixture of two means is
    not itself unbiased under a Poisson deviance; phase 1 saw the frozen P3
    carry a systematic scale bias that affine recalibration removed. One scalar
    per target is the smallest correction for that, and it is orthogonal to the
    blend weight. Both stages are cross-fitted inside the same outer loop.
    """
    started = time.time()
    table = P.build_multi_table(caches, components, points)
    all_folds = np.arange(len(table.folds))
    deployed_weights, hyperparameter = _as_pair(fitter(table, all_folds))
    deployed_scales = P.fit_calibration(caches, components, table, deployed_weights, all_folds)
    candidate_all, per_fold_scales = [], []
    for index in range(len(table.folds)):
        train = np.delete(all_folds, index)
        weights, _ = _as_pair(fitter(table, train))
        scales = P.fit_calibration(caches, components, table, weights, train)
        per_fold_scales.append(scales.tolist())
        candidate_all.append(P.calibrated_fold_score(
            caches[index], components, table, weights, scales,
        ))
        print(f"  calibrated fold {table.folds[index]}", flush=True)
    candidate_all = np.array(candidate_all)
    reasons = []
    score, base = float(np.mean(candidate_all)), float(np.mean(incumbent["all"]))
    if not score < base:
        reasons.append(f"(a) stable_score {score:.6f} did not improve on {base:.6f}")
    boot = P.bootstrap(candidate_all - incumbent["all"])
    if not (boot["p05"] < 0 and boot["p95"] < 0):
        reasons.append(f"(b) bootstrap CI [{boot['p05']:.6f}, {boot['p95']:.6f}] includes 0")
    families = _tournament_frame(table, incumbent["all"], candidate_all)
    bad = families[families["pct"] > 2.0]["tournament"].tolist()
    if bad:
        reasons.append(f"(d) tournament-family regression >2%: {', '.join(bad)}")
    extended_reasons, extended_values = extended_check(
        caches, components, _extended_weights(components), frozen_extended,
    )
    reasons += extended_reasons
    train = np.arange(TEMPORAL_SPLIT)
    held = np.arange(TEMPORAL_SPLIT, len(table.folds))
    weights, _ = _as_pair(fitter(table, train))
    scales = P.fit_calibration(caches, components, table, weights, train)
    temporal_scores = np.array([
        P.calibrated_fold_score(caches[index], components, table, weights, scales)
        for index in held
    ])
    temporal_base = incumbent["all"][held]
    temporal = {
        "candidate": float(np.mean(temporal_scores)),
        "baseline": float(np.mean(temporal_base)),
        "delta": float(np.mean(temporal_scores - temporal_base)),
        "bootstrap": P.bootstrap(temporal_scores - temporal_base),
    }
    if not (temporal["delta"] < 0):
        reasons.append(f"(f) temporal split delta {temporal['delta']:+.6f} is not an improvement")
    return {
        "candidate": name, "hypothesis": hypothesis, "components": list(components),
        "n_points": int(len(points)), "hyperparameter": hyperparameter,
        "accepted": not reasons, "reasons": reasons,
        "stable_score": score, "incumbent_score": base,
        "delta_vs_incumbent": score - base, "delta_vs_frozen": score - FROZEN,
        "bootstrap": boot, "families": families.to_dict("records"),
        "temporal": temporal, "extended": extended_values,
        "deployed_weights": {
            target: table.points[deployed_weights[position]].tolist()
            for position, target in enumerate(table.targets)
        },
        "deployed_scales": dict(zip(table.targets, deployed_scales.tolist())),
        "per_fold": {
            "folds": list(table.folds), "candidate": candidate_all.tolist(),
            "incumbent": incumbent["all"].tolist(), "scales": per_fold_scales,
        },
        "seconds": round(time.time() - started, 1),
    }


def _as_pair(fitted):
    return fitted if isinstance(fitted, tuple) else (fitted, None)
