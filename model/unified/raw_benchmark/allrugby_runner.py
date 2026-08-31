"""Greedy hill-climb driver for the champion's all-rugby raw-prediction path.

Runs a queue of candidate :class:`RawPathConfig` deltas off the current
incumbent, applies the precommitted acceptance rule, and writes the ledger.

Sweeps write to ``data/unified/raw_benchmark/allrugby_sixnations/`` and never
touch the frozen ``v1`` ledger.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import ROOT
from .allrugby_champion import (
    ENGINE, EVAL_SEASONS, MISSING_STABLE_EVENTS, RAW_EVENTS, RawPathConfig,
    SHARED_STABLE_TARGETS, load_context, run_trial,
)
from .config import STABLE_EVENTS
from .report import _bootstrap_difference

FROZEN = ROOT / "data" / "unified" / "raw_benchmark" / "v1"
OUT = ROOT / "data" / "unified" / "raw_benchmark" / "allrugby_sixnations"
FROZEN_BASELINE_SCORE = 1.1144043542089876

#: Acceptance thresholds, precommitted before any candidate was run.
COHORT_REGRESSION = 0.02
TOURNAMENT_REGRESSION = 0.02


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _fold_scores(events: pd.DataFrame, targets: tuple[str, ...], cohort: str = "all") -> pd.Series:
    block = events[events["cohort"].eq(cohort) & events["target"].isin(targets)]
    if block.empty:
        return pd.Series(dtype=float)
    return block.groupby("fold")["relative_loss"].mean()


def _covered_targets(events: pd.DataFrame) -> tuple[str, ...]:
    """Stable targets (plus minutes) the engine actually predicted on every fold."""
    block = events[events["cohort"].eq("all")]
    wanted = ("minutes", *STABLE_EVENTS)
    folds = block["fold"].nunique()
    counts = block[block["target"].isin(wanted)].groupby("target")["fold"].nunique()
    return tuple(sorted(counts.index[counts.eq(folds)]))


def _rubric_mae(rankings: pd.DataFrame) -> dict:
    stable = rankings[rankings["tier"].eq("stable")]
    output: dict = {}
    for rubric in sorted(stable["rubric"].dropna().unique()):
        block = stable[stable["rubric"].eq(rubric)]
        capture_columns = [c for c in block.columns if c.endswith("_capture")]
        output[str(rubric)] = {
            "mae": float(block["mae"].mean()),
            "spearman": float(block["spearman"].mean(skipna=True)),
            "mean_capture": float(block[capture_columns].mean(axis=1, skipna=True).mean()),
            "slates": int(block["slate_id"].nunique()),
        }
    return output


def summarise(events: pd.DataFrame, rankings: pd.DataFrame) -> dict:
    covered = _covered_targets(events)
    shared = tuple(t for t in SHARED_STABLE_TARGETS if t in covered)
    by_fold_shared = _fold_scores(events, shared)
    by_fold_full = _fold_scores(events, covered)
    per_target = (
        events[events["cohort"].eq("all") & events["target"].isin(covered)]
        .groupby("target")["relative_loss"].mean().to_dict()
    )
    cohorts = {}
    for cohort in ("north", "south"):
        scores = _fold_scores(events, shared, cohort)
        cohorts[cohort] = float(scores.mean()) if len(scores) else float("nan")
    tournaments = (
        events[events["cohort"].eq("all") & events["target"].isin(shared)]
        .groupby(["tournament", "fold"])["relative_loss"].mean()
        .groupby("tournament").mean().to_dict()
    )
    return {
        "partial_stable_score": float(by_fold_shared.mean()),
        "full_stable_score": float(by_fold_full.mean()),
        "shared_targets": list(shared),
        "covered_targets": list(covered),
        "coverage": len(covered),
        "folds": int(events["fold"].nunique()),
        "by_fold_shared": {k: float(v) for k, v in by_fold_shared.items()},
        "by_fold_full": {k: float(v) for k, v in by_fold_full.items()},
        "per_target": {k: float(v) for k, v in per_target.items()},
        "cohorts": cohorts,
        "tournaments": {k: float(v) for k, v in tournaments.items()},
        "rubric_mae": _rubric_mae(rankings),
    }


# ---------------------------------------------------------------------------
# acceptance rule
# ---------------------------------------------------------------------------

def decide(candidate: dict, incumbent: dict, candidate_events: pd.DataFrame,
           incumbent_events: pd.DataFrame) -> dict:
    """Apply the precommitted acceptance rule.  ALL clauses must hold."""
    reasons: list[str] = []

    shared = tuple(sorted(set(candidate["shared_targets"]) & set(incumbent["shared_targets"])))
    cand_shared = _fold_scores(candidate_events, shared)
    inc_shared = _fold_scores(incumbent_events, shared)

    # (e) coverage never decreases
    widened = candidate["coverage"] > incumbent["coverage"]
    if candidate["coverage"] < incumbent["coverage"]:
        reasons.append(
            f"coverage decreased {incumbent['coverage']} -> {candidate['coverage']}"
        )

    improvement = float(inc_shared.mean()) - float(cand_shared.mean())
    if widened:
        # Clause (a) read together with (e) for a coverage-widening candidate.
        # Adding a head cannot improve the like-for-like score -- it does not
        # touch the existing heads -- so requiring a like-for-like gain would
        # make coverage extension impossible, which contradicts (e). Instead:
        # the like-for-like score must not REGRESS, and every newly covered
        # target must on average beat the naive comparator, so that widening
        # cannot be used to pad the mean with heads that are worse than
        # nothing. Both scores are reported, as (e) requires.
        if improvement < -1e-12:
            reasons.append(
                f"like-for-like stable score regressed while widening coverage "
                f"({float(cand_shared.mean()):.6f} vs {float(inc_shared.mean()):.6f})"
            )
        added = sorted(set(candidate["covered_targets"]) - set(incumbent["covered_targets"]))
        added_scores = [candidate["per_target"][t] for t in added if t in candidate["per_target"]]
        if added_scores and float(np.mean(added_scores)) >= 1.0:
            reasons.append(
                f"newly covered targets do not beat the naive comparator "
                f"(mean relative loss {float(np.mean(added_scores)):.6f})"
            )
    else:
        # (a) like-for-like score improves
        if improvement <= 0:
            reasons.append(
                f"like-for-like stable score did not improve "
                f"({float(cand_shared.mean()):.6f} vs {float(inc_shared.mean()):.6f})"
            )
        # (a') at equal coverage the full score must also improve
        if candidate["coverage"] == incumbent["coverage"]:
            if candidate["full_stable_score"] >= incumbent["full_stable_score"]:
                reasons.append("full stable score did not improve")

    # (b) paired bootstrap must exclude 0.
    #
    # At equal coverage this is the protocol's paired-by-fold test on the
    # like-for-like set.
    #
    # A widening candidate needs a different pairing. Its like-for-like
    # differences are identically zero (the added heads do not touch the
    # existing ones), and pairing a 13-target fold mean against a 24-target
    # fold mean is not a test of anything -- it just measures whether the new
    # targets happen to sit above or below the old ones' average, which is the
    # apples-to-oranges comparison clause (e) warns about. So for a widening
    # candidate the bootstrap is run over the newly covered fold x target
    # cells against the naive comparator (relative loss 1.0), which is the
    # actual claim being made: the new heads are better than no head at all.
    if widened:
        added = sorted(set(candidate["covered_targets"]) - set(incumbent["covered_targets"]))
        cells = candidate_events[
            candidate_events["cohort"].eq("all") & candidate_events["target"].isin(added)
        ]
        differences = (
            cells.groupby(["fold", "target"])["relative_loss"].mean().to_numpy(float) - 1.0
        )
        bootstrap_basis = "new_targets_vs_naive_by_fold_target"
    else:
        paired = pd.concat(
            [cand_shared.rename("cand"), inc_shared.rename("inc")], axis=1
        ).dropna()
        differences = (paired["cand"] - paired["inc"]).to_numpy(float)
        bootstrap_basis = "like_for_like_by_fold"
    bootstrap = _bootstrap_difference(differences)
    excludes_zero = bool(
        np.isfinite(bootstrap["p05"]) and np.isfinite(bootstrap["p95"])
        and (bootstrap["p05"] < 0 and bootstrap["p95"] < 0)
    )
    if not excludes_zero:
        reasons.append(
            f"paired-by-fold bootstrap does not exclude 0 "
            f"(p05={bootstrap['p05']:.6f}, p95={bootstrap['p95']:.6f})"
        )

    # (c) neither hemisphere cohort regresses by >2%
    for cohort in ("north", "south"):
        cand_value = candidate["cohorts"].get(cohort, float("nan"))
        inc_value = incumbent["cohorts"].get(cohort, float("nan"))
        if np.isfinite(cand_value) and np.isfinite(inc_value):
            if cand_value > inc_value * (1.0 + COHORT_REGRESSION):
                reasons.append(f"{cohort} cohort regressed by more than 2%")

    # (d) no tournament-family stable regression >2%
    for tournament, inc_value in incumbent["tournaments"].items():
        cand_value = candidate["tournaments"].get(tournament)
        if cand_value is not None and cand_value > inc_value * (1.0 + TOURNAMENT_REGRESSION):
            reasons.append(f"tournament-family regression >2%: {tournament}")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "like_for_like_candidate": float(cand_shared.mean()),
        "like_for_like_incumbent": float(inc_shared.mean()),
        "like_for_like_targets": list(shared),
        "improvement": improvement,
        "bootstrap_by_fold": bootstrap,
        "bootstrap_basis": bootstrap_basis,
        "bootstrap_excludes_zero": excludes_zero,
    }


# ---------------------------------------------------------------------------
# candidate queue
# ---------------------------------------------------------------------------

def candidate_queue(incumbent: RawPathConfig) -> list[RawPathConfig]:
    """Small, principled deltas off the CURRENT incumbent."""
    sparse = ("drop_goals_converted", "red_cards", "yellow_cards",
              "penalty_goals", "conversion_goals")
    return [
        replace(incumbent, name="c1_calib_global",
                level_calibration="global"),
        replace(incumbent, name="c2_calib_global_sparse_only",
                level_calibration="global", calibrate_events=sparse),
        replace(incumbent, name="c3_calib_position",
                level_calibration="position"),
    ]


def run_queue(
    configs: list[RawPathConfig], *, output_dir: Path = OUT, force: bool = False,
) -> pd.DataFrame:
    """Run each config, caching its metric tables under ``trials/<name>/``."""
    store_path = FROZEN / "player_match.csv"
    benchmark, folds, native, champion_config = load_context(output_dir, store_path)
    cache_dir = output_dir / "champion_cache"
    rows = []
    for cfg in configs:
        trial_dir = output_dir / "trials" / cfg.name
        events_path = trial_dir / "event_metrics.csv"
        rankings_path = trial_dir / "ranking_metrics.csv"
        if events_path.exists() and rankings_path.exists() and not force:
            events = pd.read_csv(events_path)
            rankings = pd.read_csv(rankings_path)
        else:
            print(f"[allrugby] running trial {cfg.name}", flush=True)
            events, rankings = run_trial(
                cfg, benchmark=benchmark, folds=folds, native=native,
                champion_config=champion_config, cache_dir=cache_dir,
            )
            trial_dir.mkdir(parents=True, exist_ok=True)
            events.to_csv(events_path, index=False)
            rankings.to_csv(rankings_path, index=False)
            (trial_dir / "config.json").write_text(
                json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n"
            )
        summary = summarise(events, rankings)
        summary["name"] = cfg.name
        summary["config"] = cfg.to_dict()
        (trial_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        rows.append(summary)
        print(
            f"  {cfg.name}: partial={summary['partial_stable_score']:.6f} "
            f"full={summary['full_stable_score']:.6f} coverage={summary['coverage']}",
            flush=True,
        )
    return pd.DataFrame(rows)


def load_trial(name: str, output_dir: Path = OUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_dir = output_dir / "trials" / name
    return (
        pd.read_csv(trial_dir / "event_metrics.csv"),
        pd.read_csv(trial_dir / "ranking_metrics.csv"),
    )
