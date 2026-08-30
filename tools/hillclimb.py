"""Greedy hill-climb harness for the all-rugby raw benchmark.

Runs one ``empirical_event`` variant at a time against the frozen ``v1``
baseline, inside a single working directory so that the fold manifests (which
record the store path) stay byte-stable across trials. ``v1`` artifacts are
fitted once and reused by every trial; only the candidate's artifacts are
cleared and re-computed.

Usage:
    python -m tools.hillclimb run <variant>
    python -m tools.hillclimb bundle <variant>
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "unified" / "raw_benchmark" / "allrugby_empirical"
TRIALS = WORK / "trials"
CANDIDATE = "empirical_event"
BASELINE = "v1"
KINDS = ("models", "predictions", "manifests", "metrics/events", "metrics/rankings")


def _candidate_dirs(base: Path) -> list[Path]:
    return [base / kind / CANDIDATE for kind in KINDS]


def _aggregate_files() -> list[Path]:
    return [
        path
        for pattern in ("event_metrics*.csv", "ranking_metrics*.csv")
        for path in WORK.glob(pattern)
    ] + [WORK / name for name in ("REPORT.md", "decision.json", "selection_summary.csv")]


def archive(variant: str) -> None:
    """Move the current candidate artifacts into trials/<variant>/."""
    target = TRIALS / variant
    target.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        source = WORK / kind / CANDIDATE
        if source.exists():
            destination = target / kind / CANDIDATE
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(source), str(destination))


def clean() -> None:
    """Remove candidate artifacts and aggregates, leaving v1 caches intact."""
    for path in _candidate_dirs(WORK):
        if path.exists():
            shutil.rmtree(path)
    for path in _aggregate_files():
        if path.exists():
            path.unlink()


def restore(variant: str) -> bool:
    """Put an archived trial's artifacts back so metrics can be re-read."""
    target = TRIALS / variant
    if not target.exists():
        return False
    for kind in KINDS:
        source = target / kind / CANDIDATE
        if source.exists():
            destination = WORK / kind / CANDIDATE
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
    return True


def _read_metrics(kind: str, engine: str) -> pd.DataFrame:
    paths = sorted((WORK / "metrics" / kind / engine).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no {kind} metrics for {engine}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False)


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.concat(
        [_read_metrics("events", engine) for engine in (BASELINE, CANDIDATE)],
        ignore_index=True, sort=False,
    )
    rankings = pd.concat(
        [_read_metrics("rankings", engine) for engine in (BASELINE, CANDIDATE)],
        ignore_index=True, sort=False,
    )
    return events, rankings


def bundle(variant: str) -> dict:
    """Full metric bundle for the current on-disk candidate vs v1."""
    from model.unified.raw_benchmark.report import (
        _bootstrap_difference, _mean_capture, _stable_scores, build_decision,
    )

    events, rankings = collect()
    decision, summary = build_decision(events, rankings)
    scores = _stable_scores(events)

    def _pivot(frame: pd.DataFrame, index) -> pd.DataFrame:
        return frame.pivot_table(index=index, columns="engine", values="stable_score")

    overall = scores.groupby("engine")["stable_score"].mean()
    by_fold = _pivot(scores, "fold")
    by_tournament = _pivot(scores, "tournament")
    north = _stable_scores(events, "north").groupby("engine")["stable_score"].mean()
    south = _stable_scores(events, "south").groupby("engine")["stable_score"].mean()

    stable_events = events[events["cohort"].eq("all") & events["tier"].isin(["stable", "minutes"])]
    by_target = stable_events.pivot_table(index="target", columns="engine", values="relative_loss")
    extended = events[events["cohort"].eq("all") & events["tier"].eq("extended")]
    by_extended = extended.pivot_table(index="target", columns="engine", values="loss")

    rank = rankings[rankings["tier"].eq("stable")].copy()
    rank["mean_capture"] = _mean_capture(rank)
    rank_agg = rank.groupby(["engine", "rubric"], as_index=False).agg(
        mae=("mae", "mean"), spearman=("spearman", "mean"),
        mean_capture=("mean_capture", "mean"), slates=("slate_id", "nunique"),
    )

    paired = by_fold[[CANDIDATE, BASELINE]].dropna()
    return {
        "variant": variant,
        "config": json.loads(_config_json(variant)),
        "stable_score": {engine: float(overall[engine]) for engine in overall.index},
        "north": {engine: float(north[engine]) for engine in north.index},
        "south": {engine: float(south[engine]) for engine in south.index},
        "by_fold": by_fold.to_dict(),
        "by_tournament": by_tournament.to_dict(),
        "by_target_relative_loss": by_target.to_dict(),
        "extended_loss": by_extended.to_dict(),
        "rubrics": rank_agg.to_dict(orient="records"),
        "bootstrap_vs_v1_by_fold": _bootstrap_difference(
            paired[CANDIDATE].to_numpy(float) - paired[BASELINE].to_numpy(float)
        ),
        "gate_reasons": decision["reasons"].get(CANDIDATE, []),
        "gate_passed": bool(decision["passed"].get(CANDIDATE, False)),
        "n_folds": int(len(paired)),
    }


def _config_json(variant: str) -> str:
    from dataclasses import asdict

    from model.unified.raw_benchmark.empirical import VARIANTS

    return json.dumps(asdict(VARIANTS[variant]), sort_keys=True)


def run(variant: str) -> dict:
    from model.unified.raw_benchmark.empirical import VARIANTS

    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    archived = TRIALS / variant
    if archived.exists():
        print(f"[{variant}] reusing archived trial")
        clean()
        restore(variant)
    else:
        clean()
        os.environ["RB_EMPIRICAL_VARIANT"] = "" if variant == "base" else variant
        from model.unified.raw_benchmark.runner import run_benchmark

        run_benchmark(output_dir=WORK, engines=(BASELINE, CANDIDATE))
    result = bundle(variant)
    archive(variant)
    for path in _aggregate_files():
        if path.exists():
            path.unlink()
    out = TRIALS / variant / "bundle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def adopt(variant: str) -> dict:
    """Bundle and archive candidate artifacts already on disk, without refitting.

    Used for the control, whose artifacts the initial benchmark run produced.
    """
    result = bundle(variant)
    archive(variant)
    for path in _aggregate_files():
        if path.exists():
            path.unlink()
    out = TRIALS / variant / "bundle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    command, variant = sys.argv[1], sys.argv[2]
    if command == "run":
        result = run(variant)
    elif command == "adopt":
        result = adopt(variant)
    elif command == "bundle":
        result = bundle(variant)
    else:
        raise SystemExit(f"unknown command {command!r}")
    print(json.dumps({
        key: result[key]
        for key in ("variant", "stable_score", "north", "south", "gate_passed", "gate_reasons")
    }, indent=2))


if __name__ == "__main__":
    main()
