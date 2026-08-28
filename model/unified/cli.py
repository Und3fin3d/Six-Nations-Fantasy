"""Command-line entrypoint for canonical build, training, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import ROOT, write_canonical_store
from .evaluation import evaluate_predictions
from .features import build_pit_features
from .gbdt import UniversalGBDT
from .neural import NeuralTrainingConfig, UniversalNeuralModel
from .scoring import scorer_for

DEFAULT_STORE = ROOT / "data/unified/player_match.csv"
DEFAULT_ARTIFACT = ROOT / "data/unified/models"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build(args) -> None:
    frame = write_canonical_store(args.output, asof=args.asof)
    print(f"wrote {len(frame):,} canonical rows ({frame.fixture_id.nunique():,} fixtures) to {args.output}")


def train(args) -> None:
    raw = pd.read_csv(args.store, low_memory=False, parse_dates=["date"])
    frame = build_pit_features(raw)
    cutoff = pd.Timestamp(args.asof)
    train_frame = frame[frame.date < cutoff].copy()
    if train_frame.empty:
        raise ValueError(f"no training rows before {cutoff.date()}")
    # Latest chronological slice is validation; fixture dates never cross folds.
    dates = sorted(train_frame.date.dropna().unique())
    validation_start = dates[max(1, int(len(dates) * .85)) - 1]
    fit_frame = train_frame[train_frame.date < validation_start]
    validation = train_frame[train_frame.date >= validation_start]
    if args.engine == "gbdt":
        model = UniversalGBDT().fit(fit_frame)
    else:
        config = NeuralTrainingConfig(epochs=args.epochs, batch_size=args.batch_size)
        model = UniversalNeuralModel(config=config).fit(fit_frame, validation)
    suffix = ".pkl" if args.engine == "gbdt" else ".pt"
    artifact = args.output or (DEFAULT_ARTIFACT / f"unified_{args.engine}_{cutoff.date()}{suffix}")
    model.save(artifact)
    manifest = {
        "schema_version": 1, "engine": args.engine, "asof": str(cutoff.date()),
        "created_at": datetime.now(timezone.utc).isoformat(), "training_rows": len(fit_frame),
        "validation_rows": len(validation), "validation_start": str(pd.Timestamp(validation_start).date()),
        "store_sha256": _sha(args.store), "seed": 17,
        "scoring_rules": {"ncr": 1, "six_nations": 1},
    }
    artifact.with_suffix(artifact.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"trained {args.engine} on {len(fit_frame):,} rows; artifact {artifact}")


def evaluate(args) -> None:
    raw = pd.read_csv(args.store, low_memory=False, parse_dates=["date"])
    frame = build_pit_features(raw)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    test = frame[(frame.date >= start) & (frame.date < end)].copy()
    if args.level:
        test = test[test["competition_level"].eq(args.level)]
    if args.competition_id is not None:
        test = test[pd.to_numeric(test["competition_id"], errors="coerce").eq(args.competition_id)]
    if test.empty:
        raise ValueError("evaluation filters selected no rows")
    if args.engine == "gbdt":
        model = UniversalGBDT.load(args.model)
    else:
        model = UniversalNeuralModel.load(args.model)
    predictions = model.predict_frame(test)
    metrics = evaluate_predictions(test, predictions, scorer_for(args.competition))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def predict(args) -> None:
    """Forecast arbitrary future candidate rows without changing either incumbent."""
    raw = pd.read_csv(args.store, low_memory=False, parse_dates=["date"])
    candidates = pd.read_csv(args.candidates, low_memory=False, parse_dates=["date"])
    required = {"date", "fixture_id", "player_id", "player_name", "team", "opponent",
                "position", "is_forward", "competition_level"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate input is missing columns: {missing}")
    candidates = candidates.copy()
    candidates["source"] = "prediction_input"
    candidates["source_priority"] = 999
    for target in ("minutes", *UniversalGBDT().events):
        candidates[target] = pd.NA
        candidates[f"available__{target}"] = False
    combined = pd.concat([raw, candidates], ignore_index=True, sort=False)
    features = build_pit_features(combined)
    future = features[features.source.eq("prediction_input")].copy()
    model = (UniversalGBDT.load(args.model) if args.engine == "gbdt"
             else UniversalNeuralModel.load(args.model))
    scorer = scorer_for(args.competition)
    rows = []
    for i, prediction in enumerate(model.predict_frame(future)):
        summary = scorer.score_prediction(prediction, n=args.samples, seed=args.seed + i)
        payload = prediction.to_dict()
        payload["fantasy_points"] = {
            "rules": scorer.name, "mean": summary.mean, "p10": summary.p10,
            "median": summary.median, "p90": summary.p90,
            "ceiling_probability": summary.ceiling_probability,
        }
        rows.append(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"wrote {len(rows)} raw distribution forecasts to {args.output}")


def train_stack(args) -> None:
    """Fit the deployable stage-2 stack on all labels, using the given stage-1."""
    from .gbdt import UniversalGBDT
    from .labels import build_fantasy_labels, write_fantasy_labels
    from .rank_stack import RankStack, build_stage2_features, load_store_features

    write_fantasy_labels(include_2023=args.include_2023)
    labels = build_fantasy_labels(include_2023=args.include_2023)
    if args.holdout_2026:
        labels = labels[~((labels.competition == "six_nations") & (labels.season == 2026))]
    stage1 = UniversalGBDT.load(args.stage1)
    store = load_store_features()
    features = build_stage2_features(labels, store, stage1)
    stack = RankStack().fit(features)
    artifact = args.output or (DEFAULT_ARTIFACT / "unified_rank_stack.pkl")
    stack.save(artifact)
    manifest = {
        "schema_version": 1, "kind": "rank_stack",
        "stage1": str(args.stage1), "label_rows": int(len(labels)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "competitions": sorted(labels.competition.unique().tolist()),
    }
    artifact.with_suffix(artifact.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"trained rank stack on {len(features):,} labelled rows; artifact {artifact}")


def benchmark_v2(args) -> None:
    from . import benchmark_v2 as bench
    bench.main()


def v3(args) -> None:
    from .v3.cli import main as v3_main
    import sys
    prior = sys.argv
    try:
        sys.argv = [f"{prior[0]} v3", *args.v3_args]
        v3_main()
    finally:
        sys.argv = prior


def raw_benchmark(args) -> None:
    from .raw_benchmark.cli import main as raw_main
    prior = sys.argv
    try:
        sys.argv = [f"{prior[0]} raw-benchmark", *args.raw_args]
        raw_main()
    finally:
        sys.argv = prior


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("build-data")
    p.add_argument("--asof")
    p.add_argument("--output", type=Path, default=DEFAULT_STORE)
    p.set_defaults(func=build)
    p = sub.add_parser("train")
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--asof", required=True)
    p.add_argument("--engine", choices=("gbdt", "neural"), default="neural")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--output", type=Path)
    p.set_defaults(func=train)
    p = sub.add_parser("evaluate")
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--engine", choices=("gbdt", "neural"), required=True)
    p.add_argument("--competition", choices=("ncr", "six_nations"), required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--level", choices=("club", "international"))
    p.add_argument("--competition-id", type=int)
    p.add_argument("--output", type=Path)
    p.set_defaults(func=evaluate)
    p = sub.add_parser("predict")
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--candidates", type=Path, required=True,
                   help="future player-fixture rows; targets are always cleared before use")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--engine", choices=("gbdt", "neural"), required=True)
    p.add_argument("--competition", choices=("ncr", "six_nations"), required=True)
    p.add_argument("--samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=predict)
    p = sub.add_parser("train-stack", help="fit the deployable stage-2 ranking stack")
    p.add_argument("--stage1", type=Path, required=True, help="stage-1 UniversalGBDT artifact")
    p.add_argument("--include-2023", action="store_true", help="include the different-era 2023 6N labels")
    p.add_argument("--holdout-2026", action="store_true", help="exclude sealed 6N 2026 from training")
    p.add_argument("--output", type=Path)
    p.set_defaults(func=train_stack)
    p = sub.add_parser("benchmark-v2", help="run the honest cross-competition stack benchmark")
    p.set_defaults(func=benchmark_v2)
    p = sub.add_parser(
        "v3", help="unified supermodel v3 audit/tune/train/benchmark/shadow",
        add_help=False,
    )
    p.add_argument("v3_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=v3)
    p = sub.add_parser(
        "raw-benchmark", help="audit/run/report historical stable and extended raw events",
        add_help=False,
    )
    p.add_argument("raw_args", nargs=argparse.REMAINDER)
    p.set_defaults(func=raw_benchmark)
    return ap


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "raw-benchmark":
        from .raw_benchmark.cli import main as raw_main
        prior = sys.argv
        try:
            sys.argv = [f"{prior[0]} raw-benchmark", *prior[2:]]
            raw_main()
        finally:
            sys.argv = prior
        return
    if len(sys.argv) > 1 and sys.argv[1] == "v3":
        from .v3.cli import main as v3_main
        prior = sys.argv
        try:
            sys.argv = [f"{prior[0]} v3", *prior[2:]]
            v3_main()
        finally:
            sys.argv = prior
        return
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
