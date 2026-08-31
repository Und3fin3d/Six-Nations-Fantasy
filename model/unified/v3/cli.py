"""CLI for unified rugby supermodel v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit import run_audit
from .benchmark import run_benchmark
from .harness import OUT
from .search import assess_blend, tune_baseline, tune_gbdt, tune_neural
from .shadow import evaluate_prospective_shadows, freeze_shadow
from .train import fit_frozen


def audit_command(args):
    print(json.dumps(run_audit(output_dir=args.output), indent=2, sort_keys=True))


def tune_command(args):
    if args.engine in {"gbdt", "all"}:
        baseline, _ = tune_baseline(args.output)
        print("best baseline:", json.dumps(baseline.to_dict(), sort_keys=True))
        best, _ = tune_gbdt(args.output)
        print("best GBDT:", json.dumps(best.to_dict(), sort_keys=True))
    if args.engine in {"neural", "all"}:
        best, _ = tune_neural(args.output)
        print("best neural:", json.dumps(best.to_dict(), sort_keys=True))
        if (args.output / "best_gbdt.json").exists():
            print("blend:", json.dumps(assess_blend(args.output), sort_keys=True))


def train_command(args):
    _, manifest = fit_frozen(
        args.engine, args.cutoff, args.output, force=args.force,
    )
    if args.activate_shadow:
        fixtures = pd.read_csv(OUT.parents[1] / "ncr" / "ncr_fixtures.csv")
        current = fixtures[pd.to_numeric(fixtures["iscurrent"], errors="coerce").eq(1)]
        active = {
            "engine": args.engine,
            "model": str(args.output),
            "exclusive_cutoff": manifest["exclusive_cutoff"],
            "artifact_sha256": manifest["artifact_sha256"],
            "target_gw": int(current["gameday"].iloc[0]) if len(current) else None,
        }
        path = OUT / "shadow_active.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(active, indent=2, sort_keys=True) + "\n")
        manifest["active_shadow_config"] = str(path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def benchmark_command(args):
    engines = tuple(args.engines)
    metrics = run_benchmark(args.output, engines)
    print(f"wrote {len(metrics)} fold-metric rows to {args.output}")


def shadow_command(args):
    path = freeze_shadow(args.gw, args.engine, args.model, output_dir=args.output)
    print(f"froze immutable shadow at {path}")


def shadow_evaluate_command(args):
    payload = evaluate_prospective_shadows(
        args.engine, rounds=tuple(args.rounds), shadow_dir=args.shadow_dir,
        output_dir=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(required=True)
    p = sub.add_parser("audit", help="audit scorer reconstruction and data eligibility")
    p.add_argument("--output", type=Path, default=OUT / "audit")
    p.set_defaults(func=audit_command)
    p = sub.add_parser("tune", help="run bounded rolling-origin tuning")
    p.add_argument("--engine", choices=("gbdt", "neural", "all"), default="all")
    p.add_argument("--output", type=Path, default=OUT / "search")
    p.set_defaults(func=tune_command)
    p = sub.add_parser("train", help="fit a frozen artifact through an exclusive cutoff")
    p.add_argument(
        "--engine", choices=("baseline", "gbdt_v3", "neural_v3", "blend_v3", "gbdt_v4"),
        required=True,
    )
    p.add_argument("--cutoff", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--activate-shadow", action="store_true",
        help="make this artifact the automatic NCR shadow model",
    )
    p.set_defaults(func=train_command)
    p = sub.add_parser("benchmark", help="official-points selection/reference benchmark")
    p.add_argument(
        "--engines", nargs="+",
        choices=("baseline", "gbdt_v3", "neural_v3", "blend_v3", "gbdt_v4"),
        default=("baseline", "gbdt_v3", "neural_v3"),
    )
    p.add_argument("--output", type=Path, default=OUT / "benchmark")
    p.set_defaults(func=benchmark_command)
    p = sub.add_parser("shadow", help="freeze a write-once NCR shadow prediction")
    p.add_argument("--gw", type=int, required=True)
    p.add_argument(
        "--engine", choices=(
            "baseline", "gbdt_v3", "neural_v3", "blend_v3", "gbdt_v4", "p3_event_50",
        ),
        required=True,
    )
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, default=OUT / "shadow")
    p.set_defaults(func=shadow_command)
    p = sub.add_parser(
        "shadow-evaluate", help="apply the combined NCR GW4-GW7 promotion gates",
    )
    p.add_argument(
        "--engine", choices=(
            "baseline", "gbdt_v3", "neural_v3", "blend_v3", "gbdt_v4", "p3_event_50",
        ),
        required=True,
    )
    p.add_argument("--rounds", type=int, nargs="+", default=(4, 5, 6, 7))
    p.add_argument("--shadow-dir", type=Path, default=OUT / "shadow")
    p.add_argument("--output", type=Path, default=OUT / "prospective")
    p.set_defaults(func=shadow_evaluate_command)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
