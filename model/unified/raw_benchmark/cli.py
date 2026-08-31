"""Command-line interface for historical raw rugby benchmark v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import CORE_ENGINE_ORDER, ENGINE_ORDER, OUT
from .coverage import run_audit


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(required=True)
    audit = sub.add_parser("audit", help="build immutable corrected store and coverage ledger")
    audit.add_argument("--output", type=Path, default=OUT)
    audit.set_defaults(command="audit")
    run = sub.add_parser("run", help="fit raw candidates on historical tournament holdouts")
    run.add_argument("--output", type=Path, default=OUT)
    run.add_argument("--engines", nargs="+", choices=ENGINE_ORDER, default=CORE_ENGINE_ORDER)
    run.add_argument("--folds", nargs="*")
    run.set_defaults(command="run")
    report = sub.add_parser("report", help="aggregate metrics and apply historical selection gates")
    report.add_argument("--output", type=Path, default=OUT)
    report.add_argument("--events", type=Path)
    report.add_argument("--rankings", type=Path)
    report.set_defaults(command="report")
    challengers = sub.add_parser(
        "challengers-report", help="compare raw-eligible challengers without rewriting v1",
    )
    challengers.add_argument("--output", type=Path, default=OUT)
    challengers.set_defaults(command="challengers-report")
    champion = sub.add_parser(
        "champion-diagnostic", help="run the Six Nations champion's partial raw heads",
    )
    champion.add_argument("--output", type=Path, default=OUT)
    champion.set_defaults(command="champion-diagnostic")
    hillclimb = sub.add_parser(
        "p3-hillclimb",
        help="optimise competition-independent P3 event weights without refitting",
    )
    hillclimb.add_argument("--benchmark", type=Path, default=OUT)
    hillclimb.add_argument(
        "--output", type=Path, default=OUT.parents[1] / "p3_hillclimb",
    )
    hillclimb.add_argument(
        "--skip-official-ncr", action="store_true",
        help="skip the slower post-selection NCR GW1-3 diagnostic",
    )
    hillclimb.set_defaults(command="p3-hillclimb")
    checkpoint = sub.add_parser(
        "p3-checkpoint",
        help="reproduce the guarded retrospective P3 checkpoint search",
    )
    checkpoint.add_argument("--benchmark", type=Path, default=OUT)
    checkpoint.add_argument(
        "--base-config",
        type=Path,
        default=OUT.parents[1] / "p3_hillclimb" / "config.json",
    )
    checkpoint.add_argument(
        "--output", type=Path, default=OUT.parents[1] / "p3_checkpoint",
    )
    checkpoint.set_defaults(command="p3-checkpoint")
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.command == "audit":
        print(json.dumps(run_audit(args.output), indent=2, sort_keys=True))
    elif args.command == "run":
        from .runner import run_benchmark
        events, rankings = run_benchmark(
            output_dir=args.output, engines=tuple(args.engines),
            fold_labels=tuple(args.folds) if args.folds else None,
        )
        print(f"wrote {len(events):,} event metric rows and {len(rankings):,} ranking rows")
    elif args.command == "report":
        from .report import render_report
        print(json.dumps(render_report(args.output, args.events, args.rankings), indent=2))
    elif args.command == "challengers-report":
        from .challenge_report import render_challenger_report
        print(json.dumps(render_challenger_report(args.output), indent=2))
    elif args.command == "champion-diagnostic":
        from .champion_diagnostic import run_champion_diagnostic
        print(json.dumps(run_champion_diagnostic(args.output), indent=2))
    elif args.command == "p3-hillclimb":
        from .p3_hillclimb import run_hillclimb
        print(json.dumps(
            run_hillclimb(
                benchmark_dir=args.benchmark, output_dir=args.output,
                include_official_ncr=not args.skip_official_ncr,
            ),
            indent=2,
        ))
    elif args.command == "p3-checkpoint":
        from .p3_checkpoint import run_checkpoint
        print(json.dumps(
            run_checkpoint(
                benchmark_dir=args.benchmark,
                base_config_path=args.base_config,
                output_dir=args.output,
            ),
            indent=2,
        ))
    else:
        raise ValueError(f"unknown raw-benchmark command: {args.command}")


if __name__ == "__main__":
    main()
