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
    else:
        from .champion_diagnostic import run_champion_diagnostic
        print(json.dumps(run_champion_diagnostic(args.output), indent=2))


if __name__ == "__main__":
    main()
