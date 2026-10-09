"""Command-line interface for fitting and reporting P3 on historical folds."""

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
    else:
        raise ValueError(f"unknown raw-benchmark command: {args.command}")


if __name__ == "__main__":
    main()
