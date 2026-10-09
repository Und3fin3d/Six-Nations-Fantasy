"""CLI for the P3 NCR shadow predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness import OUT
from .shadow import evaluate_prospective_shadows, freeze_shadow


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
    p = sub.add_parser("shadow", help="freeze a write-once NCR shadow prediction")
    p.add_argument("--gw", type=int, required=True)
    p.add_argument(
        "--engine", choices=("p3_event_50",), required=True,
    )
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, default=OUT / "shadow")
    p.set_defaults(func=shadow_command)
    p = sub.add_parser(
        "shadow-evaluate", help="apply the combined NCR GW4-GW7 promotion gates",
    )
    p.add_argument(
        "--engine", choices=("p3_event_50",), required=True,
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
