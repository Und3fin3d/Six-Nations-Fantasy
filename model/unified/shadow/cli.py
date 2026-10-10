import argparse
import json

from .shadow import evaluate_prospective_shadows, freeze_shadow

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    shadow = sub.add_parser("shadow")
    shadow.add_argument("--gw", type=int, required=True)
    shadow.add_argument("--engine", choices=("p3_event_50",), required=True)
    shadow.add_argument("--model", required=True)
    sub.add_parser("shadow-evaluate").add_argument("--engine", choices=("p3_event_50",), required=True)
    args = ap.parse_args()
    if args.command == "shadow":
        print(f"froze immutable shadow at {freeze_shadow(args.gw, args.engine, args.model)}")
    else:
        print(json.dumps(evaluate_prospective_shadows(args.engine), indent=2, sort_keys=True))
