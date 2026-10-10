import argparse
import json

from .coverage import run_audit
from .runner import run_benchmark

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("run")
    if ap.parse_args().command == "audit":
        print(json.dumps(run_audit(), indent=2, sort_keys=True))
    else:
        events, rankings = run_benchmark()
        print(f"wrote {len(events):,} event metric rows and {len(rankings):,} ranking rows")
