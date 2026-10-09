from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parents[2])
from pipeline.sources.rugby_api import RugbyAPI

COMPS = {1236: "URC", 1218: "Prem", 1230: "Top14", 1464: "ChampCup", 1470: "ChalCup"}
DONE = ("Result", "Full Time")
LOG_PATH = Path("data/ncr/club_backfill_log.txt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", nargs="*", type=int, default=list(COMPS),
                    help=f"competition ids (default all): {COMPS}")
    ap.add_argument("--seasons", nargs="*", type=int, default=[2026, 2025, 2024, 2023],
                    help="season labels, newest first")
    ap.add_argument("--floor", type=int, default=60,
                    help="stop before quota drops to this (keeps a gameweek reserve)")
    ap.add_argument("--budget", type=int, default=100000,
                    help="max live /match calls this run")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what is missing; spend only cheap /fixtures calls")
    args = ap.parse_args()

    api = RugbyAPI(verbose=False, min_interval=0.7)
    log_f = LOG_PATH.open("a")

    def log(s: str) -> None:
        log_f.write(s + "\n"); log_f.flush(); print(s, flush=True)

    log(f"=== club backfill: comps={args.comps} seasons={args.seasons} "
        f"floor={args.floor} budget={args.budget} dry={args.dry_run} ===")

    total = 0
    pending = 0
    for season in args.seasons:
        for comp in args.comps:
            label = COMPS.get(comp, str(comp))
            try:
                fixtures = api.fixtures(comp, season)
            except Exception as exc:
                log(f"{label} s{season}: fixtures unavailable ({exc})")
                continue
            todo = [f for f in fixtures
                    if f.get("status") in DONE
                    and not api._cache_path(f"/match/{f['id']}").exists()]
            if args.dry_run:
                pending += len(todo)
                log(f"{label} s{season}: {len(todo)} uncached")
                continue

            got = 0
            for f in sorted(todo, key=lambda x: (x.get("date") or "")):
                if api.remaining is not None and api.remaining <= args.floor:
                    log(f"QUOTA FLOOR at {label} s{season} ({got}/{len(todo)}); total={total}")
                    log("=== run end (floor) ===")
                    return 0
                if total >= args.budget:
                    log(f"BUDGET {args.budget} reached at {label} s{season} ({got}/{len(todo)})")
                    log("=== run end (budget) ===")
                    return 0
                try:
                    api.match(f["id"]); got += 1; total += 1
                except RuntimeError as exc:
                    if "429" in str(exc):
                        time.sleep(8)
                    else:
                        log(f"  ERR {f['id']}: {str(exc)[:80]}")
            log(f"{label} s{season}: fetched {got}/{len(todo)}  quota={api.remaining}")

    if args.dry_run:
        log(f"=== dry run: {pending} matches uncached; quota={api.remaining} ===")
    else:
        log(f"=== run end: ALL DONE total={total} quota={api.remaining} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
