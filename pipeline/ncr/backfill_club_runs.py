from __future__ import annotations

import argparse
import os
import time

from pipeline.sources.rugby_api import RugbyAPI

COMPS = {1236: "URC", 1218: "Prem", 1230: "Top14", 1464: "ChampCup", 1470: "ChalCup"}
DONE = ("Result", "Full Time")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", nargs="*", type=int, default=list(COMPS))
    ap.add_argument("--seasons", nargs="*", type=int, default=[2026, 2025, 2024, 2023])
    ap.add_argument("--floor", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    api = RugbyAPI(verbose=False, min_interval=0.7)
    print(f"=== club backfill: comps={args.comps} seasons={args.seasons} floor={args.floor} dry={args.dry_run} ===", flush=True)
    total = pending = 0
    for season in args.seasons:
        for comp in args.comps:
            label = COMPS.get(comp, str(comp))
            try:
                fixtures = api.fixtures(comp, season)
            except Exception as exc:
                print(f"{label} s{season}: fixtures unavailable ({exc})", flush=True)
                continue
            todo = [f for f in fixtures if f.get("status") in DONE and not os.path.exists(api._cache_path(f"/match/{f['id']}"))]
            if args.dry_run:
                pending += len(todo)
                print(f"{label} s{season}: {len(todo)} uncached", flush=True)
                continue
            got = 0
            for f in sorted(todo, key=lambda x: x.get("date") or ""):
                if api.remaining is not None and api.remaining <= args.floor:
                    print(f"QUOTA FLOOR at {label} s{season} ({got}/{len(todo)}); total={total}\n=== run end (floor) ===", flush=True)
                    return 0
                try:
                    api.match(f["id"])
                    got, total = got + 1, total + 1
                except RuntimeError as exc:
                    if "429" in str(exc):
                        time.sleep(8)
                    else:
                        print(f"  ERR {f['id']}: {str(exc)[:80]}", flush=True)
            print(f"{label} s{season}: fetched {got}/{len(todo)}  quota={api.remaining}", flush=True)
    print(
        f"=== dry run: {pending} matches uncached; quota={api.remaining} ==="
        if args.dry_run
        else f"=== run end: ALL DONE total={total} quota={api.remaining} ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
