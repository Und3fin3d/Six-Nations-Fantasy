#!/usr/bin/env python3
"""
backfill_window.py — quota-aware monthly maintenance of the pre-6N form window
===============================================================================
On the free RapidAPI tier (~250 calls/month) the Oct→Feb club window (~200-300
matches) cannot be pulled in one burst. Run this once a month (Nov, Dec, Jan):
it fetches every eligible completed match that is not yet in data/cache/,
oldest first, and stops cleanly at --budget live calls or the quota floor.
Everything fetched is cached permanently, so each run only pays for new matches.

Eligible = completed fixtures in [--window-start, --window-end) for the form-
window competitions (club comps + internationals) that feed FORM features.

Usage:
    python backfill_window.py --dry-run          # count remaining, spend ~6 fixtures calls
    python backfill_window.py                    # fetch up to 150 matches
    python backfill_window.py --budget 80        # tighter monthly budget
    # after the last run of the season:
    python ingest_6n.py --comps 1266 1296 1326 30 1218 1230 1236 1464 1470 --seasons ...
"""
from __future__ import annotations

import argparse
from datetime import date

from rugby_api import RugbyAPI

# form-window competitions (comp_id: label). Club comps use END-year season
# labels (2027 = the 2026/27 season); International uses the same convention.
WINDOW_COMPS = {
    1218: "Top 14",
    1230: "Premiership",
    1236: "United Rugby Championship",
    1464: "Champions Cup",
    1470: "Challenge Cup",
    30:   "International (autumn tests)",
}
QUOTA_FLOOR = 50            # never spend below this remaining quota
DONE = ("Result", "Full Time")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2027,
                    help="season label = 6N year (club seasons by end year)")
    ap.add_argument("--window-start", default=None,
                    help="YYYY-MM-DD (default: Oct 1 of season-1)")
    ap.add_argument("--window-end", default=None,
                    help="YYYY-MM-DD exclusive (default: Feb 5 of season)")
    ap.add_argument("--budget", type=int, default=150,
                    help="max live /match calls this run")
    ap.add_argument("--dry-run", action="store_true",
                    help="count uncached eligible matches, fetch nothing")
    args = ap.parse_args()

    w0 = args.window_start or f"{args.season - 1}-10-01"
    w1 = args.window_end or f"{args.season}-02-05"
    api = RugbyAPI(verbose=False)

    todo: list[tuple[str, int, str]] = []       # (date, match_id, comp label)
    cached = 0
    for comp, label in WINDOW_COMPS.items():
        try:
            fixtures = api.fixtures(comp, args.season)
        except RuntimeError as e:
            print(f"  {label}: fixtures unavailable ({e})")
            continue
        for f in fixtures:
            d = (f.get("date") or "")[:10]
            if f.get("status") in DONE and w0 <= d < w1:
                if api._cache_path(f"/match/{f['id']}").exists():
                    cached += 1
                else:
                    todo.append((d, f["id"], label))
    todo.sort()                                  # oldest first -> contiguous growth

    print(f"window {w0} .. {w1}  season {args.season}")
    print(f"eligible completed: {cached + len(todo)}  "
          f"(cached {cached}, to fetch {len(todo)})  quota left: {api.remaining}")
    if args.dry_run or not todo:
        if todo:
            print(f"next run at --budget {args.budget} would fetch "
                  f"{min(args.budget, len(todo))} and leave "
                  f"{max(0, len(todo) - args.budget)}")
        return 0

    fetched = 0
    for d, mid, label in todo:
        if fetched >= args.budget:
            print(f"budget {args.budget} reached")
            break
        if api.remaining is not None and api.remaining <= QUOTA_FLOOR:
            print(f"quota floor {QUOTA_FLOOR} reached (remaining {api.remaining})")
            break
        api.match(mid)
        fetched += 1
        if fetched % 25 == 0:
            print(f"  {fetched}/{min(args.budget, len(todo))}  "
                  f"(at {d}, quota {api.remaining})")
    left = len(todo) - fetched
    print(f"fetched {fetched} matches; {left} still uncached"
          + ("" if left == 0 else " — run again next month") +
          f"; quota left: {api.remaining}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
