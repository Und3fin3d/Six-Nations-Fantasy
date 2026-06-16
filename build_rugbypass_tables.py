#!/usr/bin/env python3
"""
RugbyPass JSON  →  three tidy CSV tables  (0 API spend, pure local)
===================================================================
Walks every players/rugbypass_*.json (≈314 files) and emits:

    data/rp_bio.csv        one row / player   (key, slug, nationality, age,
                           position, height_cm, weight_kg)
    data/rp_compstats.csv  one row / (player, competition, season) — the rich
                           season aggregate (CLASS feature source)
    data/rp_matchlog.csv   one row / match_log entry — the thin per-match line
                           (mins/tries/conv/cards), date parsed to ISO

Every row is keyed by `norm_key` (column "key") and keeps the RugbyPass slug,
so it can be joined back to the official / API tables. Numerics are coerced;
missing → 0.

Reuses: slug_key / SLUG_TO_NAME (compare_three_way), norm_key
(compare_api_official). No network.

Usage:  python build_rugbypass_tables.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from compare_api_official import norm_key            # noqa: F401 (re-export)
from compare_three_way import SLUG_TO_NAME, slug_key  # noqa: F401

BASE = Path(__file__).parent

# numeric competition_stats fields to carry through (missing -> 0)
COMP_NUM_FIELDS = [
    "games", "minutes", "tries", "try_assists", "points",
    "carries", "metres", "av_gain", "post_contact_metres",
    "defenders_beaten", "clean_breaks", "offloads", "handling_errors",
    "passes", "bad_passes", "tackles", "missed_tackles",
    "tackles_success_pct", "dominant_tackles", "turnovers_won",
    "turnovers_conceded", "kicks_from_hand", "penalties_conceded",
    "yellow_cards", "red_cards",
]

# match_log fields (besides the joined keys / parsed date)
LOG_NUM_FIELDS = ["mins", "tries", "conversions", "yellow_cards", "red_cards"]
LOG_BOOL_FIELDS = ["win", "draw"]


def _num(v):
    """Coerce to int/float; None/blank/non-numeric -> 0."""
    if v is None:
        return 0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    return int(f) if f == int(f) else f


def _parse_height_cm(v) -> int:
    """'188cm' -> 188 ; missing -> 0."""
    if not isinstance(v, str):
        return _num(v)
    digits = "".join(ch for ch in v if ch.isdigit())
    return int(digits) if digits else 0


def _parse_weight_kg(v) -> int:
    """'106kg' -> 106 ; missing -> 0."""
    if not isinstance(v, str):
        return _num(v)
    digits = "".join(ch for ch in v if ch.isdigit())
    return int(digits) if digits else 0


def _parse_age(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def load_json_files():
    for f in sorted(glob.glob(str(BASE / "players" / "rugbypass_*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        slug = d.get("player") or Path(f).stem.replace("rugbypass_", "")
        yield slug, d


def build_bio(records) -> pd.DataFrame:
    rows = []
    for slug, d in records:
        bio = d.get("bio") or {}
        rows.append({
            "key": slug_key(slug),
            "slug": slug,
            "nationality": bio.get("Nationality") or "",
            "age": _parse_age(bio.get("Age")),
            "position": bio.get("Position") or "",
            "height_cm": _parse_height_cm(bio.get("Height")),
            "weight_kg": _parse_weight_kg(bio.get("Weight")),
        })
    return pd.DataFrame(rows)


def build_compstats(records) -> pd.DataFrame:
    rows = []
    for slug, d in records:
        key = slug_key(slug)
        for e in d.get("competition_stats", []):
            rec = {
                "key": key,
                "slug": slug,
                "competition": e.get("competition") or "",
                "season": e.get("season") or "",
            }
            for c in COMP_NUM_FIELDS:
                rec[c] = _num(e.get(c, 0))
            rows.append(rec)
    return pd.DataFrame(rows)


def build_matchlog(records) -> pd.DataFrame:
    rows = []
    for slug, d in records:
        key = slug_key(slug)
        for e in d.get("match_log", []):
            rec = {
                "key": key,
                "slug": slug,
                "competition": e.get("competition") or "",
                "match": e.get("match") or "",
                "date": e.get("date") or "",
                "opposition": e.get("opposition") or "",
            }
            for c in LOG_NUM_FIELDS:
                rec[c] = _num(e.get(c, 0))
            for c in LOG_BOOL_FIELDS:
                rec[c] = bool(e.get(c, False))
            rows.append(rec)
    df = pd.DataFrame(rows)
    if not df.empty:
        # "29 Nov 2025" -> ISO "2025-11-29"; unparseable -> "" (kept, not dropped)
        iso = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")
        df["date"] = iso.dt.strftime("%Y-%m-%d").where(iso.notna(), "")
    return df


def main():
    out_dir = BASE / "data"
    out_dir.mkdir(exist_ok=True)

    # materialize once; the generator is consumed three times
    records = list(load_json_files())
    print(f"loaded {len(records)} RugbyPass JSON files")

    bio = build_bio(records)
    comp = build_compstats(records)
    log = build_matchlog(records)

    for name, df in (("rp_bio", bio), ("rp_compstats", comp),
                     ("rp_matchlog", log)):
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        key_nonnull = 100.0 * df["key"].notna().mean() if len(df) else 0.0
        key_nonblank = 100.0 * (df["key"].astype(str).str.len() > 0).mean() if len(df) else 0.0
        print(f"💾  {path}  rows={len(df)}  "
              f"key non-null={key_nonnull:.1f}%  non-blank={key_nonblank:.1f}%")

    print("\nrp_compstats sample (3 rows):")
    print(comp.head(3).to_string())


if __name__ == "__main__":
    main()
