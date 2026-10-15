from __future__ import annotations

import glob
import json
import os

import pandas as pd

from pipeline.sources.compare_three_way import slug_key

COMP_NUM_FIELDS = [
    "games", "minutes", "tries", "try_assists", "points", "carries", "metres", "av_gain", "post_contact_metres", "defenders_beaten", "clean_breaks", "offloads",
    "handling_errors", "passes", "bad_passes", "tackles", "missed_tackles", "tackles_success_pct", "dominant_tackles", "turnovers_won", "turnovers_conceded",
    "kicks_from_hand", "penalties_conceded", "yellow_cards", "red_cards",
]

LOG_NUM_FIELDS = ["mins", "tries", "conversions", "yellow_cards", "red_cards"]
LOG_BOOL_FIELDS = ["win", "draw"]


def _num(v):
    if v is None:
        return 0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    return int(f) if f == int(f) else f


def _parse_height_cm(v) -> int:
    if not isinstance(v, str):
        return _num(v)
    digits = "".join(ch for ch in v if ch.isdigit())
    return int(digits) if digits else 0


def _parse_weight_kg(v) -> int:
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
    for f in sorted(glob.glob("players/rugbypass_*.json")):
        d = json.load(open(f))
        slug = d.get("player") or os.path.basename(f).removesuffix(".json").replace("rugbypass_", "")
        yield slug, d


def build_bio(records) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "key": slug_key(slug), "slug": slug, "nationality": bio.get("Nationality") or "", "age": _parse_age(bio.get("Age")),
                "position": bio.get("Position") or "", "height_cm": _parse_height_cm(bio.get("Height")), "weight_kg": _parse_weight_kg(bio.get("Weight")),
            }
            for slug, bio in ((slug, d.get("bio") or {}) for slug, d in records)
        ]
    )


def build_compstats(records) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "key": slug_key(slug),
                "slug": slug,
                "competition": e.get("competition") or "",
                "season": e.get("season") or "",
                **{c: _num(e.get(c, 0)) for c in COMP_NUM_FIELDS},
            }
            for slug, d in records
            for e in d.get("competition_stats", [])
        ]
    )


def build_matchlog(records) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "key": slug_key(slug),
                "slug": slug,
                **{c: e.get(c) or "" for c in ("competition", "match", "date", "opposition")},
                **{c: _num(e.get(c, 0)) for c in LOG_NUM_FIELDS},
                **{c: bool(e.get(c, False)) for c in LOG_BOOL_FIELDS},
            }
            for slug, d in records
            for e in d.get("match_log", [])
        ]
    )
    iso = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")
    df["date"] = iso.dt.strftime("%Y-%m-%d").where(iso.notna(), "")
    return df


def main():
    records = list(load_json_files())
    print(f"loaded {len(records)} RugbyPass JSON files")
    comp = build_compstats(records)
    for name, df in (("rp_bio", build_bio(records)), ("rp_compstats", comp), ("rp_matchlog", build_matchlog(records))):
        df.to_csv(f"data/{name}.csv", index=False)
        print(
            f"💾  data/{name}.csv  rows={len(df)}  key non-null={100.0 * df['key'].notna().mean():.1f}%  "
            f"non-blank={100.0 * (df['key'].astype(str).str.len() > 0).mean():.1f}%"
        )

    print("\nrp_compstats sample (3 rows):")
    print(comp.head(3).to_string())


if __name__ == "__main__":
    main()
