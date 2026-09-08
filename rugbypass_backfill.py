#!/usr/bin/env python3
"""
RugbyPass backfill scraper  — close the API→RugbyPass coverage gap
==================================================================
The CSV-driven `rugbypass_batch.py` only sees players in 2025/2026
`players.csv`, so it never scraped the ~100 pre-2025 (retired / dropped) 6N
players that still appear in the API per-match table. This script targets
exactly those: it reads the *current* crosswalk to find every API player not
yet linked to RugbyPass, resolves each to a RugbyPass slug by probing
candidate URLs, scrapes the (server-rendered) player page with `requests`, and
writes `players/rugbypass_<slug>.json` in the same schema as
`rugbypass_batch.py` so `build_rugbypass_tables.py` picks them up unchanged.

No Selenium: the RugbyPass player page embeds its stats JSON directly in the
HTML (`#app-comp-stats`, `#app-competitions`, `.player-details`), so a plain
HTTP GET is enough and far faster.

Usage:
    python rugbypass_backfill.py            # resolve + scrape all unmatched
    python rugbypass_backfill.py --dry-run  # resolve slugs only, no scrape
    python rugbypass_backfill.py --force    # re-scrape even if JSON exists
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import time
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Reuse the exact parsers the batch scraper uses — do NOT reinvent.
from rugbypass_batch import (
    extract_bio,
    extract_comp_stats,
    extract_match_log,
    is_six_nations,
)

BASE = Path(__file__).parent
OUT_DIR = BASE / "players"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

# Hand-verified slugs for players whose RugbyPass URL is not the naive
# "first-last" of their API name (nicknames, concatenations, -N collisions).
# Filled in after the first probe pass; key = API player_name.
SLUG_OVERRIDES: dict[str, str] = {
    "Cameron Winnett": "cam-winnett",        # RugbyPass shortens Cameron -> Cam
    "Hame Faiva": "epalahame-faiva",         # RugbyPass uses his full first name
    "Oli Kebble": "oliver-kebble",            # RugbyPass uses Oliver, not Oli
    "Noah Nene": "noah-tisie-nene",           # RugbyPass includes his middle name
    "Mohamed Haouas":  "mohammed-haouas",    # RugbyPass spells it with double-m
    "Gaetan Barlot":   "gaeton-barlot",      # RugbyPass misspells Gaëtan -> Gaeton
    # discovered in the 2026 NCR backfill sweep (see data/ncr/rp_backfill_log.txt)
    "Andy Onyeama-Christie": "andy-christie",
    "Cobus Reinach": "jacobus-reinach",
    "Faf de Klerk": "francois-de-klerk",
    "Francisco Coria Marchetti": "francisco-coria",
    "Joe Carpenter": "joseph-carpenter",
    "Keiran Williams": "kieran-williams",
    "Nicholas Champion de Crespigny": "nick-champion-de-crespigny",
    "Pete Samu": "peter-samu",
    "Pita-Gus Sowakula": "pita-sowakula",
    "Sacha Feinberg-Mngomezulu": "sacha-mngomezulu",
    "Sam Whitelock": "samuel-whitelock",
    "Sam Matavesi": "samuel-matavesi",
    "Rob Leota": "robert-leota",
    "Isaac Kailea": "isaac-aedo-kailea",
    "Phepsi Buthelezi": "phendulani-buthelezi",
    "Lood de Jager": "lodewyk-de-jager",     # base slug is a bio-only stub page
    "Sebastian Negri": "sebastian-negri-da-ollegio",
    "Tadhg Beirne": "tadgh-beirne",          # RugbyPass misspells Tadhg
    "Ollie Chessum": "oliver-chessum",
    "Mack Hansen": "mackenzie-hansen",
}

# API players confirmed to have no RugbyPass page.
# key = API player_name, value = short note. Skipped without probing.
KNOWN_ABSENT: dict[str, str] = {}


def fold(s: str) -> str:
    """ASCII-fold + lowercase: 'Sébastien' -> 'sebastien'."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower()


def slug_candidates(name: str) -> list[str]:
    """Ordered RugbyPass slug guesses for an API player name."""
    base = fold(name).replace("'", "").replace(".", "")
    base = re.sub(r"[^a-z0-9\- ]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    hyphen = base.replace(" ", "-")
    cands = [hyphen]
    # collision suffixes RugbyPass uses when a name is taken (e.g. blair-murray-1)
    cands += [f"{hyphen}-1", f"{hyphen}-2"]
    # fully concatenated surname variant (RugbyPass occasionally does this,
    # e.g. manuelzuliani, gianmarcolucchesi)
    toks = base.split(" ")
    if len(toks) >= 2:
        cands.append("".join(toks))
    # de-duplicate, preserve order
    seen: set[str] = set()
    out = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def unmatched_api_players() -> list[dict]:
    """API players present in the crosswalk but not linked to RugbyPass."""
    import build_crosswalk as bc
    cw = bc.build()
    api = cw[cw["in_api"] & ~cw["in_rp"]]
    return (api[["api_name", "key", "canonical_pos"]]
            .sort_values("api_name")
            .to_dict("records"))


def fetch(session: requests.Session, slug: str):
    """Return (status_code, html_or_None) for a RugbyPass player slug."""
    url = f"https://www.rugbypass.com/players/{slug}/"
    try:
        r = session.get(url, timeout=25)
        return r.status_code, (r.text if r.status_code == 200 else None)
    except requests.RequestException as e:
        return None, f"ERR {e}"


def resolve_and_scrape(records, *, dry_run: bool, force: bool, delay: float):
    OUT_DIR.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    resolved: list[tuple[str, str, str]] = []   # (name, slug, nationality)
    unresolved: list[str] = []
    absent: list[str] = []

    for i, rec in enumerate(records, 1):
        name = rec["api_name"]
        key = rec["key"]

        if name in KNOWN_ABSENT:
            print(f"  [{i:>3}/{len(records)}]  {name:<26} — KNOWN ABSENT "
                  f"({KNOWN_ABSENT[name]})")
            absent.append(f"{name} — {KNOWN_ABSENT[name]}")
            continue

        cands = ([SLUG_OVERRIDES[name]] if name in SLUG_OVERRIDES
                 else slug_candidates(name))

        hit_slug = hit_html = None
        for slug in cands:
            stem = slug.replace("-", "_")
            json_path = OUT_DIR / f"rugbypass_{stem}.json"
            if json_path.exists() and not force and not dry_run:
                hit_slug = slug
                hit_html = None  # already have it
                print(f"  [{i:>3}/{len(records)}]  {name:<26} ⏭  have {json_path.name}")
                break
            status, html = fetch(session, slug)
            time.sleep(delay)
            if status == 200:
                hit_slug, hit_html = slug, html
                break

        if hit_slug is None:
            print(f"  [{i:>3}/{len(records)}]  {name:<26} ❌  no page "
                  f"(tried {', '.join(cands)})")
            unresolved.append(f"{name} ({key})  tried: {', '.join(cands)}")
            continue

        # parse (or note we already had it)
        if hit_html is None and hit_slug:
            # already scraped earlier; read nationality back for the report
            try:
                d = json.loads((OUT_DIR /
                                f"rugbypass_{hit_slug.replace('-', '_')}.json").read_text())
                nat = (d.get("bio") or {}).get("Nationality", "?")
            except Exception:
                nat = "?"
            resolved.append((name, hit_slug, nat))
            continue

        soup = BeautifulSoup(hit_html, "html.parser")
        bio = extract_bio(soup)
        comp_stats = extract_comp_stats(soup)
        match_log = extract_match_log(soup)
        nat = bio.get("Nationality", "?")
        sn = sum(1 for r in comp_stats if is_six_nations(r.get("competition")))

        print(f"  [{i:>3}/{len(records)}]  {name:<26} ✅  {hit_slug:<28} "
              f"nat={nat:<10} comp_rows={len(comp_stats)} 6n={sn}")

        if not dry_run:
            out = {
                "player": hit_slug,
                "csv_name": name,
                "bio": bio,
                "competition_stats": comp_stats,
                "match_log": match_log,
            }
            (OUT_DIR / f"rugbypass_{hit_slug.replace('-', '_')}.json").write_text(
                json.dumps(out, indent=2))
        resolved.append((name, hit_slug, nat))

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print(f"RESOLVED {len(resolved)} / {len(records)}   "
          f"unresolved {len(unresolved)}   known-absent {len(absent)}")
    print("=" * 74)
    if unresolved:
        print("\nUNRESOLVED (need a manual slug in SLUG_OVERRIDES, or truly absent):")
        for u in unresolved:
            print(f"  {u}")
    if absent:
        print("\nKNOWN ABSENT (irreducible misses):")
        for a in absent:
            print(f"  {a}")
    return resolved, unresolved, absent


def refresh_existing(*, delay: float):
    """Refresh every stored profile without replacing useful data with blanks."""
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    paths = sorted(OUT_DIR.glob("rugbypass_*.json"))
    results = []
    changed = unchanged = failed = 0
    retained = {"bio": 0, "competition_stats": 0, "match_log": 0}

    for i, path in enumerate(paths, 1):
        old = json.loads(path.read_text())
        slug = old.get("player") or path.stem.replace("rugbypass_", "").replace("_", "-")
        status, html = fetch(session, slug)
        time.sleep(delay)
        if status != 200 or not html:
            failed += 1
            results.append({"slug": slug, "status": "failed", "http_status": status})
            print(f"  [{i:>4}/{len(paths)}] {slug:<35} KEEP status={status}")
            continue

        soup = BeautifulSoup(html, "html.parser")
        fresh = {
            "player": slug,
            "csv_name": old.get("csv_name") or slug,
            "bio": extract_bio(soup),
            "competition_stats": extract_comp_stats(soup),
            "match_log": extract_match_log(soup),
        }
        retained_fields = []
        for field in retained:
            if not fresh[field] and old.get(field):
                fresh[field] = old[field]
                retained[field] += 1
                retained_fields.append(field)

        if fresh != old:
            path.write_text(json.dumps(fresh, indent=2))
            changed += 1
            outcome = "changed"
        else:
            unchanged += 1
            outcome = "unchanged"
        results.append({"slug": slug, "status": outcome,
                        "retained_fields": retained_fields})
        print(f"  [{i:>4}/{len(paths)}] {slug:<35} {outcome.upper()}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "RugbyPass player profiles",
        "mode": "full_existing_profile_refresh",
        "total_profiles": len(paths),
        "changed_profiles": changed,
        "unchanged_profiles": unchanged,
        "failed_profiles_kept": failed,
        "retained_nonempty_sections": retained,
        "results": results,
    }
    manifest_dir = BASE / "data"
    manifest_dir.mkdir(exist_ok=True)
    manifest_path = manifest_dir / f"rugbypass_refresh_{datetime.now(timezone.utc).date()}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"FULL REFRESH profiles={len(paths)} changed={changed} unchanged={unchanged} "
          f"failed_kept={failed} manifest={manifest_path}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Backfill RugbyPass for unmatched API players")
    ap.add_argument("--dry-run", action="store_true", help="resolve slugs only, no JSON written")
    ap.add_argument("--force", action="store_true", help="re-scrape even if JSON exists")
    ap.add_argument("--refresh-existing", action="store_true",
                    help="refresh every stored RugbyPass profile with degradation guards")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between requests")
    args = ap.parse_args()

    if args.refresh_existing:
        refresh_existing(delay=args.delay)
        return

    records = unmatched_api_players()
    print(f"{len(records)} unmatched API players to resolve.\n")
    resolve_and_scrape(records, dry_run=args.dry_run, force=args.force, delay=args.delay)


if __name__ == "__main__":
    main()
