from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

from pipeline.sources.rugbypass_batch import (
    extract_bio,
    extract_comp_stats,
    extract_match_log,
    is_six_nations,
)

BASE = "."
OUT_DIR = "players"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

SLUG_OVERRIDES: dict[str, str] = {
    "Cameron Winnett": "cam-winnett",
    "Hame Faiva": "epalahame-faiva",
    "Oli Kebble": "oliver-kebble",
    "Noah Nene": "noah-tisie-nene",
    "Mohamed Haouas":  "mohammed-haouas",
    "Gaetan Barlot":   "gaeton-barlot",
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
    "Lood de Jager": "lodewyk-de-jager",
    "Sebastian Negri": "sebastian-negri-da-ollegio",
    "Tadhg Beirne": "tadgh-beirne",
    "Ollie Chessum": "oliver-chessum",
    "Mack Hansen": "mackenzie-hansen",
}

KNOWN_ABSENT: dict[str, str] = {}


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower()


def slug_candidates(name: str) -> list[str]:
    base = fold(name).replace("'", "").replace(".", "")
    base = re.sub(r"[^a-z0-9\- ]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    hyphen = base.replace(" ", "-")
    cands = [hyphen]
    cands += [f"{hyphen}-1", f"{hyphen}-2"]
    toks = base.split(" ")
    if len(toks) >= 2:
        cands.append("".join(toks))
    seen: set[str] = set()
    out = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def unmatched_api_players() -> list[dict]:
    import pipeline.sixn.build_crosswalk as bc
    cw = bc.build()
    api = cw[cw["in_api"] & ~cw["in_rp"]]
    return (api[["api_name", "key", "canonical_pos"]]
            .sort_values("api_name")
            .to_dict("records"))


def fetch(session: requests.Session, slug: str):
    url = f"https://www.rugbypass.com/players/{slug}/"
    try:
        r = session.get(url, timeout=25)
        return r.status_code, (r.text if r.status_code == 200 else None)
    except requests.RequestException as e:
        return None, f"ERR {e}"


def resolve_and_scrape(records, *, dry_run: bool, force: bool, delay: float):
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    resolved: list[tuple[str, str, str]] = []
    unresolved: list[str] = []
    absent: list[str] = []

    for i, rec in enumerate(records, 1):
        name = rec["api_name"]
        key = rec["key"]

        if name in KNOWN_ABSENT:
            print(f"  [{i:>3}/{len(records)}]  {name:<26} KNOWN ABSENT "
                  f"({KNOWN_ABSENT[name]})")
            absent.append(f"{name}: {KNOWN_ABSENT[name]}")
            continue

        cands = ([SLUG_OVERRIDES[name]] if name in SLUG_OVERRIDES
                 else slug_candidates(name))

        hit_slug = hit_html = None
        for slug in cands:
            stem = slug.replace("-", "_")
            json_path = f"{OUT_DIR}/rugbypass_{stem}.json"
            if os.path.exists(json_path) and not force and not dry_run:
                hit_slug = slug
                hit_html = None
                print(f"  [{i:>3}/{len(records)}]  {name:<26} ⏭  have {os.path.basename(json_path)}")
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

        if hit_html is None and hit_slug:
            try:
                d = json.load(open(f"{OUT_DIR}/rugbypass_{hit_slug.replace('-', '_')}.json"))
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
            open(f"{OUT_DIR}/rugbypass_{hit_slug.replace('-', '_')}.json", "w").write(json.dumps(out, indent=2))
        resolved.append((name, hit_slug, nat))

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
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    paths = sorted(glob.glob(f"{OUT_DIR}/rugbypass_*.json"))
    results = []
    changed = unchanged = failed = 0
    retained = {"bio": 0, "competition_stats": 0, "match_log": 0}
    retained_historical_competition_rows = 0

    for i, path in enumerate(paths, 1):
        old = json.load(open(path))
        slug = old.get("player") or os.path.basename(path).removesuffix(".json").replace("rugbypass_", "").replace("_", "-")
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

        competition_keys = {
            (row.get("competition"), row.get("season"))
            for row in fresh["competition_stats"]
        }
        historical_rows = [
            row for row in old.get("competition_stats", [])
            if (row.get("competition"), row.get("season")) not in competition_keys
        ]
        fresh["competition_stats"].extend(historical_rows)
        retained_historical_competition_rows += len(historical_rows)

        if fresh != old:
            open(path, "w").write(json.dumps(fresh, indent=2))
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
        "retained_historical_competition_rows": retained_historical_competition_rows,
        "results": results,
    }
    manifest_path = f"{BASE}/data/rugbypass_refresh_{datetime.now(timezone.utc).date()}.json"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    open(manifest_path, "w").write(json.dumps(manifest, indent=2))
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
