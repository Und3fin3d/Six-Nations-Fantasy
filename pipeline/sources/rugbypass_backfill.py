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

from pipeline.sources.rugbypass_batch import extract_bio, extract_comp_stats, extract_match_log, is_six_nations

BASE = "."
OUT_DIR = "players"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

SLUG_OVERRIDES: dict[str, str] = {
    "Cameron Winnett": "cam-winnett", "Hame Faiva": "epalahame-faiva", "Oli Kebble": "oliver-kebble", "Noah Nene": "noah-tisie-nene",
    "Mohamed Haouas": "mohammed-haouas", "Gaetan Barlot": "gaeton-barlot", "Andy Onyeama-Christie": "andy-christie", "Cobus Reinach": "jacobus-reinach",
    "Faf de Klerk": "francois-de-klerk", "Francisco Coria Marchetti": "francisco-coria", "Joe Carpenter": "joseph-carpenter",
    "Keiran Williams": "kieran-williams", "Nicholas Champion de Crespigny": "nick-champion-de-crespigny", "Pete Samu": "peter-samu",
    "Pita-Gus Sowakula": "pita-sowakula", "Sacha Feinberg-Mngomezulu": "sacha-mngomezulu", "Sam Whitelock": "samuel-whitelock",
    "Sam Matavesi": "samuel-matavesi", "Rob Leota": "robert-leota", "Isaac Kailea": "isaac-aedo-kailea", "Phepsi Buthelezi": "phendulani-buthelezi",
    "Lood de Jager": "lodewyk-de-jager", "Sebastian Negri": "sebastian-negri-da-ollegio", "Tadhg Beirne": "tadgh-beirne", "Ollie Chessum": "oliver-chessum",
    "Mack Hansen": "mackenzie-hansen",
}

KNOWN_ABSENT: dict[str, str] = {}


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def slug_candidates(name: str) -> list[str]:
    base = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\- ]", " ", fold(name).replace("'", "").replace(".", ""))).strip()
    hyphen = base.replace(" ", "-")
    return list(dict.fromkeys(c for c in [hyphen, f"{hyphen}-1", f"{hyphen}-2", *(["".join(base.split(" "))] if len(base.split(" ")) >= 2 else [])] if c))


def unmatched_api_players() -> list[dict]:
    import pipeline.sixn.build_crosswalk as bc

    cw = bc.build()
    return cw[cw["in_api"] & ~cw["in_rp"]][["api_name", "key", "canonical_pos"]].sort_values("api_name").to_dict("records")


def fetch(session: requests.Session, slug: str):
    try:
        r = session.get(f"https://www.rugbypass.com/players/{slug}/", timeout=25)
        return r.status_code, (r.text if r.status_code == 200 else None)
    except requests.RequestException as e:
        return None, f"ERR {e}"


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    return session


def _profile(slug: str, name: str, soup) -> dict:
    return {"player": slug, "csv_name": name, "bio": extract_bio(soup), "competition_stats": extract_comp_stats(soup), "match_log": extract_match_log(soup)}


def resolve_and_scrape(records, delay: float):
    session = _session()
    resolved, unresolved, absent = [], [], []
    for i, rec in enumerate(records, 1):
        name, key = rec["api_name"], rec["key"]
        if name in KNOWN_ABSENT:
            print(f"  [{i:>3}/{len(records)}]  {name:<26} KNOWN ABSENT ({KNOWN_ABSENT[name]})")
            absent.append(f"{name}: {KNOWN_ABSENT[name]}")
            continue
        cands = [SLUG_OVERRIDES[name]] if name in SLUG_OVERRIDES else slug_candidates(name)
        hit_slug = hit_html = None
        for slug in cands:
            json_path = f"{OUT_DIR}/rugbypass_{slug.replace('-', '_')}.json"
            if os.path.exists(json_path):
                hit_slug = slug
                print(f"  [{i:>3}/{len(records)}]  {name:<26} ⏭  have {os.path.basename(json_path)}")
                break
            status, html = fetch(session, slug)
            time.sleep(delay)
            if status == 200:
                hit_slug, hit_html = slug, html
                break
        if hit_slug is None:
            print(f"  [{i:>3}/{len(records)}]  {name:<26} ❌  no page (tried {', '.join(cands)})")
            unresolved.append(f"{name} ({key})  tried: {', '.join(cands)}")
            continue
        path = f"{OUT_DIR}/rugbypass_{hit_slug.replace('-', '_')}.json"
        if hit_html is None:
            resolved.append((name, hit_slug, (json.load(open(path)).get("bio") or {}).get("Nationality", "?")))
            continue
        out = _profile(hit_slug, name, BeautifulSoup(hit_html, "html.parser"))
        nat = out["bio"].get("Nationality", "?")
        print(
            f"  [{i:>3}/{len(records)}]  {name:<26} ✅  {hit_slug:<28} nat={nat:<10} comp_rows={len(out['competition_stats'])} "
            f"6n={sum(1 for r in out['competition_stats'] if is_six_nations(r.get('competition')))}"
        )
        open(path, "w").write(json.dumps(out, indent=2))
        resolved.append((name, hit_slug, nat))
    print("\n" + "=" * 74)
    print(f"RESOLVED {len(resolved)} / {len(records)}   unresolved {len(unresolved)}   known-absent {len(absent)}")
    print("=" * 74)
    for title, items in (("UNRESOLVED (need a manual slug in SLUG_OVERRIDES, or truly absent):", unresolved), ("KNOWN ABSENT (irreducible misses):", absent)):
        if items:
            print(f"\n{title}")
            for item in items:
                print(f"  {item}")


def refresh_existing(*, delay: float):
    session = _session()
    paths = sorted(glob.glob(f"{OUT_DIR}/rugbypass_*.json"))
    results = []
    changed = unchanged = failed = historical = 0
    retained = {"bio": 0, "competition_stats": 0, "match_log": 0}
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
        fresh = _profile(slug, old.get("csv_name") or slug, BeautifulSoup(html, "html.parser"))
        retained_fields = [field for field in retained if not fresh[field] and old.get(field)]
        for field in retained_fields:
            fresh[field] = old[field]
            retained[field] += 1
        keys = {(row.get("competition"), row.get("season")) for row in fresh["competition_stats"]}
        rows = [row for row in old.get("competition_stats", []) if (row.get("competition"), row.get("season")) not in keys]
        fresh["competition_stats"].extend(rows)
        historical += len(rows)
        outcome = "changed" if fresh != old else "unchanged"
        if outcome == "changed":
            open(path, "w").write(json.dumps(fresh, indent=2))
            changed += 1
        else:
            unchanged += 1
        results.append({"slug": slug, "status": outcome, "retained_fields": retained_fields})
        print(f"  [{i:>4}/{len(paths)}] {slug:<35} {outcome.upper()}")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": "RugbyPass player profiles", "mode": "full_existing_profile_refresh",
        "total_profiles": len(paths), "changed_profiles": changed, "unchanged_profiles": unchanged, "failed_profiles_kept": failed,
        "retained_nonempty_sections": retained, "retained_historical_competition_rows": historical, "results": results,
    }
    manifest_path = f"{BASE}/data/rugbypass_refresh_{datetime.now(timezone.utc).date()}.json"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    open(manifest_path, "w").write(json.dumps(manifest, indent=2))
    print(f"FULL REFRESH profiles={len(paths)} changed={changed} unchanged={unchanged} failed_kept={failed} manifest={manifest_path}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-existing", action="store_true")
    ap.add_argument("--delay", type=float, default=0.4)
    args = ap.parse_args()
    if args.refresh_existing:
        refresh_existing(delay=args.delay)
    else:
        records = unmatched_api_players()
        print(f"{len(records)} unmatched API players to resolve.\n")
        resolve_and_scrape(records, args.delay)
