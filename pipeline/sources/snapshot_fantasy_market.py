from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://fantasy.sixnationsrugby.com"
OUTPUT = "data/fantasy_market_snapshots"
CARD_GAME_RE = re.compile(r"var\s+FS_CardGame\s*=\s*(\{.*?\})\s*;", re.DOTALL)


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    try:
        with urlopen(Request(url, headers={"User-Agent": "6n-fantasy-research/1.0", **(headers or {})}), timeout=30) as response:
            body, response_headers, status = response.read(), response.headers, response.status
    except HTTPError as exc:
        body, response_headers, status = exc.read(), exc.headers, exc.code
    return status, gzip.decompress(body) if response_headers.get("Content-Encoding", "").lower() == "gzip" else body


def discover_game(game: str) -> dict:
    game_url = f"{BASE_URL}/{game}/"
    status, page = _get(game_url)
    if status >= 400:
        raise RuntimeError(f"game page returned HTTP {status}")
    match = re.search(rf"/{re.escape(game)}/assets/client/card-game\.js[^\"']*", page.decode("utf-8"))
    if not match:
        raise RuntimeError(f"could not find card-game.js on {game_url}")
    card_url = BASE_URL + match.group(0)
    status, card_js = _get(card_url)
    if status >= 400:
        raise RuntimeError(f"card-game.js returned HTTP {status}")
    config_match = CARD_GAME_RE.search(card_js.decode("utf-8"))
    if not config_match:
        raise RuntimeError(f"could not parse FS_CardGame from {card_url}")
    return {**json.loads(config_match.group(1)), "game_url": game_url, "card_url": card_url}


def fetch_catalogue(config: dict) -> tuple[dict, int]:
    headers = {
        "Accept": "application/json", "Origin": BASE_URL, "Referer": str(config["game_url"]), "X-Access-Key": f"{config['identity']}@{config['version']}@",
    }
    status, text = _get(f"{str(config['api_url']).rstrip('/')}/public/sportifs?{urlencode({'lg': 'en'})}", headers)
    if status >= 400:
        return {}, status
    payload = json.loads(text.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("fantasy API returned a non-object catalogue")
    return payload, status


def write_snapshot(game: str, config: dict, payload: dict) -> tuple[str, str | None, int]:
    now = datetime.now(timezone.utc)
    base = f"{OUTPUT}/{game}_{now.strftime('%Y%m%dT%H%M%SZ')}"
    os.makedirs(OUTPUT, exist_ok=True)
    players = payload.get("sportifs", [])
    if not isinstance(players, list):
        raise RuntimeError("fantasy API payload has no list-valued 'sportifs' field")
    metadata = {
        "snapshot_utc": now.isoformat(), "game": game, "api_identity": str(config["identity"]), "api_version": str(config["version"]),
        "site_active": bool(config.get("site_actif", False)), "game_url": str(config["game_url"]), "card_url": str(config["card_url"]),
        "player_count": len(players),
    }
    open(f"{base}.json", "w").write(json.dumps({"metadata": metadata, "response": payload}, indent=2, sort_keys=True) + "\n")
    if not players:
        return f"{base}.json", None, 0
    meta_fields = ["snapshot_utc", "game", "api_identity", "api_version", "site_active"]
    scalar_fields = sorted({k for p in players for k, v in p.items() if v is None or isinstance(v, (str, int, float, bool))})
    with open(f"{base}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=meta_fields + scalar_fields)
        writer.writeheader()
        writer.writerows({**{k: metadata[k] for k in meta_fields}, **{k: p.get(k) for k in scalar_fields}} for p in players)
    return f"{base}.json", f"{base}.csv", len(players)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["m6n", "w6n"], default="m6n")
    ap.add_argument("--allow-unavailable", action="store_true")
    args = ap.parse_args()
    try:
        config = discover_game(args.game)
        payload, status = fetch_catalogue(config)
    except (URLError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    site_active = bool(config.get("site_actif", False))
    if status >= 400:
        print(f"catalogue unavailable: HTTP {status}; site_active={site_active}", file=sys.stderr)
        return 0 if args.allow_unavailable else 2
    raw_path, csv_path, count = write_snapshot(args.game, config, payload)
    print(f"saved raw snapshot: {raw_path}")
    if csv_path:
        print(f"saved player table: {csv_path}")
    print(f"players: {count}; site_active={site_active}")
    if count == 0 and not args.allow_unavailable:
        print("catalogue is empty; snapshots must be collected while the game is active", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
