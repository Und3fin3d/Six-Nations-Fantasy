#!/usr/bin/env python3
"""Snapshot the public Six Nations fantasy player catalogue.

The fantasy API only exposes the current catalogue. Run this after team sheets
and before each lock so price, ownership, availability, and any new public
fields are preserved point-in-time for future model research.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://fantasy.sixnationsrugby.com"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "fantasy_market_snapshots"
CARD_GAME_RE = re.compile(r"var\s+FS_CardGame\s*=\s*(\{.*?\})\s*;", re.DOTALL)


@dataclass(frozen=True)
class HttpResult:
    status_code: int
    body: bytes
    headers: Any

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.text)


def _get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> HttpResult:
    if params:
        url = f"{url}?{urlencode(params)}"
    request_headers = {"User-Agent": "6n-fantasy-research/1.0"}
    request_headers.update(headers or {})
    request = Request(url, headers=request_headers)
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read()
            response_headers = response.headers
            status = response.status
    except HTTPError as exc:
        body = exc.read()
        response_headers = exc.headers
        status = exc.code
    if response_headers.get("Content-Encoding", "").lower() == "gzip":
        body = gzip.decompress(body)
    return HttpResult(status_code=status, body=body, headers=response_headers)


def discover_game(game: str) -> dict[str, Any]:
    game_url = f"{BASE_URL}/{game}/"
    page_response = _get(game_url)
    if page_response.status_code >= 400:
        raise RuntimeError(f"game page returned HTTP {page_response.status_code}")
    page = page_response.text
    match = re.search(rf"/{re.escape(game)}/assets/client/card-game\.js[^\"']*", page)
    if not match:
        raise RuntimeError(f"could not find card-game.js on {game_url}")
    card_url = BASE_URL + match.group(0)
    card_response = _get(card_url)
    if card_response.status_code >= 400:
        raise RuntimeError(f"card-game.js returned HTTP {card_response.status_code}")
    card_js = card_response.text
    config_match = CARD_GAME_RE.search(card_js)
    if not config_match:
        raise RuntimeError(f"could not parse FS_CardGame from {card_url}")
    config = json.loads(config_match.group(1))
    config["game_url"] = game_url
    config["card_url"] = card_url
    return config


def fetch_catalogue(
    config: dict[str, Any],
    language: str,
) -> tuple[dict[str, Any], HttpResult]:
    identity = str(config["identity"])
    version = str(config["version"])
    api_url = str(config["api_url"]).rstrip("/")
    headers = {
        "Accept": "application/json",
        "Origin": BASE_URL,
        "Referer": str(config["game_url"]),
        "X-Access-Key": f"{identity}@{version}@",
    }
    response = _get(
        f"{api_url}/public/sportifs",
        params={"lg": language},
        headers=headers,
    )
    if response.status_code >= 400:
        return {}, response
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("fantasy API returned a non-object catalogue")
    return payload, response


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _csv_rows(
    players: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    scalar_fields = sorted(
        {
            key
            for player in players
            for key, value in player.items()
            if _scalar(value)
        }
    )
    metadata_fields = [
        "snapshot_utc",
        "game",
        "api_identity",
        "api_version",
        "site_active",
    ]
    rows = []
    for player in players:
        row = {key: metadata[key] for key in metadata_fields}
        row.update({key: player.get(key) for key in scalar_fields})
        rows.append(row)
    return metadata_fields + scalar_fields, rows


def write_snapshot(
    output_dir: Path,
    game: str,
    config: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[Path, Path | None, int]:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"{game}_{stamp}"
    players = payload.get("sportifs", [])
    if not isinstance(players, list):
        raise RuntimeError("fantasy API payload has no list-valued 'sportifs' field")

    metadata = {
        "snapshot_utc": now.isoformat(),
        "game": game,
        "api_identity": str(config["identity"]),
        "api_version": str(config["version"]),
        "site_active": bool(config.get("site_actif", False)),
        "game_url": str(config["game_url"]),
        "card_url": str(config["card_url"]),
        "player_count": len(players),
    }
    raw_path = base.with_suffix(".json")
    raw_path.write_text(
        json.dumps({"metadata": metadata, "response": payload}, indent=2, sort_keys=True)
        + "\n"
    )

    csv_path = None
    if players:
        fields, rows = _csv_rows(players, metadata)
        csv_path = base.with_suffix(".csv")
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return raw_path, csv_path, len(players)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=["m6n", "w6n"], default="m6n")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="Return success when the closed-season API is unavailable or empty.",
    )
    args = parser.parse_args()

    try:
        config = discover_game(args.game)
        payload, response = fetch_catalogue(config, args.language)
    except (URLError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if response.status_code >= 400:
        print(
            f"catalogue unavailable: HTTP {response.status_code}; "
            f"site_active={bool(config.get('site_actif', False))}",
            file=sys.stderr,
        )
        return 0 if args.allow_unavailable else 2

    raw_path, csv_path, count = write_snapshot(
        args.output_dir, args.game, config, payload
    )
    print(f"saved raw snapshot: {raw_path}")
    if csv_path:
        print(f"saved player table: {csv_path}")
    print(f"players: {count}; site_active={bool(config.get('site_actif', False))}")
    if count == 0 and not args.allow_unavailable:
        print(
            "catalogue is empty; snapshots must be collected while the game is active",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
