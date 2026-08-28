#!/usr/bin/env python3
"""
Rugby Live Data API client (RapidAPI: rugby-live-data.p.rapidapi.com)
====================================================================
Thin, rate-limit-aware client with permanent on-disk caching.

Why caching matters: the free plan allows only ~250 requests/month, and one
match call returns all 46 players, so we never want to re-fetch a match we
already have. Every successful GET is written to data/cache/ keyed by its path;
subsequent calls for the same path are served from disk for free.

Auth: reads the API key from the RUGBY_API_KEY environment variable. The client
stops before making a request when the variable is not set.

Usage:
    from rugby_api import RugbyAPI
    api = RugbyAPI()
    comps   = api.competitions()
    fixtures= api.fixtures(comp=1266, season=2026)   # Six Nations 2026
    match   = api.match(8523943)                     # France 36-14 Ireland
    table   = api.standings(comp=1266, season=2026)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import urllib.request
import urllib.error

HOST = "rugby-live-data.p.rapidapi.com"
BASE = f"https://{HOST}"
CACHE_DIR = Path(__file__).parent / "data" / "cache"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)

class RugbyAPI:
    def __init__(self, key: str | None = None, cache_dir: Path = CACHE_DIR,
                 min_interval: float = 0.4, verbose: bool = True):
        self.key = key or os.environ.get("RUGBY_API_KEY")
        if not self.key:
            raise RuntimeError(
                "RUGBY_API_KEY is not set. Export the RapidAPI key before running this command."
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.verbose = verbose
        self._last_call = 0.0
        self.remaining: int | None = None

    # ── low-level ────────────────────────────────────────────────────────────
    def _cache_path(self, path: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip("/"))
        return self.cache_dir / f"{safe}.json"

    def get(self, path: str, refresh: bool = False) -> dict:
        """GET an endpoint path (e.g. '/match/123'), with permanent disk cache."""
        cp = self._cache_path(path)
        if cp.exists() and not refresh:
            return json.loads(cp.read_text())

        # polite throttle between live calls
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

        url = BASE + path
        req = urllib.request.Request(url, headers={
            "x-rapidapi-host": HOST,
            "x-rapidapi-key": self.key,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                rem = resp.headers.get("x-ratelimit-requests-remaining")
                if rem is not None:
                    self.remaining = int(rem)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} for {path}: {e.read().decode()[:200]}")
        finally:
            self._last_call = time.time()

        data = json.loads(raw)
        if isinstance(data, dict) and data.get("message", "").startswith("Endpoint"):
            raise RuntimeError(f"Endpoint does not exist: {path}")
        cp.write_text(raw)
        if self.verbose:
            print(f"  ↓ {path}   (quota left: {self.remaining})")
        return data

    # ── typed convenience wrappers ───────────────────────────────────────────
    def competitions(self) -> list[dict]:
        return self.get("/competitions").get("results", [])

    def fixtures(self, comp: int, season: int) -> list[dict]:
        return self.get(f"/fixtures/{comp}/{season}").get("results", [])

    def fixtures_by_team(self, team_id: int) -> list[dict]:
        return self.get(f"/fixtures-by-team/{team_id}").get("results", [])

    def match(self, match_id: int) -> dict:
        return self.get(f"/match/{match_id}").get("results", {})

    def standings(self, comp: int, season: int) -> dict:
        return self.get(f"/standings/{comp}/{season}").get("results", {})


if __name__ == "__main__":
    api = RugbyAPI()
    comps = api.competitions()
    sixn = [c for c in comps if c["name"] == "Six Nations"]
    print(f"competitions: {len(comps)}  | Six Nations seasons: "
          f"{sorted({c['season'] for c in sixn})[:8]}...")
    print(f"quota remaining: {api.remaining}")
