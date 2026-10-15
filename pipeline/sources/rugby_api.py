from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

HOST = "rugby-live-data.p.rapidapi.com"
CACHE_DIR = "data/cache"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"


class RugbyAPI:
    def __init__(self, min_interval: float = 0.4, verbose: bool = True):
        self.key = os.environ.get("RUGBY_API_KEY")
        if not self.key:
            raise RuntimeError("RUGBY_API_KEY is not set. Export the RapidAPI key before running this command.")
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.min_interval, self.verbose, self._last_call = min_interval, verbose, 0.0
        self.remaining: int | None = None

    def _cache_path(self, path: str) -> str:
        return f"{CACHE_DIR}/{re.sub(r'[^A-Za-z0-9._-]+', '_', path.strip('/'))}.json"

    def get(self, path: str, refresh: bool = False) -> dict:
        cp = self._cache_path(path)
        if os.path.exists(cp) and not refresh:
            return json.load(open(cp))
        time.sleep(max(0.0, self.min_interval - (time.time() - self._last_call)))
        req = urllib.request.Request(
            f"https://{HOST}{path}", headers={"x-rapidapi-host": HOST, "x-rapidapi-key": self.key, "Content-Type": "application/json", "User-Agent": USER_AGENT}
        )
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
        open(cp, "w").write(raw)
        if self.verbose:
            print(f"  ↓ {path}   (quota left: {self.remaining})")
        return data

    def fixtures(self, comp: int, season: int) -> list[dict]:
        return self.get(f"/fixtures/{comp}/{season}").get("results", [])

    def match(self, match_id: int) -> dict:
        return self.get(f"/match/{match_id}").get("results", {})
