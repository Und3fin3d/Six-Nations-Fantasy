#!/usr/bin/env python3
"""Rebuild per-round Six Nations fantasy prices from public third-party mirrors.

The repo's official 6N workbooks (2023/2025/2026) carry per-round stats and
points but no per-round prices, so budget-legal squad evaluation on those rounds
is impossible. Public GitHub repositories of other fantasy players hold pulls of
the official game that include prices:

  2023  david-sykes/fantasy-rugby-streamlit  data/data.csv           price_round_1..5 + points_round_1..5
  2025  alexmgl/six_nations_solver           data/example_2025_gw1.csv (GW1 value + points)
                                             test.csv (round-4 catalogue: Value, stat_moy)
  2026  fnb-software/rugby-fantasy           2026/6nations/data/players.js
                                             (statsjoueur detail: valeuravant/valeurapres per round)

These are third-party copies (no licence stated); download them at run time for
research and do not commit them. Five requests in total.

Usage:
  python research/fantasy_sources/sixn_prices.py --cache DIR --out prices.csv \
      [--official data/official_player_match.csv]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fantasy_common import USER_AGENT, initial_key  # noqa: E402
from sixn_platform import tidy_detail  # noqa: E402

RAW = "https://raw." + "githubusercontent.com/"
MIRRORS = {
    "2023": "david-sykes/fantasy-rugby-streamlit/main/data/data.csv",
    "2025_gw1": "alexmgl/six_nations_solver/main/data/example_2025_gw1.csv",
    "2025_r4": "alexmgl/six_nations_solver/main/test.csv",
    "2026": "fnb-software/rugby-fantasy/main/2026/6nations/data/players.js",
}
FR_TEAMS = {"Angleterre": "England", "Ecosse": "Scotland", "Écosse": "Scotland", "Irlande": "Ireland",
            "Italie": "Italy", "Pays de Galles": "Wales", "Galles": "Wales", "France": "France"}


def fetch(name: str, cache: Path) -> bytes:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / Path(MIRRORS[name]).name.replace(".", f"_{name}.", 1)
    if not path.exists():
        req = urllib.request.Request(RAW + MIRRORS[name], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
    return path.read_bytes()


def js_module_to_json(text: str) -> list:
    """Convert a prettier-formatted ``export default [...]`` module to Python data."""
    body = text.strip()
    body = re.sub(r"^export default\s*", "", body).rstrip(";\n ")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    import shutil
    import subprocess
    import tempfile
    if not shutil.which("node"):
        raise RuntimeError("players.js is a JS literal; install node to convert it")
    with tempfile.TemporaryDirectory() as tmp:
        mod = Path(tmp) / "m.mjs"
        mod.write_text(text, encoding="utf-8")
        out = subprocess.run(["node", "-e", f"import('{mod.as_uri()}').then(m=>process.stdout.write(JSON.stringify(m.default)))"],
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout)


def prices_2023(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    rows = []
    for _, r in df.iterrows():
        for k in range(1, 6):
            rows.append({"season": 2023, "round": k, "team": r["team"], "name": r["name"],
                         "price": r[f"price_round_{k}"], "points": r[f"points_round_{k}"],
                         "source": MIRRORS["2023"]})
    return pd.DataFrame(rows)


def prices_2025(gw1: bytes, r4: bytes) -> pd.DataFrame:
    a = pd.read_csv(io.BytesIO(gw1), encoding="utf-8-sig")
    a = pd.DataFrame({"season": 2025, "round": 1, "team": a["Club"], "name": a["Name"], "price": a["Value"],
                      "points": a["Points"], "source": MIRRORS["2025_gw1"]})
    b = pd.read_csv(io.BytesIO(r4), encoding="utf-8-sig")
    b = pd.DataFrame({"season": 2025, "round": 4, "team": b["Club"].map(lambda t: FR_TEAMS.get(t, t)),
                      "name": b["Name"], "price": b["Value"], "points": pd.NA, "source": MIRRORS["2025_r4"]})
    return pd.concat([a, b], ignore_index=True)


def prices_2026(raw: bytes) -> pd.DataFrame:
    t = tidy_detail(js_module_to_json(raw.decode("utf-8")))
    return pd.DataFrame({"season": 2026, "round": t["round"], "team": t["team"], "name": t["short_name"],
                         "price": t["price_before"], "price_after": t["price_after"],
                         "points": t["points"].where(t["played"]), "source": MIRRORS["2026"]})


def join_official(prices: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """Share of official player-rounds that get a price, and points agreement."""
    off = official.assign(k=official["name"].map(initial_key))
    pr = prices.assign(k=prices["name"].map(initial_key)).drop_duplicates(["season", "round", "team", "k"])
    m = off.merge(pr, on=["season", "round", "team", "k"], how="left", suffixes=("", "_p"))
    out = []
    for (season, rnd), g in m.groupby(["season", "round"]):
        has = g["price"].notna()
        pts = pd.to_numeric(g.loc[has, "points"], errors="coerce")
        ok = pts.notna()
        out.append({"season": season, "round": rnd, "official_rows": len(g), "with_price": int(has.sum()),
                    "price_rate": round(float(has.mean()), 4),
                    "points_exact": round(float((abs(pts[ok] - g.loc[has, "Pts"][ok]) < 0.051).mean()), 4) if ok.any() else None})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--official", type=Path)
    args = ap.parse_args(argv)
    prices = pd.concat([
        prices_2023(fetch("2023", args.cache)),
        prices_2025(fetch("2025_gw1", args.cache), fetch("2025_r4", args.cache)),
        prices_2026(fetch("2026", args.cache)),
    ], ignore_index=True)
    prices.to_csv(args.out, index=False)
    print(prices.groupby(["season", "round"]).price.count().to_string())
    if args.official:
        print(join_official(prices, pd.read_csv(args.official)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
