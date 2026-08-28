#!/usr/bin/env python3
"""Build optional Six Nations fixture weather features from Open-Meteo.

Output: data/external_fixture_weather.csv

The player model does not use these features by default.  Enable them with the
`external_weather_features_on` research candidate after this file has been built.
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover - only used on minimal Python installs
    certifi = None

BASE = Path(__file__).parent
DATA = BASE / "data"
SIX_NATIONS = 1266
TARGET_SEASONS = [2023, 2024, 2025, 2026]

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]

SSL_CONTEXT = (
    ssl.create_default_context(cafile=certifi.where())
    if certifi is not None else None
)


def _fixture_grid(team_csv: Path, stadium_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(team_csv)
    raw = raw[(raw["comp_id"] == SIX_NATIONS) & raw["season"].isin(TARGET_SEASONS)].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.date.astype(str)
    home = raw[raw["home_away"].astype(str).str.lower().eq("home")].copy()
    cols = [
        "season", "round", "fixture_id", "date", "team_id", "team",
        "opponent_id", "opponent",
    ]
    home = home[cols].drop_duplicates("fixture_id")
    stadiums = pd.read_csv(stadium_csv)
    out = home.merge(stadiums, on=["team_id", "team"], how="left", validate="many_to_one")
    if out[["latitude", "longitude"]].isna().any(axis=None):
        missing = out.loc[out["latitude"].isna() | out["longitude"].isna(), ["team", "team_id"]]
        raise ValueError(f"missing stadium coordinates: {missing.drop_duplicates().to_dict('records')}")
    return out.sort_values(["date", "fixture_id"]).reset_index(drop=True)


def _fetch_open_meteo(lat: float, lon: float, day: str, tz_name: str, sleep_s: float) -> dict:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": day,
        "end_date": day,
        "daily": ",".join(DAILY_VARS),
        "timezone": tz_name or "auto",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    with urlopen(url, timeout=30, context=SSL_CONTEXT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if sleep_s > 0:
        time.sleep(sleep_s)
    daily = payload.get("daily", {})
    return {k: (daily.get(k) or [np.nan])[0] for k in DAILY_VARS}


def build(team_csv: Path, stadium_csv: Path, sleep_s: float = 0.0) -> pd.DataFrame:
    fixtures = _fixture_grid(team_csv, stadium_csv)
    rows = []
    source_ts = datetime.now(timezone.utc).isoformat()
    for r in fixtures.itertuples(index=False):
        weather = _fetch_open_meteo(
            float(r.latitude), float(r.longitude), str(r.date), str(r.timezone), sleep_s
        )
        rows.append({
            "season": int(r.season),
            "round": int(r.round),
            "fixture_id": int(r.fixture_id),
            "date": r.date,
            "venue": r.venue,
            "weather_source_timestamp": source_ts,
            "weather_source_kind": "open_meteo_archive_daily",
            "weather_temp_c": weather.get("temperature_2m_mean"),
            "weather_temp_min_c": weather.get("temperature_2m_min"),
            "weather_temp_max_c": weather.get("temperature_2m_max"),
            "weather_rain_mm": weather.get("rain_sum"),
            "weather_precip_mm": weather.get("precipitation_sum"),
            "weather_wind_kph": weather.get("wind_speed_10m_max"),
            "weather_wind_gust_kph": weather.get("wind_gusts_10m_max"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-csv", default=str(DATA / "api_team_match.csv"))
    ap.add_argument("--stadium-csv", default=str(DATA / "sixn_stadiums.csv"))
    ap.add_argument("--sleep", type=float, default=0.05,
                    help="seconds to sleep between Open-Meteo requests")
    args = ap.parse_args()

    out = build(Path(args.team_csv), Path(args.stadium_csv), args.sleep)
    path = DATA / "external_fixture_weather.csv"
    out.to_csv(path, index=False)
    print(f"OK {len(out)} fixtures -> {path}")
    print(out[["season", "round", "fixture_id", "venue", "weather_temp_c",
               "weather_rain_mm", "weather_wind_kph"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
