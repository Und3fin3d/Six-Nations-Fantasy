from __future__ import annotations

import json
import ssl
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

import certifi

SIX_NATIONS = 1266
TARGET_SEASONS = [2023, 2024, 2025, 2026]

DAILY_VARS = ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "precipitation_sum", "rain_sum", "wind_speed_10m_max", "wind_gusts_10m_max"]

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def _fixture_grid() -> pd.DataFrame:
    raw = pd.read_csv("data/api_team_match.csv")
    raw = raw[(raw["comp_id"] == SIX_NATIONS) & raw["season"].isin(TARGET_SEASONS)].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.date.astype(str)
    home = raw[raw["home_away"].astype(str).str.lower().eq("home")].copy()
    cols = ["season", "round", "fixture_id", "date", "team_id", "team", "opponent_id", "opponent"]
    home = home[cols].drop_duplicates("fixture_id")
    stadiums = pd.read_csv("data/sixn_stadiums.csv")
    out = home.merge(stadiums, on=["team_id", "team"], how="left")
    return out.sort_values(["date", "fixture_id"]).reset_index(drop=True)


def _fetch_open_meteo(lat: float, lon: float, day: str, tz_name: str) -> dict:
    params = {
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": day, "end_date": day, "daily": ",".join(DAILY_VARS), "timezone": tz_name or "auto",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    with urlopen(url, timeout=30, context=SSL_CONTEXT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    time.sleep(0.05)
    daily = payload.get("daily", {})
    return {k: (daily.get(k) or [np.nan])[0] for k in DAILY_VARS}


def build() -> pd.DataFrame:
    fixtures = _fixture_grid()
    rows = []
    source_ts = datetime.now(timezone.utc).isoformat()
    for r in fixtures.itertuples(index=False):
        weather = _fetch_open_meteo(float(r.latitude), float(r.longitude), str(r.date), str(r.timezone))
        rows.append(
            {
                "season": int(r.season), "round": int(r.round), "fixture_id": int(r.fixture_id), "date": r.date, "venue": r.venue,
                "weather_source_timestamp": source_ts, "weather_source_kind": "open_meteo_archive_daily", "weather_temp_c": weather.get("temperature_2m_mean"),
                "weather_temp_min_c": weather.get("temperature_2m_min"), "weather_temp_max_c": weather.get("temperature_2m_max"),
                "weather_rain_mm": weather.get("rain_sum"), "weather_precip_mm": weather.get("precipitation_sum"),
                "weather_wind_kph": weather.get("wind_speed_10m_max"), "weather_wind_gust_kph": weather.get("wind_gusts_10m_max"),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = build()
    out.to_csv("data/external_fixture_weather.csv", index=False)
    print(f"OK {len(out)} fixtures -> data/external_fixture_weather.csv")
    print(out[["season", "round", "fixture_id", "venue", "weather_temp_c", "weather_rain_mm", "weather_wind_kph"]].head(10).to_string(index=False))
