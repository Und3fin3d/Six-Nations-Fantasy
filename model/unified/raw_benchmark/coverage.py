"""Rebuild event truth and availability from permanent match JSON caches."""

from __future__ import annotations

import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ingest_6n import derive_minutes, player_minutes

from ..data import _stable_id
from ..schema import EVENTS
from .config import (
    CACHE, DENSE_THRESHOLD, DERIVED_EVENTS, DIAGNOSTIC_ONLY_EVENTS, DIRECT_EVENTS,
    EXTENDED_EVENTS, LEGACY_STORE, OFFICIAL, OFFICIAL_COLUMNS, OUT, STABLE_EVENTS,
    UNOBSERVED_RUBRIC_EVENTS,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _name_key(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    parts = plain.strip().split()
    initial = parts[0][0].lower()
    surname = "".join(parts[1:] if len(parts) > 1 else parts)
    return initial + "|" + re.sub(r"[^a-z]", "", surname.lower())


def _hemisphere(team: object) -> str:
    from .config import NORTH, SOUTH
    if str(team) in NORTH:
        return "north"
    if str(team) in SOUTH:
        return "south"
    return "other"


def parse_cache_fixture(path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Return fixture metadata, player truth rows and fixture-event coverage."""
    payload = json.loads(path.read_text()).get("results", {})
    match = payload.get("match") or {}
    fixture_id = str(match.get("id") or path.stem.removeprefix("match_"))
    match_at = pd.to_datetime(match.get("date"), errors="coerce", utc=True)
    metadata = {
        "fixture_id": fixture_id,
        "match_at": match_at,
        "date": match_at.tz_convert(None).normalize() if pd.notna(match_at) else pd.NaT,
        "competition_id_cache": match.get("comp_id"),
        "competition_cache": match.get("comp_name"),
        "calendar_year": int(match_at.year) if pd.notna(match_at) else None,
        "source_season": match.get("season"),
        "source_game_week": match.get("game_week"),
        "source_round": match.get("round_id"),
    }
    off, on, cards = derive_minutes(payload.get("events") or [])
    players: list[dict] = []
    teamsheet_players: list[dict] = []
    seen_players: set[tuple[str, str]] = set()
    for side in ("home", "away"):
        opponent_side = "away" if side == "home" else "home"
        side_payload = payload.get(side) or {}
        for player in side_payload.get("teamsheet") or []:
            stats = player.get("match_stats")
            if stats is not None:
                teamsheet_players.append(player)
            pid = player.get("player_id")
            team = match.get(f"{side}_team")
            stable_pid = _stable_id(pid, player.get("name"), team)
            player_key = (str(team), stable_pid)
            if player_key in seen_players:
                continue
            seen_players.add(player_key)
            started = not bool(player.get("substitute", False))
            row = {
                **{key: value for key, value in metadata.items() if key != "date"},
                "player_id": stable_pid,
                "player_name_cache": player.get("name"),
                "team": team,
                "opponent_cache": match.get(f"{opponent_side}_team"),
                "jersey_cache": player.get("position"),
                "started_cache": started,
                "minutes_cache": player_minutes(pid, started, off, on),
                "yellow_cards_cache": cards[pid]["yellow_cards"],
                "red_cards_cache": cards[pid]["red_cards"],
                "available__minutes_cache": True,
                "available__yellow_cards_cache": True,
                "available__red_cards_cache": True,
            }
            stats = stats or {}
            for event in DIRECT_EVENTS:
                present = event in stats
                row[f"{event}_cache"] = _number(stats.get(event)) if present else np.nan
                row[f"available__{event}_cache"] = present
            players.append(row)

    n_stats = len(teamsheet_players)
    coverage: list[dict] = []
    for event in DIRECT_EVENTS:
        present = sum(event in (player.get("match_stats") or {}) for player in teamsheet_players)
        fraction = present / n_stats if n_stats else 0.0
        status = (
            "recorded" if fraction >= DENSE_THRESHOLD
            else "unobserved" if present == 0 else "ambiguous"
        )
        coverage.append({
            **metadata, "event": event, "status": status, "source_kind": "match_stats",
            "players_total": n_stats, "players_with_key": present,
            "coverage_fraction": fraction,
        })
    for event in DERIVED_EVENTS:
        coverage.append({
            **metadata, "event": event, "status": "derived", "source_kind": "event_log",
            "players_total": len(players), "players_with_key": len(players),
            "coverage_fraction": 1.0,
        })
    return metadata, players, coverage


def build_cache_tables(fixture_ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_rows, player_rows, coverage_rows = [], [], []
    missing = []
    for fixture_id in sorted(fixture_ids):
        path = CACHE / f"match_{fixture_id}.json"
        if not path.exists():
            missing.append(fixture_id)
            continue
        metadata, players, coverage = parse_cache_fixture(path)
        metadata_rows.append(metadata)
        player_rows.extend(players)
        coverage_rows.extend(coverage)
    if missing:
        raise FileNotFoundError(f"{len(missing)} canonical fixtures lack cache JSON: {missing[:10]}")
    return pd.DataFrame(metadata_rows), pd.DataFrame(player_rows), pd.DataFrame(coverage_rows)


def _attach_official_extensions(store: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = store.copy()
    for event in OFFICIAL_COLUMNS.values():
        out[event] = np.nan
        out[f"available__{event}"] = False
        out[f"provenance__{event}"] = "unobserved"
    report = {"official_rows": 0, "matched_rows": 0, "ambiguous_keys": 0}
    if not OFFICIAL.exists():
        return out, report
    official = pd.read_csv(OFFICIAL, low_memory=False)
    official["name_key"] = official["name"].map(_name_key)
    official["competition_id"] = 1266
    report["official_rows"] = int(len(official))

    six = out[pd.to_numeric(out["competition_id"], errors="coerce").eq(1266)].copy()
    six["name_key"] = six["player_name"].map(_name_key)
    fixture_keys = six[["season", "round", "team", "fixture_id"]].drop_duplicates()
    duplicate = fixture_keys.duplicated(["season", "round", "team"], keep=False)
    if duplicate.any():
        report["ambiguous_keys"] = int(duplicate.sum())
        raise ValueError("official Six Nations fixture map is ambiguous")
    official = official.merge(
        fixture_keys, on=["season", "round", "team"], how="left", validate="many_to_one",
    )
    official = official.dropna(subset=["fixture_id"])
    keep = ["fixture_id", "team", "name_key", *OFFICIAL_COLUMNS]
    official = official[keep].drop_duplicates(["fixture_id", "team", "name_key"])

    out["name_key"] = out["player_name"].map(_name_key)
    merged = out[["fixture_id", "team", "name_key"]].merge(
        official, on=["fixture_id", "team", "name_key"], how="left", validate="many_to_one",
    )
    matched = pd.Series(False, index=out.index)
    for source_col, event in OFFICIAL_COLUMNS.items():
        values = pd.to_numeric(merged[source_col], errors="coerce")
        valid = values.notna()
        out.loc[valid, event] = values[valid].to_numpy(float)
        out.loc[valid, f"available__{event}"] = True
        out.loc[valid, f"provenance__{event}"] = "official_fixture_player"
        matched |= valid
    report["matched_rows"] = int(matched.sum())
    return out.drop(columns="name_key"), report


def build_corrected_store(legacy_path: Path = LEGACY_STORE) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    legacy = pd.read_csv(legacy_path, low_memory=False)
    legacy["fixture_id"] = legacy["fixture_id"].astype(str)
    legacy["player_id"] = legacy["player_id"].astype(str)
    metadata, cache_players, coverage = build_cache_tables(set(legacy["fixture_id"]))
    keys = ["fixture_id", "player_id", "team"]
    if cache_players.duplicated(keys).any():
        raise ValueError("cache player truth is not unique at fixture/player grain")
    out = legacy.merge(cache_players, on=keys, how="left", validate="one_to_one")
    if out["match_at"].isna().any():
        raise ValueError("corrected store contains rows without exact cached kickoff")

    for target in ("minutes", "yellow_cards", "red_cards", *DIRECT_EVENTS):
        value_col = f"{target}_cache"
        mask_col = f"available__{target}_cache"
        out[target] = pd.to_numeric(out[value_col], errors="coerce")
        out[f"available__{target}"] = out[mask_col].fillna(False).astype(bool)
        out[f"provenance__{target}"] = np.where(
            out[f"available__{target}"],
            "event_log" if target in DERIVED_EVENTS else "match_stats_key",
            "unobserved",
        )
        out = out.drop(columns=[value_col, mask_col])

    metadata_columns = [
        "fixture_id", "match_at", "competition_id_cache", "competition_cache",
        "calendar_year", "source_season", "source_game_week", "source_round",
    ]
    # Metadata already arrived through cache_players; remove duplicate cache identity helpers.
    drop_helpers = [
        "player_name_cache", "team_cache", "opponent_cache", "jersey_cache", "started_cache",
        "date_cache",
    ]
    out = out.drop(columns=[c for c in drop_helpers if c in out], errors="ignore")
    out["match_at"] = pd.to_datetime(out["match_at"], utc=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["calendar_year"] = out["match_at"].dt.year.astype(int)
    out["hemisphere"] = out["team"].map(_hemisphere)
    out, official_report = _attach_official_extensions(out)

    for event in EVENTS:
        if event not in out:
            out[event] = np.nan
        if f"available__{event}" not in out:
            out[f"available__{event}"] = False
        unavailable = ~out[f"available__{event}"].fillna(False).astype(bool)
        out.loc[unavailable, event] = np.nan
    out["benchmark_row_id"] = np.arange(len(out), dtype=int)

    legacy_ids = set(legacy["fixture_id"])
    international = out[out["competition_level"].eq("international")]
    report = {
        "canonical_fixtures": len(legacy_ids),
        "cached_fixtures": int(metadata["fixture_id"].nunique()),
        "international_fixtures": int(international["fixture_id"].nunique()),
        "official": official_report,
        "legacy_store": str(legacy_path),
        "legacy_store_sha256": sha256(legacy_path),
        "unobserved_rubric_events": list(UNOBSERVED_RUBRIC_EVENTS),
        "diagnostic_only_events": list(DIAGNOSTIC_ONLY_EVENTS),
    }
    return out.sort_values(["match_at", "fixture_id", "team", "player_id"]), coverage, report


def _event_catalog(store: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    intl = store[store["competition_level"].eq("international")]
    for event in (*STABLE_EVENTS, *EXTENDED_EVENTS, *DIAGNOSTIC_ONLY_EVENTS):
        mask = intl[f"available__{event}"].fillna(False).astype(bool)
        fixture_mask = intl.assign(_available=mask).groupby("fixture_id")["_available"].any()
        rows.append({
            "event": event,
            "tier": (
                "stable" if event in STABLE_EVENTS else
                "extended" if event in EXTENDED_EVENTS else "diagnostic_only"
            ),
            "player_rows": int(mask.sum()),
            "player_coverage": float(mask.mean()),
            "fixtures": int(fixture_mask.sum()),
            "fixture_coverage": float(fixture_mask.mean()),
            "first_year": (
                int(intl.loc[mask, "calendar_year"].min()) if mask.any() else None
            ),
            "last_year": (
                int(intl.loc[mask, "calendar_year"].max()) if mask.any() else None
            ),
        })
    return pd.DataFrame(rows)


def run_audit(output_dir: Path = OUT) -> dict:
    store, coverage, report = build_corrected_store()
    eligible = store[
        store["competition_level"].eq("international")
        & store["match_at"].ge(pd.Timestamp("2022-01-01", tz="UTC"))
    ]
    failures = []
    if report["canonical_fixtures"] != 3327 or report["cached_fixtures"] != 3327:
        failures.append("expected 3,327/3,327 cached canonical fixtures")
    if report["international_fixtures"] != 458:
        failures.append("expected 458 cached international fixtures")
    eligible_fixtures = set(eligible["fixture_id"].astype(str))
    eligible_coverage = coverage[coverage["fixture_id"].astype(str).isin(eligible_fixtures)]
    for event in STABLE_EVENTS:
        supported = eligible_coverage[eligible_coverage["event"].eq(event)]
        expected_status = "derived" if event in DERIVED_EVENTS else "recorded"
        if (
            supported["fixture_id"].nunique() != len(eligible_fixtures)
            or not supported["status"].eq(expected_status).all()
        ):
            failures.append(f"stable event lacks complete eligible fixture coverage: {event}")
    catalog = _event_catalog(store, coverage)
    report["stable_events"] = list(STABLE_EVENTS)
    report["extended_events"] = list(EXTENDED_EVENTS)
    report["eligible_international_fixtures"] = int(eligible["fixture_id"].nunique())
    report["failures"] = failures
    report["passed"] = not failures

    if failures:
        raise AssertionError("; ".join(failures))

    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "player_match.csv"
    coverage_path = output_dir / "fixture_event_coverage.csv"
    catalog_path = output_dir / "event_catalog.csv"
    audit_path = output_dir / "audit.json"
    for path in (store_path, coverage_path, catalog_path, audit_path):
        if path.exists():
            raise FileExistsError(f"immutable raw benchmark artifact exists: {path}")
    store.to_csv(store_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    catalog.to_csv(catalog_path, index=False)
    report["artifacts"] = {
        "store": str(store_path), "coverage": str(coverage_path), "catalog": str(catalog_path),
    }
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
