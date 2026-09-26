"""Rebuild event truth and availability from permanent match JSON caches."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from official_labels import match_official

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


def _hemisphere(team: object) -> str:
    from .config import NORTH, SOUTH
    if str(team) in NORTH:
        return "north"
    if str(team) in SOUTH:
        return "south"
    return "other"


def _minutes_observed(pid, started, on, stats) -> bool:
    if started or pid in on:
        return True
    return not any(
        _number((stats or {}).get(event)) > 0
        for event in STABLE_EVENTS if event not in {"yellow_cards", "red_cards"}
    )


def _fixture_event_coverage(metadata, players, teamsheet_players):
    n_stats = len(teamsheet_players)
    coverage = []
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
        observed = sum(player[f"available__{event}_cache"] for player in players)
        coverage.append({
            **metadata, "event": event, "status": "derived", "source_kind": "event_log",
            "players_total": len(players), "players_with_key": observed,
            "coverage_fraction": observed / len(players) if players else 0.0,
        })
    return coverage


def _cache_players(payload, metadata, match):
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
            minutes_observed = _minutes_observed(pid, started, on, stats)
            row = {
                **{key: value for key, value in metadata.items() if key != "date"},
                "player_id": stable_pid,
                "player_name_cache": player.get("name"),
                "team": team,
                "opponent_cache": match.get(f"{opponent_side}_team"),
                "team_id_cache": match.get(f"{side}_id"),
                "opponent_id_cache": match.get(f"{opponent_side}_id"),
                "team_score_cache": match.get(f"{side}_score"),
                "opp_score_cache": match.get(f"{opponent_side}_score"),
                "jersey_cache": player.get("position"),
                "started_cache": started,
                "minutes_cache": player_minutes(pid, started, off, on) if minutes_observed else np.nan,
                "yellow_cards_cache": cards[pid]["yellow_cards"],
                "red_cards_cache": cards[pid]["red_cards"],
                "available__minutes_cache": minutes_observed,
                "available__yellow_cards_cache": True,
                "available__red_cards_cache": True,
            }
            stats = stats or {}
            for event in DIRECT_EVENTS:
                present = event in stats
                row[f"{event}_cache"] = _number(stats.get(event)) if present else np.nan
                row[f"available__{event}_cache"] = present
            players.append(row)

    return players, teamsheet_players


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
        "match_status": match.get("status"),
    }
    players, teamsheet_players = _cache_players(payload, metadata, match)
    return metadata, players, _fixture_event_coverage(metadata, players, teamsheet_players)


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
    report['official_rows'] = int(len(official))
    six = out[pd.to_numeric(out['competition_id'], errors='coerce').eq(1266)].copy()
    fixture_keys = six[['season', 'round', 'team', 'fixture_id']].drop_duplicates()
    if fixture_keys.duplicated(['season', 'round', 'team']).any():
        raise ValueError('official Six Nations fixture map is ambiguous')
    labels = match_official(six, official)
    merged = labels.reindex(out.index)
    matched = pd.Series(False, index=out.index)
    for source_col, event in OFFICIAL_COLUMNS.items():
        values = pd.to_numeric(merged[source_col], errors="coerce")
        valid = values.notna()
        out.loc[valid, event] = values[valid].to_numpy(float)
        out.loc[valid, f"available__{event}"] = True
        out.loc[valid, f"provenance__{event}"] = "official_fixture_player"
        matched |= valid
    report["matched_rows"] = int(matched.sum())
    return out, report


def complete_cache_population(legacy, metadata, players, asof):
    keys = ["fixture_id", "player_id", "team"]
    inventory = metadata.copy()
    inventory['cached_players'] = inventory.fixture_id.map(players.groupby('fixture_id').size()).fillna(0).astype(int)
    inventory['included'] = (inventory.match_status.eq('Result')
        & inventory.match_at.lt(pd.Timestamp(asof, tz='UTC')) & inventory.cached_players.gt(0))
    included = inventory.loc[inventory.included, 'fixture_id']
    if not set(legacy.fixture_id).issubset(set(included)):
        raise ValueError('Canonical fixtures lack completed cached teamsheets before the research cutoff')
    players = players[players.fixture_id.isin(included)].copy()
    extra = players.merge(legacy[keys], on=keys, how='left', indicator=True, validate='one_to_one')
    extra = extra[extra._merge.eq('left_only')].copy()
    fields = ('player_name', 'opponent', 'team_id', 'opponent_id', 'team_score', 'opp_score', 'jersey', 'started')
    additions = extra[keys + [f'{name}_cache' for name in fields]].rename(
        columns={f'{name}_cache': name for name in fields})
    additions['date'] = extra.match_at.dt.tz_convert(None).dt.normalize()
    additions['competition_id'] = extra.competition_id_cache
    additions['competition'] = extra.competition_cache
    levels = legacy[['competition_id', 'competition_level']].drop_duplicates()
    if levels.competition_id.duplicated().any():
        raise ValueError('Competition level is ambiguous')
    additions['competition_level'] = additions.competition_id.map(levels.set_index('competition_id').competition_level)
    if additions.competition_level.isna().any():
        raise ValueError('Cached additions include an unclassified competition')
    additions['season'] = extra.calendar_year
    additions['round'] = extra.source_game_week
    additions['source'] = 'permanent_cache_teamsheet'
    additions['source_count'] = 1
    additions['source_priority'] = 0
    complete = pd.concat([legacy, additions], ignore_index=True, sort=False)
    if set(map(tuple, complete[keys].to_numpy())) != set(map(tuple, players[keys].to_numpy())):
        raise ValueError('Research population differs from completed cached teamsheets')
    report = {'asof_exclusive': asof, 'cached_player_rows_added': len(additions),
              'cached_fixtures_added': len(set(included) - set(legacy.fixture_id)),
              'inventory': json.loads(inventory.to_json(orient='records', date_format='iso'))}
    return complete, players, report


def build_corrected_store(legacy_path: Path = LEGACY_STORE, *, complete_asof=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    legacy = pd.read_csv(legacy_path, low_memory=False)
    legacy["fixture_id"] = legacy["fixture_id"].astype(str)
    legacy["player_id"] = legacy["player_id"].astype(str)
    fixture_ids = (set(path.stem.removeprefix('match_') for path in CACHE.glob('match_*.json'))
                   if complete_asof else set(legacy["fixture_id"]))
    metadata, cache_players, coverage = build_cache_tables(fixture_ids)
    population_report = {}
    canonical_ids = set(legacy.fixture_id)
    if complete_asof:
        legacy, cache_players, population_report = complete_cache_population(legacy, metadata, cache_players, complete_asof)
        metadata = metadata[metadata.fixture_id.isin(legacy.fixture_id)]
        coverage = coverage[coverage.fixture_id.isin(legacy.fixture_id)]
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
            "unobserved_entry_time" if target == "minutes" else "unobserved",
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
        "team_id_cache", "opponent_id_cache", "team_score_cache", "opp_score_cache",
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
        "canonical_fixtures": len(canonical_ids),
        "cached_fixtures": int(metadata["fixture_id"].nunique()),
        "international_fixtures": int(international["fixture_id"].nunique()),
        "official": official_report,
        "legacy_store": str(legacy_path),
        "legacy_store_sha256": sha256(legacy_path),
        "unobserved_rubric_events": list(UNOBSERVED_RUBRIC_EVENTS),
        "diagnostic_only_events": list(DIAGNOSTIC_ONLY_EVENTS),
        "population": population_report,
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
