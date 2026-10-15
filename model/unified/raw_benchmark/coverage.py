from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from pipeline.sixn.ingest_6n import derive_minutes, player_minutes
from pipeline.sixn.official_labels import match_official

from ..data import _stable_id
from ..schema import EVENTS
from .config import (
    CACHE,
    DENSE_THRESHOLD,
    DERIVED_EVENTS,
    DIAGNOSTIC_ONLY_EVENTS,
    DIRECT_EVENTS,
    EXTENDED_EVENTS,
    LEGACY_STORE,
    NORTH,
    OFFICIAL,
    OFFICIAL_COLUMNS,
    OUT,
    SOUTH,
    STABLE_EVENTS,
    UNOBSERVED_RUBRIC_EVENTS,
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_cache_fixture(path: str) -> tuple[dict, list[dict], list[dict]]:
    payload = json.load(open(path)).get("results", {})
    match = payload.get("match") or {}
    match_at = pd.to_datetime(match.get("date"), errors="coerce", utc=True)
    known = pd.notna(match_at)
    metadata = {
        "fixture_id": str(match.get("id") or os.path.basename(path).removesuffix(".json").removeprefix("match_")), "match_at": match_at,
        "date": match_at.tz_convert(None).normalize() if known else pd.NaT, "competition_id_cache": match.get("comp_id"),
        "competition_cache": match.get("comp_name"), "calendar_year": int(match_at.year) if known else None, "source_season": match.get("season"),
        "source_game_week": match.get("game_week"), "source_round": match.get("round_id"), "match_status": match.get("status"),
    }
    off, on, cards = derive_minutes(payload.get("events") or [])
    players, teamsheet_players, seen = [], [], set()
    for side, opp in (("home", "away"), ("away", "home")):
        team = match.get(f"{side}_team")
        for player in (payload.get(side) or {}).get("teamsheet") or []:
            if player.get("match_stats") is not None:
                teamsheet_players.append(player)
            pid = player.get("player_id")
            stable_pid = _stable_id(pid, player.get("name"), team)
            if (str(team), stable_pid) in seen:
                continue
            seen.add((str(team), stable_pid))
            started = not bool(player.get("substitute", False))
            stats = player.get("match_stats") or {}
            players.append(
                {
                    **{k: v for k, v in metadata.items() if k != "date"},
                    "player_id": stable_pid,
                    "player_name_cache": player.get("name"),
                    "team": team,
                    "opponent_cache": match.get(f"{opp}_team"),
                    "team_id_cache": match.get(f"{side}_id"),
                    "opponent_id_cache": match.get(f"{opp}_id"),
                    "team_score_cache": match.get(f"{side}_score"),
                    "opp_score_cache": match.get(f"{opp}_score"),
                    "jersey_cache": player.get("position"),
                    "started_cache": started,
                    "minutes_cache": player_minutes(pid, started, off, on),
                    "yellow_cards_cache": cards[pid]["yellow_cards"],
                    "red_cards_cache": cards[pid]["red_cards"],
                    "available__minutes_cache": True,
                    "available__yellow_cards_cache": True,
                    "available__red_cards_cache": True,
                    **{
                        k: v
                        for event in DIRECT_EVENTS
                        for k, v in ((f"{event}_cache", _number(stats[event]) if event in stats else np.nan), (f"available__{event}_cache", event in stats))
                    },
                }
            )
    n_stats = len(teamsheet_players)
    coverage = []
    for event in DIRECT_EVENTS:
        present = sum(event in (player.get("match_stats") or {}) for player in teamsheet_players)
        fraction = present / n_stats if n_stats else 0.0
        status = "recorded" if fraction >= DENSE_THRESHOLD else "unobserved" if present == 0 else "ambiguous"
        coverage.append(
            {
                **metadata,
                "event": event,
                "status": status,
                "source_kind": "match_stats",
                "players_total": n_stats,
                "players_with_key": present,
                "coverage_fraction": fraction,
            }
        )
    coverage += [
        {
            **metadata,
            "event": event,
            "status": "derived",
            "source_kind": "event_log",
            "players_total": len(players),
            "players_with_key": len(players),
            "coverage_fraction": 1.0,
        }
        for event in DERIVED_EVENTS
    ]
    return metadata, players, coverage


def _attach_official_extensions(store: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = store.copy()
    for event in OFFICIAL_COLUMNS.values():
        out[event], out[f"available__{event}"], out[f"provenance__{event}"] = np.nan, False, "unobserved"
    official = pd.read_csv(OFFICIAL, low_memory=False)
    merged = match_official(out[pd.to_numeric(out["competition_id"], errors="coerce").eq(1266)].copy(), official).reindex(out.index)
    matched = pd.Series(False, index=out.index)
    for source_col, event in OFFICIAL_COLUMNS.items():
        values = pd.to_numeric(merged[source_col], errors="coerce")
        valid = values.notna()
        out.loc[valid, event] = values[valid].to_numpy(float)
        out.loc[valid, f"available__{event}"] = True
        out.loc[valid, f"provenance__{event}"] = "official_fixture_player"
        matched |= valid
    return out, {"official_rows": int(len(official)), "matched_rows": int(matched.sum()), "ambiguous_keys": 0}


def build_corrected_store() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    legacy = pd.read_csv(LEGACY_STORE, low_memory=False)
    legacy["fixture_id"], legacy["player_id"] = legacy["fixture_id"].astype(str), legacy["player_id"].astype(str)
    parsed = [parse_cache_fixture(f"{CACHE}/match_{fixture_id}.json") for fixture_id in sorted(set(legacy["fixture_id"]))]
    metadata = pd.DataFrame([m for m, _, _ in parsed])
    coverage = pd.DataFrame([row for _, _, c in parsed for row in c])
    out = legacy.merge(pd.DataFrame([row for _, p, _ in parsed for row in p]), on=["fixture_id", "player_id", "team"], how="left")
    for target in ("minutes", "yellow_cards", "red_cards", *DIRECT_EVENTS):
        out[target] = pd.to_numeric(out[f"{target}_cache"], errors="coerce")
        out[f"available__{target}"] = out[f"available__{target}_cache"].fillna(False).astype(bool)
        out[f"provenance__{target}"] = np.where(out[f"available__{target}"], "event_log" if target in DERIVED_EVENTS else "match_stats_key", "unobserved")
        out = out.drop(columns=[f"{target}_cache", f"available__{target}_cache"])
    out = out.drop(
        columns=[
            "player_name_cache", "opponent_cache", "jersey_cache", "started_cache", "team_id_cache", "opponent_id_cache", "team_score_cache", "opp_score_cache",
        ]
    )
    out["match_at"] = pd.to_datetime(out["match_at"], utc=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["calendar_year"] = out["match_at"].dt.year.astype(int)
    out["hemisphere"] = out["team"].map(lambda team: "north" if str(team) in NORTH else "south" if str(team) in SOUTH else "other")
    out, official_report = _attach_official_extensions(out)
    for event in EVENTS:
        out.loc[~out[f"available__{event}"].fillna(False).astype(bool), event] = np.nan
    out["benchmark_row_id"] = np.arange(len(out), dtype=int)
    report = {
        "canonical_fixtures": len(set(legacy["fixture_id"])), "cached_fixtures": int(metadata["fixture_id"].nunique()),
        "international_fixtures": int(out[out["competition_level"].eq("international")]["fixture_id"].nunique()), "official": official_report,
        "legacy_store": LEGACY_STORE, "legacy_store_sha256": sha256(LEGACY_STORE), "unobserved_rubric_events": list(UNOBSERVED_RUBRIC_EVENTS),
        "diagnostic_only_events": list(DIAGNOSTIC_ONLY_EVENTS), "population": {},
    }
    return out.sort_values(["match_at", "fixture_id", "team", "player_id"]), coverage, report


def _event_catalog(store: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    intl = store[store["competition_level"].eq("international")]
    for event in (*STABLE_EVENTS, *EXTENDED_EVENTS, *DIAGNOSTIC_ONLY_EVENTS):
        mask = intl[f"available__{event}"].fillna(False).astype(bool)
        fixture_mask = intl.assign(_available=mask).groupby("fixture_id")["_available"].any()
        rows.append(
            {
                "event": event,
                "tier": ("stable" if event in STABLE_EVENTS else "extended" if event in EXTENDED_EVENTS else "diagnostic_only"),
                "player_rows": int(mask.sum()),
                "player_coverage": float(mask.mean()),
                "fixtures": int(fixture_mask.sum()),
                "fixture_coverage": float(fixture_mask.mean()),
                "first_year": (int(intl.loc[mask, "calendar_year"].min()) if mask.any() else None),
                "last_year": (int(intl.loc[mask, "calendar_year"].max()) if mask.any() else None),
            }
        )
    return pd.DataFrame(rows)


def run_audit() -> dict:
    store, coverage, report = build_corrected_store()
    eligible = store[store["competition_level"].eq("international") & store["match_at"].ge(pd.Timestamp("2022-01-01", tz="UTC"))]
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
        if supported["fixture_id"].nunique() != len(eligible_fixtures) or not supported["status"].eq(expected_status).all():
            failures.append(f"stable event lacks complete eligible fixture coverage: {event}")
    catalog = _event_catalog(store, coverage)
    report["stable_events"] = list(STABLE_EVENTS)
    report["extended_events"] = list(EXTENDED_EVENTS)
    report["eligible_international_fixtures"] = int(eligible["fixture_id"].nunique())
    report["failures"] = failures
    report["passed"] = not failures

    if failures:
        raise AssertionError("; ".join(failures))

    os.makedirs(OUT, exist_ok=True)
    store_path, coverage_path, catalog_path, audit_path = (
        f"{OUT}/player_match.csv", f"{OUT}/fixture_event_coverage.csv", f"{OUT}/event_catalog.csv", f"{OUT}/audit.json",
    )
    for path in (store_path, coverage_path, catalog_path, audit_path):
        if os.path.exists(path):
            raise FileExistsError(f"immutable raw benchmark artifact exists: {path}")
    store.to_csv(store_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    catalog.to_csv(catalog_path, index=False)
    report["artifacts"] = {"store": str(store_path), "coverage": str(coverage_path), "catalog": str(catalog_path)}
    open(audit_path, "w").write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
