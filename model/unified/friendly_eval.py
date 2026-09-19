"""Fixed, fantasy-label-free international fixture evaluation.

The headline is match-balanced MAE over count statistics. Metres and minutes
have separate units and separate headlines. Per-stat MAEs are always retained.
This is a retrospective test with oracle teamsheets, not a prospective claim.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import RawPrediction
from .raw_benchmark.config import STABLE_EVENTS
from .raw_benchmark.coverage import sha256

KEY = ["fixture_id", "player_id", "team"]
TARGETS = ("minutes", *STABLE_EVENTS)
COUNT_TARGETS = tuple(event for event in STABLE_EVENTS if event != "metres")
MANIFEST_VERSION = 1


def select_fixtures(
    store: pd.DataFrame, cache: Path, *, forbidden: Iterable[str],
    asof: str, count: int = 15,
) -> dict:
    """Latest completed cached friendlies, chosen without inspecting errors.

    Competition 30 is the cache's international-friendly collection. Exclude
    fixtures with repository fantasy labels. No selection by score, margin,
    players' results, model predictions, or event values is performed.
    """
    if count <= 0:
        raise ValueError("fixture count must be positive")
    end = pd.Timestamp(asof)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    forbidden = set(map(str, forbidden))
    source = store.loc[
        store["competition_level"].eq("international")
        & pd.to_numeric(store["competition_id_cache"], errors="coerce").eq(30)
    ].drop_duplicates("fixture_id")
    eligible = []
    for fixture in source["fixture_id"].astype(str):
        if fixture in forbidden:
            continue
        path = cache / f"match_{fixture}.json"
        match = json.loads(path.read_text())["results"]["match"]
        kickoff = pd.to_datetime(match.get("date"), errors="coerce", utc=True)
        if str(match.get("comp_id")) != "30" or match.get("status") != "Result":
            continue
        if pd.isna(kickoff) or kickoff + pd.Timedelta(hours=3) >= end:
            continue
        eligible.append({
            "fixture_id": fixture, "kickoff": kickoff.isoformat(),
            "home": str(match["home_team"]), "away": str(match["away_team"]),
            "cache_sha256": sha256(path),
        })
    eligible.sort(key=lambda item: (pd.Timestamp(item["kickoff"]), item["fixture_id"]))
    if len(eligible) < count:
        raise ValueError(f"need {count} eligible friendlies; found {len(eligible)}")
    return {
        "version": MANIFEST_VERSION, "count": count, "asof": end.isoformat(),
        "selection": "latest completed canonical comp_id=30, excluding fantasy-labelled fixtures",
        "eligible_count": len(eligible), "targets": list(TARGETS),
        "count_targets": list(COUNT_TARGETS), "fixtures": eligible[-count:],
        "lineup_basis": "historical oracle teamsheets; kickoff is the prediction lock proxy",
        "availability": "earlier results require kickoff plus strictly more than three hours",
        "label_scope": "no official fantasy labels in this repository; not a claim about all platforms",
    }


def write_manifest(path: Path, manifest: dict) -> None:
    """Write once: a rerun cannot silently change the 15 selected fixtures."""
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"frozen friendly manifest differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def validate_manifest(
    manifest: dict, store: pd.DataFrame, cache: Path, *, forbidden: Iterable[str],
    expected_count: int = 15,
) -> None:
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("unsupported friendly manifest version")
    fixtures = manifest["fixtures"]
    ids = [str(row["fixture_id"]) for row in fixtures]
    if manifest["count"] != expected_count or len(ids) != expected_count or len(set(ids)) != len(ids):
        raise ValueError("wrong or duplicate friendly fixture count")
    if manifest["targets"] != list(TARGETS) or manifest["count_targets"] != list(COUNT_TARGETS):
        raise ValueError("friendly statistic set changed")
    if set(ids) & set(map(str, forbidden)):
        raise ValueError("a fantasy-labelled fixture entered the friendly test")
    for row in fixtures:
        fixture = str(row["fixture_id"])
        path = cache / f"match_{fixture}.json"
        if sha256(path) != row["cache_sha256"]:
            raise ValueError(f"friendly cache changed: {fixture}")
        local = store[store["fixture_id"].astype(str).eq(fixture)]
        if local.empty or local.duplicated(KEY).any():
            raise ValueError(f"empty or duplicate friendly cohort: {fixture}")
        if not local["competition_level"].eq("international").all() or not pd.to_numeric(
            local["competition_id_cache"], errors="coerce"
        ).eq(30).all():
            raise ValueError(f"not an international friendly: {fixture}")
        if not pd.to_datetime(local["match_at"], utc=True).eq(pd.Timestamp(row["kickoff"])).all():
            raise ValueError(f"friendly kickoff changed: {fixture}")


def score_stats(
    truth: pd.DataFrame, predictions: list[RawPrediction], *, engine: str,
    targets: tuple[str, ...] = TARGETS,
) -> pd.DataFrame:
    """Per-match, per-stat MAE with exact player-key parity and missing masks.

    Missing labels are excluded, not converted to zero. An observed zero,
    including an unused substitute, remains an observation. Missing forecasts
    for observed labels fail the evaluation, rather than improving coverage.
    """
    truth = truth.copy()
    truth[KEY] = truth[KEY].astype(str)
    if truth.empty or truth.duplicated(KEY).any():
        raise ValueError("truth requires unique, nonempty fixture/player/team keys")
    keys = [(p.fixture_id, p.player_id, p.team) for p in predictions]
    wanted = list(truth[KEY].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)) or set(keys) != set(wanted):
        raise ValueError("raw prediction cohort differs from truth")
    by_key = dict(zip(keys, predictions))
    aligned = [by_key[key] for key in wanted]
    rows = []
    for target in targets:
        mask = truth[f"available__{target}"].fillna(False).astype(bool).to_numpy()
        actual = pd.to_numeric(truth[target], errors="coerce").to_numpy(float)
        mask &= np.isfinite(actual)
        values = []
        for prediction in aligned:
            dist = prediction.minutes if target == "minutes" else prediction.events.get(target)
            values.append(np.nan if dist is None else dist.mean)
        predicted = np.asarray(values, dtype=float)
        if not np.isfinite(predicted[mask]).all():
            raise ValueError(f"{engine} has missing predictions for observed {target}")
        for fixture in truth["fixture_id"].unique():
            valid = mask & truth["fixture_id"].eq(fixture).to_numpy()
            rows.append({
                "engine": engine, "fixture_id": fixture, "target": target,
                "n": int(valid.sum()),
                "mae": float(np.abs(actual[valid] - predicted[valid]).mean()) if valid.any() else np.nan,
            })
    return pd.DataFrame(rows)


def summarise_stats(metrics: pd.DataFrame, fixture_ids: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal weight for each fixture and count target, not each player row."""
    fixture_ids = set(map(str, fixture_ids))
    if not fixture_ids or metrics.empty:
        raise ValueError("a nonempty fixed fixture cohort is required")
    if metrics.duplicated(["engine", "fixture_id", "target"]).any():
        raise ValueError("duplicate metric rows")
    expected = {(fixture, target) for fixture in fixture_ids for target in TARGETS}
    reference_support = None
    for engine, block in metrics.groupby("engine"):
        keys = set(zip(block["fixture_id"].astype(str), block["target"]))
        if keys != expected:
            raise ValueError(f"{engine}: incomplete fixed fixture/stat coverage")
        support = block.set_index(["fixture_id", "target"])["n"].sort_index()
        if reference_support is not None and not support.equals(reference_support):
            raise ValueError("model metric supports differ")
        reference_support = support
    per_stat = metrics.groupby(["engine", "target"], sort=True).agg(
        mae=("mae", "mean"), observations=("n", "sum"),
        observed_fixtures=("mae", "count"),
    ).reset_index()
    rows = []
    for engine, block in metrics.groupby("engine", sort=True):
        count = block[block["target"].isin(COUNT_TARGETS)]
        # Never hide a completely unobserved match/stat behind skipna averaging.
        complete = bool(count["mae"].notna().all() and count["n"].gt(0).all())
        row = {"engine": engine, "fixtures": len(fixture_ids), "count_support_complete": complete,
               "count_stat_mae": float(count["mae"].mean()) if complete else np.nan}
        for target in ("minutes", "metres"):
            values = block.loc[block["target"].eq(target), "mae"]
            row[f"{target}_mae"] = float(values.mean()) if values.notna().all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows), per_stat
