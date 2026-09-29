from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from ..data import ROOT
from .harness import OUT, sha256, write_once

DATA = ROOT / "data"
P3_PROTOCOL = "p3_event_50_gw4_7_v1"
LEGACY_PROTOCOL = "unified_v3_gw4_7_v1"
GRAIN = ["fixture_id", "player_id", "team"]


def protocol_for(engine):
    p3 = engine == "p3_event_50"
    return {
        "model_sha256": "79da46093fc497cb1f694da32b951bb221c162e955a9ce064d05b4318d857439" if p3 else None,
        "history_sha256": "7db55ba3a7c3de38908acb90bf23eb499c649d01c4094df1597b776bce67fe74" if p3 else None,
        "id": P3_PROTOCOL if p3 else LEGACY_PROTOCOL,
        "engine": engine, "rounds": [4, 5, 6, 7], "single_look": True,
        "scoring_version": "ncr_unallocated_scrums_v1",
        "samples": 4000, "seed_start": 1701, "max_snapshot_age_hours": 24,
        "history_policy": "tournament_frozen" if p3 else "prior_only",
        "mae_regression": 0.05 if p3 else 0.02,
        "capture_regression": 0.05 if p3 else 0.02,
        "capture_gate": "mean" if p3 else "each_cutoff",
        "top_ns": [10, 25, 50, 100],
        "missing_outcomes": "withhold_evaluation",
    }


def round_fixtures(gw):
    fixtures = pd.read_csv(DATA / "ncr/ncr_fixtures.csv")
    rows = fixtures[pd.to_numeric(fixtures.gameday, errors="raise").eq(gw)].copy()
    teams = list(rows.home) + list(rows.away)
    if len(rows) != 6 or len(set(teams)) != 12 or "TBD" in teams:
        raise ValueError(f"GW{gw} requires six confirmed fixtures and twelve distinct teams")
    lock = pd.to_datetime(rows.lock_date, utc=True, errors="raise").min()
    if pd.isna(lock):
        raise ValueError("missing fantasy lock")
    return rows, lock


def players_for(gw, content=None):
    path = DATA / "ncr/feeds" / f"players_gw{gw}.json"
    content = path.read_text() if content is None else content
    players = pd.DataFrame(json.loads(content)["Data"]["Value"]["Players"])
    if set(pd.to_numeric(players.gameday_id, errors="raise")) != {gw}:
        raise ValueError(f"GW{gw} feed has stale gameday identifiers")
    players["id"] = pd.to_numeric(players.id, errors="raise").astype(int)
    if players.id.duplicated().any():
        raise ValueError("duplicate fantasy player IDs in feed")
    return players


def validate_cohort(gw, projection, *, retrospective=False):
    fixtures, lock = round_fixtures(gw)
    players = players_for(gw)
    if retrospective:
        if projection.id.duplicated().any() or not set(projection.id).issubset(set(players.id)):
            raise ValueError("historical projection has duplicate or unmapped identities")
        return lock
    named = players[players.player_status.isin(["P", "B"])].copy()
    teams = set(fixtures.home) | set(fixtures.away)
    counts = named.groupby(["team_name", "player_status"]).size().unstack(fill_value=0)
    if set(counts.index) != teams or not counts.reindex(columns=["P", "B"]).eq([15, 8]).all().all():
        raise ValueError("complete teamsheets require 15 starters and 8 bench players per team")
    if projection.id.duplicated().any() or set(projection.id) != set(named.id):
        raise ValueError("projection does not cover the complete named cohort exactly once")
    from model.ncr_project import SKILL2POS
    joined = projection.merge(named, on="id", validate="one_to_one", suffixes=("", "_feed"))
    joined["pos_feed"] = joined.skill_desc.map(SKILL2POS)
    for left, right in (("team", "team_name"), ("status", "player_status"), ("pos", "pos_feed"), ("hemi", "hemisphere"), ("value", "value_feed")):
        if not joined[left].astype(str).equals(joined[right].astype(str)):
            if left not in {"hemi", "value"} or not np.allclose(joined[left].astype(float), joined[right].astype(float)):
                raise ValueError(f"projection and source feed differ for {left}")
    if not pd.to_numeric(fixtures.iscurrent, errors="raise").eq(1).all():
        raise ValueError("live capture requires the current fantasy round")
    validate_lineup_freshness(gw, named, lock)
    return lock


def validate_lineup_freshness(gw, named, lock):
    from model.ncr_project import SKILL2POS
    if named.skill_desc.map(SKILL2POS).isna().any():
        raise ValueError("unrecognised fantasy position in source")
    receipt = json.loads((DATA / "ncr/ncr_snapshot.manifest.json").read_text())
    fetched = pd.to_datetime(receipt["fetched_at"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    if receipt["round"] != gw or fetched >= lock or fetched > now or now - fetched > pd.Timedelta(hours=24):
        raise ValueError("refresh the current source snapshot before live capture")
    for name, digest in receipt["files"].items():
        if sha256(DATA / "ncr" / name) != digest:
            raise ValueError(f"snapshot source changed after retrieval: {name}")
    expected = {"ncr_players.csv", "ncr_fixtures.csv", f"feeds/players_gw{gw}.json", "feeds/fixtures_latest.json"}
    if set(receipt["files"]) != expected:
        raise ValueError("snapshot receipt does not cover all required catalogue sources")
    signature = dict(zip(named.id, named.player_status))
    for path in (DATA / "ncr/feeds").glob("players_gw*.json"):
        if path.name == f"players_gw{gw}.json":
            continue
        prior = pd.DataFrame(json.loads(path.read_text())["Data"]["Value"]["Players"])
        prior = prior[prior.player_status.isin(["P", "B"])]
        old = dict(zip(pd.to_numeric(prior.id).astype(int), prior.player_status))
        if signature == old:
            raise ValueError(f"carryover teamsheets match {path.name}")


def source_paths(gw, engine, model_path, retrospective):
    history = DATA / "unified/raw_benchmark/v1/player_match.csv" if engine == "p3_event_50" else DATA / "unified/player_match.csv"
    paths = [
        model_path, history, DATA / "ncr/ncr_fixtures.csv", DATA / "ncr/ncr_players.csv",
        DATA / "ncr/ncr_player_crosswalk.csv", DATA / "ncr/ncr_teams.csv",
        DATA / "ncr/ncr_player_match.csv", DATA / "wr_rankings.csv",
        DATA / "ncr/feeds" / f"players_gw{gw}.json", DATA / "rp_compstats.csv",
        DATA / "ncr" / f"ncr_gw{gw}_projections.csv", DATA / "ncr" / f"ncr_gw{gw}_squad.csv",
    ]
    if not retrospective:
        paths.extend([DATA / "ncr/ncr_snapshot.manifest.json", DATA / "ncr/feeds/fixtures_latest.json"])
    paths.extend(path for path in (DATA / "ncr/club_player_match.csv", DATA / "ncr/ncr_fixture_weather.csv") if path.exists())
    gate = DATA / "unified/raw_benchmark/v1/decision.json" if engine == "p3_event_50" else OUT / "benchmark/decision.json"
    paths.append(gate)
    paths.extend(ROOT.glob("model/**/*.py"))
    paths.extend(ROOT / name for name in ("requirements-model.txt", "requirements-unified.txt", "gw_update.sh", "ncr_snapshot.py"))
    return list(dict.fromkeys(paths))


def snapshot_inputs(directory, paths):
    for path in paths:
        with path.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(f"hydrate the Git LFS object before capture: {path}")
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in paths:
        relative = str(path.resolve().relative_to(ROOT.resolve()))
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[relative] = sha256(destination)
        if sha256(path) != hashes[relative]:
            raise RuntimeError(f"source changed during capture: {relative}")
    return hashes


def validate_sources(hashes):
    for relative, digest in hashes.items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"source changed since capture: {relative}")


def source_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def same_projection(actual, expected):
    columns = ["id", "team", "pos", "status", "value", "hemi", "starter_exp", "supersub_exp"]
    a = actual[columns].sort_values("id").reset_index(drop=True)
    b = expected[columns].sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10)


def validate_decision(squad, projection):
    from model.ncr_project import BUDGET, MAX_HEMI, MAX_NATION, REQUIRED
    if len(squad) != 16 or squad.id.duplicated().any() or not set(squad.id).issubset(set(projection.id)):
        raise ValueError("selected decisions do not contain sixteen unique cohort players")
    same_projection(squad, projection[projection.id.isin(squad.id)])
    if squad.is_capt.sum() != 1 or squad.is_sub.sum() != 1 or (squad.is_capt & squad.is_sub).any():
        raise ValueError("selected decisions have invalid multiplier assignments")
    if not squad.loc[squad.is_capt, "status"].eq("P").all() or not squad.loc[squad.is_sub, "status"].eq("B").all():
        raise ValueError("selected decisions violate captain or super-sub eligibility")
    counts = squad.loc[~squad.is_sub].groupby("pos").size().to_dict()
    if counts != REQUIRED or squad.value.sum() > BUDGET + 1e-9:
        raise ValueError("selected decisions violate position or budget constraints")
    if squad.groupby("team").size().max() > MAX_NATION or squad.groupby("hemi").size().max() > MAX_HEMI:
        raise ValueError("selected decisions violate country or hemisphere limits")


def validate_capture_files(manifest, stem, output_dir, model_path):
    required = {stem.with_suffix(".csv").name, *[manifest[f"{role}_{kind}_file"] for role in ("candidate", "incumbent") for kind in ("projection", "decision")]}
    if not required.issubset(manifest["files"]):
        raise ValueError("capture manifest omits required forecast or decision hashes")
    for filename, digest in manifest["files"].items():
        if os.path.isabs(filename) or ".." in filename.split("/"):
            raise ValueError("capture manifest contains an unsafe path")
        if sha256(output_dir / filename) != digest:
            raise ValueError(f"capture hash mismatch: {filename}")
    if model_path is not None:
        if sha256(model_path) != manifest["model_sha256"]:
            raise ValueError("existing capture does not match configured model")
        prefix = f"{stem.name}.inputs/"
        sources = {name.removeprefix(prefix): digest for name, digest in manifest["files"].items() if name.startswith(prefix)}
        sources.pop("data/ncr/ncr_snapshot.manifest.json", None)
        validate_sources(sources)

    protocol = manifest["protocol"]
    if protocol["id"] == P3_PROTOCOL:
        history = f"{stem.name}.inputs/data/unified/raw_benchmark/v1/player_match.csv"
        if manifest["model_sha256"] != protocol["model_sha256"] or manifest["files"].get(history) != protocol["history_sha256"]:
            raise ValueError("protected P3 model or tournament history changed")


def validate_capture(gw, engine, *, output_dir=OUT / "shadow", model_path=None, prospective=True):
    stem = output_dir / f"ncr_gw{gw}_{engine}"
    if stem.with_suffix(".rejected.json").exists():
        raise ValueError("capture crossed its lock and was rejected")
    manifest = json.loads(stem.with_suffix(".manifest.json").read_text())
    protocol = protocol_for(engine)
    if manifest["schema_version"] != 2 or manifest["protocol"] != protocol or manifest["round"] != gw:
        raise ValueError("capture manifest does not match the registered experiment")
    if prospective and manifest["evidence_kind"] != "prospective":
        raise ValueError("retrospective rehearsal cannot enter prospective evaluation")
    if manifest["evidence_kind"] == "prospective":
        if pd.to_datetime(manifest["completed_at"], utc=True) >= pd.to_datetime(manifest["lock_at"], utc=True):
            raise ValueError("capture was completed after the lock")
    validate_capture_files(manifest, stem, output_dir, model_path)
    validate_capture_cohort(manifest, stem, output_dir)
    return manifest


def validate_capture_cohort(manifest, stem, output_dir):
    frame = pd.read_csv(stem.with_suffix(".csv"), dtype={"fixture_id": str, "player_id": str})
    incumbent = pd.read_csv(output_dir / manifest["incumbent_file"])
    if len(frame) != manifest["rows"] or frame.duplicated(GRAIN).any() or frame.fantasy_id.duplicated().any():
        raise ValueError("capture has incomplete or duplicate candidate keys")
    if set(frame.fantasy_id) != set(incumbent.id) or incumbent.id.duplicated().any():
        raise ValueError("candidate and incumbent capture cohorts differ")
    if not np.isfinite(frame.expected_points).all() or not np.isfinite(incumbent.starter_exp).all():
        raise ValueError("capture contains missing forecasts")
    for role in ("candidate", "incumbent"):
        projection = pd.read_csv(output_dir / manifest[f"{role}_projection_file"])
        squad = pd.read_csv(output_dir / manifest[f"{role}_decision_file"])
        validate_decision(squad, projection)


def completed_sources(gw, manifest, shadow_dir):
    receipt = json.loads((DATA / "ncr/ncr_snapshot.manifest.json").read_text())
    now = pd.Timestamp.now(tz="UTC")
    fetched = pd.to_datetime(receipt["fetched_at"], utc=True)
    if receipt["results_round"] != gw or fetched > now or now - fetched > pd.Timedelta(hours=24):
        raise ValueError("refresh the completed round source before archiving outcomes")
    names = {f"feeds/players_gw{gw}.json", "feeds/fixtures_latest.json"}
    if set(receipt["results_files"]) != names:
        raise ValueError("results receipt lacks player or fixture evidence")
    sources = {name: (DATA / "ncr" / name).read_bytes() for name in names}
    for name, content in sources.items():
        if hashlib.sha256(content).hexdigest() != receipt["results_files"][name]:
            raise ValueError(f"result source changed since retrieval: {name}")
    fixtures = pd.DataFrame(json.loads(sources["feeds/fixtures_latest.json"])["Data"]["Value"])
    fixtures = fixtures[pd.to_numeric(fixtures.gameday).eq(gw)]
    frozen = pd.read_csv(shadow_dir / f"ncr_gw{gw}_{manifest['engine']}.inputs/data/ncr/ncr_fixtures.csv")
    expected = set(frozen.loc[pd.to_numeric(frozen.gameday).eq(gw), "match_id"])
    if set(fixtures.match_id) != expected or len(fixtures) != 6:
        raise ValueError("completed fixture source does not match the captured round")
    if not fixtures[["match_status", "islive"]].apply(pd.to_numeric).eq(2).all().all():
        raise ValueError("official fixture source does not mark every match completed")
    latest = pd.to_datetime(fixtures.game_date, utc=True, errors="raise").max()
    if fetched <= latest + pd.Timedelta(days=1):
        raise ValueError("wait 24 hours after the final recorded kickoff before freezing results")
    completion = json.dumps({"receipt": receipt, "fixtures": json.loads(sources["feeds/fixtures_latest.json"])}, indent=2, sort_keys=True)
    return sources[f"feeds/players_gw{gw}.json"].decode(), completion


def archive_outcomes(gw, engine, *, shadow_dir=OUT / "shadow"):
    manifest = validate_capture(gw, engine, output_dir=shadow_dir)
    content, completion = completed_sources(gw, manifest, shadow_dir)
    players = players_for(gw, content)
    frame = players[["id", "team_name", "cur_gd_points", "player_status", "min_in_game"]].copy()
    frame = frame.rename(columns={"team_name": "team", "cur_gd_points": "official_pts", "player_status": "status"})
    frame["official_pts"] = pd.to_numeric(frame.official_pts, errors="coerce")
    frame["min_in_game"] = pd.to_numeric(frame.min_in_game, errors="coerce")
    cohort = pd.read_csv(shadow_dir / manifest["incumbent_file"])[["id", "team"]]
    frame = cohort.merge(frame, on=["id", "team"], how="left", validate="one_to_one")
    if not frame.groupby("team").min_in_game.max().gt(0).all():
        raise ValueError("official outcomes do not yet show every team playing")
    if frame[["official_pts", "min_in_game"]].isna().any().any():
        raise ValueError("required official outcomes remain unknown")
    stem = shadow_dir / f"ncr_gw{gw}_{engine}.outcomes"
    csv_path, record_path = stem.with_suffix(".outcomes.csv"), stem.with_suffix(".outcomes.json")
    feed_path = stem.with_suffix(".outcomes.feed.json")
    completion_path = stem.with_suffix(".outcomes.completion.json")
    if any(path.exists() for path in (csv_path, record_path, feed_path, completion_path)):
        raise FileExistsError("outcomes already archived; preserve the original record")
    write_once(csv_path, frame.to_csv(index=False))
    write_once(feed_path, content)
    write_once(completion_path, completion + "\n")
    record = {"round": gw, "engine": engine, "rows": len(frame), "source_sha256": sha256(feed_path), "completion_sha256": sha256(completion_path), "outcomes_sha256": sha256(csv_path), "capture_sha256": sha256(shadow_dir / f"ncr_gw{gw}_{engine}.manifest.json")}
    write_once(record_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
