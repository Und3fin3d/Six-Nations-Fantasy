"""Immutable NCR shadow predictions at the published fantasy lock."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..benchmark_v2 import _group_metrics
from ..data import ROOT
from ..features import build_pit_features
from ..gbdt import UniversalGBDT
from ..schema import EVENTS, FORWARD_POSITIONS
from ..scoring import scorer_for
from ..raw_benchmark.blend import EventBlend50
from ..raw_benchmark.config import OUT as RAW_BENCHMARK_OUT
from ..raw_benchmark.features import build_frozen_feature_frames
from ..raw_benchmark.folds import build_folds, strict_training_frame
from .blend import EventBlendModel
from .context import augment_context
from .gbdt import ExposureRateGBDT
from .harness import OUT, attach_match_timestamps, sha256, write_once
from .neural import ExposureRateNeural
from .capture import (
    P3_PROTOCOL, archive_outcomes, protocol_for, round_fixtures, same_projection,
    snapshot_inputs, source_paths, source_revision, validate_capture,
    validate_cohort, validate_decision, validate_sources,
)

DATA = ROOT / "data"

POSITION = {
    "Prop": "Prop", "Hooker": "Hooker", "Lock": "Second-row",
    "Loose Forward": "Back-row", "Scrum Half": "Scrum-half",
    "Fly Half": "Fly-half", "Centre": "Centre", "Back Three": "Back-three",
}
START_JERSEY = {
    "Prop": 1, "Hooker": 2, "Lock": 4, "Loose Forward": 6,
    "Scrum Half": 9, "Fly Half": 10, "Centre": 12, "Back Three": 11,
}
BENCH_JERSEY = {
    "Prop": 17, "Hooker": 16, "Lock": 19, "Loose Forward": 20,
    "Scrum Half": 21, "Fly Half": 22, "Centre": 23, "Back Three": 23,
}


def ncr_candidates(gw: int, projection: pd.DataFrame | None = None) -> pd.DataFrame:
    if projection is not None:
        projection = projection.copy()
    else:
        projection_path = DATA / "ncr" / f"ncr_gw{gw}_projections.csv"
        if projection_path.exists():
            projection = pd.read_csv(projection_path)
        else:
            # An --exclude incumbent run writes only its tagged projection. Build
            # the full shadow cohort in memory from the same current team sheets.
            from model.ncr_project import build_projection
            projection = build_projection(gameday=gw)
            if projection.empty:
                raise RuntimeError(
                    f"GW{gw} projection is empty; wait for real team sheets before freezing"
                )
    crosswalk = pd.read_csv(DATA / "ncr" / "ncr_player_crosswalk.csv")
    crosswalk = crosswalk[["fantasy_id", "api_player_id", "full_name"]]
    if crosswalk.fantasy_id.duplicated().any() or crosswalk.api_player_id.dropna().duplicated().any():
        raise ValueError('NCR crosswalk contains duplicate fantasy or rugby identities')
    crosswalk["fantasy_id"] = pd.to_numeric(crosswalk["fantasy_id"], errors="coerce")
    projection["id"] = pd.to_numeric(projection["id"], errors="raise")
    frame = projection.merge(
        crosswalk, left_on="id", right_on="fantasy_id", how="left", validate="one_to_one",
    )
    fixtures = pd.read_csv(DATA / "ncr" / "ncr_fixtures.csv")
    fixtures = fixtures[pd.to_numeric(fixtures["gameday"], errors="coerce").eq(gw)]
    by_team = {}
    for row in fixtures.itertuples(index=False):
        by_team[row.home] = (row.away, str(row.match_id), row.game_date, "home")
        by_team[row.away] = (row.home, str(row.match_id), row.game_date, "away")
    records = []
    for row in frame.itertuples(index=False):
        opponent, fixture_id, date, home_away = by_team[str(row.team)]
        pos = POSITION[str(row.pos)]
        started = str(row.status).upper() == "P"
        api_id = (
            str(int(row.api_player_id)) if pd.notna(row.api_player_id)
            else f"fantasy_{int(row.id)}"
        )
        records.append({
            "date": date, "competition": "Nations Championship",
            "competition_id": 9999, "competition_level": "international",
            "season": 2026, "round": gw, "fixture_id": fixture_id,
            "player_id": api_id, "fantasy_id": int(row.id),
            "player_name": row.full_name if isinstance(row.full_name, str) else row.name,
            "team": row.team, "opponent": opponent, "position": pos, "home_away": home_away,
            "is_forward": pos in FORWARD_POSITIONS, "started": started,
            "jersey": START_JERSEY[str(row.pos)] if started else BENCH_JERSEY[str(row.pos)],
            "source": "v3_shadow_candidate", "source_priority": 999,
        })
    candidates = pd.DataFrame(records)
    for target in ("minutes", *EVENTS):
        candidates[target] = np.nan
        candidates[f"available__{target}"] = False
    return candidates


def _load_model(engine: str, path: Path):
    if engine == "p3_event_50":
        model = EventBlend50.load(path)
        if model.weight_v4 != 0.5:
            raise ValueError("P3 artifact does not use the fixed 50/50 blend")
        return model
    if engine == "baseline":
        return UniversalGBDT.load(path)
    if engine == "gbdt_v4":
        from ..v4.gbdt import V4GBDT
        return V4GBDT.load(path)
    if engine == "gbdt_v3":
        return ExposureRateGBDT.load(path)
    if engine == "neural_v3":
        return ExposureRateNeural.load(path)
    if engine == "blend_v3":
        return EventBlendModel.load(path)
    raise ValueError(engine)


def _p3_features(candidate: pd.DataFrame) -> pd.DataFrame:
    """Build the saved P3 artifact's frozen-history NCR candidate features."""
    store = pd.read_csv(
        RAW_BENCHMARK_OUT / "player_match.csv",
        low_memory=False,
        parse_dates=["date", "match_at"],
    )
    fold = next(
        fold for fold in build_folds(store)
        if fold.label == "nations_championship_2026"
    )
    train = strict_training_frame(store, fold)
    _, features = build_frozen_feature_frames(train, candidate, v4=True)
    return features


def validate_shadow_write(
    csv_path: Path, manifest_path: Path, lock_at: pd.Timestamp,
    *, now: pd.Timestamp | None = None,
) -> None:
    """Enforce both the lock cutoff and write-once shadow contract."""
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError(f"shadow snapshot already frozen: {csv_path.stem}")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    if current >= lock_at:
        raise RuntimeError(f"lock passed at {lock_at}; refusing retrospective shadow")


def predict_shadow(candidate, engine, model_path, lock_at, samples):
    model = _load_model(engine, model_path)
    if engine == "p3_event_50":
        protocol = protocol_for(engine)
        if sha256(model_path) != protocol["model_sha256"] or sha256(RAW_BENCHMARK_OUT / "player_match.csv") != protocol["history_sha256"]:
            raise ValueError("protected P3 requires its exact registered model and history")
        future = _p3_features(candidate)
    else:
        raw = pd.read_csv(
            DATA / "unified" / "player_match.csv", low_memory=False, parse_dates=["date"]
        )
        # The store must contain no rows from the lock day or later: a same-day row
        # can only be a post-lock result, so its presence means hindsight leakage.
        latest = pd.to_datetime(raw["date"], errors="coerce").max()
        if pd.Timestamp(latest).date() >= lock_at.date():
            raise RuntimeError(
                f"store contains rows dated {latest.date()} on/after the fantasy lock "
                f"({lock_at.date()}); refusing shadow freeze"
            )
        combined = pd.concat([raw, candidate], ignore_index=True, sort=False)
        timed = attach_match_timestamps(combined)
        blocks = ()
        if engine == "blend_v3":
            blocks = tuple(dict.fromkeys(
                (*model.gbdt.config.context_blocks, *model.neural.config.context_blocks)
            ))
        elif engine not in ("baseline", "gbdt_v4"):
            blocks = tuple(model.config.context_blocks)
        features = build_pit_features(augment_context(timed, blocks))
        if engine == "gbdt_v4":
            from ..v4.features import add_v4_base_stats, apply_eb_features
            features = add_v4_base_stats(features)
            k_by_event = getattr(model, "k_by_event", None)
            if k_by_event:
                features = apply_eb_features(features, k_by_event)
        future = features[features["source"].eq("v3_shadow_candidate")].copy()
    scorer = scorer_for("ncr", version=protocol_for(engine)["scoring_version"])
    predictions = model.predict_frame(future)
    candidate_grain = candidate[["fixture_id", "player_id", "team"]].astype(str)
    if candidate_grain.duplicated().any():
        raise ValueError("NCR shadow candidates violate the fixture/player/team grain")
    prediction_grain = pd.DataFrame([
        (item.fixture_id, item.player_id, item.team) for item in predictions
    ], columns=["fixture_id", "player_id", "team"]).astype(str)
    if len(prediction_grain) != len(candidate_grain) or not prediction_grain.equals(
        candidate_grain.reset_index(drop=True)
    ):
        raise ValueError("shadow predictions changed the fixture/player/team grain or order")
    rows = []
    for i, (source, prediction) in enumerate(zip(future.itertuples(index=False), predictions)):
        summary = scorer.score_prediction(prediction, n=samples, seed=1701 + i)
        rows.append({
            "fantasy_id": int(source.fantasy_id), "fixture_id": prediction.fixture_id,
            "player_id": prediction.player_id,
            "player_name": prediction.player_name, "team": prediction.team,
            "opponent": prediction.opponent, "position": prediction.position,
            "started": bool(source.started), "engine": engine,
            "expected_points": summary.mean, "p10": summary.p10,
            "median": summary.median, "p90": summary.p90,
            "p_play": prediction.metadata.get("p_play"),
            "expected_minutes": prediction.minutes.mean,
            "career_matches": float(
                pd.to_numeric(getattr(source, "career_matches", 0), errors="coerce")
            ),
        })
    return pd.DataFrame(rows)


def validate_capture_destination(csv_path, manifest_path, output_dir, lock_at, retrospective):
    if not retrospective:
        validate_shadow_write(csv_path, manifest_path, lock_at)
        return
    if output_dir.resolve() == (OUT / "shadow").resolve():
        raise ValueError("rehearsal requires a separate output directory")
    if csv_path.exists() or manifest_path.exists():
        raise FileExistsError("rehearsal already exists")


def capture_shadow(gw, engine, model_path, output_dir, samples, retrospective):
    from model.ncr_project import BUDGET, MAX_HEMI, MAX_NATION, REQUIRED, build_projection, optimise
    fixtures, lock_at = round_fixtures(gw)
    protocol = protocol_for(engine)
    if samples != protocol["samples"]:
        raise ValueError("capture must use the registered sampling settings")
    if not retrospective and gw not in protocol["rounds"]:
        raise ValueError("round is outside the registered prospective protocol")
    stem = output_dir / f"ncr_gw{gw}_{engine}"
    csv_path, manifest_path = stem.with_suffix(".csv"), stem.with_suffix(".manifest.json")
    validate_capture_destination(csv_path, manifest_path, output_dir, lock_at, retrospective)
    started_at = datetime.now(timezone.utc).isoformat()
    inputs = stem.with_suffix(".inputs")
    incumbent = pd.read_csv(DATA / "ncr" / f"ncr_gw{gw}_projections.csv")
    validate_cohort(gw, incumbent, retrospective=retrospective)
    hashes = snapshot_inputs(inputs, source_paths(gw, engine, model_path, retrospective))
    validate_cohort(gw, incumbent, retrospective=retrospective)
    if not retrospective:
        fresh = build_projection(gameday=gw, asof=pd.to_datetime(fixtures.game_date, utc=True).min().normalize())
        same_projection(incumbent, fresh)
    incumbent_squad = pd.read_csv(DATA / "ncr" / f"ncr_gw{gw}_squad.csv")
    validate_decision(incumbent_squad, incumbent)
    candidate = ncr_candidates(gw, projection=incumbent)
    frame = predict_shadow(candidate, engine, model_path, lock_at, samples)
    candidate_projection = incumbent.drop(columns=["starter_exp", "supersub_exp"]).merge(
        frame[["fantasy_id", "expected_points"]], left_on="id", right_on="fantasy_id", validate="one_to_one",
    )
    candidate_projection["starter_exp"] = candidate_projection.expected_points
    candidate_projection["supersub_exp"] = candidate_projection.expected_points * np.where(candidate_projection.status.eq("B"), 3.0, 0.5)
    candidate_squad, _, _ = optimise(candidate_projection)
    validate_decision(candidate_squad, candidate_projection)
    validate_sources(hashes)
    if not retrospective:
        validate_shadow_write(csv_path, manifest_path, lock_at)
    payloads = {
        csv_path.name: frame.sort_values(["expected_points", "fantasy_id"], ascending=[False, True]),
        f"{stem.name}.incumbent.csv": incumbent,
        f"{stem.name}.incumbent_decision.csv": incumbent_squad,
        f"{stem.name}.candidate_projection.csv": candidate_projection,
        f"{stem.name}.candidate_decision.csv": candidate_squad,
    }
    for name, table in payloads.items():
        write_once(output_dir / name, table.to_csv(index=False))
    completed_at = datetime.now(timezone.utc).isoformat()
    if not retrospective and pd.Timestamp(completed_at) >= lock_at:
        raise RuntimeError("capture crossed the lock; partial files cannot enter evaluation")
    files = {name: sha256(output_dir / name) for name in payloads}
    files.update({f"{inputs.name}/{name}": digest for name, digest in hashes.items()})
    manifest = {
        "schema_version": 2, "competition": "ncr", "round": gw, "engine": engine,
        "protocol": protocol, "evidence_kind": "retrospective_rehearsal" if retrospective else "prospective",
        "started_at": started_at, "completed_at": completed_at, "lock_at": lock_at.isoformat(),
        "source_revision": source_revision(), "model_path": str(model_path),
        "model_sha256": sha256(model_path), "rows": len(frame), "immutable": True,
        "files": files, "prediction_sha256": sha256(csv_path),
        "incumbent_file": f"{stem.name}.incumbent.csv",
        "incumbent_projection_file": f"{stem.name}.incumbent.csv",
        "incumbent_decision_file": f"{stem.name}.incumbent_decision.csv",
        "candidate_projection_file": f"{stem.name}.candidate_projection.csv",
        "candidate_decision_file": f"{stem.name}.candidate_decision.csv",
        "optimiser": {"budget": BUDGET, "max_nation": MAX_NATION, "max_hemi": MAX_HEMI, "required": REQUIRED, "captain_status": "P", "super_sub_status": "B"},
        "samples": samples, "seed_start": 1701,
        "cohort_readiness": "historical_archived_cohort_not_lock_ready" if retrospective else "complete_current_teamsheets",
    }
    manifest["blend_weight_v4"] = 0.5 if engine == "p3_event_50" else None
    finish_capture(stem, manifest, output_dir, lock_at, retrospective)
    return csv_path


def finish_capture(stem, manifest, output_dir, lock_at, retrospective):
    write_once(stem.with_suffix(".manifest.json"), json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not retrospective and pd.Timestamp.now(tz="UTC") >= lock_at:
        write_once(stem.with_suffix(".rejected.json"), json.dumps({"reason": "manifest crossed lock"}))
        raise RuntimeError("capture manifest crossed the lock and was rejected")
    validate_capture(manifest["round"], manifest["engine"], output_dir=output_dir, prospective=not retrospective)


def freeze_shadow(gw: int, engine: str, model_path: Path,
                  *, output_dir: Path = OUT / "shadow", samples: int = 4000) -> Path:
    return capture_shadow(gw, engine, model_path, output_dir, samples, False)


def rehearse_shadow(gw: int, engine: str, model_path: Path, *, output_dir: Path, samples: int = 4000) -> Path:
    return capture_shadow(gw, engine, model_path, output_dir, samples, True)


def evaluated_round(gw, engine, shadow_dir):
    manifest = validate_capture(gw, engine, output_dir=shadow_dir)
    stem = shadow_dir / f"ncr_gw{gw}_{engine}"
    outcome_path = stem.with_suffix(".outcomes.csv")
    record = json.loads(stem.with_suffix(".outcomes.json").read_text())
    if record["completion_sha256"] != sha256(stem.with_suffix(".outcomes.completion.json")):
        raise ValueError("archived outcome completion evidence changed")
    if record["source_sha256"] != sha256(stem.with_suffix(".outcomes.feed.json")):
        raise ValueError("archived outcome source hash changed")
    if record["outcomes_sha256"] != sha256(outcome_path) or record["capture_sha256"] != sha256(stem.with_suffix(".manifest.json")):
        raise ValueError("outcomes are not bound to this capture")
    frame = pd.read_csv(stem.with_suffix(".csv"))
    incumbent = pd.read_csv(shadow_dir / manifest["incumbent_file"])
    outcomes = pd.read_csv(outcome_path)
    frame = frame.merge(incumbent[["id", "starter_exp"]], left_on="fantasy_id", right_on="id", validate="one_to_one")
    frame = frame.merge(outcomes, on=["id", "team"], how="left", validate="one_to_one")
    if frame[["official_pts", "starter_exp", "expected_points", "min_in_game"]].isna().any().any():
        raise ValueError("required prospective outcomes or predictions are incomplete")
    frame["incumbent_points"] = frame.starter_exp
    frame["round"] = gw
    return frame, manifest


def squad_outcomes(gw, engine, shadow_dir, frame, manifest):
    totals = {"round": gw}
    for role in ("candidate", "incumbent"):
        squad = pd.read_csv(shadow_dir / manifest[f"{role}_decision_file"])
        rows = squad[["id", "is_sub", "is_capt"]].merge(frame[["id", "official_pts", "status", "min_in_game"]], on="id", validate="one_to_one")
        sub_multiplier = np.where(rows.min_in_game.gt(0), np.where(rows.status.eq("B"), 3.0, 0.5), 0.0)
        ordinary = float(rows.loc[~rows.is_sub, "official_pts"].sum())
        captain = float(rows.loc[rows.is_capt, "official_pts"].sum())
        sub = float((rows.official_pts * sub_multiplier * rows.is_sub).sum())
        totals[role] = {"ordinary_xv": ordinary, "captain_extra": captain, "super_sub": sub, "total": ordinary + captain + sub}
    return totals


def protocol_decision(evaluated, protocol, historical):
    candidate = _group_metrics(evaluated, "expected_points")
    incumbent = _group_metrics(evaluated, "incumbent_points")
    keys = [f"top_{n}_capture" for n in protocol["top_ns"]]
    candidate["mean_capture"] = float(np.mean([candidate[key] for key in keys]))
    incumbent["mean_capture"] = float(np.mean([incumbent[key] for key in keys]))
    reasons = []
    if candidate["mae"] > incumbent["mae"] * (1 + protocol["mae_regression"]):
        reasons.append("pooled MAE exceeds the registered regression margin")
    capture_keys = ["mean_capture"] if protocol["capture_gate"] == "mean" else keys
    if any(candidate[key] < incumbent[key] - protocol["capture_regression"] for key in capture_keys):
        reasons.append("pooled capture exceeds the registered regression margin")
    if protocol["id"] == P3_PROTOCOL:
        historical_passed = historical.get("selected") == "p3_event_50" and historical.get("passed", {}).get("p3_event_50") is True
    else:
        historical_passed = historical.get("retrospective_gate_passed") is True
        for mask in (evaluated.status.eq("B"), pd.to_numeric(evaluated.career_matches).lt(5)):
            rows = evaluated[mask]
            if len(rows) and (rows.expected_points - rows.official_pts).abs().mean() > (rows.incumbent_points - rows.official_pts).abs().mean() + 0.5:
                reasons.append("registered subgroup MAE regression exceeds 0.5")
    if not historical_passed:
        reasons.append("the declared historical gate did not pass")
    return {"candidate": candidate, "incumbent": incumbent, "reasons": reasons, "gate_passed": not reasons}


def evaluate_prospective_shadows(
    engine: str, *, rounds: tuple[int, ...] = (4, 5, 6, 7),
    shadow_dir: Path = OUT / "shadow", output_dir: Path = OUT / "prospective",
) -> dict:
    protocol = protocol_for(engine)
    if tuple(rounds) != tuple(protocol["rounds"]):
        raise ValueError("single-look evaluation requires exactly the registered GW4-GW7 rounds")
    decision_path = output_dir / f"{engine}_decision.json"
    rows_path = output_dir / f"{engine}_gw4_gw7_rows.csv"
    if decision_path.exists() or rows_path.exists():
        raise FileExistsError("single-look evaluation already recorded")
    pieces, manifests = zip(*(evaluated_round(gw, engine, shadow_dir) for gw in rounds))
    if len({record["model_sha256"] for record in manifests}) != 1:
        raise ValueError("prospective rounds used different frozen model artifacts")
    evaluated = pd.concat(pieces, ignore_index=True)
    gate = "data/unified/raw_benchmark/v1/decision.json" if protocol["id"] == P3_PROTOCOL else "data/unified/v3/benchmark/decision.json"
    gate_paths = [shadow_dir / f"ncr_gw{gw}_{engine}.inputs" / gate for gw in rounds]
    if len({sha256(path) for path in gate_paths}) != 1:
        raise ValueError("historical decision changed between prospective captures")
    payload = protocol_decision(evaluated, protocol, json.loads(gate_paths[0].read_text()))
    protected = protocol["id"] == P3_PROTOCOL
    payload.update({
        "schema_version": 2, "engine": engine, "rounds": list(rounds), "protocol": protocol,
        "promotion": False,
        "deployment_action": "retain_incumbent_pending_corrected_scoring_version" if protected else "manual_review_required",
        "interpretation": "registered severe-failure veto; legacy scoring cannot authorise deployment" if protected else "registered v3 non-inferiority gates",
        "squad_outcomes": [squad_outcomes(gw, engine, shadow_dir, frame, manifest) for gw, frame, manifest in zip(rounds, pieces, manifests)],
        "capture_manifests": {str(gw): sha256(shadow_dir / f"ncr_gw{gw}_{engine}.manifest.json") for gw in rounds},
    })
    write_once(rows_path, evaluated.to_csv(index=False))
    write_once(decision_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
