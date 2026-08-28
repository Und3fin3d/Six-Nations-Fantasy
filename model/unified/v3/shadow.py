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
from ..labels import build_fantasy_labels
from ..schema import EVENTS, FORWARD_POSITIONS
from ..scoring import scorer_for
from .blend import EventBlendModel
from .context import augment_context
from .gbdt import ExposureRateGBDT
from .harness import OUT, attach_match_timestamps, sha256, write_once
from .neural import ExposureRateNeural

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
    projection = (
        projection.copy() if projection is not None
        else pd.read_csv(DATA / "ncr" / f"ncr_gw{gw}_projections.csv")
    )
    crosswalk = pd.read_csv(DATA / "ncr" / "ncr_player_crosswalk.csv")
    crosswalk = crosswalk[["fantasy_id", "api_player_id", "full_name"]].drop_duplicates("fantasy_id")
    crosswalk["fantasy_id"] = pd.to_numeric(crosswalk["fantasy_id"], errors="coerce")
    projection["id"] = pd.to_numeric(projection["id"], errors="raise")
    frame = projection.merge(
        crosswalk, left_on="id", right_on="fantasy_id", how="left", validate="one_to_one",
    )
    fixtures = pd.read_csv(DATA / "ncr" / "ncr_fixtures.csv")
    fixtures = fixtures[pd.to_numeric(fixtures["gameday"], errors="coerce").eq(gw)]
    by_team = {}
    for row in fixtures.itertuples(index=False):
        by_team[row.home] = (row.away, str(row.match_id), row.game_date)
        by_team[row.away] = (row.home, str(row.match_id), row.game_date)
    records = []
    for row in frame.itertuples(index=False):
        opponent, fixture_id, date = by_team[str(row.team)]
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
            "team": row.team, "opponent": opponent, "position": pos,
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


def freeze_shadow(gw: int, engine: str, model_path: Path,
                  *, output_dir: Path = OUT / "shadow", samples: int = 4000) -> Path:
    fixtures = pd.read_csv(DATA / "ncr" / "ncr_fixtures.csv")
    round_rows = fixtures[pd.to_numeric(fixtures["gameday"], errors="coerce").eq(gw)]
    lock_at = pd.to_datetime(round_rows["lock_date"], utc=True).min()
    stem = output_dir / f"ncr_gw{gw}_{engine}"
    csv_path = stem.with_suffix(".csv")
    manifest_path = stem.with_suffix(".manifest.json")
    validate_shadow_write(csv_path, manifest_path, lock_at)
    candidate = ncr_candidates(gw)
    raw = pd.read_csv(DATA / "unified" / "player_match.csv", low_memory=False, parse_dates=["date"])
    # The store must contain no rows from the lock day or later: a same-day row
    # can only be a post-lock result, so its presence means hindsight leakage.
    latest = pd.to_datetime(raw["date"], errors="coerce").max()
    if pd.Timestamp(latest).date() >= lock_at.date():
        raise RuntimeError(
            f"store contains rows dated {latest.date()} on/after the GW{gw} lock "
            f"({lock_at.date()}); refusing shadow freeze"
        )
    combined = pd.concat([raw, candidate], ignore_index=True, sort=False)
    timed = attach_match_timestamps(combined)
    model = _load_model(engine, model_path)
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
    scorer = scorer_for("ncr")
    predictions = model.predict_frame(future)
    rows = []
    for i, (source, prediction) in enumerate(zip(future.itertuples(index=False), predictions)):
        summary = scorer.score_prediction(prediction, n=samples, seed=1701 + i)
        rows.append({
            "fantasy_id": int(source.fantasy_id), "player_id": prediction.player_id,
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
    payload = pd.DataFrame(rows).sort_values(
        ["expected_points", "fantasy_id"], ascending=[False, True],
    ).to_csv(index=False)
    write_once(csv_path, payload)
    manifest = {
        "schema_version": 1, "competition": "ncr", "round": gw,
        "engine": engine, "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_at": lock_at.isoformat(), "model_path": str(model_path),
        "model_sha256": sha256(model_path), "prediction_sha256": sha256(csv_path),
        "rows": len(rows), "immutable": True,
    }
    write_once(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return csv_path


def evaluate_prospective_shadows(
    engine: str, *, rounds: tuple[int, ...] = (4, 5, 6, 7),
    shadow_dir: Path = OUT / "shadow", output_dir: Path = OUT / "prospective",
) -> dict:
    """Apply the frozen NCR non-inferiority gates after all four shadow rounds."""
    labels = build_fantasy_labels()
    labels = labels[
        labels["competition"].eq("ncr") & labels["round"].isin(rounds)
    ].copy()
    missing_labels = sorted(set(rounds) - set(labels["round"].astype(int)))
    if missing_labels:
        raise RuntimeError(f"official NCR labels are not available for GW{missing_labels}")
    pieces = []
    for gw in rounds:
        path = shadow_dir / f"ncr_gw{gw}_{engine}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing immutable shadow: {path}")
        frame = pd.read_csv(path)
        frame["round"] = gw
        pieces.append(frame)
    shadow = pd.concat(pieces, ignore_index=True)
    shadow["key_player"] = pd.to_numeric(
        shadow["fantasy_id"], errors="coerce",
    ).astype("Int64").astype(str)
    labels["key_player"] = labels["key_player"].astype(str)
    evaluated = labels.merge(
        shadow[[
            "round", "key_player", "expected_points", "career_matches",
        ]],
        on=["round", "key_player"], how="left", validate="one_to_one",
    )
    if evaluated["expected_points"].isna().any():
        raise RuntimeError(
            f"{int(evaluated['expected_points'].isna().sum())} official rows lack a shadow"
        )
    incumbents = []
    for gw, block in evaluated.groupby("round", sort=True):
        projection = pd.read_csv(DATA / "ncr" / f"ncr_gw{int(gw)}_projections.csv")
        projection["key_player"] = projection["id"].astype(int).astype(str)
        incumbent = block[["round", "key_player"]].merge(
            projection[["key_player", "starter_exp"]],
            on="key_player", how="left", validate="one_to_one",
        )
        incumbents.append(incumbent)
    evaluated = evaluated.merge(
        pd.concat(incumbents, ignore_index=True).rename(
            columns={"starter_exp": "incumbent_points"}
        ),
        on=["round", "key_player"], how="left", validate="one_to_one",
    )
    if evaluated["incumbent_points"].isna().any():
        raise RuntimeError("incumbent projections are incomplete for prospective cohort")
    candidate_metrics = _group_metrics(
        evaluated.rename(columns={"official_pts": "actual"}),
        "expected_points", actual_col="actual",
    )
    incumbent_metrics = _group_metrics(evaluated, "incumbent_points")
    reasons = []
    if candidate_metrics["mae"] > incumbent_metrics["mae"] * 1.02:
        reasons.append("four-round NCR MAE regression exceeds 2%")
    for n in (10, 25, 50, 100):
        key = f"top_{n}_capture"
        if candidate_metrics[key] < incumbent_metrics[key] - 0.02:
            reasons.append(f"four-round NCR top-{n} capture regression exceeds 2pp")
    subgroup = {}
    for name, mask in (
        ("bench", evaluated["status"].astype(str).eq("B")),
        (
            "low_history",
            pd.to_numeric(evaluated["career_matches"], errors="coerce").fillna(0).lt(5),
        ),
    ):
        rows = evaluated[mask]
        candidate_mae = float(np.mean(np.abs(rows["expected_points"] - rows["official_pts"])))
        incumbent_mae = float(np.mean(np.abs(rows["incumbent_points"] - rows["official_pts"])))
        subgroup[name] = {
            "rows": int(len(rows)), "candidate_mae": candidate_mae,
            "incumbent_mae": incumbent_mae,
        }
        if len(rows) and candidate_mae > incumbent_mae + 0.5:
            reasons.append(f"four-round NCR {name} MAE regression exceeds 0.5")
    retrospective_path = OUT / "benchmark" / "decision.json"
    retrospective = (
        json.loads(retrospective_path.read_text()) if retrospective_path.exists() else {}
    )
    if not retrospective.get("retrospective_gate_passed"):
        reasons.append("retrospective safe-overall gate did not pass")
    payload = {
        "schema_version": 1, "engine": engine, "rounds": list(rounds),
        "candidate": candidate_metrics, "incumbent": incumbent_metrics,
        "subgroups": subgroup, "reasons": reasons,
        "promotion": not reasons,
        "deployment_action": (
            "promote_unified" if not reasons else "retain_specialists_and_stop"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(output_dir / f"{engine}_gw4_gw7_rows.csv", index=False)
    (output_dir / f"{engine}_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Unified v3 NCR GW4-GW7 prospective decision", "",
        f"- Engine: **{engine}**.",
        f"- Candidate MAE: {candidate_metrics['mae']:.2f}; incumbent: "
        f"{incumbent_metrics['mae']:.2f}.",
        f"- Promotion: **{'YES' if not reasons else 'NO'}**.",
    ]
    lines.extend(f"- {reason}" for reason in reasons)
    lines.append("")
    (output_dir / f"{engine}_report.md").write_text("\n".join(lines))
    return payload
