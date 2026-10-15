from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..evaluation import tie_aware_top_n
from ..schema import EVENTS, FORWARD_POSITIONS
from ..scoring import NationsChampionshipScorer
from ..raw_benchmark.blend import EventBlend50
from ..raw_benchmark.config import OUT as RAW_BENCHMARK_OUT
from ..raw_benchmark.features import build_frozen_feature_frames
from ..raw_benchmark.folds import build_folds, strict_training_frame
from model.ncr.score_gw import load_official_actuals
from .harness import OUT, sha256, write_once

CUTOFFS = (10, 25, 50, 100)
POSITION = {
    "Prop": "Prop", "Hooker": "Hooker", "Lock": "Second-row", "Loose Forward": "Back-row", "Scrum Half": "Scrum-half", "Fly Half": "Fly-half",
    "Centre": "Centre", "Back Three": "Back-three",
}
START_JERSEY = {"Prop": 1, "Hooker": 2, "Lock": 4, "Loose Forward": 6, "Scrum Half": 9, "Fly Half": 10, "Centre": 12, "Back Three": 11}
BENCH_JERSEY = {"Prop": 17, "Hooker": 16, "Lock": 19, "Loose Forward": 20, "Scrum Half": 21, "Fly Half": 22, "Centre": 23, "Back Three": 23}


def build_fantasy_labels(ncr_rounds: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    t = pd.read_csv("data/model_targets.csv", low_memory=False)
    t = t[t["season"].isin({2025, 2026}) & t["official_pts"].notna()]
    six = pd.DataFrame(
        {
            "competition": "six_nations", "rubric_era": "modern", "season": t["season"].astype(int), "round": t["round"].astype(int),
            "key_fixture": t["fixture_id"].astype(str), "key_player": t["player_id"].astype(str), "team": t["team"].astype(str),
            "player_name": t["player_name"].astype(str), "position": t["canonical_pos"].astype(str), "is_forward": t["is_forward"].astype(bool),
            "official_pts": t["official_pts"].astype(float),
        }
    )
    six["group_id"] = "6n-" + six["season"].astype(str) + "-r" + six["round"].astype(str)
    ncr = []
    for gw in ncr_rounds:
        official = load_official_actuals(gw)
        actuals = pd.DataFrame(
            [{"id": player, "actual": p["points"], "actual_status": p["status"]} for player, p in official.items()]
        )
        c = pd.read_csv(f"data/ncr/ncr_gw{gw}_projections.csv")[["id", "name", "team", "pos"]].merge(actuals, on="id", how="left", validate="one_to_one")
        if not np.isfinite(c["actual"]).all():
            raise ValueError(f"GW{gw} projections lack finite official points")
        ncr.append(
            pd.DataFrame(
                {
                    "competition": "ncr", "rubric_era": "modern", "season": 2026, "round": gw, "key_fixture": f"ncr-gw{gw}",
                    "key_player": c["id"].astype(int).astype(str), "team": c["team"].astype(str), "player_name": c["name"].astype(str),
                    "position": c["pos"].astype(str), "is_forward": c["pos"].astype(str).str.lower().str.contains("prop|hook|row|forward|flank|lock|number"),
                    "official_pts": c["actual"].astype(float), "status": c["actual_status"].astype(str), "group_id": f"ncr-2026-gw{gw}",
                }
            )
        )
    pooled = pd.concat([six, *ncr], ignore_index=True, sort=False)
    grp = pooled.groupby("group_id")["official_pts"]
    pooled["label_percentile"] = grp.rank(pct=True, method="average").to_numpy()
    pooled["label_z"] = ((pooled["official_pts"] - grp.transform("mean")) / grp.transform("std").replace(0, 1.0)).to_numpy()
    pooled["status"] = pooled["status"].fillna("")
    return pooled


def _group_metrics(cohort: pd.DataFrame, rank_col: str) -> dict:
    rank, actual = cohort[rank_col].reset_index(drop=True), cohort["official_pts"].reset_index(drop=True)
    row = {"n": len(cohort), "mae": float(np.mean(np.abs(rank - actual))), "spearman": float(spearmanr(rank, actual).statistic) if len(cohort) > 2 else np.nan}
    for n in CUTOFFS:
        if len(cohort) >= n:
            row[f"top_{n}_overlap"], row[f"top_{n}_capture"] = tie_aware_top_n(rank, actual, n)
    return row


def _fixture_rows(gw: int) -> pd.DataFrame:
    fixtures = pd.read_csv("data/ncr/ncr_fixtures.csv")
    return fixtures[pd.to_numeric(fixtures["gameday"], errors="coerce").eq(gw)]


def ncr_candidates(gw: int, projection: pd.DataFrame | None = None) -> pd.DataFrame:
    if projection is not None:
        projection = projection.copy()
    elif os.path.exists(f"data/ncr/ncr_gw{gw}_projections.csv"):
        projection = pd.read_csv(f"data/ncr/ncr_gw{gw}_projections.csv")
    else:
        from model.ncr.project import build_projection

        projection = build_projection(gameday=gw)
        if projection.empty:
            raise RuntimeError(f"GW{gw} projection is empty; wait for real team sheets before freezing")
    crosswalk = pd.read_csv("data/ncr/ncr_player_crosswalk.csv")[["fantasy_id", "api_player_id", "full_name"]]
    crosswalk["fantasy_id"] = pd.to_numeric(crosswalk["fantasy_id"], errors="coerce")
    projection["id"] = pd.to_numeric(projection["id"])
    by_team = {}
    for r in _fixture_rows(gw).itertuples(index=False):
        by_team[r.home], by_team[r.away] = (r.away, str(r.match_id), r.game_date, "home"), (r.home, str(r.match_id), r.game_date, "away")
    records = []
    for row in projection.merge(crosswalk, left_on="id", right_on="fantasy_id", how="left").itertuples(index=False):
        opponent, fixture_id, date, home_away = by_team[str(row.team)]
        pos, started = POSITION[str(row.pos)], str(row.status).upper() == "P"
        records.append(
            {
                "date": date, "competition": "Nations Championship", "competition_id": 9999, "competition_level": "international", "season": 2026, "round": gw,
                "fixture_id": fixture_id, "player_id": str(int(row.api_player_id)) if pd.notna(row.api_player_id) else f"fantasy_{int(row.id)}",
                "fantasy_id": int(row.id), "player_name": row.full_name if isinstance(row.full_name, str) else row.name, "team": row.team, "opponent": opponent,
                "position": pos, "home_away": home_away, "is_forward": pos in FORWARD_POSITIONS, "started": started,
                "jersey": (START_JERSEY if started else BENCH_JERSEY)[str(row.pos)], "source": "v3_shadow_candidate", "source_priority": 999,
            }
        )
    candidates = pd.DataFrame(records)
    for target in ("minutes", *EVENTS):
        candidates[target], candidates[f"available__{target}"] = np.nan, False
    return candidates


def _load_model(engine: str, path: str):
    return EventBlend50.load(path)


def validate_shadow_write(csv_path: str, manifest_path: str, lock_at: pd.Timestamp, *, now: pd.Timestamp | None = None) -> None:
    if os.path.exists(csv_path) or os.path.exists(manifest_path):
        raise FileExistsError(f"shadow snapshot already frozen: {os.path.basename(csv_path).removesuffix('.csv')}")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    if current >= lock_at:
        raise RuntimeError(f"lock passed at {lock_at}; refusing retrospective shadow")


def freeze_shadow(gw: int, engine: str, model_path: str) -> str:
    lock_at = pd.to_datetime(_fixture_rows(gw)["lock_date"], utc=True).min()
    stem = f"{OUT}/shadow/ncr_gw{gw}_{engine}"
    csv_path, manifest_path = f"{stem}.csv", f"{stem}.manifest.json"
    validate_shadow_write(csv_path, manifest_path, lock_at)
    candidate = ncr_candidates(gw)
    model = _load_model(engine, model_path)
    store = pd.read_csv(f"{RAW_BENCHMARK_OUT}/player_match.csv", low_memory=False, parse_dates=["date", "match_at"])
    fold = next(fold for fold in build_folds(store) if fold.label == "nations_championship_2026")
    future = build_frozen_feature_frames(strict_training_frame(store, fold), candidate, v4=True)[1]
    scorer = NationsChampionshipScorer()
    rows = []
    for i, (source, p) in enumerate(zip(future.itertuples(index=False), model.predict_frame(future))):
        s = scorer.score_prediction(p, n=4000, seed=1701 + i)
        rows.append(
            {
                "fantasy_id": int(source.fantasy_id), "player_id": p.player_id, "player_name": p.player_name, "team": p.team, "opponent": p.opponent,
                "position": p.position, "started": bool(source.started), "engine": engine, "expected_points": s.mean, "p10": s.p10, "median": s.median,
                "p90": s.p90, "p_play": None, "expected_minutes": p.minutes.mean, "career_matches": float(source.career_matches),
            }
        )
    write_once(csv_path, pd.DataFrame(rows).sort_values(["expected_points", "fantasy_id"], ascending=[False, True]).to_csv(index=False))
    manifest = {
        "schema_version": 1, "competition": "ncr", "round": gw, "engine": engine, "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_at": lock_at.isoformat(), "model_path": str(model_path), "model_sha256": sha256(model_path), "prediction_sha256": sha256(csv_path),
        "rows": len(rows), "immutable": True, "blend_weight_v4": model.weight_v4,
    }
    write_once(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return csv_path


def evaluate_prospective_shadows(engine: str) -> dict:
    rounds = (4, 5, 6, 7)
    labels = build_fantasy_labels(ncr_rounds=rounds)
    labels = labels[labels["competition"].eq("ncr") & labels["round"].isin(rounds)].copy()
    missing_labels = sorted(set(rounds) - set(labels["round"].astype(int)))
    if missing_labels:
        raise RuntimeError(f"official NCR labels are not available for GW{missing_labels}")
    pieces = []
    for gw in rounds:
        path = f"{OUT}/shadow/ncr_gw{gw}_{engine}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing immutable shadow: {path}")
        pieces.append(pd.read_csv(path).assign(round=gw))
    shadow = pd.concat(pieces, ignore_index=True)
    shadow["key_player"] = pd.to_numeric(shadow["fantasy_id"], errors="coerce").astype("Int64").astype(str)
    labels["key_player"] = labels["key_player"].astype(str)
    evaluated = labels.merge(shadow[["round", "key_player", "expected_points", "career_matches"]], on=["round", "key_player"], how="left", validate="one_to_one")
    if not np.isfinite(evaluated["expected_points"]).all():
        raise RuntimeError(f"{int((~np.isfinite(evaluated['expected_points'])).sum())} official rows lack finite shadow points")
    incumbents = []
    for gw, block in evaluated.groupby("round", sort=True):
        projection = pd.read_csv(f"data/ncr/ncr_gw{int(gw)}_projections.csv")
        projection["key_player"] = projection["id"].astype(int).astype(str)
        incumbents.append(block[["round", "key_player"]].merge(projection[["key_player", "starter_exp"]], on="key_player", how="left"))
    evaluated = evaluated.merge(
        pd.concat(incumbents, ignore_index=True).rename(columns={"starter_exp": "incumbent_points"}), on=["round", "key_player"], how="left"
    )
    if not np.isfinite(evaluated["incumbent_points"]).all():
        raise RuntimeError("incumbent projections are incomplete for prospective cohort")
    candidate, incumbent = _group_metrics(evaluated, "expected_points"), _group_metrics(evaluated, "incumbent_points")
    reasons = ["four-round NCR MAE regression exceeds 2%"] if candidate["mae"] > incumbent["mae"] * 1.02 else []
    reasons += [
        f"four-round NCR top-{n} capture regression exceeds 2pp"
        for n in (10, 25, 50, 100)
        if candidate[f"top_{n}_capture"] < incumbent[f"top_{n}_capture"] - 0.02
    ]
    subgroup = {}
    for name, mask in (
        ("bench", evaluated["status"].astype(str).eq("B")), ("low_history", pd.to_numeric(evaluated["career_matches"], errors="coerce").fillna(0).lt(5)),
    ):
        rows = evaluated[mask]
        candidate_mae = float(np.mean(np.abs(rows["expected_points"] - rows["official_pts"])))
        incumbent_mae = float(np.mean(np.abs(rows["incumbent_points"] - rows["official_pts"])))
        subgroup[name] = {"rows": int(len(rows)), "candidate_mae": candidate_mae, "incumbent_mae": incumbent_mae}
        if len(rows) and candidate_mae > incumbent_mae + 0.5:
            reasons.append(f"four-round NCR {name} MAE regression exceeds 0.5")
    retrospective = f"{OUT}/benchmark/decision.json"
    if not (json.load(open(retrospective)) if os.path.exists(retrospective) else {}).get("retrospective_gate_passed"):
        reasons.append("retrospective safe-overall gate did not pass")
    payload = {
        "schema_version": 1, "engine": engine, "rounds": list(rounds), "candidate": candidate, "incumbent": incumbent, "subgroups": subgroup,
        "reasons": reasons, "promotion": not reasons, "deployment_action": "promote_unified" if not reasons else "retain_specialists_and_stop",
    }
    out = f"{OUT}/prospective"
    os.makedirs(out, exist_ok=True)
    evaluated.to_csv(f"{out}/{engine}_gw4_gw7_rows.csv", index=False)
    open(f"{out}/{engine}_decision.json", "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Unified v3 NCR GW4-GW7 prospective decision", "", f"- Engine: **{engine}**.",
        f"- Candidate MAE: {candidate['mae']:.2f}; incumbent: {incumbent['mae']:.2f}.", f"- Promotion: **{'YES' if not reasons else 'NO'}**.",
        *(f"- {reason}" for reason in reasons), "",
    ]
    open(f"{out}/{engine}_report.md", "w").write("\n".join(lines))
    return payload
