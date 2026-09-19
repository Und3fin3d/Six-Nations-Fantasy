"""Matched rolling histories; old frozen tournament reports remain untouched.

Every round refits the same fixed model settings using all available prior-day
matches, including earlier rounds. This is retrospective: final NCR feeds and
Six Nations observed lineups are oracle inputs, not historical publication logs.
Six Nations team points are positional diagnostics until historical prices and
complete eligible pools are supplied; they cannot satisfy the deployment gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from model import empirical_unified, ncr_project
from model.ncr_eval import load_actuals, team_points
from model.pit import day_cutoff, history_before
from .data import ROOT, DEFAULT_SOURCES, build_canonical_store
from .labels import _six_nations_rows
from .contracts import RawPrediction
from .raw_benchmark.blend import EventBlend50, EventWeightedBlend
from .raw_benchmark.config import STABLE_EVENTS, EXTENDED_EVENTS
from .raw_benchmark.coverage import build_corrected_store
from .raw_benchmark.empirical import EmpiricalEventModel
from .raw_benchmark.features import build_frozen_feature_frames
from .raw_benchmark.folds import HistoricalFold, masked_candidates, strict_training_frame
from .scoring import NationsChampionshipScorer, SixNationsScorer
from .v3.shadow import ncr_candidates
from .v4.gbdt import V4GBDT

DATA = ROOT / "data"
OUT = DATA / "unified" / "rolling_history"
GRAIN = ["fixture_id", "player_id", "team"]
MODELS = ("empirical_baseline", "p3_rolling", "p3_weighted", "p3_shrunk")
ROUND_IDS = tuple(f"6n-{year}-r{r}" for year in (2025, 2026) for r in range(1, 6)) + tuple(f"ncr-2026-gw{r}" for r in (1, 2, 3))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_store(output: Path = OUT) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    paths = [p for p, *_ in DEFAULT_SOURCES] + [DATA / "official_player_match.csv"]
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    manifest_path, store_path = output / "store_manifest.json", output / "player_match.csv"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["sources"] != sources or saved["store_sha256"] != sha256(store_path):
            raise ValueError("rolling store does not match its sources; use a new output directory")
    else:
        canonical = build_canonical_store()
        canonical_path = output / "canonical_player_match.csv"
        canonical.to_csv(canonical_path, index=False)
        store, coverage, report = build_corrected_store(canonical_path)
        store.to_csv(store_path, index=False)
        coverage.to_csv(output / "fixture_event_coverage.csv", index=False)
        (output / "source_report.json").write_text(json.dumps(report, indent=2))
        manifest_path.write_text(json.dumps({"sources": sources, "store_sha256": sha256(store_path)}, indent=2))
    store = pd.read_csv(store_path, low_memory=False, dtype={k: str for k in GRAIN}, parse_dates=["date", "match_at"])
    if store.duplicated(GRAIN).any():
        raise ValueError("rolling store violates player/fixture/team grain")
    return store


def cohort(store: pd.DataFrame, round_id: str):
    if round_id.startswith("6n-"):
        season = int(round_id.split("-")[1])
        labels = _six_nations_rows((season,), include_2023=False)
        labels = labels[labels["group_id"].eq(round_id)].reset_index(drop=True)
        if labels.empty:
            raise ValueError(f"no official labels for {round_id}")
        wanted = labels[["key_fixture", "key_player", "team"]].rename(columns={"key_fixture": "fixture_id", "key_player": "player_id"})
        rows = wanted.merge(store, on=GRAIN, how="left", validate="one_to_one", indicator=True)
        if not rows["_merge"].eq("both").all():
            raise ValueError("Six Nations official cohort lacks match rows")
        rows = rows.drop(columns="_merge")
        cutoff = day_cutoff(rows["match_at"].min())
        return cutoff, masked_candidates(rows), labels, None
    gw = int(round_id.rsplit("gw", 1)[1])
    fixtures = pd.read_csv(DATA / "ncr" / "ncr_fixtures.csv")
    fixtures = fixtures[pd.to_numeric(fixtures["gameday"]).eq(gw)]
    cutoff = day_cutoff(pd.to_datetime(fixtures["lock_date"], utc=True).min())
    feed = DATA / "ncr" / "feeds" / f"players_gw{gw}.json"
    pool = pd.DataFrame(json.loads(feed.read_text())["Data"]["Value"]["Players"])
    pool["id"] = pd.to_numeric(pool["id"], errors="raise").astype(int)
    pool = pool[pool["player_status"].isin(["P", "B"])].copy()
    baseline = ncr_project.build_projection(pool_override=pool, asof=cutoff, gameday=gw, use_weather=False)
    baseline["id"] = pd.to_numeric(baseline["id"], errors="raise").astype(int)
    rows = ncr_candidates(gw, projection=baseline)
    homes = set(fixtures["home"])
    rows["home_away"] = np.where(rows["team"].isin(homes), "home", "away")
    rows["date"] = pd.to_datetime(rows["date"], errors="raise")
    actuals = load_actuals(feed)
    labels = pd.DataFrame({"official_pts": [actuals[str(int(i))][0] for i in baseline["id"]]})
    if not np.array_equal(rows["fantasy_id"].to_numpy(), baseline["id"].to_numpy()):
        raise ValueError("NCR cohort ordering changed")
    return cutoff, masked_candidates(rows), labels, baseline


def expected_points(predictions: list[RawPrediction], competition: str) -> np.ndarray:
    scorer = NationsChampionshipScorer() if competition == "ncr" else SixNationsScorer()
    scores = []
    for p in predictions:
        value = sum(weight * p.events[event].mean for event, weight in scorer.weights.items() if event in p.events)
        if competition == "ncr":
            value += 2.0 * p.events["scrums_won"].mean if "scrums_won" in p.events else 0.0
        else:
            value += (15.0 if p.is_forward else 10.0) * p.events["tries"].mean if "tries" in p.events else 0.0
            metres = p.events.get("metres")
            if metres and metres.mean > 0:
                sigma2 = np.log1p(metres.dispersion / max(metres.mean ** 2, 1e-9))
                sigma = np.sqrt(sigma2)
                scale = np.exp(np.log(metres.mean) - sigma2 / 2.0)
                stop = int(np.ceil(lognorm.isf(1e-12, sigma, scale=scale) / 10))
                if stop > 2_000_000:
                    raise ValueError("metres distribution tail is too large for reliable deterministic scoring")
                value += float(lognorm.sf(10.0 * np.arange(1, stop + 1), sigma, scale=scale).sum())
        scores.append(float(value))
    return np.asarray(scores)


def positional_team(rows: pd.DataFrame, scores: np.ndarray, actual: np.ndarray):
    from model.evaluate import _pick_xv
    frame = rows.assign(canonical_pos=rows["position"], predicted=scores, official_pts=actual)
    xv = _pick_xv(frame, "predicted")
    if len(xv) != 15:
        raise ValueError("incomplete positional XV")
    bench = frame.loc[~frame.index.isin(xv.index) & ~frame["started"].astype(bool)]
    if bench.empty:
        raise ValueError("no bench candidate for positional diagnostic")
    captain, sub = xv["predicted"].idxmax(), bench["predicted"].idxmax()
    selected = pd.concat([xv, frame.loc[[sub]]]).copy()
    selected["multiplier"] = 1
    selected.loc[captain, "multiplier"] = 2
    selected.loc[sub, "multiplier"] = 3
    return float((selected["official_pts"] * selected["multiplier"]).sum()), selected


def run(round_id: str, output: Path = OUT) -> pd.DataFrame:
    if round_id not in ROUND_IDS:
        raise ValueError(f"unsupported round {round_id}")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    start = time.monotonic()
    store = prepare_store(output)
    cutoff, candidates, labels, baseline = cohort(store, round_id)
    candidates = candidates.reset_index(drop=True)
    competition = "ncr" if baseline is not None else "six_nations"
    current_fixtures = tuple(candidates["fixture_id"].astype(str).unique())
    fold = HistoricalFold(round_id, competition, cutoff.year, cutoff.isoformat(), current_fixtures, (round_id,) * len(current_fixtures), "mixed")
    train = strict_training_frame(store, fold).reset_index(drop=True)
    train = history_before(train, cutoff)
    if train.empty or set(train["fixture_id"]) & set(current_fixtures):
        raise AssertionError("invalid training boundary")
    if candidates.duplicated(GRAIN).any():
        raise ValueError("candidate grain is not unique")
    for col in candidates.columns:
        if col.startswith("available__") and candidates[col].any():
            raise AssertionError("unmasked candidate outcome")
    round_dir = output / round_id
    round_dir.mkdir(exist_ok=True)
    if (round_dir / "metrics.json").exists():
        raise FileExistsError("refusing to overwrite completed round results")
    sources = [output / "player_match.csv", DATA / "rp_compstats.csv", DATA / "wr_rankings.csv", DATA / "unified" / "p3_hillclimb" / "config.json"]
    if competition == "ncr":
        sources += [DATA / "ncr" / "feeds" / f"players_gw{int(round_id.rsplit('gw', 1)[1])}.json", DATA / "ncr" / "ncr_player_crosswalk.csv", DATA / "ncr" / "ncr_fixtures.csv"]
    else:
        sources += [DATA / "model_targets.csv"]
    code_paths = [ROOT / "model" / "pit.py", ROOT / "model" / "ncr_project.py", ROOT / "model" / "rp_rates.py", ROOT / "model" / "empirical_unified.py", *sorted((ROOT / "model" / "unified").rglob("*.py"))]
    manifest = {
        "round": round_id, "training_cutoff": cutoff.isoformat(),
        "training_rows": len(train), "training_date_max": str(train["date"].max()),
        "training_fixture_ids": sorted(train["fixture_id"].astype(str).unique()),
        "evaluation_fixture_ids": list(current_fixtures), "evaluation_rows": len(candidates),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
        "python": platform.python_version(), "rolling": True,
        "official_outcomes_used_for_fitting": False, "lineup_basis": "retrospective observed lineup",
        "six_nations_budget_verified": False, "retrospective_only": True,
        "model_settings": {"trees": 180, "leaves": 23, "minutes_prior_matches_challenger": 3.0},
    }
    (round_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{round_id}: {len(train)} training rows before {cutoff}; {len(candidates)} candidates", flush=True)
    train_features, future_features = build_frozen_feature_frames(train, candidates, v4=True)
    print(f"{round_id}: fitting raw-event trees", flush=True)
    events = (*STABLE_EVENTS, *EXTENDED_EVENTS)
    tree = V4GBDT(events=events, weighting="natural", pool_player_id=True, player_effects=True).fit(train_features)
    empirical = EmpiricalEventModel(asof=cutoff).fit(train)
    shrunk = EmpiricalEventModel(asof=cutoff, minutes_prior_matches=3.0, exposure_weighted_priors=True).fit(train)
    weights = json.loads((DATA / "unified" / "p3_hillclimb" / "config.json").read_text())["event_weights_v4"]
    models = {
        "p3_rolling": EventBlend50(empirical=empirical, v4=tree),
        "p3_weighted": EventWeightedBlend(empirical=empirical, v4=tree, event_weights_v4=weights),
        "p3_shrunk": EventWeightedBlend(empirical=shrunk, v4=tree, event_weights_v4=weights),
    }
    if baseline is not None:
        baseline_scores = baseline["starter_exp"].to_numpy(float)
    else:
        wr = pd.read_csv(DATA / "wr_rankings.csv", parse_dates=["snapshot_date"])
        prediction = empirical_unified.project_round(labels, store, "six_nations", wr)
        prediction = prediction.set_index("label_row_id").reindex(range(len(labels)))
        baseline_scores = prediction["predicted_points"].to_numpy(float)
    predictions_by_model = {"empirical_baseline": baseline_scores}
    for name, model in models.items():
        predictions = model.predict_frame(future_features)
        grain = pd.DataFrame([(p.fixture_id, p.player_id, p.team) for p in predictions], columns=GRAIN)
        if not grain.astype(str).equals(candidates[GRAIN].astype(str).reset_index(drop=True)):
            raise AssertionError("prediction row alignment failed")
        (round_dir / f"{name}_raw.jsonl").write_text("".join(json.dumps(p.to_dict()) + "\n" for p in predictions))
        predictions_by_model[name] = expected_points(predictions, competition)
    actual = labels["official_pts"].to_numpy(float)
    metrics = []
    for name, scores in predictions_by_model.items():
        if not np.isfinite(scores).all() or not np.isfinite(actual).all():
            raise ValueError("non-finite prediction or actual")
        if baseline is None:
            total, squad = positional_team(candidates, scores, actual)
            team_basis = "positional diagnostic; prices and complete eligible pool unavailable"
        else:
            projected = baseline.copy()
            projected["starter_exp"] = scores
            projected["supersub_exp"] = np.where(projected["status"].eq("B"), 3.0, 0.5) * scores
            squad, _, _ = ncr_project.optimise(projected)
            actuals = load_actuals(DATA / "ncr" / "feeds" / f"players_gw{int(round_id.rsplit('gw', 1)[1])}.json")
            total = team_points(squad, actuals)
            team_basis = "same budget and squad constraints; retrospective final feed"
        squad.to_csv(round_dir / f"{name}_squad.csv", index=False)
        pd.DataFrame({"player_id": candidates["player_id"], "team": candidates["team"], "actual": actual, "predicted": scores}).to_csv(round_dir / f"{name}_predictions.csv", index=False)
        row = {"round": round_id, "competition": competition, "season": cutoff.year, "model": name, "n": len(scores), "mae": float(np.mean(np.abs(scores - actual))), "team_points": float(total), "team_basis": team_basis}
        metrics.append(row)
        print(json.dumps(row), flush=True)
    (round_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"{round_id}: complete in {time.monotonic() - start:.1f}s", flush=True)
    return pd.DataFrame(metrics)


def summarize(output: Path = OUT) -> dict:
    rows = []
    missing = []
    for round_id in ROUND_IDS:
        path = output / round_id / "metrics.json"
        if path.exists():
            rows.extend(json.loads(path.read_text()))
        else:
            missing.append(round_id)
    if not rows:
        raise ValueError("no completed rolling comparisons")
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    summary = metrics.groupby(["competition", "season", "model"]).agg(mae=("mae", "mean"), team_points=("team_points", "sum"), rounds=("round", "nunique")).reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    checks = {}
    for model in MODELS[1:]:
        failed = []
        for (comp, year), group in summary.groupby(["competition", "season"]):
            indexed = group.set_index("model")
            if model not in indexed.index or "empirical_baseline" not in indexed.index:
                failed.append(f"{comp}/{year}: model or baseline missing")
                continue
            candidate, baseline = indexed.loc[model], indexed.loc["empirical_baseline"]
            if not candidate.mae < baseline.mae:
                failed.append(f"{comp}/{year}: MAE did not improve")
            if not candidate.team_points > baseline.team_points:
                failed.append(f"{comp}/{year}: team points did not improve")
        checks[model] = {"all_available_metric_checks_passed": not failed and not missing, "failures": failed}
    result = {"pr_eligible": False, "missing_rounds": missing, "candidates": checks,
              "limitations": ["Six Nations budget-valid team comparison unavailable", "All lineups are retrospective observed inputs", "Existing 2026 repository results are not an untouched prospective holdout"]}
    (output / "decision.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", choices=ROUND_IDS)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.prepare:
        store = prepare_store(args.output)
        print(f"Prepared {len(store)} rows through {store['date'].max()}")
    elif args.summarize:
        summarize(args.output)
    elif args.round:
        run(args.round, args.output)
    else:
        parser.error("choose --prepare, --round, or --summarize")


if __name__ == "__main__":
    main()
