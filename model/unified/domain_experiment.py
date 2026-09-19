"""Bounded shared-domain experiment and fixed Friendly-15 evaluation.

Reuses the matched-history runner's data, scorers and squad optimiser. Output
is research evidence only; this module never changes a model route or a PR.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from model.history import past_matches
from model.empirical_unified import project_candidates
from .contracts import RawPrediction
from .data import ROOT
from .features import build_pit_features
from .friendly_eval import score_stats, summarise_stats, validate_manifest
from .raw_benchmark.blend import EventWeightedBlend
from .raw_benchmark.config import STABLE_EVENTS, EXTENDED_EVENTS
from .raw_benchmark.coverage import sha256
from .raw_benchmark.empirical import EmpiricalEventModel
from .raw_benchmark.robust_empirical import RobustEmpiricalEventModel
from .raw_benchmark.features import build_frozen_feature_frames
from .raw_benchmark.folds import masked_candidates
from .rolling_eval import prepare_store, official_slates, evaluate, expected_points
from .v4.features import add_v4_base_stats
from .v4.gbdt import V4GBDT

DATA = ROOT / "data"
CONFIG_DIR = DATA / "unified" / "friendly15"
KEY = ["fixture_id", "player_id", "team"]
CONTROL = "p3_robust_native"


@dataclass(frozen=True)
class Candidate:
    weighting: str
    half_life_days: float | None = None


CANDIDATES = {
    CONTROL: Candidate("natural"),
    "p3_balanced_native": Candidate("level_balanced"),
    "p3_balanced_recent_native": Candidate("level_balanced", 730.0),
}


def forbidden_fixtures(store: pd.DataFrame) -> set[str]:
    labels = pd.read_csv(DATA / "model_targets.csv", dtype={"fixture_id": str})
    forbidden = set(labels.loc[labels["official_pts"].notna(), "fixture_id"])
    forbidden.update(store.loc[pd.to_numeric(store["competition_id_cache"]).eq(696), "fixture_id"].astype(str))
    return forbidden


def prepare(output: Path) -> pd.DataFrame:
    store = prepare_store(output)
    path = output / "inputs" / "domain_features.pkl"
    feature_manifest = {
        "store_sha256": sha256(output / "inputs" / "player_match.csv"),
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in (
            ROOT / "model/unified/features.py", ROOT / "model/unified/v4/features.py")},
        "numpy": version("numpy"), "pandas": version("pandas"),
    }
    manifest_path = path.with_suffix(".json")
    if path.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != feature_manifest:
            raise ValueError("stale training features; choose a new output directory")
    else:
        print("Building shared prior-only features", flush=True)
        add_v4_base_stats(build_pit_features(store)).to_pickle(path)
        _record(manifest_path, feature_manifest)
    return store


def _record(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"incompatible immutable experiment manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _save_raw(path: Path, raw: list[RawPrediction]) -> None:
    path.write_text("".join(json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in raw))


def run_job(output: Path, job: str, engines: tuple[str, ...]) -> None:
    """A job is dev:<fixture>, test:<fixture>, or an official slate name."""
    if not engines or set(engines) - set(CANDIDATES):
        raise ValueError("unknown or empty candidate set")
    from model_env_preflight import check
    errors = check(ROOT / "requirements-model.txt")
    if errors:
        raise RuntimeError("incompatible model environment: " + "; ".join(errors))
    selection_path = CONFIG_DIR / "selection.json"
    if not job.startswith("dev:"):
        allowed = {CONTROL}
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection["selected"] is not None:
                allowed.add(selection["selected"])
        if set(engines) - allowed:
            raise ValueError("test/fantasy engines must be admitted by frozen development selection")
    start = time.monotonic()
    store = prepare(output)
    prepared = pd.read_pickle(output / "inputs" / "domain_features.pkl")
    source_manifest = {
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "model").rglob("*.py"))},
        "store_sha256": sha256(output / "inputs" / "player_match.csv"),
        "features_sha256": sha256(output / "inputs" / "domain_features.pkl"),
        "weight_config_sha256": sha256(DATA / "unified" / "p3_hillclimb" / "config.json"),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")},
        "candidate_configs": {name: asdict(CANDIDATES[name]) for name in engines},
        "job": job,
    }
    if not job.startswith("dev:") and selection_path.exists():
        source_manifest["selection_sha256"] = sha256(selection_path)
    slate = None
    if job.startswith(("dev:", "test:")):
        scope, fixture = job.split(":", 1)
        manifest_path = CONFIG_DIR / ("development_fixtures.json" if scope == "dev" else "fixtures.json")
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest, store, DATA / "cache", forbidden=forbidden_fixtures(store),
                          expected_count=12 if scope == "dev" else 15)
        chosen = next((row for row in manifest["fixtures"] if row["fixture_id"] == fixture), None)
        if chosen is None:
            raise ValueError("fixture is not in the frozen manifest")
        cutoff = pd.Timestamp(chosen["kickoff"])
        truth = store[store.fixture_id.eq(fixture)].sort_values(KEY).reset_index(drop=True)
        candidates = masked_candidates(truth.drop(columns=["team_score", "opp_score"], errors="ignore"))
        source_manifest["fixture_manifest_sha256"] = sha256(manifest_path)
    else:
        slates = official_slates(store, ("ncr", "six_nations"))
        slate = next((s for s in slates if s.name == job), None)
        if slate is None:
            raise ValueError("unknown official slate")
        cutoff, candidates, truth = slate.cutoff, slate.candidates, None
    train = past_matches(store, cutoff).sort_values(["date", "fixture_id", "team", "player_id"])
    if set(train.fixture_id.astype(str)) & set(candidates.fixture_id.astype(str)):
        raise ValueError("current fixture leaked into history")
    features, candidate_features = build_frozen_feature_frames(
        train, candidates, v4=True, prepared_train=prepared.loc[train.index])
    del prepared
    directory = output / "jobs" / job.replace(":", "_")
    source_manifest.update({
        "cutoff": cutoff.isoformat(), "training_rows": len(train),
        "training_fixtures": int(train.fixture_id.nunique()),
        "training_match_at_max": train.match_at.max().isoformat(),
        "candidate_keys": list(candidates[KEY].astype(str).itertuples(index=False, name=None)),
        "history_keys_sha256": __import__("hashlib").sha256(train[KEY].to_csv(index=False).encode()).hexdigest(),
    })
    _record(directory / "manifest.json", source_manifest)
    robust = RobustEmpiricalEventModel(asof=cutoff).fit(train)
    blend_config = json.loads((DATA / "unified" / "p3_hillclimb" / "config.json").read_text())
    metrics = []
    if slate is None:
        control_raw = EmpiricalEventModel(asof=cutoff).fit(train).predict_frame(candidate_features)
        _save_raw(directory / "empirical_raw.jsonl", control_raw)
        metrics.append(score_stats(truth, control_raw, engine="empirical_raw"))
        truth[[*KEY, "position", "started", "minutes", *STABLE_EVENTS,
               "available__minutes", *(f"available__{e}" for e in STABLE_EVENTS)]].to_csv(directory / "truth.csv", index=False)
    else:
        if slate.baseline is None:
            wr = pd.read_csv(DATA / "wr_rankings.csv", parse_dates=["snapshot_date"])
            baseline = project_candidates(slate.candidates, train, slate.competition, wr, asof=cutoff)
            slate.baseline = baseline.set_index("label_row_id").reindex(slate.candidates.label_row_id).predicted_points.to_numpy(float)
        metrics.append(pd.DataFrame([evaluate(slate, "empirical_baseline", slate.baseline, directory)]))
    for name in engines:
        settings = CANDIDATES[name]
        print(f"Fitting {job}: {name}", flush=True)
        tree = V4GBDT(events=(*STABLE_EVENTS, *EXTENDED_EVENTS), weighting=settings.weighting,
            time_half_life_days=settings.half_life_days, pool_player_id=True,
            player_effects=True, native_categories=True).fit(features)
        model = EventWeightedBlend(robust, tree, weight_v4=blend_config["default_weight_v4"],
                                  event_weights_v4=blend_config["event_weights_v4"])
        raw = model.predict_frame(candidate_features)
        _save_raw(directory / f"{name}.jsonl", raw)
        if slate is None:
            metrics.append(score_stats(truth, raw, engine=name))
        else:
            metrics.append(pd.DataFrame([evaluate(slate, name, expected_points(raw, slate.competition), directory)]))
        pd.concat(metrics, ignore_index=True).to_csv(directory / "metrics.csv", index=False)
        del model, tree
    print(f"Completed {job} in {time.monotonic()-start:.1f}s", flush=True)


def select_development(metrics: pd.DataFrame, fixture_ids: list[str]) -> dict:
    development = json.loads((CONFIG_DIR / "development_fixtures.json").read_text())
    expected = {row["fixture_id"] for row in development["fixtures"]}
    if len(fixture_ids) != 12 or set(fixture_ids) != expected:
        raise ValueError("selection must use exactly the frozen 12 development fixtures")
    summary, _ = summarise_stats(metrics, fixture_ids)
    board = summary.set_index("engine")
    if not set(CANDIDATES).issubset(board.index):
        raise ValueError("both predeclared candidates and control must complete development")
    control = board.loc[CONTROL]
    if not bool(control.count_support_complete):
        raise ValueError("development control has incomplete statistic support")
    eligible = []
    for name in CANDIDATES:
        if name == CONTROL:
            continue
        row = board.loc[name]
        if (bool(row.count_support_complete) and row.count_stat_mae < control.count_stat_mae
            and row.minutes_mae <= 1.02 * control.minutes_mae
            and row.metres_mae <= 1.02 * control.metres_mae):
            eligible.append(name)
    selected = min(eligible, key=lambda name: (board.loc[name].count_stat_mae, name)) if eligible else None
    return {"selected": selected, "eligible": eligible, "development_fixture_ids": fixture_ids,
            "selection_used_friendly15": False, "selection_used_fantasy_outcomes": False,
            "summary": summary.to_dict(orient="records")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--engines", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output)
    elif args.job:
        run_job(args.output, args.job, tuple(args.engines))
    else:
        parser.error("supply --job or --prepare-only")


if __name__ == "__main__":
    main()
