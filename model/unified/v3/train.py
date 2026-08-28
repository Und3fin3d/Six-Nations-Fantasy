"""Fit a frozen v3 artifact through an explicit exclusive lock timestamp."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..data import ROOT
from ..features import build_pit_features
from ..gbdt import UniversalGBDT
from .blend import EventBlendModel
from .config import BaselineConfig, GBDTV3Config, NeuralV3Config
from .context import augment_context, restrict_context
from .gbdt import ExposureRateGBDT
from .harness import OUT, attach_match_timestamps, sha256
from .neural import ExposureRateNeural

DATA = ROOT / "data"


def _configs() -> tuple[BaselineConfig, GBDTV3Config, NeuralV3Config]:
    search = OUT / "search"
    b_path = search / "best_baseline.json"
    g_path, n_path = search / "best_gbdt.json", search / "best_neural.json"
    baseline = (
        BaselineConfig.from_dict(json.loads(b_path.read_text()))
        if b_path.exists() else BaselineConfig()
    )
    gbdt = (
        GBDTV3Config.from_dict(json.loads(g_path.read_text()))
        if g_path.exists() else GBDTV3Config()
    )
    neural = (
        NeuralV3Config.from_dict(json.loads(n_path.read_text()))
        if n_path.exists() else NeuralV3Config()
    )
    audit_path = OUT / "audit" / "audit.json"
    appearance = False
    if audit_path.exists():
        appearance = bool(
            json.loads(audit_path.read_text()).get("zero_minutes", {})
            .get("appearance_head_enabled")
        )
    gbdt = replace(gbdt, use_appearance=appearance)
    neural = replace(neural, use_appearance=appearance)
    return baseline, gbdt, neural


def fit_frozen(engine: str, cutoff: str | pd.Timestamp, output: Path,
               *, force: bool = False):
    if output.exists() and not force:
        raise FileExistsError(f"artifact exists; pass --force to replace: {output}")
    raw_path = DATA / "unified" / "player_match.csv"
    raw = pd.read_csv(raw_path, low_memory=False, parse_dates=["date"])
    timed = attach_match_timestamps(raw)
    baseline_config, gbdt_config, neural_config = _configs()
    if engine == "neural_v3":
        track_path = OUT / "search" / "neural_track.json"
        if track_path.exists() and json.loads(track_path.read_text()).get("status") == "stopped":
            raise ValueError("neural-v3 track was stopped by the material-regression rule")
    blend_payload = {}
    if engine == "blend_v3":
        blend_path = OUT / "search" / "blend.json"
        if not blend_path.exists():
            raise FileNotFoundError("blend eligibility has not been assessed")
        blend_payload = json.loads(blend_path.read_text())
        if not blend_payload.get("eligible"):
            raise ValueError("global blend did not pass its eligibility gates")
    config = {
        "baseline": baseline_config, "gbdt_v3": gbdt_config, "neural_v3": neural_config,
        "blend_v3": (gbdt_config, neural_config),
    }[engine]
    if engine == "blend_v3":
        blocks = tuple(dict.fromkeys(
            (*gbdt_config.context_blocks, *neural_config.context_blocks)
        ))
    else:
        blocks = () if engine == "baseline" else tuple(config.context_blocks)
    features = build_pit_features(augment_context(timed, blocks))
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    match_at = pd.to_datetime(features["match_at"], utc=True)
    train = features[match_at < cutoff].copy()
    if train.empty:
        raise ValueError(f"no rows before {cutoff}")
    if engine == "baseline":
        model = UniversalGBDT(
            random_state=baseline_config.seed, weighting=baseline_config.weighting,
            time_half_life_days=baseline_config.time_half_life_days,
            n_estimators=baseline_config.n_estimators,
            num_leaves=baseline_config.num_leaves,
        ).fit(train)
    elif engine == "gbdt_v3":
        model = ExposureRateGBDT(config=gbdt_config).fit(train)
    elif engine == "neural_v3":
        dates = sorted(pd.to_datetime(train["date"]).dropna().unique())
        validation_start = dates[max(1, int(len(dates) * 0.90)) - 1]
        fit = train[pd.to_datetime(train["date"]) < validation_start]
        validation = train[pd.to_datetime(train["date"]) >= validation_start]
        model = ExposureRateNeural(config=neural_config).fit(fit, validation)
    else:
        g_train = restrict_context(train, tuple(gbdt_config.context_blocks))
        n_train = restrict_context(train, tuple(neural_config.context_blocks))
        dates = sorted(pd.to_datetime(n_train["date"]).dropna().unique())
        validation_start = dates[max(1, int(len(dates) * 0.90)) - 1]
        fit = n_train[pd.to_datetime(n_train["date"]) < validation_start]
        validation = n_train[pd.to_datetime(n_train["date"]) >= validation_start]
        model = EventBlendModel(
            ExposureRateGBDT(config=gbdt_config).fit(g_train),
            ExposureRateNeural(config=neural_config).fit(fit, validation),
            float(blend_payload["neural_weight"]),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    manifest = {
        "schema_version": 1, "engine": engine, "created_at": datetime.now(timezone.utc).isoformat(),
        "exclusive_cutoff": cutoff.isoformat(), "training_rows": int(len(train)),
        "training_match_at_max": pd.to_datetime(train["match_at"], utc=True).max().isoformat(),
        "training_fixture_count": int(train["fixture_id"].nunique()),
        "store_path": str(raw_path), "store_sha256": sha256(raw_path),
        "artifact_sha256": sha256(output),
        "config": (
            None if config is None else
            {
                "gbdt": gbdt_config.to_dict(), "neural": neural_config.to_dict(),
                "neural_weight": float(blend_payload["neural_weight"]),
            }
            if engine == "blend_v3" else config.to_dict()
        ),
    }
    if engine == "blend_v3":
        manifest["component_artifact_sha256"] = {
            "gbdt": sha256(output.with_suffix(output.suffix + ".gbdt.pkl")),
            "neural": sha256(output.with_suffix(output.suffix + ".neural.pt")),
        }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    registry = OUT / "model_registry.csv"
    row = pd.DataFrame([{
        "created_at": manifest["created_at"], "engine": engine,
        "exclusive_cutoff": cutoff.isoformat(), "artifact": str(output),
        "artifact_sha256": manifest["artifact_sha256"], "status": "shadow_only",
    }])
    if registry.exists():
        prior = pd.read_csv(registry)
        prior = prior[prior["artifact"].ne(str(output))]
        row = pd.concat([prior, row], ignore_index=True)
    row.to_csv(registry, index=False)
    return model, manifest
