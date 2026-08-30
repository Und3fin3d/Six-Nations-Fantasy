from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.unified.features import build_pit_features
from model.unified.labels import build_fantasy_labels
from model.unified.schema import EVENTS
from model.unified.scoring import scorer_for
from model.unified.v3.audit import appearance_eligibility, run_audit
from model.unified.v3.blend import blend_predictions
from model.unified.v3.config import GBDTV3Config, NeuralV3Config
from model.unified.v3.gbdt import ExposureRateGBDT
from model.unified.v3.harness import (
    fold_for_block, load_timed_store, strict_training_frame, write_once,
)
from model.unified.v3.neural import ExposureRateNeural
from model.unified.raw_benchmark.blend import EventBlend50
from model.unified.v3.shadow import _load_model, validate_shadow_write


def _frame(rows: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    minutes = np.where(np.arange(rows) % 5 == 0, 0.0, rng.integers(18, 81, rows))
    data = {
        "date": pd.date_range("2023-01-01", periods=rows, freq="7D"),
        "competition": "Test",
        "competition_id": 1,
        "competition_level": "international",
        "season": 2023,
        "round": np.arange(rows) + 1,
        "fixture_id": [f"f{i // 2}" for i in range(rows)],
        "player_id": [f"p{i % 12}" for i in range(rows)],
        "player_name": [f"Player {i % 12}" for i in range(rows)],
        "team": np.where(np.arange(rows) % 2, "B", "A"),
        "team_id": np.where(np.arange(rows) % 2, "2", "1"),
        "opponent": np.where(np.arange(rows) % 2, "A", "B"),
        "opponent_id": np.where(np.arange(rows) % 2, "1", "2"),
        "position": np.where(np.arange(rows) % 2, "Centre", "Back-row"),
        "is_forward": np.arange(rows) % 2 == 0,
        "jersey": np.where(np.arange(rows) % 5 == 0, 20, 7),
        "started": np.arange(rows) % 5 != 0,
        "minutes": minutes,
        "team_score": rng.integers(10, 40, rows),
        "opp_score": rng.integers(10, 40, rows),
        "source": "test",
        "source_priority": 1,
        "available__minutes": True,
    }
    frame = pd.DataFrame(data)
    for event in EVENTS:
        if event == "metres":
            values = rng.gamma(2.0, 15.0, rows)
        elif event == "potm":
            values = (np.arange(rows) % 29 == 0).astype(float)
        else:
            values = rng.poisson(0.35, rows).astype(float)
        values[minutes == 0] = 0
        frame[event] = values
        frame[f"available__{event}"] = True
    return build_pit_features(frame)


def test_france_wales_cannot_enter_2025_opening_fold():
    store = load_timed_store()
    labels = build_fantasy_labels()
    block = labels[
        labels["competition"].eq("six_nations")
        & labels["season"].eq(2025)
        & labels["round"].eq(1)
    ]
    fold = fold_for_block(block, store)
    train = strict_training_frame(store, fold)
    france_wales = store[
        store["date"].eq(pd.Timestamp("2025-01-31"))
        & store["team"].isin(["France", "Wales"])
        & store["opponent"].isin(["France", "Wales"])
    ]
    assert not france_wales.empty
    assert fold.cutoff == pd.Timestamp("2025-01-31", tz="UTC")
    assert not set(france_wales["fixture_id"].astype(str)) & set(
        train["fixture_id"].astype(str)
    )
    assert pd.to_datetime(train["match_at"], utc=True).max() < fold.cutoff


def test_point_in_time_features_do_not_change_when_future_rows_are_appended():
    base = _frame(48)
    raw_columns = [c for c in base if not c.startswith(("form_per80__", "history_count__"))]
    raw_columns = [
        c for c in raw_columns
        if c not in {
            "days_since_last", "career_matches", "recent_minutes",
            "recent_start_rate", "team_recent_margin",
        }
    ]
    raw = base[raw_columns]
    historical = build_pit_features(raw.iloc[:36].copy())
    extended = build_pit_features(raw.copy())
    feature_columns = [
        c for c in historical
        if c.startswith(("form_per80__", "history_count__"))
        or c in {
            "days_since_last", "career_matches", "recent_minutes",
            "recent_start_rate", "team_recent_margin",
        }
    ]
    pd.testing.assert_frame_equal(
        historical[feature_columns].reset_index(drop=True),
        extended.iloc[:36][feature_columns].reset_index(drop=True),
    )


def test_zero_minute_rows_are_selected_but_unused():
    store = pd.DataFrame({
        "minutes": np.zeros(1200),
        "jersey": np.tile(np.arange(1, 24), 53)[:1200],
        "started": False,
    })
    result = appearance_eligibility(store)
    assert result["appearance_head_enabled"]
    assert result["zero_minute_jersey_1_23_share"] == 1.0


def test_real_data_audit_covers_all_labelled_rounds(tmp_path: Path):
    payload = run_audit(output_dir=tmp_path)
    correlations = pd.read_csv(tmp_path / "observable_points_correlation.csv")
    assert payload["official_label_rows"] == 2144
    # GW3 official labels are present, but its raw match-event rows have not
    # yet been ingested into the canonical store, so observable reconstruction
    # correlations still cover the prior 12 groups only.
    assert correlations["group_id"].nunique() == 12
    assert payload["observable_points_enabled"]
    assert payload["zero_minutes"]["appearance_head_enabled"]
    assert not payload["context"]["weather"]["pit_safe"]
    assert (tmp_path / "scorer_reconstruction.csv").exists()
    assert (tmp_path / "scoring_event_coverage.csv").exists()


def test_gbdt_is_deterministic_serializable_and_scores_both_rules(tmp_path: Path):
    frame = _frame()
    config = GBDTV3Config(
        n_estimators=12, num_leaves=7, min_child_samples=5,
        context_blocks=(), seed=19,
    )
    first = ExposureRateGBDT(config=config, events=("tries", "metres", "potm")).fit(frame)
    second = ExposureRateGBDT(config=config, events=("tries", "metres", "potm")).fit(frame)
    candidate = frame.tail(3).copy()
    candidate.loc[candidate.index[-1], "player_id"] = "never-seen"
    predicted_first = first.predict_frame(candidate)
    predicted_second = second.predict_frame(candidate)
    assert [p.minutes.mean for p in predicted_first] == pytest.approx(
        [p.minutes.mean for p in predicted_second]
    )
    artifact = tmp_path / "gbdt.pkl"
    first.save(artifact)
    restored = ExposureRateGBDT.load(artifact).predict_frame(candidate)
    assert [p.events["tries"].mean for p in restored] == pytest.approx(
        [p.events["tries"].mean for p in predicted_first]
    )
    for competition in ("six_nations", "ncr"):
        summary = scorer_for(competition).score_prediction(restored[-1], n=100, seed=2)
        assert np.isfinite(summary.mean)


def test_global_blend_preserves_identity_and_blends_raw_means():
    frame = _frame()
    config = GBDTV3Config(
        n_estimators=8, num_leaves=7, min_child_samples=5,
        context_blocks=(), seed=29,
    )
    model = ExposureRateGBDT(config=config, events=("tries",)).fit(frame)
    predictions = model.predict_frame(frame.tail(2))
    blended = blend_predictions(predictions, predictions, 0.5)
    assert [p.events["tries"].mean for p in blended] == pytest.approx(
        [p.events["tries"].mean for p in predictions]
    )
    changed = list(predictions)
    changed[0] = replace(changed[0], player_id="wrong")
    with pytest.raises(ValueError, match="identity mismatch"):
        blend_predictions(predictions, changed, 0.5)


def test_neural_trains_predicts_and_serializes(tmp_path: Path):
    frame = _frame()
    config = NeuralV3Config(
        hidden=16, depth=2, dropout=0.1, epochs=1, batch_size=32,
        patience=1, context_blocks=(), seed=23,
    )
    model = ExposureRateNeural(
        config=config, events=("tries", "metres", "potm"),
    ).fit(frame.iloc[:56], frame.iloc[56:])
    predictions = model.predict_frame(frame.tail(2))
    repeated = ExposureRateNeural(
        config=config, events=("tries", "metres", "potm"),
    ).fit(frame.iloc[:56], frame.iloc[56:]).predict_frame(frame.tail(2))
    assert len(predictions) == 2
    assert all(0 <= prediction.minutes.mean <= 80 for prediction in predictions)
    assert [p.minutes.mean for p in repeated] == pytest.approx(
        [p.minutes.mean for p in predictions]
    )
    artifact = tmp_path / "neural.pt"
    model.save(artifact)
    restored = ExposureRateNeural.load(artifact).predict_frame(frame.tail(2))
    assert [p.minutes.mean for p in restored] == pytest.approx(
        [p.minutes.mean for p in predictions]
    )


def test_immutable_outputs_and_shadow_lock_are_enforced(tmp_path: Path):
    path = tmp_path / "manifest.json"
    write_once(path, json.dumps({"ok": True}))
    with pytest.raises(FileExistsError):
        write_once(path, "{}")

    csv_path = tmp_path / "shadow.csv"
    manifest_path = tmp_path / "shadow.manifest.json"
    lock = pd.Timestamp("2026-07-18T06:40:00Z")
    validate_shadow_write(
        csv_path, manifest_path, lock, now=pd.Timestamp("2026-07-18T06:39:00Z"),
    )
    csv_path.write_text("frozen\n")
    with pytest.raises(FileExistsError):
        validate_shadow_write(
            csv_path, manifest_path, lock, now=pd.Timestamp("2026-07-18T06:41:00Z"),
        )
    csv_path.unlink()
    with pytest.raises(RuntimeError):
        validate_shadow_write(
            csv_path, manifest_path, lock, now=pd.Timestamp("2026-07-18T06:41:00Z"),
        )


def test_saved_p3_shadow_artifact_keeps_fixed_global_blend():
    artifact = Path(
        "data/unified/raw_benchmark/v1/models/p3_event_50/"
        "nations_championship_2026.pkl"
    )
    model = _load_model("p3_event_50", artifact)

    assert isinstance(model, EventBlend50)
    assert model.weight_v4 == 0.5
