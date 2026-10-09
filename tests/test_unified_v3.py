from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.unified.features import build_pit_features
from model.unified.schema import EVENTS
from model.unified.shadow.harness import write_once
from model.unified.raw_benchmark.blend import EventBlend50
from model.unified.shadow.shadow import _load_model, validate_shadow_write


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
