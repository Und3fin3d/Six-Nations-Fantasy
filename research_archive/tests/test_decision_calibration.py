from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.baselines import SCORED
from model.research import (
    Config,
    _decision_calibration_delta,
    _shrunk_volume_factors,
)


def test_shrunk_volume_factors_ratios_clip_and_position_shrinkage() -> None:
    sums = {
        "global": {
            "tries": (10.0, 20.0),
            "offload": (1.0, 10.0),
        },
        "position": {
            ("tries", "Back-row"): (10.0, 1.0),
            ("tries", "Back-three"): (1000.0, 1500.0),
        },
    }

    factors = _shrunk_volume_factors(sums, kappa=100.0, clip=3.0)

    assert factors["global"]["tries"] == 2.0
    assert factors["global"]["offload"] == 3.0
    assert factors["position"][("tries", "Back-row")] == pytest.approx(
        (1.0 / 101.0) * 0.1 + (100.0 / 101.0) * 2.0
    )
    assert factors["position"][("tries", "Back-three")] == pytest.approx(
        (1500.0 / 1600.0) * 1.5 + (100.0 / 1600.0) * 2.0
    )


def _toy_prediction() -> pd.DataFrame:
    pred = pd.DataFrame({
        "canonical_pos": ["Back-row", "Back-three"],
        "is_forward": [True, False],
    })
    for comp in SCORED:
        pred[f"hat_{comp}"] = 0.0
    pred["hat_tries"] = 1.0
    pred["hat_tackles"] = 2.0
    pred["hat_metres"] = 15.0
    return pred


def test_decision_calibration_delta_matches_scoring_and_does_not_mutate() -> None:
    pred = _toy_prediction()
    original = pred.copy(deep=True)
    factors = {
        "global": {comp: 1.0 for comp in SCORED},
        "position": {},
    }
    factors["global"].update(tries=2.0, tackles=0.5, metres=2.0)

    delta = _decision_calibration_delta(
        pred, factors, Config(decision_calibration="volume_global"))

    np.testing.assert_allclose(delta, [16.0, 11.0])
    pd.testing.assert_frame_equal(pred, original)


def test_decision_calibration_delta_uses_position_factors() -> None:
    pred = _toy_prediction()
    factors = {
        "global": {comp: 1.0 for comp in SCORED},
        "position": {
            ("tries", "Back-row"): 3.0,
            ("tries", "Back-three"): 0.5,
        },
    }

    global_delta = _decision_calibration_delta(
        pred, factors, Config(decision_calibration="volume_global"))
    position_delta = _decision_calibration_delta(
        pred, factors, Config(decision_calibration="volume_position"))

    np.testing.assert_allclose(global_delta, [0.0, 0.0])
    np.testing.assert_allclose(position_delta, [30.0, -5.0])
