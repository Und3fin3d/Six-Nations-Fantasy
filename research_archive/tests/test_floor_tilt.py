from __future__ import annotations

import numpy as np
import pandas as pd

from model.research import _spike_points, _SPIKE_POINT_WEIGHTS
from model.baselines import SCORED


def _toy() -> pd.DataFrame:
    rows = pd.DataFrame({
        "hat_tries": [0.5, 0.0],
        "is_forward": [True, False],
    })
    for comp in SCORED:
        if comp not in rows.columns and f"hat_{comp}" not in rows.columns:
            rows[f"hat_{comp}"] = 0.0
    rows["hat_tries"] = [0.5, 0.0]
    rows["hat_tackle_turnover"] = [1.0, 0.0]
    rows["hat_try_assists"] = [0.0, 2.0]
    return rows


def test_spike_points_uses_positional_try_weight_and_component_weights() -> None:
    pred = _toy()
    spike = _spike_points(pred)
    # forward: 15*0.5 (tries) + 5*1.0 (TTO) = 12.5
    # back:    10*0.0 (tries) + 4*2.0 (assists) = 8.0
    np.testing.assert_allclose(spike, [12.5, 8.0])


def test_spike_points_excludes_floor_components() -> None:
    """Tackles and metres must never enter the spike total."""
    assert "tackles" not in _SPIKE_POINT_WEIGHTS
    assert "metres" not in _SPIKE_POINT_WEIGHTS
    pred = _toy()
    pred["hat_tackles"] = [20.0, 3.0]
    pred["hat_metres"] = [50.0, 90.0]
    baseline = _spike_points(_toy())
    with_floor = _spike_points(pred)
    np.testing.assert_allclose(baseline, with_floor)
