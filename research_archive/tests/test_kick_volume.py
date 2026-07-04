from __future__ import annotations

import numpy as np
import pandas as pd

from model.research import (
    Config,
    KICK_VOLUME_FEATURES,
    _apply_kick_volume,
)


def _toy_frame(n_fixtures: int = 6, players_per_team: int = 3) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    rows = []
    for fx in range(n_fixtures):
        season = 2023 if fx < n_fixtures - 2 else 2025
        for team in (0, 1):
            for p in range(players_per_team):
                is_kicker = p == 0
                rows.append(dict(
                    season=season,
                    fixture_id=fx,
                    team_id=fx * 2 + team,
                    player_id=fx * 100 + team * 10 + p,
                    y_conversion_goals=(2 + (fx % 3)) if is_kicker else 0,
                    y_penalty_goals=(1 + (fx % 2)) if is_kicker else 0,
                    **{f: float(rng.normal()) for f in KICK_VOLUME_FEATURES},
                ))
    df = pd.DataFrame(rows)
    train_idx = np.where((df["season"] < 2025).to_numpy())[0]
    test_idx = np.where((df["season"] == 2025).to_numpy())[0]
    return df, train_idx, test_idx


def _toy_pred(df: pd.DataFrame, test_idx: np.ndarray) -> pd.DataFrame:
    sub = df.iloc[test_idx]
    pred = pd.DataFrame(index=sub.index)
    pred["is_forward"] = False
    pred["hat_conversion_goals"] = np.where(
        sub["player_id"].to_numpy() % 10 == 0, 2.0, 0.0)
    pred["hat_penalty_goals"] = np.where(
        sub["player_id"].to_numpy() % 10 == 0, 1.5, 0.0)
    pred["target_pts_hat"] = 10.0
    pred["sel_score"] = 1.0
    pred["captain_score"] = 1.0
    pred["supersub_score"] = 1.0
    return pred


def test_kick_volume_off_is_noop() -> None:
    df, train_idx, test_idx = _toy_frame()
    pred = _toy_pred(df, test_idx)
    out = _apply_kick_volume(
        df, train_idx, test_idx, pred, Config(), stage="post_target")
    pd.testing.assert_frame_equal(out, pred)


def test_kick_volume_post_target_moves_target_only_and_respects_clip() -> None:
    df, train_idx, test_idx = _toy_frame()
    pred = _toy_pred(df, test_idx)
    cfg = Config(kick_volume_mode="ridge_tp", kick_volume_weight=1.0,
                 kick_volume_clip=2.0, kick_volume_stage="post_target")
    out = _apply_kick_volume(df, train_idx, test_idx, pred, cfg,
                             stage="post_target")

    # decision scores are exactly frozen
    for col in ("sel_score", "captain_score", "supersub_score"):
        np.testing.assert_array_equal(out[col].to_numpy(), pred[col].to_numpy())

    # target moved by exactly the component-point delta
    d_conv = out["hat_conversion_goals"] - pred["hat_conversion_goals"]
    d_pen = out["hat_penalty_goals"] - pred["hat_penalty_goals"]
    expected = pred["target_pts_hat"] + 2.0 * d_conv + 3.0 * d_pen
    np.testing.assert_allclose(out["target_pts_hat"], expected)

    # non-kickers (zero mass) never gain kick mass; factors respect the clip
    zero = pred["hat_conversion_goals"].to_numpy() == 0.0
    assert np.all(out["hat_conversion_goals"].to_numpy()[zero] == 0.0)
    kick = ~zero
    ratio = (out["hat_conversion_goals"].to_numpy()[kick]
             / pred["hat_conversion_goals"].to_numpy()[kick])
    assert np.all(ratio <= 2.0 + 1e-9) and np.all(ratio >= 0.5 - 1e-9)

    # wrong stage is a no-op
    same = _apply_kick_volume(df, train_idx, test_idx, pred, cfg,
                              stage="pre_selection")
    pd.testing.assert_frame_equal(same, pred)
