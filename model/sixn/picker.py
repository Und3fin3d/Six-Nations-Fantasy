from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

from model.sixn.assemble import assemble_predictions, rank_scores
from model.sixn.baselines import MIN_MINUTES, _rate_target, expected_minutes
from model.sixn.data import CATEGORICAL_COLS, feature_view
from model.sixn.splits import component_train, latent_calib_rows, round_iter
from model.sixn.train_components import LGBM_ONLY, predict_rates_lgbm_only, predict_rates_registry

MODE = "post_team_sheet"
CFG = json.load(open("model/sixn/promoted_config.json"))
META_COLS = [
    "team", "opponent", "started", "jersey", "rolecert_goal_kicker_prob", "rolecert_team_kicker_confidence", "rolecert_bench_fh_kick_share",
    "rolecert_bench_kick_takeover_risk", "rolecert_replacement_role_uncertainty",
]
XGB_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, max_depth=3, min_child_weight=10.0, reg_lambda=5.0, colsample_bytree=0.8, subsample=0.9, random_state=0, n_jobs=1,
    verbosity=0, tree_method="hist", max_bin=64,
)


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def _xgb_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_view(df, MODE)].copy()
    X["bio_position"] = X["bio_position"].cat.codes.replace(-1, np.nan).astype(float)
    return X


def _xgb_objective(comp: str) -> dict:
    if comp == "metres":
        return dict(objective="reg:tweedie", tweedie_variance_power=1.3)
    if comp == "tackles":
        return dict(objective="reg:squarederror")
    return dict(objective="count:poisson", max_delta_step=1.0)


def _xgb_rates(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str) -> pd.DataFrame:
    X = _xgb_frame(df)
    use = train_idx[(df.iloc[train_idx]["minutes"] >= MIN_MINUTES).to_numpy()]

    def model(comp: str) -> np.ndarray:
        y = np.clip(np.nan_to_num(_rate_target(df.iloc[use], comp), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        regressor = xgb.XGBRegressor(**XGB_PARAMS, **_xgb_objective(comp))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regressor.fit(X.iloc[use], y)
        return np.clip(regressor.predict(X.iloc[test_idx]), 0.0, None)

    return predict_rates_registry(df, train_idx, test_idx, mode, LGBM_ONLY, model)


def _frontrow_prior(df: pd.DataFrame, test_idx: np.ndarray, pred: pd.DataFrame, season: int) -> np.ndarray:
    out = pred["target_pts_hat"].to_numpy(float).copy()
    base, w = out.copy(), CFG["target_prior_blend"]
    for _, asof, ridx in round_iter(df, season):
        hist_mask = latent_calib_rows(df, asof)
        if hist_mask.sum() < 40:
            continue
        local = np.searchsorted(test_idx, ridx)
        hist = df.loc[hist_mask]
        rows = df.iloc[ridx]["canonical_pos"]
        prior = rows.map(hist.groupby("canonical_pos")["official_pts"].mean()).fillna(float(hist["official_pts"].mean())).to_numpy(float)
        eligible = rows.isin({"Prop", "Hooker"}).to_numpy()
        out[local[eligible]] = (1.0 - w) * base[local[eligible]] + w * prior[eligible]
    return np.clip(out, 0.0, None)


def _ridge_predict(Xtr: pd.DataFrame, y: np.ndarray, Xte: pd.DataFrame) -> np.ndarray:
    scale = Xtr.std(axis=0, ddof=0).replace(0.0, 1.0)
    center = Xtr.mean(axis=0)
    return Ridge(alpha=CFG["bench_alpha"]).fit(((Xtr - center) / scale).to_numpy(), y).predict(((Xte - center) / scale).to_numpy())


def _bench_design(rows: pd.DataFrame) -> pd.DataFrame:
    X = rows[[c for c in feature_view(rows, MODE) if c not in CATEGORICAL_COLS]].astype(float)
    return pd.concat([X.reset_index(drop=True), pd.get_dummies(rows["canonical_pos"], prefix="pos", dtype=float).reset_index(drop=True)], axis=1)


def _align_design(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xtr = _bench_design(train)
    Xte = _bench_design(test).reindex(columns=Xtr.columns, fill_value=0.0)
    means = Xtr.mean(numeric_only=True)
    return Xtr.fillna(means).fillna(0.0), Xte.fillna(means).fillna(0.0)


def _bench_minutes(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, np.ndarray]:
    train = df.iloc[train_idx]
    train = train[~train["started"].astype(bool)]
    test = df.iloc[test_idx]
    minutes = train["minutes"].to_numpy(float)
    Xtr, Xte = _align_design(train, test)
    play = np.clip(_ridge_predict(Xtr, (minutes >= 10.0).astype(float), Xte), 0.0, 1.0)
    high = np.clip(_ridge_predict(Xtr, (minutes >= CFG["bench_high_minutes_threshold"]).astype(float), Xte), 0.0, 1.0)
    on = minutes >= 10.0
    Xon, Xte_on = _align_design(train.loc[on], test)
    if_on = np.clip(_ridge_predict(Xon, minutes[on], Xte_on), 0.0, 80.0)
    expected = np.clip(play * if_on, 0.0, 80.0)
    uncertainty = play * (1.0 - play) + np.abs(expected - test["form_minutes_recent"].fillna(0).to_numpy(float)) / 80.0
    return {
        "bench_play_prob": play, "bench_high_minutes_prob": high, "bench_low_minutes_prob": np.clip(1.0 - play, 0.0, 1.0), "bench_minutes_if_on": if_on,
        "bench_expected_minutes": expected, "bench_minutes_uncertainty": uncertainty,
    }


def _target(df, train_idx, test_idx, season, minutes_hat, xgb_target, bench_minutes, teamplay) -> pd.DataFrame:
    pred = assemble_predictions(df, train_idx, season, MODE, lambda *args: predict_rates_lgbm_only(*args, teamplay), minutes_hat=minutes_hat)
    for col in META_COLS:
        pred[col] = df.iloc[test_idx][col].to_numpy()
    lgbm = _frontrow_prior(df, test_idx, pred, season)
    pred["lgbm_target_pts_hat"] = lgbm
    pred["xgb_component_pts_hat"] = xgb_target
    te = df.iloc[test_idx]
    diff = np.abs(lgbm - xgb_target)
    disagreement = diff >= float(np.nanquantile(diff, 0.75))
    pred["uncertainty_proxy"] = (
        (te["form_n_prior"].fillna(0).to_numpy(float) <= 4).astype(float)
        + (te["class_minutes_prior"].fillna(0).to_numpy(float) <= 160).astype(float)
        + disagreement.astype(float)
    )
    w = CFG["xgb_point_weight"]
    pred["target_pts_hat"] = np.clip((1.0 - w) * lgbm + w * xgb_target, 0.0, None)
    for col, vals in bench_minutes.items():
        pred[col] = vals
    bench_hat = np.clip(
        pred["target_pts_hat"].to_numpy(float) * bench_minutes["bench_expected_minutes"] / np.maximum(pred["minutes_hat"].to_numpy(float), 1.0), 0.0, None
    )
    pred["bench_kick_pts_hat"] = np.asarray(2.0 * pred["hat_conversion_goals"] + 3.0 * pred["hat_penalty_goals"], dtype=float)
    pred["bench_pts_hat"] = bench_hat
    w = CFG["bench_points_weight"]
    bench = ~te["started"].astype(bool).to_numpy()
    target = pred["target_pts_hat"].to_numpy(float).copy()
    target[bench] = (1.0 - w) * target[bench] + w * bench_hat[bench]
    pred["target_pts_hat"] = np.clip(target, 0.0, None)
    return pred


def _upside(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, pred: pd.DataFrame) -> np.ndarray:
    hat = {c: pred[f"hat_{c}"].to_numpy(float) for c in ("tries", "try_assists", "defenders_beaten", "offload", "metres")}
    attack = (
        np.where(pred["is_forward"].to_numpy(bool), 15.0, 10.0) * hat["tries"]
        + 4.0 * hat["try_assists"]
        + 2.0 * hat["defenders_beaten"]
        + 2.0 * hat["offload"]
        + np.floor(hat["metres"] / 10.0)
    )
    disagreement = np.abs(pred["lgbm_target_pts_hat"].to_numpy(float) - pred["xgb_component_pts_hat"].to_numpy(float))
    train = df.iloc[train_idx]
    official = train["official_pts"].to_numpy(float)
    target = np.where(
        (train["has_label"].astype(bool) & train["is_modern"].astype(bool)).to_numpy() & np.isfinite(official), official,
        train["target_pts"].fillna(train["recon_pts"]).to_numpy(float),
    )
    by = train.assign(_t=target).groupby
    player_std = by("player_id")["_t"].std().where(by("player_id")["_t"].size() >= 3)
    pos = by("canonical_pos")["_t"]
    te = df.iloc[test_idx]
    std = te["player_id"].map(player_std).fillna(te["canonical_pos"].map(pos.std())).fillna(float(np.nanstd(target))).to_numpy(float)
    ceiling = te["canonical_pos"].map(pos.quantile(0.80) - pos.mean()).fillna(float(np.nanstd(target))).to_numpy(float)
    return _zscore(_zscore(attack) + 0.50 * _zscore(disagreement) + 0.50 * _zscore(std) + 0.25 * _zscore(ceiling))


def _residual(pred: pd.DataFrame) -> pd.DataFrame:
    target = pred["target_pts_hat"].to_numpy(float).copy()
    lab = (pred["has_label"].astype(bool) & pred["is_modern"].astype(bool)).to_numpy()
    rounds = pred["round"].to_numpy()
    residual = pred["official_pts"].to_numpy(float) - target
    keys = pred["team"].astype(str).to_numpy() + "|" + pred["canonical_pos"].astype(str).to_numpy()
    scope = pred["started"].astype(bool).to_numpy() & ~pred["is_forward"].astype(bool).to_numpy()
    clip = CFG["post_target_residual_clip"]
    applied = np.zeros(len(pred), dtype=float)
    for r in np.unique(rounds):
        prev, cur = lab & (rounds < r), (rounds == r) & scope
        if prev.sum() < CFG["post_target_residual_min_n"] or not cur.any():
            continue
        stats = pd.DataFrame({"key": keys[prev], "resid": residual[prev]}).groupby("key")["resid"].agg(["mean", "count"])
        adj = (
            pd.Series(keys[cur])
            .map(stats[stats["count"] >= CFG["post_target_residual_group_min_n"]]["mean"])
            .fillna(float(np.nanmean(residual[prev])))
            .to_numpy(float)
        )
        applied[cur] = CFG["post_target_residual_weight"] * np.clip(adj, -clip, clip)
        target[cur] = target[cur] + applied[cur]
    pred["post_target_residual_adj"] = applied
    pred["target_pts_hat"] = np.clip(target, 0.0, None)
    return pred


def predict(df: pd.DataFrame, season: int) -> pd.DataFrame:
    train_idx = np.where(component_train(df, season))[0]
    test_idx = np.where((df["season"] == season).to_numpy())[0]
    minutes_hat = expected_minutes(df, train_idx, test_idx, MODE)
    xgb_pred = assemble_predictions(df, train_idx, season, MODE, _xgb_rates, minutes_hat=minutes_hat)
    xgb_target = _frontrow_prior(df, test_idx, xgb_pred, season)
    bench_minutes = _bench_minutes(df, train_idx, test_idx)
    pred = _target(df, train_idx, test_idx, season, minutes_hat, xgb_target, bench_minutes, "off")
    pred["selector_pts_hat"] = pred["target_pts_hat"].to_numpy(float)
    pred["upside_score"] = _upside(df, train_idx, test_idx, pred)
    pred["lgbm_rank_score"] = pred["rank_score"] = rank_scores(df, train_idx, test_idx, MODE)
    pred["sel_score"] = _zscore(pred["target_pts_hat"].to_numpy(float)) + CFG["selector_tilt"] * _zscore(pred["rank_score"].to_numpy(float))
    pred["supersub_score"] = _zscore(pred["sel_score"].to_numpy(float)) + CFG["bench_supersub_weight"] * _zscore(pred["bench_pts_hat"].to_numpy(float))
    pred["captain_score"] = _zscore(pred["target_pts_hat"].to_numpy(float)) + CFG["captain_upside_weight"] * pred["upside_score"].to_numpy(float)
    teamplay = _target(df, train_idx, test_idx, season, minutes_hat, xgb_target, bench_minutes, "aspects")
    starters = pred["started"].astype(bool).to_numpy()
    w = CFG["post_target_xgb_weight"]
    teamplay_target = teamplay["target_pts_hat"].to_numpy(float)
    teamplay_target[starters] = (1.0 - w) * teamplay_target[starters] + w * xgb_target[starters]
    target = pred["target_pts_hat"].to_numpy(float).copy()
    target[starters] = np.clip(teamplay_target, 0.0, None)[starters]
    pred["target_pts_hat"] = np.clip(target, 0.0, None)
    pred = _residual(pred)
    delta = pred["target_pts_hat"].to_numpy(float) - pred["selector_pts_hat"].to_numpy(float)
    pred["sel_score"] = pred["sel_score"].to_numpy(float) + CFG["post_target_selector_delta_weight"] * _zscore(delta)
    return pred
