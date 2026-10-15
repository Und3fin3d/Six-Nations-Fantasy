from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline.sixn.build_targets import _fit_latent
from model.sixn.baselines import SCORED, expected_minutes, rates_to_components, score_recon
from model.sixn.splits import latent_calib_rows, round_iter
from model.sixn.train_components import _lgbm_frame

POTM_PP_WEIGHT, POTM_TAU_FLOOR = 0.5, 1.0


def fit_latent_asof(df: pd.DataFrame, asof) -> dict | None:
    calib = df.loc[latent_calib_rows(df, asof)].copy()
    if calib.empty:
        return None
    calib["sw_drive"] = calib["is_forward"].astype(float) * calib["team_scrums_won"] * calib["min_share"]
    calib["ls_drive"] = calib["is_jumper"].astype(float) * calib["team_lineout_steal"] * calib["min_share"]
    return _fit_latent(calib)


def _expected_potm(sub: pd.DataFrame, recon_local: np.ndarray) -> np.ndarray:
    fixtures, prior = sub["fixture_id"].to_numpy(), sub["role_potm_rate"].fillna(0.0).to_numpy(float)
    out = np.zeros(len(sub))
    for fx in np.unique(fixtures):
        m = fixtures == fx
        pp, pr = recon_local[m], prior[m]
        e = np.exp((pp - pp.max()) / max(float(pp.std()), POTM_TAU_FLOOR))
        out[m] = 15.0 * (POTM_PP_WEIGHT * (e / e.sum()) + (1.0 - POTM_PP_WEIGHT) * (pr / pr.sum() if pr.sum() > 0 else np.full(m.sum(), 1.0 / m.sum())))
    return out


def assemble_predictions(
    df: pd.DataFrame, train_idx: np.ndarray, test_season: int, mode: str, rate_predictor, *, minutes_hat: np.ndarray | None = None
) -> pd.DataFrame:
    test_idx = np.where((df["season"] == test_season).to_numpy())[0]
    if minutes_hat is None:
        minutes_hat = expected_minutes(df, train_idx, test_idx, mode)
    comp = rates_to_components(rate_predictor(df, train_idx, test_idx, mode), minutes_hat)
    sub = df.iloc[test_idx]
    isf = sub["is_forward"].to_numpy()
    recon_hat = score_recon(comp, isf)
    latent_hat = np.zeros(len(test_idx))
    for _, asof, ridx in round_iter(df, test_season):
        local = np.searchsorted(test_idx, ridx)
        fit = fit_latent_asof(df, asof)
        if fit is None:
            continue
        rows, msf = df.iloc[ridx], minutes_hat[local] / 80.0
        sw = rows["is_forward"].astype(float).to_numpy() * rows["ownteam_scrums_won"].fillna(0).to_numpy(float) * msf
        ls = rows["is_jumper"].astype(float).to_numpy() * rows["ownteam_lineout_steal"].fillna(0).to_numpy(float) * msf
        other = rows["canonical_pos"].map(fit["other_const"]).fillna(fit["other_global"]).to_numpy(float)
        latent_hat[local] = fit["a"] * sw + fit["b"] * ls + other + _expected_potm(rows, recon_hat[local])
    pred = pd.DataFrame(
        {
            "season": sub["season"].to_numpy(), "round": sub["round"].to_numpy(), "fixture_id": sub["fixture_id"].to_numpy(),
            "player_id": sub["player_id"].to_numpy(), "player_name": sub["player_name"].to_numpy(), "canonical_pos": sub["canonical_pos"].to_numpy(),
            "is_forward": isf, "minutes_hat": minutes_hat, "recon_pts_hat": recon_hat, "latent_hat": latent_hat, "target_pts_hat": recon_hat + latent_hat,
            "official_pts": sub["official_pts"].to_numpy(), "has_label": sub["has_label"].to_numpy(), "is_modern": sub["is_modern"].to_numpy(),
        },
        index=sub.index,
    )
    for c in SCORED:
        pred[f"hat_{c}"] = comp[c].to_numpy()
    return pred


def rank_scores(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str) -> np.ndarray:
    X = _lgbm_frame(df, mode)
    tr = df.iloc[train_idx].copy()
    order = np.argsort(tr["fixture_id"].to_numpy(), kind="stable")
    grade = tr.iloc[order].groupby("fixture_id")["recon_pts"].rank(pct=True).mul(4.999).astype(int).to_numpy()
    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50, reg_lambda=5.0, random_state=0, n_jobs=1,
        verbosity=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker.fit(X.iloc[train_idx[order]], grade, group=tr.iloc[order].groupby("fixture_id", sort=False).size().to_numpy())
    return ranker.predict(X.iloc[test_idx])
