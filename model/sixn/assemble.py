from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pipeline.sixn.build_targets import _fit_latent
from model.sixn.baselines import SCORED, expected_minutes, rates_to_components, score_recon
from model.sixn.splits import latent_calib_rows, round_iter

POTM_PP_WEIGHT = 0.5
POTM_TAU_FLOOR = 1.0
LATENT_SHRINK = 1.0
OTHER_CONST_SHRINK = 1.0
ZERO_BACK_SETPIECE = False


def fit_latent_asof(df: pd.DataFrame, asof) -> dict | None:
    mask = latent_calib_rows(df, asof)
    if mask.sum() == 0:
        return None
    calib = df.loc[mask].copy()
    calib["sw_drive"] = calib["is_forward"].astype(float) * calib["team_scrums_won"] * calib["min_share"]
    calib["ls_drive"] = calib["is_jumper"].astype(float) * calib["team_lineout_steal"] * calib["min_share"]
    return _fit_latent(calib)


def _expected_drives(df: pd.DataFrame, idx: np.ndarray, minutes_hat_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = df.iloc[idx]
    msf = minutes_hat_local / 80.0
    sw = sub["is_forward"].astype(float).to_numpy() * sub["ownteam_scrums_won"].fillna(0).to_numpy(float) * msf
    ls = sub["is_jumper"].astype(float).to_numpy() * sub["ownteam_lineout_steal"].fillna(0).to_numpy(float) * msf
    return sw, ls


def _expected_potm(df: pd.DataFrame, idx: np.ndarray, recon_local: np.ndarray) -> np.ndarray:
    sub = df.iloc[idx]
    fixtures = sub["fixture_id"].to_numpy()
    prior = sub["role_potm_rate"].fillna(0.0).to_numpy(float)
    out = np.zeros(len(idx))
    for fx in np.unique(fixtures):
        m = fixtures == fx
        pp = recon_local[m]
        tau = max(float(pp.std()), POTM_TAU_FLOOR)
        e = np.exp((pp - pp.max()) / tau)
        a_pp = e / e.sum()
        pr = prior[m]
        a_pr = pr / pr.sum() if pr.sum() > 0 else np.full(m.sum(), 1.0 / m.sum())
        out[m] = 15.0 * (POTM_PP_WEIGHT * a_pp + (1.0 - POTM_PP_WEIGHT) * a_pr)
    return out


def assemble_predictions(
    df: pd.DataFrame, train_idx: np.ndarray, test_season: int, mode: str, rate_predictor, *, minutes_hat: np.ndarray | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_idx = np.where((df["season"] == test_season).to_numpy())[0]
    if minutes_hat is None:
        minutes_hat = expected_minutes(df, train_idx, test_idx, mode)

    rates = rate_predictor(df, train_idx, test_idx, mode)
    comp = rates_to_components(rates, minutes_hat)
    isf = df.iloc[test_idx]["is_forward"].to_numpy()
    recon_hat = score_recon(comp, isf)

    latent_hat = np.zeros(len(test_idx))
    diag = []
    for r, asof, ridx in round_iter(df, test_season):
        local = np.searchsorted(test_idx, ridx)
        fit = fit_latent_asof(df, asof)
        n_calib = int(latent_calib_rows(df, asof).sum())
        if fit is None:
            diag.append(dict(round=r, asof=asof, n_calib=0, a=0.0, b=0.0))
            continue
        sw, ls = _expected_drives(df, ridx, minutes_hat[local])
        sub_round = df.iloc[ridx]
        other = sub_round["canonical_pos"].map(fit["other_const"]).fillna(fit["other_global"]).to_numpy(float)
        other = OTHER_CONST_SHRINK * other
        potm = _expected_potm(df, ridx, recon_hat[local])
        setpiece = LATENT_SHRINK * (fit["a"] * sw + fit["b"] * ls)
        if ZERO_BACK_SETPIECE:
            setpiece = np.where(sub_round["is_forward"].to_numpy(bool), setpiece, 0.0)
        latent_hat[local] = setpiece + other + potm
        diag.append(dict(round=r, asof=asof, n_calib=n_calib, a=fit["a"], b=fit["b"]))

    sub = df.iloc[test_idx]
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
    return pred, pd.DataFrame(diag)


def rank_scores(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str) -> np.ndarray:
    import lightgbm as lgb

    from model.sixn.train_components import _lgbm_frame

    X = _lgbm_frame(df, mode)
    tr = df.iloc[train_idx].copy()
    order = np.argsort(tr["fixture_id"].to_numpy(), kind="stable")
    tr_idx_sorted = train_idx[order]
    grp = tr.iloc[order].groupby("fixture_id", sort=False).size().to_numpy()
    grade = tr.iloc[order].groupby("fixture_id")["recon_pts"].rank(pct=True).mul(4.999).astype(int).to_numpy()
    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50, reg_lambda=5.0, random_state=0, n_jobs=1,
        verbosity=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker.fit(X.iloc[tr_idx_sorted], grade, group=grp)
    return ranker.predict(X.iloc[test_idx])
