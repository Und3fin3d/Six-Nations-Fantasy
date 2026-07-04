#!/usr/bin/env python3
"""model/assemble.py  —  assemble points + forward-chained latent (Phase 5).

  target_pts_hat = recon_pts_hat + latent_hat

recon_pts_hat = score_components(predicted component dict) via the frozen scorer.

latent_hat is rebuilt round-by-round, forward-chained:
  for each test round R, fit `_fit_latent` on modern-labelled matches with
  date < R's date, then apply
     a * sw_drive_hat + b * ls_drive_hat + other_const[pos] + 15 * P(POTM)
  where the SW/LS drives at INFERENCE use the PIT EXPECTED team set-piece
  (`ownteam_scrums_won`/`ownteam_lineout_steal`) x role x minutes_hat/80 — never
  the realised `team_scrums_won`/`team_lineout_steal`/`potm_winner` (label-only).
  POTM is an expected value: a within-match softmax of predicted points blended
  with the `role_potm_rate` prior, x 15.

2025 R1 has no prior modern labels -> cold-start latent_hat = 0 (the only
no-leak choice; there is genuinely no modern data to calibrate on yet).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from build_targets import _fit_latent
from model.baselines import (
    SCORED,
    expected_minutes,
    rates_to_components,
    score_recon,
)
from model.splits import latent_calib_rows, round_iter

# --- tunable knobs (defaults reproduce the frozen `main` behaviour exactly) ---
# Set by model.research via apply_config(); never changes the deployed defaults.
POTM_PP_WEIGHT = 0.5     # weight on the softmax-of-points term vs the role prior
POTM_TAU_FLOOR = 1.0     # softmax temperature floor (points scale)
LATENT_SHRINK = 1.0      # shrink set-piece latent a*sw + b*ls toward 0 (fallback step 4)
OTHER_CONST_SHRINK = 1.0  # shrink position/global latent constants toward 0
ZERO_BACK_SETPIECE = False


def fit_latent_asof(df: pd.DataFrame, asof) -> dict | None:
    """Fit the latent decomposition on the forward-chained modern subset
    (date < asof).  Returns None at cold-start (no prior modern labels)."""
    mask = latent_calib_rows(df, asof)
    if mask.sum() == 0:
        return None
    calib = df.loc[mask].copy()
    calib["sw_drive"] = (calib["is_forward"].astype(float)
                         * calib["team_scrums_won"] * calib["min_share"])
    calib["ls_drive"] = (calib["is_jumper"].astype(float)
                         * calib["team_lineout_steal"] * calib["min_share"])
    return _fit_latent(calib)


def _expected_drives(
    df: pd.DataFrame, idx: np.ndarray, minutes_hat_local: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """PIT expected SW / LS drives (NO realised team counts)."""
    sub = df.iloc[idx]
    msf = minutes_hat_local / 80.0
    sw = (sub["is_forward"].astype(float).to_numpy()
          * sub["ownteam_scrums_won"].fillna(0).to_numpy(float) * msf)
    ls = (sub["is_jumper"].astype(float).to_numpy()
          * sub["ownteam_lineout_steal"].fillna(0).to_numpy(float) * msf)
    return sw, ls


def _expected_potm(
    df: pd.DataFrame, idx: np.ndarray, recon_local: np.ndarray
) -> np.ndarray:
    """Expected POTM points = 15 * P(POTM), where P is a per-match blend of a
    softmax over predicted points and the `role_potm_rate` prior (sums to 1 per
    match -> exactly one expected POTM)."""
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
    df: pd.DataFrame, train_idx: np.ndarray, test_season: int, mode: str,
    rate_predictor, *, minutes_hat: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full Strategy-C assembly for one season.

    rate_predictor(df, train_idx, test_idx, mode) -> per-80 rate DataFrame.
    Returns (predictions, latent_diag).
    """
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
        assert np.array_equal(test_idx[local], ridx)            # ridx ⊂ test_idx
        fit = fit_latent_asof(df, asof)
        n_calib = int(latent_calib_rows(df, asof).sum())
        if fit is None:                                          # cold-start
            diag.append(dict(round=r, asof=asof, n_calib=0, a=0.0, b=0.0))
            continue
        sw, ls = _expected_drives(df, ridx, minutes_hat[local])
        sub_round = df.iloc[ridx]
        other = (sub_round["canonical_pos"].map(fit["other_const"])
                 .fillna(fit["other_global"]).to_numpy(float))
        other = OTHER_CONST_SHRINK * other
        potm = _expected_potm(df, ridx, recon_hat[local])
        setpiece = LATENT_SHRINK * (fit["a"] * sw + fit["b"] * ls)
        if ZERO_BACK_SETPIECE:
            setpiece = np.where(sub_round["is_forward"].to_numpy(bool), setpiece, 0.0)
        latent_hat[local] = setpiece + other + potm
        diag.append(dict(round=r, asof=asof, n_calib=n_calib,
                         a=fit["a"], b=fit["b"]))

    sub = df.iloc[test_idx]
    pred = pd.DataFrame({
        "season": sub["season"].to_numpy(),
        "round": sub["round"].to_numpy(),
        "fixture_id": sub["fixture_id"].to_numpy(),
        "player_id": sub["player_id"].to_numpy(),
        "player_name": sub["player_name"].to_numpy(),
        "canonical_pos": sub["canonical_pos"].to_numpy(),
        "is_forward": isf,
        "minutes_hat": minutes_hat,
        "recon_pts_hat": recon_hat,
        "latent_hat": latent_hat,
        "target_pts_hat": recon_hat + latent_hat,
        "official_pts": sub["official_pts"].to_numpy(),
        "has_label": sub["has_label"].to_numpy(),
        "is_modern": sub["is_modern"].to_numpy(),
    }, index=sub.index)
    for c in SCORED:
        pred[f"hat_{c}"] = comp[c].to_numpy()
    return pred, pd.DataFrame(diag)


# ---------------------------------------------------------------------------
# optional selection-rank head (LightGBM lambdarank) — comparison overlay
# ---------------------------------------------------------------------------
def rank_scores(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, mode: str
) -> np.ndarray:
    """Within-match LightGBM lambdarank score for the test rows.

    Trained to rank players by realised modern-scale `recon_pts` within each
    training fixture (a label available for every season, no leakage), grouped
    by fixture.  A pure ordering overlay — not a calibrated point forecast.
    """
    import lightgbm as lgb

    from model.train_components import _lgbm_frame

    X = _lgbm_frame(df, mode)
    tr = df.iloc[train_idx].copy()
    order = np.argsort(tr["fixture_id"].to_numpy(), kind="stable")
    tr_idx_sorted = train_idx[order]
    grp = tr.iloc[order].groupby("fixture_id", sort=False).size().to_numpy()
    # relevance grade 0..4 by within-fixture realised recon_pts rank
    grade = (tr.iloc[order].groupby("fixture_id")["recon_pts"]
             .rank(pct=True).mul(4.999).astype(int).to_numpy())
    ranker = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=300, learning_rate=0.05,
        num_leaves=15, min_child_samples=50, reg_lambda=5.0,
        random_state=0, n_jobs=1, verbosity=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker.fit(X.iloc[tr_idx_sorted], grade, group=grp)
    return ranker.predict(X.iloc[test_idx])


def _selfcheck() -> None:
    from model.baselines import predict_rates
    from model.data import load
    from model.splits import component_train
    from model.train_components import predict_rates_registry, select_components

    df = load()
    mode = "post_team_sheet"
    train_mask = component_train(df, 2025)
    train_idx = np.where(train_mask)[0]
    reg = select_components(df, train_mask, mode)

    def ens(d, ti, te, mo):
        return predict_rates_registry(d, ti, te, mo, reg)

    pred, diag = assemble_predictions(df, train_idx, 2025, mode, ens)
    print(diag.to_string(index=False))

    # forward-chain assertion already enforced inside latent_calib_rows;
    # additionally: round 1 must be cold-start (a=b=0, n_calib=0)
    assert diag.iloc[0]["n_calib"] == 0 and diag.iloc[0]["a"] == 0.0
    assert not pred["target_pts_hat"].isna().any()

    lab = pred[pred["has_label"] & pred["is_modern"]]
    mae = float((lab["official_pts"] - lab["target_pts_hat"]).abs().mean())
    mae_recon = float((lab["official_pts"] - lab["recon_pts_hat"]).abs().mean())
    print(f"\n2025 ensemble: recon-only MAE={mae_recon:.3f}  +latent MAE={mae:.3f}")
    print(f"mean latent_hat={pred['latent_hat'].mean():.2f} "
          f"(fwd {pred[pred.is_forward].latent_hat.mean():.2f} / "
          f"back {pred[~pred.is_forward].latent_hat.mean():.2f})")

    # naive comparison through the same assembly
    pred_n, _ = assemble_predictions(
        df, train_idx, 2025, mode,
        lambda d, ti, te, mo: predict_rates(d, ti, te, mo, "naive"))
    lab_n = pred_n[pred_n["has_label"] & pred_n["is_modern"]]
    print(f"2025 naive   : +latent MAE="
          f"{(lab_n['official_pts']-lab_n['target_pts_hat']).abs().mean():.3f}")
    print("Phase 5 assemble OK")


if __name__ == "__main__":
    _selfcheck()
