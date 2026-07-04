#!/usr/bin/env python3
"""OOF gate audit: fixture-team kicking-volume rescaling of naive kick rates.

Hypothesis: the naive kicker-gated kick heads carry NO fixture context, so a
PIT team-level model of fixture-team kicking totals (conversion_goals,
penalty_goals), used only to rescale the existing player kick rates within
each fixture-team, improves fixture-group OOF per-80 rate MAE by >=2% on the
2023+2024 component-train window.

Protocol mirrors model.train_components._oof_candidate_mae:
  GroupKFold(5) on fixture_id over the train window, per-80 rate MAE measured
  on minutes>=10 validation rows.  Fold minutes model = promoted ridge
  (fit on the fold's train rows only).  No 2025/2026 rows are touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from model.baselines import MIN_MINUTES, _rate_target, predict_rates
from model.data import load
from model.research import Config, build_minutes
from model.splits import component_train, group_kfold_indices

MODE = "post_team_sheet"
KICK_COMPS = ["conversion_goals", "penalty_goals"]

FEATS_BASIC = [
    "wr_pts_gap", "wr_rank_gap", "team_wr_pts", "opp_wr_pts",
    "h2h_last3_margin", "h2h_last3_winrate",
    "opp_pens_conceded", "opp_concede_tries", "opp_concede_metres",
    "opp_possession", "ownteam_tries", "ownteam_metres", "ownteam_possession",
]
FEATS_TEAMPLAY = [
    "teamplay_points_for_hat", "teamplay_points_against_hat",
    "teamplay_margin_hat", "teamplay_win_prob_hat",
    "teamplay_penalty_goal_opportunity_hat", "teamplay_kicks_from_hand_hat",
    "teamplay_aspect_kicking_opportunity_hat", "teamplay_aspect_points_edge_hat",
]


def team_rows(df: pd.DataFrame, idx: np.ndarray, feats: list[str]) -> pd.DataFrame:
    """One row per fixture-team with team-level features + realised kick totals."""
    sub = df.iloc[idx]
    agg = {f: (f, "first") for f in feats}
    g = sub.groupby(["fixture_id", "team_id"]).agg(
        y_conv=("y_conversion_goals", "sum"),
        y_pen=("y_penalty_goals", "sum"),
        **agg,
    ).reset_index()
    return g


def fit_volume_model(
    tr_teams: pd.DataFrame, feats: list[str], engine: str
) -> dict:
    X = tr_teams[feats].to_numpy(float)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    models = {}
    for tgt, col in [("conv", "y_conv"), ("pen", "y_pen")]:
        y = tr_teams[col].to_numpy(float)
        if engine == "ridge":
            m = Ridge(alpha=10.0).fit(Xs, y)
        elif engine == "poisson":
            m = PoissonRegressor(alpha=1.0, max_iter=500).fit(Xs, y)
        else:
            raise ValueError(engine)
        models[tgt] = m
    return dict(models=models, scaler=scaler, med=med, feats=feats,
                mean=dict(conv=tr_teams["y_conv"].mean(), pen=tr_teams["y_pen"].mean()))


def predict_volume(fit: dict, te_teams: pd.DataFrame, engine: str) -> pd.DataFrame:
    out = te_teams[["fixture_id", "team_id"]].copy()
    if engine == "const_mean":
        out["hat_conv"] = fit["mean"]["conv"]
        out["hat_pen"] = fit["mean"]["pen"]
        return out
    X = te_teams[fit["feats"]].to_numpy(float)
    X = np.where(np.isfinite(X), X, fit["med"])
    Xs = fit["scaler"].transform(X)
    out["hat_conv"] = np.clip(fit["models"]["conv"].predict(Xs), 0.0, None)
    out["hat_pen"] = np.clip(fit["models"]["pen"].predict(Xs), 0.0, None)
    return out


def main() -> None:
    df = load()
    cfg = Config()
    train_mask = component_train(df, 2025)
    train_pos = np.where(train_mask)[0]
    dft = df.iloc[train_pos].reset_index(drop=True)

    all_feats = FEATS_BASIC + FEATS_TEAMPLAY
    variants = [
        ("ridge_basic", "ridge", FEATS_BASIC),
        ("ridge_teamplay", "ridge", all_feats),
        ("ridge_tp_only", "ridge", FEATS_TEAMPLAY),
        ("poisson_teamplay", "poisson", all_feats),
        ("const_mean", "const_mean", FEATS_BASIC),
    ]
    weights = [0.25, 0.5, 0.75, 1.0]
    CLIP = 2.0

    # abs-error accumulators: err[comp][label] -> list of arrays
    err: dict[str, dict[str, list]] = {c: {} for c in KICK_COMPS}

    def add(comp, label, y_true, y_pred):
        err[comp].setdefault(label, []).append(np.abs(y_true - y_pred))

    fold_corr = []
    for tr_local, va_local in group_kfold_indices(dft, 5):
        tr_idx = train_pos[tr_local]
        va_idx = train_pos[va_local]
        vmask = (df.iloc[va_idx]["minutes"] >= MIN_MINUTES).to_numpy()
        vu = va_idx[vmask]
        if len(vu) == 0:
            continue

        naive_vu = predict_rates(df, tr_idx, vu, MODE, "naive")
        naive_all = predict_rates(df, tr_idx, va_idx, MODE, "naive")
        minutes_hat_all = build_minutes(df, tr_idx, va_idx, cfg)

        # naive team totals (count space) over ALL validation rows
        va_sub = df.iloc[va_idx]
        counts = pd.DataFrame({
            "fixture_id": va_sub["fixture_id"].to_numpy(),
            "team_id": va_sub["team_id"].to_numpy(),
            "conv": naive_all["conversion_goals"].to_numpy(float) * minutes_hat_all / 80.0,
            "pen": naive_all["penalty_goals"].to_numpy(float) * minutes_hat_all / 80.0,
        })
        team_naive = counts.groupby(["fixture_id", "team_id"], as_index=False).sum()

        tr_teams = team_rows(df, tr_idx, all_feats)
        va_teams = team_rows(df, va_idx, all_feats)

        # ground truth: baseline naive
        for comp in KICK_COMPS:
            ytrue = np.nan_to_num(_rate_target(df.iloc[vu], comp), nan=0.0,
                                  posinf=0.0, neginf=0.0)
            add(comp, "naive", ytrue, naive_vu[comp].to_numpy(float))

        for vname, engine, feats in variants:
            fit = fit_volume_model(tr_teams, feats, "ridge" if engine == "const_mean" else engine)
            vol = predict_volume(fit, va_teams, engine)
            m = team_naive.merge(vol, on=["fixture_id", "team_id"], how="left")
            m["factor_conv"] = np.clip(m["hat_conv"] / m["conv"].clip(lower=0.25), 1.0 / CLIP, CLIP)
            m["factor_pen"] = np.clip(m["hat_pen"] / m["pen"].clip(lower=0.25), 1.0 / CLIP, CLIP)
            fmap_c = dict(zip(zip(m["fixture_id"], m["team_id"]), m["factor_conv"]))
            fmap_p = dict(zip(zip(m["fixture_id"], m["team_id"]), m["factor_pen"]))
            vu_sub = df.iloc[vu]
            keys = list(zip(vu_sub["fixture_id"], vu_sub["team_id"]))
            fc = np.array([fmap_c.get(k, 1.0) for k in keys])
            fp = np.array([fmap_p.get(k, 1.0) for k in keys])
            for w in weights:
                fc_w = 1.0 + w * (fc - 1.0)
                fp_w = 1.0 + w * (fp - 1.0)
                for comp, fac in [("conversion_goals", fc_w), ("penalty_goals", fp_w)]:
                    ytrue = np.nan_to_num(_rate_target(df.iloc[vu], comp), nan=0.0,
                                          posinf=0.0, neginf=0.0)
                    add(comp, f"{vname}_w{int(w*100):03d}", ytrue,
                        naive_vu[comp].to_numpy(float) * fac)

            # fixture-level correlation diagnostic (100% weight, realised totals)
            va_real = team_rows(df, va_idx, [])
            mm = m.merge(va_real, on=["fixture_id", "team_id"])
            fold_corr.append(dict(
                variant=vname,
                corr_naive_conv=mm["conv"].corr(mm["y_conv"]),
                corr_hat_conv=mm["hat_conv"].corr(mm["y_conv"]),
                corr_naive_pen=mm["pen"].corr(mm["y_pen"]),
                corr_hat_pen=mm["hat_pen"].corr(mm["y_pen"]),
            ))

    print("=== OOF per-80 rate MAE (2023+2024 window, minutes>=10 rows) ===")
    for comp in KICK_COMPS:
        base = float(np.concatenate(err[comp]["naive"]).mean())
        print(f"\n{comp}: naive = {base:.6f}   (2% gate: <= {base*0.98:.6f})")
        rows = []
        for label, v in err[comp].items():
            if label == "naive":
                continue
            mae = float(np.concatenate(v).mean())
            rows.append((mae, label))
        for mae, label in sorted(rows):
            gain = 100.0 * (1.0 - mae / base)
            flag = " <-- PASSES 2% GATE" if gain >= 2.0 else ""
            print(f"  {label:28s} {mae:.6f}  ({gain:+.2f}%){flag}")

    fc = pd.DataFrame(fold_corr).groupby("variant").mean()
    print("\n=== mean fixture-team total correlation across folds ===")
    print(fc.round(3).to_string())


if __name__ == "__main__":
    main()
