#!/usr/bin/env python3
"""OOF gate audit: fixture-team VOLUME rescaling for the big-mass components.

Generalisation of audit_kick_volume_oof.py (kicks passed OOF at +14.8%/+9.9%
but their aggregate error mass is too small to clear the promotion gate).
Here the same team-volume x unchanged-allocation mechanism is tested on the
components that actually carry the 2025 residual budget:

  tries (|resid| 2.55 pts/row), tackles (3.01), defenders_beaten (2.13),
  metres (1.33), offload (1.04), try_assists (0.75)

Baseline engine per component = the production engine (LGBM for LGBM_COMPS
under the promoted lgbm_only deploy).  GroupKFold(5) on fixture_id over the
2023+2024 window, per-80 rate MAE on minutes>=10 validation rows.  The volume
model sees only PIT team-level features; targets are realised team totals
summed over TRAIN-fold rows.  Gate: >=2% relative OOF improvement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from model.baselines import MIN_MINUTES, _rate_target
from model.data import load
from model.research import Config, build_minutes
from model.splits import component_train, group_kfold_indices
from model.train_components import _fit_lgbm_component, _lgbm_predict

MODE = "post_team_sheet"
COMPS = ["tries", "tackles", "metres", "defenders_beaten", "offload", "try_assists"]

TEAM_FEATS = [
    # strength / points context
    "wr_pts_gap", "wr_rank_gap", "team_wr_pts", "opp_wr_pts",
    "h2h_last3_margin", "h2h_last3_winrate",
    # attack context
    "teamplay_points_for_hat", "teamplay_points_against_hat",
    "teamplay_margin_hat", "teamplay_win_prob_hat",
    "teamplay_possession_hat", "teamplay_runs_hat", "teamplay_metres_hat",
    "teamplay_aspect_attack_volume_hat", "teamplay_aspect_attack_balance_hat",
    "teamplay_aspect_open_game_hat", "teamplay_aspect_points_edge_hat",
    # defensive load context (for tackles)
    "teamplay_tackles_required_hat", "teamplay_aspect_defensive_load_hat",
    "teamplay_aspect_pressure_hat",
    # opponent permissiveness
    "opp_concede_tries", "opp_concede_metres", "opp_concede_db",
    "opp_possession", "opp_runs",
]


def team_frame(df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    sub = df.iloc[idx]
    agg = {f: (f, "first") for f in TEAM_FEATS}
    for c in COMPS:
        agg[f"team_{c}"] = (f"y_{c}", "sum")
    return sub.groupby(["fixture_id", "team_id"]).agg(**agg).reset_index()


def main() -> None:
    df = load()
    cfg = Config()
    train_mask = component_train(df, 2025)
    train_pos = np.where(train_mask)[0]
    dft = df.iloc[train_pos].reset_index(drop=True)

    weights = [0.25, 0.5, 0.75, 1.0]
    CLIP = 2.0
    err: dict[str, dict[str, list]] = {c: {} for c in COMPS}

    def add(comp, label, y_true, y_pred):
        err[comp].setdefault(label, []).append(np.abs(y_true - y_pred))

    corr_rows = []
    for tr_local, va_local in group_kfold_indices(dft, 5):
        tr_idx = train_pos[tr_local]
        va_idx = train_pos[va_local]
        vmask = (df.iloc[va_idx]["minutes"] >= MIN_MINUTES).to_numpy()
        vu = va_idx[vmask]
        if len(vu) == 0:
            continue

        minutes_hat_all = build_minutes(df, tr_idx, va_idx, cfg)
        va_sub = df.iloc[va_idx]

        tr_teams = team_frame(df, tr_idx)
        va_teams = team_frame(df, va_idx)
        Xtr = tr_teams[TEAM_FEATS].to_numpy(float)
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xva = va_teams[TEAM_FEATS].to_numpy(float)
        Xva = np.where(np.isfinite(Xva), Xva, med)
        Xtr_s, Xva_s = (Xtr - mu) / sd, (Xva - mu) / sd

        # positional map from va rows to fixture-team
        va_keys = list(zip(va_sub["fixture_id"], va_sub["team_id"]))
        vu_local = np.searchsorted(va_idx, vu)  # vu subset positions in va_idx
        assert np.array_equal(va_idx[vu_local], vu)

        for comp in COMPS:
            # production engine baseline: LGBM rate model on the fold
            m = _fit_lgbm_component(df, tr_idx, va_idx, comp, MODE)
            rate_all = _lgbm_predict(m, df, va_idx, MODE)
            ytrue = np.nan_to_num(_rate_target(df.iloc[vu], comp), nan=0.0,
                                  posinf=0.0, neginf=0.0)
            add(comp, "lgbm", ytrue, rate_all[vu_local])

            # naive-aggregated team totals from the LGBM rates
            counts = rate_all * minutes_hat_all / 80.0
            cdf = pd.DataFrame({
                "fixture_id": va_sub["fixture_id"].to_numpy(),
                "team_id": va_sub["team_id"].to_numpy(),
                "c": counts,
            })
            team_base = cdf.groupby(["fixture_id", "team_id"], as_index=False)["c"].sum()

            for label, engine in [("ridge", "ridge"), ("mean", "const")]:
                y = tr_teams[f"team_{comp}"].to_numpy(float)
                if engine == "ridge":
                    hat = np.clip(Ridge(alpha=10.0).fit(Xtr_s, y).predict(Xva_s), 0.0, None)
                else:
                    hat = np.full(len(va_teams), float(y.mean()))
                vt = va_teams[["fixture_id", "team_id"]].copy()
                vt["hat"] = hat
                mm = team_base.merge(vt, on=["fixture_id", "team_id"], how="left")
                mm["factor"] = np.clip(
                    mm["hat"] / mm["c"].clip(lower=0.25), 1.0 / CLIP, CLIP)
                fmap = dict(zip(zip(mm["fixture_id"], mm["team_id"]), mm["factor"]))
                frow = np.array([fmap.get(k, 1.0) for k in va_keys])
                for w in weights:
                    fac = 1.0 + w * (frow - 1.0)
                    add(comp, f"{label}_w{int(w*100):03d}", ytrue,
                        rate_all[vu_local] * fac[vu_local])

                if label == "ridge" and w == 1.0:
                    pass

            # fixture-level correlation diagnostic
            real = va_teams[["fixture_id", "team_id", f"team_{comp}"]].merge(
                team_base, on=["fixture_id", "team_id"])
            vt = va_teams[["fixture_id", "team_id"]].copy()
            y = tr_teams[f"team_{comp}"].to_numpy(float)
            vt["hat"] = np.clip(Ridge(alpha=10.0).fit(Xtr_s, y).predict(Xva_s), 0.0, None)
            real = real.merge(vt, on=["fixture_id", "team_id"])
            corr_rows.append(dict(
                comp=comp,
                corr_lgbm=real["c"].corr(real[f"team_{comp}"]),
                corr_ridge=real["hat"].corr(real[f"team_{comp}"]),
            ))

    print("=== OOF per-80 rate MAE (2023+2024, minutes>=10; baseline = production LGBM) ===")
    for comp in COMPS:
        base = float(np.concatenate(err[comp]["lgbm"]).mean())
        print(f"\n{comp}: lgbm = {base:.6f}   (2% gate: <= {base*0.98:.6f})")
        rows = []
        for label, v in err[comp].items():
            if label == "lgbm":
                continue
            mae = float(np.concatenate(v).mean())
            rows.append((mae, label))
        for mae, label in sorted(rows):
            gain = 100.0 * (1.0 - mae / base)
            flag = " <-- PASSES 2% GATE" if gain >= 2.0 else ""
            print(f"  {label:14s} {mae:.6f}  ({gain:+.2f}%){flag}")

    cr = pd.DataFrame(corr_rows).groupby("comp").mean()
    print("\n=== mean fixture-team total correlation with realised (across folds) ===")
    print(cr.round(3).to_string())


if __name__ == "__main__":
    main()
