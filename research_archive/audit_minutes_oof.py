#!/usr/bin/env python3
"""OOF gate audit: position-aware starter/bench minutes head.

Evidence (audit_component_residuals.py, 2025 dev): the global minutes ridge
([started, jersey, is_forward, form-prior]) over-predicts front-row starters
by ~8 minutes (prop -8.1, hooker -8.5 bias) and under-predicts front-row bench
by ~7 (prop +6.5, hooker +7.8) because front-row substitution is structural.

Protocol: GroupKFold(5) on fixture_id over the 2023+2024 window.  Candidate
designs are fit on each fold's train rows and scored on validation rows:
minutes MAE overall and bias by position x started.  No 2025/2026 rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from model.baselines import MIN_MINUTES
from model.data import load
from model.splits import component_train, group_kfold_indices

POSITIONS = ["Prop", "Hooker", "Second-row", "Back-row",
             "Scrum-half", "Fly-half", "Centre", "Back-three"]


def base_design(rows: pd.DataFrame, pos_mean: pd.Series, glob: float) -> np.ndarray:
    base = rows["form_minutes_recent"].to_numpy(float).copy()
    fill = rows["canonical_pos"].map(pos_mean).fillna(glob).to_numpy(float)
    prior = np.where(np.isnan(base), fill, base)
    return np.column_stack([
        rows[["started", "jersey", "is_forward"]].astype(float).to_numpy(),
        prior[:, None],
    ])


def pos_design(rows: pd.DataFrame, pos_mean: pd.Series, glob: float) -> np.ndarray:
    X = base_design(rows, pos_mean, glob)
    started = rows["started"].astype(float).to_numpy()
    dummies = [
        ((rows["canonical_pos"] == p).astype(float).to_numpy() * started)[:, None]
        for p in POSITIONS
    ]
    return np.column_stack([X] + dummies)


def main() -> None:
    df = load()
    train_mask = component_train(df, 2025)
    train_pos = np.where(train_mask)[0]
    dft = df.iloc[train_pos].reset_index(drop=True)

    variants = {
        "ridge_global(a=10)": ("base", 10.0),
        "ridge_pos_started(a=10)": ("pos", 10.0),
        "ridge_pos_started(a=30)": ("pos", 30.0),
    }
    errs = {v: [] for v in variants}
    seg_rows = []

    for tr_local, va_local in group_kfold_indices(dft, 5):
        tr_idx = train_pos[tr_local]
        va_idx = train_pos[va_local]
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        pos_mean = tr.loc[tr["minutes"] >= MIN_MINUTES].groupby("canonical_pos")["minutes"].mean()
        glob = float(tr.loc[tr["minutes"] >= MIN_MINUTES, "minutes"].mean())
        y_tr = tr["minutes"].to_numpy(float)
        y_va = va["minutes"].to_numpy(float)

        for name, (kind, alpha) in variants.items():
            dz = base_design if kind == "base" else pos_design
            reg = Ridge(alpha=alpha).fit(dz(tr, pos_mean, glob), y_tr)
            hat = np.clip(reg.predict(dz(va, pos_mean, glob)), 0, 80)
            errs[name].append(np.abs(y_va - hat))
            seg = pd.DataFrame({
                "variant": name,
                "pos": va["canonical_pos"].to_numpy(),
                "started": va["started"].astype(bool).to_numpy(),
                "err": y_va - hat,
            })
            seg_rows.append(seg)

    print("=== OOF minutes MAE (2023+2024 window, all rows) ===")
    base_mae = float(np.concatenate(errs["ridge_global(a=10)"]).mean())
    for name, v in errs.items():
        mae = float(np.concatenate(v).mean())
        gain = 100.0 * (1.0 - mae / base_mae)
        print(f"  {name:26s} {mae:.4f}  ({gain:+.2f}%)")

    seg = pd.concat(seg_rows)
    print("\n=== OOF minutes bias (mean err = realised - predicted) by position x started ===")
    piv = seg.groupby(["variant", "pos", "started"])["err"].mean().unstack("variant").round(2)
    print(piv.to_string())


if __name__ == "__main__":
    main()
