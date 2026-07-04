#!/usr/bin/env python3
"""Do the adopted changes (position minutes + shrink-0.20) help, and do they
help the 2026 OOF rounds specifically?

Runs the mixed all-years round-level-OOF CV twice — ORIGINAL (global minutes,
no shrink) vs IMPROVED (position minutes + naive-shrink-0.20) — and reports
recon-MAE and value_xv broken out by year, averaged over seeds. 2026 rounds are
ordinary OOF test rounds (held out of their own fold), so this is exactly "the
2026 OOF model".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.data import load
from autoresearch_mixed import _recon, _ALL_COMPS

ORIGINAL = dict(minutes="global", lgbm={}, shrink={})
IMPROVED = dict(minutes="position", lgbm={},
                shrink={c: 0.20 for c in _ALL_COMPS}, shrink_target="naive")
SEEDS = [0, 1, 2]


def summarise(df, pool_mask, cfg):
    """Return (per-year MAE dict, per-year value_xv dict, overall MAE, overall vxv)."""
    yr = df["season"].to_numpy()
    mae_by_year = {y: [] for y in (2023, 2024, 2025, 2026)}
    vxv_by_year = {y: [] for y in (2023, 2024, 2025, 2026)}
    all_err, all_vxv = [], []
    for s in SEEDS:
        row_err, per_round = _recon(df, pool_mask, cfg, seed=s)
        m = ~np.isnan(row_err)
        all_err.append(row_err[m])
        for y in mae_by_year:
            sel = m & (yr == y)
            mae_by_year[y].append(row_err[sel].mean())
        for (y, r), v in per_round.items():
            vxv_by_year[y].append(v); all_vxv.append(v)
    return (
        {y: float(np.mean(v)) for y, v in mae_by_year.items()},
        {y: float(np.nanmean(v)) for y, v in vxv_by_year.items()},
        float(np.concatenate(all_err).mean()),
        float(np.nanmean(all_vxv)),
    )


def main() -> None:
    df = load().reset_index(drop=True)
    pool_mask = df["recon_pts"].notna().to_numpy() & (df["minutes"].fillna(0) > 0).to_numpy()

    om, ov, oM, oV = summarise(df, pool_mask, ORIGINAL)
    im, iv, iM, iV = summarise(df, pool_mask, IMPROVED)

    print("=== ORIGINAL (global minutes, no shrink) vs IMPROVED (position + shrink-0.20) ===")
    print("recon value_xv over the 20-round mixed OOF (round-level isolation):")
    print(f"  OVERALL: original {oV:.4f} -> improved {iV:.4f}  ({iV-oV:+.4f})")
    print(f"\n{'year':>6} {'MAE orig':>9} {'MAE impr':>9} {'dMAE':>8}   "
          f"{'vxv orig':>9} {'vxv impr':>9} {'dvxv':>8}")
    for y in (2023, 2024, 2025, 2026):
        print(f"{y:>6} {om[y]:>9.3f} {im[y]:>9.3f} {im[y]-om[y]:>+8.3f}   "
              f"{ov[y]:>9.4f} {iv[y]:>9.4f} {iv[y]-ov[y]:>+8.4f}")
    print(f"\n  OVERALL recon MAE: original {oM:.4f} -> improved {iM:.4f}  ({iM-oM:+.4f})")
    print(f"\n2026 OOF specifically: MAE {om[2026]:.3f} -> {im[2026]:.3f} "
          f"({im[2026]-om[2026]:+.3f}),  value_xv {ov[2026]:.4f} -> {iv[2026]:.4f} "
          f"({iv[2026]-ov[2026]:+.4f})")


if __name__ == "__main__":
    main()
