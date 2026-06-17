# Autoresearch ledger — 2026-06-17

Dev season 2025 (post_team_sheet). Incumbent value_xv=0.707 / MAE=7.650.
2026 sealed — not evaluated here.

| # | candidate | value_xv | MAE | top15 | top30 | cap3 | spear_sel | accepted | reason |
|---|---|---|---|---|---|---|---|---|---|
| 0 | target_frontrow_20_selector_raw_promoted | 0.707 | 7.650 | 0.360 | 0.560 | 0.000 | 0.489 | yes | incumbent |
| 1 | target_frontrow_15_selector_raw | 0.707 | 7.653 | 0.360 | 0.560 | 0.000 | 0.489 | no | reject (no Pareto improvement): dval=+0.000 dmae=+0.002 dtop15=+0.000 dspear=+0.000 value_rounds_up=0/5 mae_rounds_up=1/5 |
| 2 | target_frontrow_25_selector_raw | 0.707 | 7.652 | 0.360 | 0.560 | 0.000 | 0.489 | no | reject (no Pareto improvement): dval=+0.000 dmae=+0.002 dtop15=+0.000 dspear=+0.000 value_rounds_up=0/5 mae_rounds_up=1/5 |
| 3 | target_prop_20_selector_raw | 0.707 | 7.653 | 0.360 | 0.560 | 0.000 | 0.489 | no | reject (no Pareto improvement): dval=+0.000 dmae=+0.003 dtop15=+0.000 dspear=+0.000 value_rounds_up=0/5 mae_rounds_up=2/5 |
| 4 | target_hooker_20_selector_raw | 0.707 | 7.677 | 0.360 | 0.560 | 0.000 | 0.489 | no | reject (no Pareto improvement): dval=+0.000 dmae=+0.027 dtop15=+0.000 dspear=+0.000 value_rounds_up=0/5 mae_rounds_up=0/5 |

**Best:** `target_frontrow_20_selector_raw_promoted` — value_xv 0.707, MAE 7.650, top15 0.360.

```json
{
  "name": "target_frontrow_20_selector_raw_promoted",
  "note": "Promoted: 20% front-row target calibration for MAE, while XV selector uses pre-calibration point signal.",
  "deploy_engine": "lgbm_only",
  "lgbm_margin": 0.02,
  "blend_weight": 0.5,
  "minutes_model": "ridge",
  "minutes_alpha": 10.0,
  "potm_pp_weight": 0.5,
  "potm_tau_floor": 1.0,
  "latent_shrink": 1.0,
  "recon_calib": "none",
  "target_prior_blend": 0.2,
  "target_prior_scope": "frontrow",
  "extra_prior_blend": 0.0,
  "extra_prior_scope": "all",
  "selector_tilt": 0.25,
  "selector_point_source": "pre_calib"
}
```