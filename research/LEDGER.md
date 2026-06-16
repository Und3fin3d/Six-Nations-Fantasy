# Autoresearch ledger — 2026-06-16

Dev season 2025 (post_team_sheet). Baseline value_xv=0.680 / MAE=7.680.
2026 sealed — not evaluated here.

| # | candidate | value_xv | MAE | top15 | accepted | reason |
|---|---|---|---|---|---|---|
| 0 | baseline | 0.680 | 7.680 | 0.360 | yes | incumbent |
| 1 | target_posmean_10 | 0.680 | 7.628 | 0.360 | yes | ACCEPT via mae: dval=+0.000 dmae=-0.052 value_rounds_up=0/5 mae_rounds_up=4/5 |
| 2 | target_posmean_20 | 0.680 | 7.620 | 0.360 | no | reject (no Pareto improvement): dval=+0.000 dmae=-0.008 value_rounds_up=0/5 mae_rounds_up=3/5 |
| 3 | selector_tilt_025 | 0.707 | 7.628 | 0.387 | yes | ACCEPT via value_xv: dval=+0.027 dmae=+0.000 value_rounds_up=3/5 mae_rounds_up=0/5 |

**Best:** `selector_tilt_025` — value_xv 0.707, MAE 7.628, top15 0.387.

Promotion note: this is the 2025-dev winner, not the deployable promotion. The sealed
generalisation check kept `research/promoted_config.json` on `selector_tilt_025_promoted`
because the target-prior MAE gain did not hold on 2026 MAE/value_xv.

```json
{
  "name": "selector_tilt_025",
  "note": "tilt XV pick toward rank head",
  "deploy_engine": "lgbm_only",
  "lgbm_margin": 0.02,
  "blend_weight": 0.5,
  "minutes_model": "ridge",
  "minutes_alpha": 10.0,
  "potm_pp_weight": 0.5,
  "potm_tau_floor": 1.0,
  "latent_shrink": 1.0,
  "recon_calib": "none",
  "target_prior_blend": 0.1,
  "selector_tilt": 0.25
}
```
