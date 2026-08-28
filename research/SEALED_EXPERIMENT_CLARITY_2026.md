# Sealed 2026 Experiment Clarity Check

Date: 2026-06-18

All candidates below are evaluated on sealed 2026 from the original `points_lgbm80_xgb20` base. This is a retrospective clarity check, not the promotion gate.

Old base: value_team=0.677667, value_xv=0.714459, MAE=7.348995.
Current promoted `supersub_two_stage_minutes`: value_team=0.687513, value_xv=0.714459, MAE=7.348995.

| candidate | team | dteam | XV | dXV | MAE | dMAE | rounds up | sealed pareto? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `selector_upside_halfbacks_backs_025` | 0.692 | +0.015 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | yes |
| `supersub_bench_lgbm_025` | 0.688 | +0.010 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | yes |
| `supersub_bench_lgbm_050` | 0.688 | +0.010 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | yes |
| `supersub_two_stage_minutes` | 0.688 | +0.010 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | yes |
| `supersub_minutes_uncertainty_025` | 0.688 | +0.010 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | yes |
| `captain_mean_only` | 0.678 | +0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 0/5 | no |
| `captain_mean_plus_upside_010` | 0.678 | +0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 0/5 | no |
| `captain_mean_plus_upside_020` | 0.678 | +0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 0/5 | no |
| `captain_kicker_floor_plus_upside` | 0.678 | +0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 0/5 | no |
| `captain_rank_overlay_025` | 0.678 | +0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 0/5 | no |
| `selector_mean_plus_upside_010` | 0.678 | +0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 0/5 | no |
| `supersub_bench_ridge_025` | 0.678 | -0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
| `supersub_value_minus_risk_025` | 0.678 | -0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
| `bench_kick_shrink_25` | 0.678 | -0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
| `bench_kick_shrink_50` | 0.678 | -0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
| `bench_kick_learned_gate` | 0.678 | -0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
| `bench_kick_lgbm_share` | 0.678 | -0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
| `team_specialist_bench025_capt010` | 0.678 | -0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
| `team_specialist_bench025_upside010` | 0.678 | -0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
| `team_specialist_bench025_capt010_upside010` | 0.678 | -0.000 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
| `selector_mean_plus_upside_020` | 0.677 | -0.000 | 0.714 | -0.000 | 7.349 | +0.000 | 0/5 | no |
| `selector_upside_backthree_025` | 0.674 | -0.004 | 0.710 | -0.004 | 7.349 | +0.000 | 0/5 | no |
| `selector_upside_backrow_backs_025` | 0.674 | -0.004 | 0.710 | -0.005 | 7.349 | +0.000 | 0/5 | no |
| `supersub_bench_ridge_050` | 0.672 | -0.005 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
| `bench_kick_ridge_share` | 0.672 | -0.005 | 0.714 | +0.000 | 7.349 | +0.000 | 1/5 | no |
