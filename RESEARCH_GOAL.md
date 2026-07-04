# Autoresearch Goal — improve MAE and XV value

Branch: `codex/xgboost-model-architecture` in worktree `../6n-xgboost`.

This worktree is an XGBoost architecture experiment branched from `cdx`: the
former `lgbm_only` component path is now `xgb_only`, and the rank overlay uses
`XGBRanker`. The 2025-only harness selected `xgb_only` with value_xv `0.718`
and MAE `7.515`; the current accepted XGBoost dev-promoted config improves that
to selector-score value_xv `0.768` and MAE `7.291`. Sealed 2026 has now been
checked as a promotion veto: the XGBoost dev winner did not beat the saved
LightGBM sealed incumbent, so it remains dev-promoted only.

This branch runs a Karpathy-style local research loop: try one scoped modelling or
selection change, evaluate it on the 2025 backtest, keep it only if it improves the
chosen metrics, and log every accepted/rejected trial. The loop costs 0 API calls.

## Current Baseline

Baseline to beat is the current `cdx` deployable picker: `lgbm_only`, post-team-sheet.
Within this worktree, the comparable deployable engine is `xgb_only`.

| metric | cdx baseline |
|---|---:|
| 2025 value_xv | 0.680 |
| 2025 MAE | 7.680 |
| 2025 top15 | 0.360 |
| 2025 spearman_pos | 0.475 |

2026 is not used by the loop. It is a sealed reporting check for a final accepted
configuration, and the 2026-06-17 check vetoed global promotion of the XGBoost
dev winner.

## Acceptance Rule

Accept a candidate only if it improves the 2025 objective without obvious fragility:

- value_xv improves by at least 0.003 with MAE not worse by more than 0.02, or
- MAE improves by at least 0.02 with value_xv not worse by more than 0.003, and
- the relevant gain appears in at least 3 of 5 rounds. Value_xv candidates need
  per-round XV value improvement; MAE candidates need per-round MAE improvement.

Rejected candidates stay in the ledger; accepted candidates become the new incumbent.

## Current Best

The current XGBoost-branch incumbent is 2025-dev promoted only. It was evaluated
once on sealed 2026 on 2026-06-17 and did not generalise strongly enough for
global promotion.

| config | value_xv | MAE | top15 | top30 | spearman_pos |
|---|---:|---:|---:|---:|---:|
| cdx baseline `lgbm_only` | 0.680 | 7.680 | 0.360 | n/a | 0.475 |
| xgboost baseline `xgb_only` | 0.718 | 7.515 | 0.373 | 0.540 | 0.503 |
| accepted `latent_shrink_05` | 0.722 | 7.366 | 0.373 | 0.553 | 0.513 |
| accepted `target_posmean_10` | 0.722 | 7.346 | 0.387 | 0.547 | 0.509 |
| accepted `latent_shrink_025_posmean_10` | 0.721 | 7.312 | 0.400 | 0.553 | 0.513 |
| promoted `potm10_latent010_tilt0025` | **0.768** | **7.291** | **0.427** | **0.573** | **0.517** |

Sealed 2026 promotion check:

| config | value_xv | MAE | top15 | top30 | capt_top3 | spearman_pos |
|---|---:|---:|---:|---:|---:|---:|
| saved LightGBM incumbent `selector_tilt_025_promoted` | **0.711** | **7.447** | 0.373 | 0.547 | 0.200 | 0.578 |
| xgboost baseline `xgb_only` | 0.684 | 7.529 | **0.387** | 0.547 | 0.200 | 0.573 |
| `target_posmean_10` | 0.684 | 7.461 | 0.347 | **0.573** | 0.200 | 0.577 |
| `latent_shrink_025_posmean_10` | 0.691 | 7.469 | 0.373 | 0.547 | 0.200 | 0.579 |
| dev-promoted `potm10_latent010_tilt0025` | 0.690 | 7.500 | 0.360 | 0.533 | 0.200 | **0.579** |

Result: the XGBoost loop produced real 2025 gains, and some intermediate
changes partially generalise within XGBoost (`target_posmean_10` has the best
sealed MAE/top30), but the final dev-promoted candidate is not a deployable
replacement for the saved sealed incumbent.

Accepted changes:

- `latent_shrink_05`: shrink set-piece latent contribution by 50%; accepted via
  MAE improvement across 4/5 rounds.
- `target_posmean_10`: blend 10% toward forward-chained prior position means;
  accepted via additional MAE improvement across 3/5 rounds.
- `latent_shrink_025_posmean_10`: shrink set-piece latent contribution to 25%
  while keeping the 10% position-mean blend; accepted via MAE improvement across
  3/5 rounds.
- `potm10_latent010_tilt0025`: make POTM probability points-only, shrink latent
  to 10%, and add a tiny rank tilt for XV selection; accepted via value_xv
  improvement across 5/5 rounds.

Rejected in the latest loop: `minutes_alpha_3`, `minutes_alpha_30`,
`latent_shrink_0`, `potm_tau_2`, and `target_posmean_20`.

Continuation batch on 2026-06-17 started from `research/promoted_config.json`
and wrote `research/continuation_2026_06_17.{json,md}`. Rejected:
`potm_pp_07`, `potm_pp_03`, `xgb_margin_05`, `xgb_margin_01`,
`recon_calib_linear`, `selector_tilt_010`, `selector_tilt_025`, and
`selector_tilt_05`. The aggregate value bumps from `potm_pp_07`,
`selector_tilt_010`, and `selector_tilt_05` failed the 3-of-5 round robustness
guard, so that batch left the promoted config unchanged.

Latent-shrink continuation on 2026-06-17 wrote
`research/continuation_2026_06_17_latent.{json,md}`. It accepted
`latent_shrink_025_posmean_10` and rejected `minutes_alpha_3`,
`latent_shrink_025_posmean_05`, `latent_shrink_075_posmean_10`, and
`minutes_alpha_3_potm07`.

POTM/tilt continuation on 2026-06-17 wrote
`research/continuation_2026_06_17_potm_tilt.{json,md}`. It accepted
`potm10_latent010_tilt0025`. Because this config uses `selector_tilt`, XV,
captain, top15, and top30 are measured on `sel_score`; MAE and Spearman are
measured on `target_pts_hat`. The target-points-only XV value is `0.757`.
The intermediate `research/continuation_2026_06_17_potm.{json,md}` ledger
records the POTM-only ordering pass that accepted `potm_pp_10` before the
combined candidate was run cleanly.

Post-seal XGBoost continuation on 2026-06-17 did not read 2026 again. It tested
two additional 2025-only robustness checks from the dev-promoted incumbent:
`deploy_registry` in `research/continuation_2026_06_17_registry.{json,md}` and
`target_posmean_20` in
`research/continuation_2026_06_17_posmean20_from_potm.{json,md}`. Both were
rejected. `deploy_registry` lifted top30 to `0.580` but collapsed value_xv to
`0.677` and worsened MAE to `7.468`; `target_posmean_20` lifted top15 to
`0.440` but worsened MAE to `7.317` with no value gain.

## Artifacts

- `model/research.py` — research loop and candidate queue.
- `research/ledger.json` and `research/LEDGER.md` — trial log.
- `research/continuation_2026_06_17.json` and `.md` — continuation batch from
  the current promoted XGBoost incumbent.
- `research/continuation_2026_06_17_latent.json` and `.md` — latent-shrink
  continuation batch from the prior promoted XGBoost incumbent.
- `research/continuation_2026_06_17_potm_tilt.json` and `.md` — POTM/selector
  continuation batch from the prior promoted XGBoost incumbent.
- `research/continuation_2026_06_17_potm.json` and `.md` — intermediate
  POTM-only ordering batch.
- `research/continuation_2026_06_17_registry.json` and `.md` — rejected
  OOF-registry fallback from the dev-promoted XGBoost incumbent.
- `research/continuation_2026_06_17_posmean20_from_potm.json` and `.md` —
  rejected stronger position-prior calibration from the dev-promoted XGBoost
  incumbent.
- `research/best_config.json` — best 2025-dev candidate.
- `research/promoted_config.json` — current dev-promoted XGBoost config.
- `research/promotion_report.json` — sealed comparison and fallback reason.
- `research/sealed_xgboost_generalization_2026.json` and `.md` — detailed
  sealed XGBoost generalisation check.
- `research/sealed_best_check.json` — earlier sealed check from the pre-XGBoost
  promotion fallback; not the current XGBoost generalisation report.
- `data/model_predictions_2025.csv` — dev predictions for the current
  dev-promoted config.
- `data/model_predictions_2026.csv` — promoted sealed-season predictions.

## Commands

```bash
/usr/local/bin/python3.11 -m model.research
/usr/local/bin/python3.11 -m model.research --seal-2026 --seal-config promoted
```
