# Autoresearch Goal — improve MAE, XV value, and selection health

Branch: `cdx`.

This branch runs a Karpathy-style local research loop: try one scoped modelling or
selection change, evaluate it on the 2025 backtest, keep it only if it improves the
primary metrics without damaging secondary selection-health metrics, and log every
accepted/rejected trial. The loop costs 0 API calls.

## Current Baseline

Baseline to beat is the current promoted `cdx` deployable picker:
`target_frontrow_20_selector_raw_promoted`, post-team-sheet.

| metric | 2025 | 2026 sealed |
|---|---:|---:|
| selector value_xv | 0.707 | 0.711 |
| target MAE | 7.650 | 7.383 |
| selector top15 | 0.360 | 0.373 |
| selector top30 | 0.560 | 0.547 |
| selector Spearman | 0.489 | 0.584 |

2026 is not used by the loop. It is a sealed reporting check for a final accepted
configuration. For 2026 prediction, 2025 rows are valid training history; no future
2026 rows may enter a 2026 prediction.

## Acceptance Rule

Primary objectives remain MAE and XV value:

- value_xv improves by at least 0.003 with MAE not worse by more than 0.02, or
- MAE improves by at least 0.02 with value_xv not worse by more than 0.003, and
- the relevant gain appears in at least 3 of 5 rounds. Value_xv candidates need
  per-round XV value improvement; MAE candidates need per-round MAE improvement.

Secondary metrics are guardrails and tie-breakers:

- `top15` and selector-based within-position Spearman must not collapse; the loop
  rejects otherwise acceptable candidates if `top15` drops by more than 0.04 or
  `spearman_sel` drops by more than 0.03.
- `top30`, `capt_top1`, `capt_top3`, and `capt_top5` are reported for diagnosis.
- The JSON ledger also persists `round_xv`, `round_mae`, `round_bias`, and
  by-position MAE/bias diagnostics.

Rejected candidates stay in the ledger; accepted candidates become the new incumbent.

## Current Best

| config | 2025 value_xv | 2025 MAE | 2025 top15 | 2026 value_xv | 2026 MAE |
|---|---:|---:|---:|---:|---:|
| baseline `lgbm_only` | 0.680 | 7.680 | 0.360 | 0.703 | 7.447 |
| previous promoted `selector_tilt_025` | 0.707 | 7.680 | 0.360 | 0.711 | 7.447 |
| **promoted `target_frontrow_20_selector_raw`** | **0.707** | **7.650** | **0.360** | **0.711** | **7.383** |

Latest dev-only vetoes:

- `target_frontrow20_back5_20` improved 2025 MAE to 7.609 with unchanged XV value, but sealed
  2026 MAE worsened to 7.529. It is not promoted.
- `minutes_alpha_4` improved 2025 MAE to 7.621 and top15 to 0.373, but sealed 2026 slipped to
  MAE 7.391 and XV value 0.706 versus the promoted 7.383 / 0.711. It is not promoted.
- Front-row split variants (`frontrow_15`, `frontrow_25`, `prop_20`, `hooker_20`) failed the 2025
  gate, so the promoted 20% combined front-row calibration remains the incumbent.

The promoted improvement has two pieces:

- `target_pts_hat` applies a 20% front-row-only calibration toward prior position means.
- `sel_score` still uses the pre-calibration point signal plus the rank head:
  `sel_score = zscore(selector_pts_hat) + 0.25 * zscore(rank_score)`.

This keeps the MAE calibration from perturbing XV ranking.

## Artifacts

- `model/research.py` — research loop and candidate queue.
- `research/ledger.json` and `research/LEDGER.md` — latest trial log.
- `research/history.jsonl` — append-only history across overwritten trial logs.
- `research/best_config.json` — latest loop best.
- `research/promoted_config.json` — deployable config after sealed fallback checks.
- `research/promotion_report.json` — sealed comparison and fallback reason.
- `research/sealed_best_check.json` — sealed check for `research/best_config.json`; prediction CSV
  is not overwritten.
- `research/sealed_veto_target_frontrow20_back5_20.json` — sealed veto evidence for the stacked
  back-five prior dev candidate.
- `research/sealed_veto_minutes_alpha_4.json` — sealed veto evidence for the tighter minutes-ridge
  dev candidate.
- `research/dev_predictions_2025.csv` — dev-best prediction snapshot from the latest loop.
- `data/model_predictions_2025.csv` — promoted dev predictions.
- `data/model_predictions_2026.csv` — promoted sealed-season predictions.

## Commands

```bash
/usr/local/bin/python3.11 -m model.research --base-config promoted --candidate target_frontrow_20_selector_raw
/usr/local/bin/python3.11 -m model.research --base-config promoted --candidate target_frontrow20_back5_10 --candidate target_frontrow20_back5_20 --candidate target_frontrow20_nonfront_10 --candidate target_frontrow20_nonfront_20
/usr/local/bin/python3.11 -m model.research --base-config promoted --candidate minutes_alpha_4 --candidate minutes_alpha_5 --candidate minutes_alpha_6 --candidate minutes_alpha_7 --candidate minutes_alpha_8 --candidate minutes_alpha_9
/usr/local/bin/python3.11 -m model.research --base-config promoted --candidate target_frontrow_15_selector_raw --candidate target_frontrow_25_selector_raw --candidate target_prop_20_selector_raw --candidate target_hooker_20_selector_raw
/usr/local/bin/python3.11 -m model.research --seal-2026 --seal-config best
/usr/local/bin/python3.11 -m model.research --seal-2026 --seal-config promoted
```
