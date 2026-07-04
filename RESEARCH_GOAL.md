# Autoresearch Goal — improve MAE and XV value

Branch: `cdx`.

This branch runs a Karpathy-style local research loop: try one scoped modelling or
selection change, evaluate it on the 2025 backtest, keep it only if it improves the
chosen metrics, and log every accepted/rejected trial. The loop costs 0 API calls.

## Current Baseline

Baseline to beat is the current `cdx` deployable picker: `lgbm_only`, post-team-sheet.

| metric | value |
|---|---:|
| 2025 value_xv | 0.680 |
| 2025 MAE | 7.680 |
| 2025 top15 | 0.360 |
| 2025 spearman_pos | 0.475 |

2026 is not used by the loop. It is a sealed reporting check for a final accepted
configuration.

## Acceptance Rule

Accept a candidate only if it improves the 2025 objective without obvious fragility:

- value_xv improves by at least 0.003 with MAE not worse by more than 0.02, or
- MAE improves by at least 0.02 with value_xv not worse by more than 0.003, and
- the relevant gain appears in at least 3 of 5 rounds. Value_xv candidates need
  per-round XV value improvement; MAE candidates need per-round MAE improvement.

Rejected candidates stay in the ledger; accepted candidates become the new incumbent.

## Current Best

There are now two useful labels:

- **Dev winner:** best 2025-only candidate found by the loop.
- **Promoted config:** deployable config after the sealed 2026 generalisation veto.

| config | value_xv | MAE | top15 |
|---|---:|---:|---:|
| baseline `lgbm_only` | 0.680 | 7.680 | 0.360 |
| promoted `selector_tilt_025` | 0.707 | 7.680 | 0.360 |
| dev winner `target_posmean_10 + selector_tilt_025` | 0.707 | 7.628 | 0.387 |

The promoted improvement comes from using calibrated points for MAE, but a separate
`sel_score` for XV picking:

`sel_score = zscore(target_pts_hat) + 0.25 * zscore(rank_score)`

The current dev winner adds a 10% forward-chained blend toward prior position means, but it is
not promoted: it regressed sealed 2026 versus the selector-only incumbent (MAE 7.473 vs 7.447,
value_xv 0.706 vs 0.711). The earlier tighter-minutes dev winner was also vetoed. The fallback
rule keeps `research/promoted_config.json` on the selector-only config.

## Artifacts

- `model/research.py` — research loop and candidate queue.
- `research/ledger.json` and `research/LEDGER.md` — trial log.
- `research/best_config.json` — best 2025-dev candidate.
- `research/promoted_config.json` — deployable config after fallback checks.
- `research/promotion_report.json` — sealed comparison and fallback reason.
- `research/sealed_best_check.json` — sealed check for the current dev winner; not deployed.
- `data/model_predictions_2025.csv` — dev predictions, including `rank_score` and
  `sel_score` when selector tilt is active.
- `data/model_predictions_2026.csv` — promoted sealed-season predictions.

## Commands

```bash
/usr/local/bin/python3.11 -m model.research
/usr/local/bin/python3.11 -m model.research --seal-2026 --seal-config promoted
```
