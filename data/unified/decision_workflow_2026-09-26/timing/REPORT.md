# Empirical-component timing ablation result

The timing-label repair is retained as a correctness change. Its fixed-tree empirical-component ablation does not advance. It changes none of the 13 selected squads, captains or super-subs relative to corrected robust. It therefore adds no squad points in any round. No production route or protected NCR GW4–7 experiment changes.

## Complete historical comparison

| Competition | Timing points | Robust points | Incumbent points | Equal blend points | Timing MAE | Robust MAE |
|---|---:|---:|---:|---:|---:|---:|
| Six Nations 2025 | 2356 | 2356 | 2260 | 2311 | 7.241811 | 7.242949 |
| Six Nations 2026 | 2496 | 2496 | 2539 | 2569 | 7.249493 | 7.243352 |
| NCR 2026 GW1–3 | 1654 | 1654 | 1578 | 1576 | 8.251969 | 8.251674 |

The timing correction changes 736 of 2,138 candidate forecasts above absolute tolerance 1e-12. All material changes affect bench players. The largest fantasy-point change is 0.887318. Starter forecasts differ by at most 7.11e-15. Fantasy MAE improves by 0.001138 in Six Nations 2025, worsens by 0.006141 in Six Nations 2026, and worsens by 0.000295 in NCR.

The candidate fails advancement against robust because its development gain is zero. Against the incumbent, its Six Nations development/audit deltas are +96/−43, and its pooled advantage fails the single-round-removal rule. Against the equal blend, they are +45/−73, also failing round removal. NCR deltas are +76 against incumbent and +78 against the blend; these existing robust advantages do not result from the timing correction. All three comparator admissions are false.

`summary.csv` retains points, XV, additional captain, super-sub, MAE, bias and RMSE. `metrics.csv`, `paired_rounds.csv`, `paired_seasons.csv`, `paired_competitions.csv` and `admission.csv` retain every comparison and its uncertainty. All selected outcomes are known. These are already-consulted historical results; Six Nations remains price-free diagnosis, and NCR retains its retrospective-input limitations.

## Method and verification

The protocol was registered before fitting. The accepted run fitted 39 empirical components and zero tree models. Original and corrected robust empirical models used the same prior-only cutoff for each slate. Legacy empirical fits verified the archived shared-tree decomposition. Only the empirical contribution changed; archived V4 contributions remained fixed. Metres and minutes used consistent mixture second moments. Count means were used for deterministic scoring without a count-tail calibration claim.

Recovery passed for all 67,450 player-event/minutes cells. Maximum absolute mean error was 2.91e-14; maximum second-moment error was 5.23e-12. All 39 comparator slate results reproduced their numeric metrics. Additional direct verification matched all 624 comparator squad rows for IDs, names, teams, positions, statuses, actuals and captain/sub flags. Cutoffs, denominators, unknown counts and named bonus roles also matched. All 208 candidate squad rows matched robust across all 13 slates.

Input, archive and runtime source hashes remained unchanged during the accepted run. See `manifest.json` and `verification.json`. The timing module and overlay builder passed complexity checks. The timing repair passed the 38 existing raw-benchmark and rolling-history tests. No tests were created or changed.

Independent supplemental review found no mathematical, cutoff, cohort, scoring or admission defect. It identified that the runner's numeric comparator check did not itself compare exact squad identities and unknown counts. The direct verification above closed that evidence gap for this run. `SUPPLEMENTAL_REVIEW.md` retains the review and resolution.

## Timing repair evidence and failed attempt

The `minutes` directory preserves the original repair-only report and four diagnostic tables. That report correctly states that no model had been fitted at the repair stage. This report records the subsequent registered ablation. The repair changes 1,224 unsupported zero-minute labels, including 211 international rows, to missing minutes with false availability. Raw event values and permanent cache remain unchanged. All 85 late-entry zeros and 48 separate unresolved pre-80-entry anomalies remain unchanged.

The first execution mistakenly selected an older archive whose training-store hash differed from the registered store. The recovery guard rejected its first slate before scoring. Its three empirical fits, source snapshot and failure record remain under `rejected_wrong_archive_attempt`. The shared archive loader now checks the registered store hash before reading predictions. The accepted retry used the correct archive and the unchanged candidate formula, parameters and recovery tolerance.

Original canonical hash: `2b70331c0cf215def67831cd27c987362e2ec8330b2afb41c8d204f48ef17313`.

Corrected canonical hash: `e4b20637e2b514d622bb35cc7c43762e0eddb4f3b4e50f1ca2e39155ae96e6be`.

Protocol hash: `3021e3876b0c11d6113a6c0ea9d6b1fb6a67bbf124a8600ed280135972f2e49e`.

Timing module hash: `57493b4f486d151b36510e59eeabe4a55cd350ffce20bb0b6a787e1fc1472559`.

## Reproduction

Run from the repository root with a new output directory:

```sh
/tmp/6n-pr-decision/venv/bin/python -m research.timing_ablation --old-store /tmp/6n-complete-labels/all-cache-final-prepared/inputs/player_match.csv --new-store /tmp/6n-decision-workflow/bench/timing_v1/player_match.csv --evidence /tmp/6n-methodology-review/replay --comparison /tmp/6n-decision-workflow/remedies/complete --output /tmp/6n-decision-workflow/bench/empirical_timing_reproduced
```

The runner refuses to overwrite an output directory. A full refit on corrected timing data remains untested. This result closes the registered empirical-component branch without a promotion claim.
