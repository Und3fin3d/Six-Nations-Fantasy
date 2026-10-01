# Hill-climb selection protocol, round 2 (frozen before any round-2 test results)

Written 2 October 2026. Every choice below was made on development data only:

- the 14 archived international tournament blocks whose cutoff precedes 2025 (`components_raw.pkl`), scored on API-observable events;
- new Six Nations 2023–2024 development slates (`research/dev_six_nations_current_rubric.py`), all 10 rounds labelled under the **current** rubric. 2023 rounds 1–4 rescore the official per-player statistics (the same construction reproduces 2025 official points with r = 0.99); other rows use `target_pts`.

No round-2 candidate has been scored on Six Nations 2025, Six Nations 2026, NCR 2026 or Friendly-25. Late archived blocks (2025–26) were printed in two exploratory runs as block-level confirmation (archived tournament-start forecasts, not the official slates). They did not change any choice below; every choice follows the development numbers in `research/hillclimb_2026-10-02/`.

## New component: matchup calibration (`model/unified/matchup.py`)

Each event mean is multiplied by `exp(b_edge*edge/400 + b_opp*log(opp_conceded/mean))`. `opp_conceded` is the opponent's recency-weighted (half-life 365 days, 3-year window, 3-match shrinkage) conceded total for that event in internationals before the lock. Coefficients are team-level Poisson fits on the 14 development blocks, multiplied by 0.75.

Development evidence (leave-one-block-out, mean over blocks; `matchup_dev_block_metrics.csv`):

| Variant | 6N-observable MAE | NCR-observable MAE | count-stat MAE | metres MAE |
|---|---:|---:|---:|---:|
| robust P3 | 5.9352 | 5.7650 | 0.6350 | 11.531 |
| edge only (C2 form) | 5.9431 | 5.7805 | 0.6341 | 11.460 |
| matchup, shrink 1.0 | 5.9213 | 5.7590 | 0.6342 | 11.357 |
| **matchup, shrink 0.75** | **5.9169** | **5.7519** | **0.6339** | 11.377 |

Shrink 0.75 was chosen over {0.25, 0.5, 0.75, 1.0} because it is best or near-best on three of the four metrics. Also rejected on development:

- forward/back-specific coefficients (no gain);
- non-negative opponent coefficients (no gain);
- per-event power (spread) calibration: worse on every metric;
- schedule-strength adjustment of player history: worse on every metric.

Kicking concentration stays at γ = 1.5. On top of matchup calibration, the development blocks give γ = 1.5 and γ = 2.0 within 0.0007 of each other.

## Fantasy-level blend weight

The rule is unchanged from the 1 October protocol, but uses the new current-rubric development slates. `w` is chosen from {0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0} by the lowest mean of the 2023 and 2024 mean-round MAEs. Any `w` within 0.005 of the best counts as tied, and ties go to the largest `w`.

| Raw forecasts | Best w (MAE) | Tied set | **Selected w** | Development MAE at selected w | Smoothed dev points (2023+2024) |
|---|---|---|---:|---:|---:|
| matchup + K15 (MK) | 0.6 (6.4597) | 0.5, 0.6, 0.7 | **0.7** | 6.4639 | 4650 |
| K15 only | 0.6 (6.4634) | 0.5–0.75 | **0.75** | 6.4652 | 4587 |
| C2 + K15 (1 October candidate) | 0.6 (6.4667) | 0.5–0.75 | **0.75** | 6.4718 | 4640 |
| robust P3 (reference) | – | – | 1.0 | 6.5128 | 4557 |

The development rule independently selects w = 0.75 for the 1 October candidate (C2 + K15), the weight that was previously chosen after seeing the test seasons.

## Candidates fixed now

- **H2 (primary):** robust P3 → matchup calibration → K15 → `0.7*model + 0.3*empirical baseline`.
- **H2-K:** robust P3 → K15 → `0.75*model + 0.25*empirical`.
- **H1-dev:** the 1 October candidate (C2 + K15, w = 0.75), now with a development-selected weight.
- **Raw-only rows** for Friendly-25 and diagnostics: MK, M, K15.

## Evaluation

All candidates are reported on all four benchmarks, including failures:

- MAE, realised and smoothed squad points, and paired fixture bootstraps (90%) against robust P3;
- Friendly-25 count, metres and minutes MAE.

"Real improvement" keeps the 1 October definition: lower MAE in all three fantasy competitions and lower Friendly-25 count MAE, with no competition losing squad points. Smoothed points are reported because realised totals carry a 1% jitter SD of 22–37 points per season.

If H2 fails, later iterations will be logged as new, explicitly post-hoc rounds. They will not replace this selection.
