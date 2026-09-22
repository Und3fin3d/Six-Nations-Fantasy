# Native-category rolling evaluation — 2026-09-19

## Decision

Do not open a PR or promote a model. `p3_robust_native` is a stronger research candidate than the previous robust blend: it improves MAE and team points in all three tournament-seasons relative to that candidate. It now exceeds the empirical baseline in total squad points in all three, but its Six Nations 2025 MAE is worse. The two Six Nations point leads are only 6 and 2 points. This is not clear universal superiority.

Main, deployed incumbents and the frozen prospective P3 shadow are unchanged. Native categorical splits remain opt-in and disabled by default.

## What was tested

The shared tree encoder produces integer codes for player identity, position, team, opponent and club/international level. These were concatenated into a NumPy matrix and passed to LightGBM without categorical-feature indices. They were therefore treated as ordered numerical inputs. The experiment explicitly marks those columns categorical for minutes, count events, binary events and optional v4 hurdle heads. No competition identity or fantasy target was added.

All tree hyperparameters, histories, scoring adapters, empirical components and existing event weights were held fixed. The plan was committed before result generation (`aea1d904433b6832bbcdc99e9401e10de6f0c319`). Three shared rolling blends were evaluated with the new representation: 50/50, event-weighted and robust-empirical event-weighted. The tournament-frozen variant is a control, not eligible to replace rolling-history deployment. No parameter search followed this run.

## Complete scoreboard

Lower MAE and higher points are better. MAE is the equal-weight mean of per-round player MAEs. Points include the same optimiser constraints, captain and super-sub rules as the previous matched-history experiment.

| Competition | Season | Engine | MAE | Team points | Rounds |
|---|---:|---|---:|---:|---:|
| ncr | 2026 | empirical_baseline | 8.401189 | 1578 | 3 |
| ncr | 2026 | p3_robust_native | 8.364626 | 1740 | 3 |
| ncr | 2026 | p3_rolling_native | 8.416834 | 1776 | 3 |
| ncr | 2026 | p3_tournament_frozen_native | 8.408939 | 1748 | 3 |
| ncr | 2026 | p3_weighted_native | 8.359950 | 1712 | 3 |
| six_nations | 2025 | empirical_baseline | 7.139481 | 2410 | 5 |
| six_nations | 2025 | p3_robust_native | 7.231897 | 2416 | 5 |
| six_nations | 2025 | p3_rolling_native | 7.271267 | 2347 | 5 |
| six_nations | 2025 | p3_tournament_frozen_native | 7.158488 | 2295 | 5 |
| six_nations | 2025 | p3_weighted_native | 7.322023 | 2260 | 5 |
| six_nations | 2026 | empirical_baseline | 7.389271 | 2548 | 5 |
| six_nations | 2026 | p3_robust_native | 7.254336 | 2550 | 5 |
| six_nations | 2026 | p3_rolling_native | 7.333754 | 2462 | 5 |
| six_nations | 2026 | p3_tournament_frozen_native | 7.299764 | 2409 | 5 |
| six_nations | 2026 | p3_weighted_native | 7.339431 | 2531 | 5 |

## Robust candidate versus the previous robust candidate

| Competition-season | MAE change | Team-points change |
|---|---:|---:|
| ncr 2026 | -0.004133 | +75 |
| six_nations 2025 | -0.011295 | +149 |
| six_nations 2026 | -0.072146 | +45 |

## Robust candidate versus empirical

| Competition-season | MAE difference | Team-points difference | Both metrics improve? |
|---|---:|---:|---|
| ncr 2026 | -0.036563 | +162 | True |
| six_nations 2025 | +0.092415 | +6 | False |
| six_nations 2026 | -0.134935 | +2 | True |

## Uncertainty and individual rounds

The following paired 90% bootstrap intervals resample complete rounds, with seed 17 and 10,000 draws. They describe these existing retrospective samples, not independent prospective evidence; there are only three NCR rounds and five rounds per Six Nations season. They do not account for all prior research choices or dependence across seasons.

| Competition-season | Metric (candidate minus empirical) | Mean per-round difference | 5th percentile | 95th percentile |
|---|---|---:|---:|---:|
| ncr 2026 | mae | -0.036563 | -0.084724 | +0.011598 |
| ncr 2026 | team_points | +54.000000 | -20.666667 | +128.666667 |
| six_nations 2025 | mae | +0.092415 | +0.015017 | +0.177101 |
| six_nations 2025 | team_points | +1.200000 | -31.000000 | +30.000000 |
| six_nations 2026 | mae | -0.134935 | -0.285492 | +0.012378 |
| six_nations 2026 | team_points | +0.400000 | -32.200000 | +32.200000 |

NCR robust squad scores are 513, 694 and 533 versus empirical 529, 493 and 556. The candidate loses two rounds and gains 201 points in round 2. Do not describe the 162-point aggregate gain as an every-round win. All three squad-points intervals include zero. Six Nations 2025 MAE remains a failed slice, not an acceptance rule to relax.

## Decision accounting

These are accounting decompositions, not controlled causal interventions. The normal XV term includes the captain at 1x; captain bonus is the extra 1x; super-sub is its complete realised contribution.

| Competition-season | Engine | Ordinary XV points | Captain bonus | Super-sub points |
|---|---|---:|---:|---:|
| ncr 2026 | empirical_baseline | 1354 | 116 | 108 |
| ncr 2026 | p3_robust_native | 1342 | 137 | 261 |
| six_nations 2025 | empirical_baseline | 1987 | 171 | 252 |
| six_nations 2025 | p3_robust_native | 1969 | 171 | 276 |
| six_nations 2026 | empirical_baseline | 2167 | 225 | 156 |
| six_nations 2026 | p3_robust_native | 2211 | 177 | 162 |

In Six Nations 2026, ordinary XV selection now gains 44 points over empirical, but captain bonuses lose 48 points. In Six Nations 2025, ordinary XV still loses 18 points and the aggregate win comes from substitutes. The earlier robust candidate lost 83 ordinary-XV points and 57 substitute points in 2025. These distinctions identify the next diagnostic targets without changing objectives.

## Verification

- 29 focused tests passed locally and in each of the 13 round jobs.
- All 121 repository tests passed in 47.35 seconds, including tests using the genuine frozen LFS P3 artifact.
- Four new tests verify category indices on minutes/count/binary heads, both hurdle heads, disabled-default equivalence, and unseen-category/artifact-round-trip behavior.
- All 65 MAEs and squad totals were independently recalculated from saved predictions, squads and official outcomes.
- All 65 squads satisfy roster size, positions, nation limits, captain and substitute constraints; all 15 NCR squads also satisfy budget and hemisphere limits.
- All 39 raw-prediction files independently rescore to the saved expected points within 1e-10, with the same ordered fixture/player/team keys as the previous cohorts.
- All baseline predictions, candidate IDs and official outcomes match the previous run exactly.
- All 13 job source manifests match the tested local source. All input/cache/store hashes and round-history manifests match the previous experiment. Only irrelevant workspace paths are ignored in source-data audit comparison.
- The combined history-plus-native patch applies cleanly to the original main source and reproduces all 16 changed source/test files.
- The rebuilt store is unchanged: 174,508 player-match rows and 3,841 fixtures. Store SHA-256 is `e87ec4666fcef24de8c6bbf333ba1483e0ed268b965099cace8fe5dfc5b93145`.

## Limits

Six Nations remains a price-free diagnostic: archived round prices are not available in this evaluation. The season player CSVs inspected contain performance statistics, not price archives; their AP column was not treated as a price. NCR GW1–2 use archived prices/status; GW3 uses the previously disclosed retrospectively corrected final lineup/prices. NCR GW1 excludes New Zealand and France for all engines.

Historical source files are unversioned. The common availability convention remains kickoff plus more than three hours, and date-only histories exclude the cutoff day. Earlier rounds enter later histories. This is not proof of original source-publication times.

Existing event weights used 2022–2025, and the historical 2026 results were already known. Nothing here is a fresh prospective validation set. No NCR GW4–7 labels or rugby API calls were used. No fitted model binaries were committed.

## Reproduce

Use the pinned Python 3.11 model environment and required LFS inputs. The existing runner now exposes the opt-in research flag:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
python -m model.unified.rolling_eval --native-categories \
  --output data/unified/native_categories_reproduction
```

`--round-job six_nations_2025_r3` limits fitting to one round. Use a new output directory; source/config/input manifests prevent incompatible cache reuse. The default command retains the original numerical-category behavior unless the flag is supplied.

Tested source commit: `2f7deeda95cb5c2f8e2924271b964d4ce49238dc`.
Comparison and full-test workflow run: `35470275137`.
Numerical environment: Python 3.11.16, NumPy 1.24.3, pandas 2.1.0, SciPy 1.11.2, scikit-learn 1.3.1, LightGBM 4.6.0 and torch 2.7.1+cpu.

The next research bottleneck is the Six Nations 2025 prediction error while preserving the new squad-selection gains. Do not open a PR from this result.
