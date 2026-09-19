# Matched rolling-history evaluation — 2026-09-19

## Decision

**Do not open a PR or promote a model.** The history corrections are implemented and tested on the working branch. No tested candidate improves both MAE and team points in every evaluated tournament-season. Main, deployed incumbents, and the frozen P3 prospective shadow are unchanged.

## What changed

The new `model.unified.rolling_eval` runner refits before each round using the same historical match rows and candidate cohort for the empirical reference and unified challengers. Earlier tournament rounds enter the next training set. Current/future matches do not.

The audit also reproduced and corrected two independent leaks: the empirical player-profile function weighted but did not exclude future international/club matches, and team-margin form shifted player rows rather than whole fixtures. A teammate could therefore inherit the current match result. Regression tests cover both.

All engines use a shared conservative availability convention: exact kickoffs must be more than three hours before the lock; date-only tables exclude the cutoff day. Undated or unfinished season aggregates are excluded. This is not proof of original publication timestamps: historical source files are unversioned. NCR uses published round lock times; Six Nations uses first kickoff as a lock proxy.

Current cache-derived sources rebuilt 174,508 player-match rows across 3,841 fixtures. Training history, RugbyPass availability, player pool, and optimiser constraints are matched. Current-round outcomes are masked before feature generation. No rugby API requests or NCR GW4–7 labels were used.

## Completed results

MAE is the equal-weight mean of round-level player MAEs over the same candidate rows, including unused bench players. Team points are official realised outcomes for optimised squads, including captain and super-sub multipliers. Lower MAE and higher team points are better.

| Competition-season | Model | MAE | Team points | Rounds |
|---|---|---:|---:|---:|
| NCR 2026 | Empirical reference | 8.401189 | 1578 | 3 |
| NCR 2026 | P3, tournament-frozen control | 8.435569 | 1519 | 3 |
| NCR 2026 | P3, rolling history | 8.436286 | 1507 | 3 |
| NCR 2026 | Existing weighted P3, rolling | 8.356977 | 1664 | 3 |
| NCR 2026 | Robust empirical / weighted P3 challenger | 8.368760 | 1665 | 3 |
| Six Nations 2025* | Empirical reference | 7.139481 | 2410 | 5 |
| Six Nations 2025* | P3, tournament-frozen control | 7.184178 | 2283 | 5 |
| Six Nations 2025* | P3, rolling history | 7.287564 | 2230 | 5 |
| Six Nations 2025* | Existing weighted P3, rolling | 7.324201 | 2211 | 5 |
| Six Nations 2025* | Robust empirical / weighted P3 challenger | 7.243192 | 2267 | 5 |
| Six Nations 2026* | Empirical reference | 7.389271 | 2548 | 5 |
| Six Nations 2026* | P3, tournament-frozen control | 7.406651 | 2410 | 5 |
| Six Nations 2026* | P3, rolling history | 7.421377 | 2445 | 5 |
| Six Nations 2026* | Existing weighted P3, rolling | 7.409380 | 2509 | 5 |
| Six Nations 2026* | Robust empirical / weighted P3 challenger | 7.326482 | 2505 | 5 |

*Six Nations has no archived round prices in this evaluation. Its squads obey position, nation, captain and substitute constraints, but are price-free diagnostics, not evidence of budget-feasible fantasy-team superiority. NCR budgets were checked against archived prices; GW3 uses retrospectively corrected final lineups/prices. GW1 excludes New Zealand and France for every model. These are retrospective comparisons, not fully reproduced prospective deployments.*

The empirical reference is recomputed with corrected cutoffs and matched inputs. Do not mix its 1578 NCR points with the old saved 1595-point report. Likewise, the tournament-frozen control above is newly refit through the corrected pipeline, not the old serialized shadow model.

## History fix versus model improvement

History access is now correct under the stated availability convention, but freshness alone does not improve every metric. Rolling versus tournament-frozen P3 changes team points by -12 in NCR, -53 in Six Nations 2025, and +35 in Six Nations 2026. MAE is slightly or materially worse in all three. Correct evaluation is necessary; it is not itself a guarantee of better forecasts.

The new robust empirical component uses exposure-weighted positional rates and international-role playing time shrunk toward position/start-status averages with a fixed four-appearance prior. It retains the existing v4 component and previously selected global event weights. No competition identifier or new official outcome search sets these parameters.

The robust challenger improves on rolling P3 in aggregate MAE and team points in all three tournament-seasons, but does not meet the stronger empirical-baseline gate. Against that baseline it gains 87 NCR points, loses 143 Six Nations 2025 diagnostic points, and loses 43 Six Nations 2026 diagnostic points. Its MAE improves in NCR and Six Nations 2026, but regresses in Six Nations 2025.

Even in NCR the aggregate gain is not an every-round win: robust scores are 502, 638, 525 versus empirical 529, 493, 556. With only three rounds, the paired round-bootstrap 90% intervals include zero for both robust-minus-empirical mean MAE (-0.0942 to +0.0293) and mean team points (-29.67 to +87.67). This does not establish clear superiority.

Existing event weights were selected using 2022–2025 data, including Six Nations 2025. The historical 2026 outcomes were already known repository evidence. Neither is a new prospective validation set. No further tuning on these results was performed.

## Verification

- Original source failed three newly written leakage tests; corrected source passes them.
- All 13 round jobs passed the 57-test focused suite and completed model fitting/evaluation.
- The entire repository test suite passed: **117 passed in 43.95 seconds**, with the genuine frozen LFS artifact and required parser dependencies present.
- Independently recalculated all 65 reported MAEs and team totals from saved predictions, squads and official outcomes.
- Verified all 65 positional/nation/role-constrained squads and all 15 NCR budget/hemisphere checks.
- All 13 round manifests exclude ongoing/current matches and admit prior rounds. Same-tournament prior rows are 0, 138, 276, 414, 552 for each Six Nations season and 0, 276, 552 for NCR.
- All independent job source and data manifests match. The patch applies to the original source and reproduces all 13 changed source/test files.

Base: `64485d2e374178b51108a298683b9ba43ab9628b`.
Full-test commit: `796d55244ed30e3694cab4b16540b510e8dccb0e`.
Comparison run: `35468499645`; full-validation run: `35468831878`.
Store SHA-256: `e87ec4666fcef24de8c6bbf333ba1483e0ed268b965099cace8fe5dfc5b93145`.

Python 3.11.16; NumPy 1.24.3; pandas 2.1.0; SciPy 1.11.2; scikit-learn 1.3.1; LightGBM 4.6.0; torch 2.7.1+cpu. The pinned model preflight passed. Full source/input hashes, per-round predictions and squad evidence are retained in the comparison artifact; temporary execution workflows were removed after completion.

## Reproduction

Use Python 3.11 with `requirements-model.txt` and `requirements-unified.txt`. Parser tests additionally need pytest, requests and beautifulsoup4. Pull the LFS historical tables and the retained frozen P3 artifact before running the full test suite.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
python -m model.unified.rolling_eval --output data/unified/rolling_eval_reproduction
```

The default command fits all 13 rounds. `--competitions ncr` limits it to NCR. `--round-job ncr_2026_r2` fits just that round; season-first jobs also emit all tournament-frozen controls. Results are written to a separate output directory and never activate a model or create a PR.
