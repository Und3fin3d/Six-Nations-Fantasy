# Hill-climb report, round 2 (2 October 2026)

This round follows `HILLCLIMB_REPORT_2026-10-01.md` and was selected under `HILLCLIMB_PROTOCOL_2026-10-02.md`. The protocol was committed (eee7c1e) before any round-2 candidate was scored on an evaluation season.

## Outcome

The protocol's primary candidate, `p3_hillclimb_2026_10b` (**H2**), has lower MAE than robust P3 on all four benchmarks:

- Six Nations 2025 by −0.084; the 90% interval excludes zero.
- Six Nations 2026 by −0.033.
- NCR 2026 by only −0.0006. This is a tie, not an improvement.
- Friendly-25 count MAE by −0.0011, with metres MAE falling 2.2%.

**This is not the real improvement the brief asked for.** H2 does not raise selected-squad points: smoothed totals change by −26, −15 and −1, all inside the noise floor.

Squad points cannot separate these candidates at all; the evidence is in a section below. The supported claims are therefore narrower:

1. The settings for both the October 1 candidate and H2 are now chosen on development data. The October 1 candidate's blend weight (0.75) is reproduced by the development rule.
2. A new competition-independent raw-event step, **matchup calibration**, improves raw-event accuracy on development blocks, on held-out blocks and on Friendly-25 (metres especially). It also improves fantasy MAE in all three fantasy competitions when used without the blend (MK row).

H2 is robust P3, then three steps:

1. **Matchup calibration** (`model/unified/matchup.py`). Each event mean is multiplied by `exp(b_edge·edge/400 + b_opp·log(opp_conceded/mean))`.
   - The second term uses the opponent's recency-weighted, shrunken history of what it has conceded for that event in internationals before the lock.
   - Coefficients come from team-level Poisson fits on the 14 tournament blocks before 2025, multiplied by 0.75.
2. **Kicking concentration** with γ = 1.5, unchanged from round 1.
3. A **fantasy-points blend**: `0.7 × model + 0.3 × empirical fantasy baseline`. The weight was chosen on the new current-rubric Six Nations 2023–24 development slates.

## Scoreboard

All rows are scored under one contract: corrected NCR front-row scrums, the same pools and optimiser, and the same 40 jitter draws for smoothed points. Each cell is MAE / realised squad points / smoothed squad points.

| Candidate | Selection | NCR 2026 | Six Nations 2025 | Six Nations 2026 | Friendly-25 count / metres / minutes |
|---|---|---|---|---|---|
| p3_robust_native (reference) | – | 8.2513 / 1654 / 1651 | 7.2404 / 2356 / 2342 | 7.2385 / 2513 / 2510 | 0.65414 / 10.857 / 9.899 |
| **H2** = MK, w = 0.7 | protocol primary | **8.2507** / 1625 / 1626 | **7.1565** / 2319 / 2327 | **7.2051** / 2569 / 2509 | **0.65308** / **10.617** / 9.899 |
| MK = matchup + K15, no blend | protocol raw row | **8.2319** / 1706 / 1703 | **7.2373** / 2310 / 2306 | **7.2054** / 2483 / 2464 | **0.65308** / **10.617** / 9.899 |
| M = matchup only | protocol raw row | **8.2308** / 1706 / 1700 | **7.2350** / 2273 / 2319 | **7.2324** / 2477 / 2489 | **0.65384** / **10.617** / 9.899 |
| H1-dev = C2 + K15, w = 0.75 (1 Oct candidate) | weight now dev-selected | **8.2224** / 1611 / 1630 | **7.1936** / 2333 / 2351 | **7.2178** / 2533 / 2519 | **0.65322** / **10.826** / 9.899 |
| H2-K = K15, w = 0.75 | protocol alternative | 8.2647 / 1664 / 1628 | **7.1961** / 2365 / 2350 | **7.2087** / 2565 / 2531 | **0.65337** / 10.857 / 9.899 |
| K15 only | round 1 | 8.2531 / 1654 / 1651 | 7.2626 / 2313 / 2320 | **7.2119** / 2499 / 2476 | **0.65337** / 10.857 / 9.899 |
| empirical baseline | – | 8.4012 / 1578 / 1584 | 7.1674 / 2410 / 2419 | 7.3507 / 2552 / 2556 | 0.68068 (raw comparator) |

Bold marks a better MAE than the reference.

Three candidates have lower MAE than robust P3 in all four benchmarks: H2, MK and H1-dev. M also does, more weakly. No candidate raises both realised and smoothed squad points in all three competitions.

### MAE uncertainty

Paired 90% fixture-bootstrap intervals, resampling fixtures within rounds, for the MAE change against robust P3 (`hillclimb_2026-10-02/round2_mae_bootstrap.txt`):

| Candidate | NCR 2026 | Six Nations 2025 | Six Nations 2026 |
|---|---|---|---|
| H2 | −0.001 [−0.035, +0.033], 1/3 rounds better | **−0.084 [−0.118, −0.051]**, 5/5 | −0.033 [−0.073, +0.010], 4/5 |
| MK | −0.019 [−0.053, +0.012], 3/3 | −0.003 [−0.045, +0.037], 4/5 | −0.033 [−0.070, +0.003], 4/5 |
| M | −0.021 [−0.060, +0.018], 3/3 | −0.005 [−0.036, +0.024], 2/5 | −0.006 [−0.040, +0.032], 2/5 |
| H1-dev | −0.029 [−0.059, +0.001], 3/3 | **−0.047 [−0.075, −0.019]**, 5/5 | −0.021 [−0.061, +0.022], 3/5 |
| H2-K | +0.013 [−0.013, +0.041], 1/3 | **−0.044 [−0.064, −0.024]**, 5/5 | −0.030 [−0.065, +0.010], 4/5 |
| K15 | +0.002 [−0.007, +0.011], 2/3 | +0.022 [+0.003, +0.042], 1/5 | **−0.027 [−0.041, −0.012]**, 5/5 |

Friendly-25 count-MAE change, bootstrapping over games:

| Candidate | Count-MAE change | 90% interval | Games better | Metres |
|---|---:|---|---:|---|
| MK and H2 | −0.0011 | [−0.0032, +0.0011] | 15/25 | −0.240 MAE, 15/25 games better |
| H1-dev | −0.0009 | [−0.0032, +0.0015] | 16/25 | – |
| K15 alone | −0.0008 | [−0.0010, −0.0006] | 22/25 | – |

**Discrimination, not only level.** Mean within-round Pearson correlation with official points:

| Model | NCR 2026 | Six Nations 2025 | Six Nations 2026 |
|---|---:|---:|---:|
| Robust P3 | 0.5705 | 0.5944 | 0.6637 |
| H2 | 0.5749 | 0.5994 | 0.6627 |
| M | 0.5750 | 0.5955 | 0.6650 |

Spearman correlation rises for H2 in all three competitions (0.6032 / 0.6562 / 0.6970 against 0.5971 / 0.6500 / 0.6963). On the development slates, the blend with the empirical baseline improves MAE without improving Pearson correlation: 0.6244 to 0.6243. Its development gain is therefore a level effect, not better ranking.

## Why squad points cannot certify any of these candidates

MK's forecasts differ from robust P3's by about 3.4% per player (correlation 0.996–0.998). As a null comparison, robust P3's own forecasts were perturbed by a fixed random 3% multiplicative error, 12 independent "equally informed" copies, and each copy's smoothed squad points were computed:

| Competition | Robust P3 smoothed | Mean of perturbed copies | SD across copies |
|---|---:|---:|---:|
| NCR 2026 | 1651 | 1641 | 41 |
| Six Nations 2025 | 2342 | 2369 | 43 |
| Six Nations 2026 | 2510 | 2464 | **148** |

The SD is not a confidence interval; it is the spread among copies of the reference that are no better than it.

- **Six Nations 2026.** Robust P3 sits 46 points above the average of its own perturbed copies, so almost any change to it loses Six Nations 2026 points in expectation.
- **Six Nations 2025.** Robust P3 sits 27 points below the average of its copies.

Measured against this null, every round-2 smoothed difference is within about 1.6 SD (MK NCR +52, Six Nations 2025 −36, Six Nations 2026 −46; H2 −26, −15, −1).

Squad decisions turn on near-ties. For example:

- In Six Nations 2025 round 2, robust P3 picked Doris (forecast 30.0, scored 53) and MK picked Alldritt (29.9, scored 27).
- In Six Nations 2026 round 1, Jalibert (29.6 → 52) was swapped for Russell (26.8 → 14).

Thirteen rounds cannot measure a forecast improvement of a few hundredths of a point through squad totals. A claim that a candidate "beats the reference on squad points" would be luck either way.

## Development evidence used for selection

**New development slates.** `research/dev_six_nations_current_rubric.py` builds Six Nations 2023 rounds 1–5 and 2024 rounds 1–5 at their own round locks, with robust P3 and the empirical baseline refitted on strictly earlier history. Labels follow the current rubric:

- 2023 rounds 1–4 rescore the official per-player statistics. API values fill defenders beaten, offloads and conceded penalties, and scrums won is approximated. On 2025, this construction gives r = 0.99 against official points, and it is exact with complete columns.
- All other rows use `target_pts` (r = 0.97 against official points in 2025–26).
- This replaces the round-1 2023 slates, whose older rubric (mean 24.5 points) confounded MAE.

**Development-slate MAE** is the mean of the 2023 and 2024 round-mean MAEs (`hillclimb_2026-10-02/dev_seasons.csv`):

| Model | Development MAE |
|---|---:|
| Robust P3 | 6.513 |
| K15 | 6.494 |
| C2 + K15 | 6.505 |
| MK | 6.506 |
| M | 6.527 |
| C2 | 6.526 |
| MK, w = 0.6 / 0.7 | 6.460 / 6.464 |
| K15, w = 0.75 | 6.465 |
| C2 + K15, w = 0.75 | 6.472 |

Under the round-1 tie rule (within 0.005 counts as tied; ties go to the larger weight), the selected weights are 0.7 for MK, 0.75 for K15 and **0.75 for C2 + K15**. This independently confirms the weight round 1 had chosen after seeing the tests.

**Matchup calibration on development blocks** (leave-one-block-out, 14 blocks before 2025; observable-event points; `matchup_dev_block_metrics.csv`):

| Variant | 6N MAE | NCR MAE | Count MAE | Metres MAE |
|---|---:|---:|---:|---:|
| Robust P3 | 5.935 | 5.765 | 0.6350 | 11.53 |
| Edge only | 5.943 | 5.781 | 0.6341 | 11.46 |
| Matchup, shrink 0.75 | 5.917 | 5.752 | 0.6339 | 11.38 |

With the frozen all-development fit on the 8 later blocks, the matchup variant scores 6.372, 6.267, 0.6634 and 11.03, against 6.401, 6.281, 0.6646 and 11.24 for robust P3.

- The gain is not uniform: Pacific Nations Cup 2022 gets worse by +0.28.
- On the Six Nations development slates, M alone worsens fantasy MAE by +0.014. Only combined with kicking and the blend is MK better than K15.

**Coefficient stability.** The main opponent coefficients are stable when refitted on blocks before mid-2023, 2024, 2025 and 2026:

| Event | Opponent coefficient range |
|---|---|
| Tries | 0.46–0.69 |
| Metres | 0.40–0.67 |
| Tackles | 0.23–0.29 |
| Missed tackles | 0.39–0.49 |

Rucks won (−0.9 to −1.6) is the least stable; it likely reflects differences between data sources rather than teams.

## Ideas tried this round and rejected

All of these were judged on development data before any test use:

| Idea | Development result |
|---|---|
| Linear and GBDT stackers on robust P3, its two components and the empirical baseline (leave-one-block-out) | No gain: MAE 6.030–6.049 against 6.024; correlation unchanged |
| Opponent/team "for" profiles, home term, forward/back-specific coefficients, non-negative coefficients, other half-lives and priors | All within ±0.003 of the chosen matchup form, or worse |
| Per-event power (spread) calibration | Worse on all four metrics (6N +0.019, count +0.005 at full strength) |
| Schedule-strength adjustment of player history | Worse on all metrics |
| Team-by-jersey recent-minutes blend for minutes | Minutes MAE 9.67 to 9.69–9.74, no fantasy gain |
| Kicking: recent team kicking-share weighting | Worse. An oracle kicker would gain only −0.035 MAE, so the remaining kicker-identity headroom is small. γ = 2 is equal to γ = 1.5 |
| Raw-level blend with the non-robust empirical event model | Worse at every weight |
| Status-specific event multipliers | Full calibration raises bench forecasts (bench under-forecast: tries ×1.25, metres ×1.13–1.24) and worsens MAE. Calibrating starters only improves MAE but is an asymmetric, MAE-gaming choice; rejected on principle |
| Starter/bench-specific blend weights; experience-specific weights | Gains ≤ 0.003; not adopted |
| Lineup-strength signals (opponent's or own fielded-lineup strength relative to its tournament norm) | Even a leaky diagnostic correlates only 0.03–0.07 with team residuals |
| Elo-margin minutes adjustment (blowouts) | 1–2 minute effects that are inconsistent between development and later blocks |

**Information ceiling.** On development blocks, about 89% of player-level fantasy residual variance is individual: tries alone account for 31% of it. Removing the whole team-level component would cut variance by about 7% after Poisson noise. Every stacker using existing signals failed. The remaining gains need new information, such as bookmaker lines, confirmed kickers or bench-usage plans, rather than recombining the current signals.

## Selection history and caveats

- Choices made before the protocol was frozen:
  - the matchup form;
  - shrink 0.75;
  - γ;
  - the blend rule and tie rule;
  - the current-rubric development slates.
- Two exploratory runs also printed the eight later archived blocks: block-level, tournament-start forecasts, not the official slates. Neither changed a choice; every choice follows the development numbers shown.
- The round-1 caveat still applies: the official-slate blend grid for robust P3 was seen in round 1.
- Development 2024 labels use `target_pts`, whose latent official-only parts are estimated, not realised. Starters' levels differ from official seasons there, by up to +4 for locks and back-rows. Blend weights act partly as level corrections, so they may not transfer.
- H2 fails the protocol's "real improvement" definition:
  - NCR MAE is a tie;
  - smoothed squad points are lower in all three competitions, by 1–26 points;
  - realised points are lower in two competitions.
- After seeing the results, w = 0.8–0.85 would very likely balance NCR, but choosing it would be test selection. It was not done.

## Recommendation

- Treat **MK** as the best-supported model change, since it is raw-level and competition-independent:
  - fantasy MAE is better in 3 of 3 competitions;
  - Friendly-25 count and metres MAE improve;
  - development and held-out blocks agree;
  - no fantasy-level weight is involved.
- Treat the fantasy blend as optional:
  - it reliably helps Six Nations MAE;
  - it is level-driven and does not help NCR.
- Freeze H2, MK and H1-dev before NCR GW4–7 and compare them prospectively against robust P3.
- Squad-point claims need many more rounds or a lower-variance decision metric.
- The most promising remaining direction is new information, such as bookmaker match lines, named goal-kickers and published bench splits, not further recombination.

## Reproduction

```bash
# Development slates (Six Nations 2023-24, current rubric)
python -m research.dev_six_nations_current_rubric --base BASE --output RUNS/devcr
# Matchup coefficients and development-block check
python -m research.matchup_fit --base BASE --components components_raw.pkl --cutoffs stack_data.pkl --output OUT
# Candidates on development, official and Friendly-25 sets (from saved robust P3 forecasts)
python -m research.hillclimb2_eval --runs RUNS --specs SPECS.json --set dev|official|friendly --output OUT
# End-to-end refit including both hill-climb candidates
python -m model.unified.rolling_eval --native-categories --hillclimb --output OUT
```

## Verification

- **Fresh end-to-end refit.** `model.unified.rolling_eval --native-categories --hillclimb --round-job six_nations_2026_r3` was run into a new output directory. It rebuilt the research store from the cache, and the store is byte-identical to the one used here (same SHA-256). It refitted every model and reproduced the ledger exactly:
  - robust P3: 7.423671 MAE / 463 points;
  - `p3_hillclimb_2026_10`: 7.428887 / 483;
  - `p3_hillclimb_2026_10b` (H2): **7.398380 / 486**, identical to the post-processed ledger value.
- **Code paths agree.** The model-code path (`model.unified.hillclimb.adjust_raw_matchup`) and the evaluator transform give identical points on Six Nations 2026 R3 and NCR GW2.
- **Test suite.** The full suite, on the pinned stack, gives 156 passed and 1 failed. The failure is the known fixed-count audit assertion (2,144 labels expected, 2,164 present).
- **No external calls or side effects.** No RapidAPI requests were made, `data/cache` is unchanged, production routing is unchanged, and no model was promoted.
