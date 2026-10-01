# Positional biases and super-sub decisions (October 2026)

Protocol: `POSITION_SUPERSUB_PROTOCOL_2026-10.md`. Candidate S was frozen in 453ab7a and candidate SK in af9f362, each before its test results were read. Evidence is in `position_supersub_2026-10/`. Unless stated otherwise, bias means predicted − actual, and the reference model is robust P3.

## Verdict

1. **Most of the large official-label positional biases are not modelling errors in the raw-event model.**
   - They come from official-only statistics: breakdown steals, scrums won, official metres and player of the match. The official-stats agent owns these.
   - Some are season-specific: Six Nations 2025–26 replacement back-rows played 2.4 more minutes than forecast, a shift no development set shows.
   - The rest is noise.
2. **One systematic raw-event bias is fixable, and is fixed: replacement production per minute.**
   - Within the same player, per-minute rates off the bench are 1.1–1.5× his starting rates.
   - This holds for tries, metres, defenders beaten, carries and tackles, in club and international rugby, and in both eras.
   - The robust empirical component ignores it. Its bench forecasts were 7–20% low on events in every development set; the tree component was not.
   - `StatusAwareEmpiricalEventModel` (S) removes the gap without any tuned parameter.
3. **A second systematic bias is partly fixed: starting hookers' tries are too high.**
   - The hooker try forecast was forecast/actual 1.13–1.35 in every block era.
   - Try forecasts are over-dispersed within position, with calibration slope 0.5–0.65, because player profiles are shrunk with a 220-minute prior for every event. Within-position empirical Bayes puts the right prior strength for tries at about 1,200 minutes.
   - SK (S plus event-specific within-position shrinkage) roughly halves the excess and improves Poisson deviance and MSE in 13 of 14 events on both block eras.
4. **The super-sub weakness is noise-limited.**
   - The hindsight gap is 876 against 345 tripled points over Six Nations 2025–26, but perfect knowledge of every replacement's true mean would add only about +24 in expectation (90% range +0 to +60).
   - At least ~85% of the hindsight gap is unforecastable match noise.
   - The current pick is usually a near-tie: in 7 of 10 rounds the top two replacements are within 1 point.
   - No better-EV decision rule exists for a linear score. "Pick the top back-row" loses on observable points.
   - SK raised smoothed super-sub points in all three test competitions (+39, +15, −1) and on 2025–26 block weekends (+58 per 10 weekends, 90% interval [19, 108]). But it lost on the 2023–24 development slates (−12, −7). This is suggestive, not established.
5. **Fantasy MAE stays neutral** (within ±0.03 everywhere), as expected: raising skewed bench means cannot help a median metric.
   - SK improves Six Nations 2025 MAE by −0.030 (90% interval [−0.056, −0.004]).
   - Discrimination improves slightly: Pearson +0.001 to +0.004, Spearman +0.001 to +0.003, bench Pearson +0.002 to +0.012 on all three test competitions.

Recommendation: adopt SK as an opt-in research candidate (`rolling_eval --status-rates` scores `p3_status_rates_native` and `p3_status_shrunk_native`). Treat squad and super-sub gains as unproven until prospective rounds confirm them. Production routing is unchanged.

## 1. Diagnosis

### 1.1 Which bias belongs to whom (official labels, Six Nations 2025–26, robust P3)

Component decomposition from the official per-player statistics (`official_component_bias.csv`):

| Group (n) | Total | Tries | Official metres¹ | Tackles | DB | BS² | Scrums won³ | POTM | Cards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Bench back-row (93) | −2.95 | −0.58 | −0.58 | −0.77 | −0.30 | −0.54 | −0.39 | +0.10 | +0.21 |
| Bench hooker (59) | −1.17 | +0.71 | −0.32 | −1.15 | −0.03 | −0.10 | −0.29 | +0.04 | +0.07 |
| Start centre (111) | −2.92 | −0.38 | −1.54 | −0.48 | −0.61 | +0.10 | +0.05 | +0.02 | +0.03 |
| Start fly-half (60) | −2.81 | −0.32 | −1.05 | −0.33 | −0.35 | +0.51 | +0.04 | −0.18 | +0.17 |
| Start hooker (59) | +1.66 | **+1.73** | −0.22 | +0.39 | +0.10 | −0.23 | −0.18 | −0.16 | +0.16 |
| Start scrum-half (64) | +1.97 | +0.56 | +0.24 | +0.35 | +0.15 | +0.35 | +0.02 | −0.18 | +0.04 |

Notes:

1. Official metres run at about 1.42× API metres. Their definition shifts between seasons, so they are not positional modelling error.
2. BS is the breakdown-steal proxy (API tackle-turnover, r ≈ 0.1 with official BS). The official-stats agent's fix moves bench back-rows from 0.32 to 0.62 points, against 0.86 actual.
3. Official scrums won are credited to all forwards. This is official-only.

Bench back-row, −2.95 in total:

| Source | Points | Owner |
|---|---:|---|
| BS proxy | ≈ −0.54 | official-stats agent, which recovers ~0.3 |
| Scrums won | ≈ −0.39 | official-only |
| Metres definition | ≈ −0.38 | official-only |
| Cards and POTM | ≈ +0.3 | – |
| API-observable events | ≈ −1.2 to −1.45 | this work |

The API-observable part splits into two:

- **Minutes, about −0.75.** Forecast 22.9 minutes, actual 25.3. This is not systematic: development 22.3 vs 22.8, pre-2025 blocks 21.7 vs 22.3, 2025–26 blocks 23.2 vs 23.4.
- **Per-minute rate, about −0.5.** Mainly tackles (−0.58) and tries (−0.54), against +0.25 from cards.

S removes about +0.4 of the API part, leaving −0.84.

### 1.2 Stability: API-observable bias across sets (`api_component_bias.csv`)

| Group | Official 6N 25–26 | Dev 6N 23–24 | Pre-2025 blocks | 2025–26 blocks | Systematic? |
|---|---:|---:|---:|---:|---|
| Bench back-row | −1.22 | −0.36 | −0.54 | −0.98 | yes (small) |
| Bench back-three | −2.12 | −1.24 | −1.86 | −1.67 | **yes** |
| Bench centre | −0.66 | −0.10 | −1.20 | −0.38 | mostly |
| Bench scrum-half | +0.39 | −0.18 | −0.90 | +0.07 | no |
| Start hooker | +2.78 | +0.97 | +1.44 | +0.59 | **yes** (tries) |
| Start scrum-half | +2.28 | +0.57 | +1.15 | −0.28 | weak |
| Start centre | −1.22 | −0.44 | +0.20 | +0.25 | no |
| Start fly-half | −1.23 | −1.72 | −0.76 | +0.18 | weak |
| Start back-three | −1.10 | +0.01 | +0.23 | +0.50 | no |

Starting backs' large official-label deficits (−1.7 to −2.9) are official metres plus noise; they are not systematic on API events. The systematic raw-event patterns are:

- bench under-forecast, by component: tries, metres, defenders beaten and tackles for forwards;
- starting front-row tries over-forecast.

### 1.3 Where the bench bias comes from

- **Component split.** Recovered component means on 22 archived blocks (`scripts/d4.py`) show that the robust empirical component under-forecasts bench events at forecast/actual 0.5–0.93 for tries, metres, defenders beaten and assists, across all positions. The v4 trees are near 1.0 because they condition on starting status. Blend weights put 90% on the empirical component for tries and metres, so robust P3 inherits the bias.
- **Within-player bench/starter rate ratios** (`scripts/d7.py`) are stable:
  - club pre-2025: mostly 1.0–1.5 (range 0.88–1.55);
  - club 2025+: mostly 1.0–1.3 (range 0.79–1.33);
  - international pre-2025: 0.5–1.85, noisy, mostly 1.1–1.5.
  - At the Six Nations 2024 R1 lock (`status_factors_dev_2024r1.csv`), the all-position bench factors are 1.21 for tries, 1.20 metres, 1.24 defenders beaten, 1.15 tackles and 1.18 carries.
- **Minutes** (`scripts/d6.py`). Mean bench minutes are right by jersey (16–23) in every set. Discrimination is weak: bench minutes MAE about 9.2, correlation 0.21–0.34. Bench-split effects (5-3, 6-2, 7-1) do not replicate between sets. There is no minutes fix with development support, which agrees with round 2 (jersey blend minutes MAE 9.67 → 9.69–9.74).
- **Team strength** (`bias_by_strength.csv`). On pre-2025 blocks, favourites' replacements are under-forecast (backs −1.83, forwards −0.97 observable points) and underdogs' are not. On 2025–26 blocks the pattern is not stable. This is left to matchup calibration (MK); no extra term was added.
- **Hooker tries** (`scripts/d10.py`, `scripts/d11.py`):
  - The starting hooker excess is in international-heavy players (0.26 forecast against 0.15 actual tries per start before 2025), not club-heavy ones, so club calibration is not the cause.
  - Within-position calibration slopes for tries are 0.50–0.65 for forwards, and for every event most slopes are below 1.
  - The empirical profile uses K = 220 minutes for all events. Within-position gamma-Poisson moment matching gives K ≈ 1,200 for tries, 1,500 for try assists, 1,800 for the turnover proxy and 1,500 for conceded penalties, but about 200 for tackles and about 130 for carries (`shrinkage_k_dev_2024r1.csv`).

## 2. Fixes and validation

**S** (`model/unified/raw_benchmark/status_rates.py`):

1. At each lock, fit the within-player multiplicative model `rate_player × factor[position, started]` by alternating ratio updates. It uses recency-weighted pre-lock club and international rows.
2. Shrink bench factors towards the all-position ratio with 20 pseudo-events.
3. Fit the unchanged robust model on starter-equivalent counts.
4. Multiply replacement forecasts by the bench factor.

Kicking, cards and official-only events are untouched. There is no tuned or competition-specific parameter.

**SK** adds `within_position_shrinkage_k`, an event-specific prior strength for player profiles:

- K bounded to [40, 4000] minutes, the existing v4 bounds;
- the weighted effective-sample correction;
- metres keep 220.

Both reuse the saved tree component unchanged. The model-code path `EventWeightedBlend(StatusAware…, v4)` reproduces the research forecasts exactly (maximum difference 0.0 on Six Nations 2026 R5).

### 2.1 Development data (used for the freeze)

Pre-2025 blocks, observable truth (`blocks_pre2025_with_shrunk.txt`):

| Metric | Robust P3 | S | SK |
|---|---:|---:|---:|
| Bench event ratio: tries / metres / defenders beaten / tackles | 0.80 / 0.81 / 0.87 / 0.93 | 0.92 / 0.93 / 0.97 / 0.95 | 0.90 / 0.93 / 0.97 / 0.95 |
| Events with lower deviance / MSE (of 14) | – | 13 / 12 | 13 / 13 |
| Count-stat MAE | 0.63949 | 0.64018 | 0.63982 |
| Observable points bias, bench / start | −0.53 / +0.26 | −0.15 / +0.09 | −0.17 / +0.00 |
| Observable points MSE, bench / start | 33.83 / 92.44 | 33.52 / 92.32 | 33.50 / 92.06 |
| Observable points MAE, bench / start | 3.977 / 7.152 | 4.043 / 7.116 | 4.035 / 7.089 |
| Within-weekend Pearson (all / bench / start) | 0.6036 / 0.219 / 0.469 | 0.6035 / 0.225 / 0.470 | 0.6058 / 0.226 / 0.473 |
| Bias: bench back-three / bench centre / start hooker | −1.86 / −1.20 / +1.44 | −1.24 / −0.45 / +1.07 | −1.19 / −0.48 / +1.03 |

Six Nations 2023–24 current-rubric slates (`dev_seasons.csv`):

| Metric | Robust P3 | S | SK |
|---|---|---|---|
| MAE, 2023 / 2024 | 7.182 / 5.844 | 7.163 / 5.847 | 7.149 / 5.812 |
| MSE, 2023 / 2024 | 104.10 / 65.54 | 103.92 / 65.54 | 104.20 / 65.50 |
| Bench Pearson, 2023 / 2024 | 0.185 / 0.242 | 0.185 / 0.242 | 0.168 / 0.235 |
| Smoothed squad points, 2023 / 2024 | 2357 / 2201 | 2327 / 2205 | 2310 / 2181 |
| Smoothed super-sub points, 2023 / 2024 | 164.6 / 103.4 | 159.2 / 102.7 | 152.2 / 96.4 |

Excluding the suspect 2023 R1 rows (coordinator warning: official rows shifted by one player) leaves the conclusions unchanged (`dev_bias_excl_2023r1.txt`):

| Engine | Bench bias | Bench MSE |
|---|---:|---:|
| Robust P3 | −1.51 | 36.15 |
| S | −1.16 | 35.30 |
| SK | −1.14 | 35.34 |

Starters move down by about 0.15 and their MSE rises slightly (101.4 → 102.0). The development labels under-forecast starters by 2.5 points from official-only components, which S is not designed to fix.

### 2.2 Test data (13 official slates, Friendly-25, 2025–26 blocks)

Each cell is MAE / MSE / Pearson / bench Pearson / realised squad points / smoothed squad points / smoothed super-sub points (`test_official_seasons.csv`):

| Engine | NCR 2026 | Six Nations 2025 | Six Nations 2026 |
|---|---|---|---|
| Robust P3 | 8.251 / 147.39 / .571 / .339 / 1654 / 1651 / 110 | 7.240 / 111.01 / .594 / .347 / 2356 / 2342 / 227 | 7.238 / 101.61 / .664 / .335 / 2513 / 2510 / 138 |
| S | 8.256 / 147.45 / .573 / .342 / 1654 / 1654 / 112 | 7.231 / 110.64 / .595 / .349 / 2307 / 2378 / 258 | 7.255 / 101.80 / .664 / .336 / 2538 / 2516 / 154 |
| **SK** | 8.263 / 147.75 / .574 / .351 / 1682 / 1662 / 109 | **7.210** / 110.39 / .598 / .354 / 2554 / 2455 / 266 | 7.243 / 101.95 / .665 / .337 / 2507 / 2514 / 153 |
| MK | 8.232 / 148.30 / .574 / .341 / 1706 / 1703 / 110 | 7.237 / 111.75 / .593 / .343 / 2310 / 2306 / 214 | 7.205 / 101.80 / .666 / .333 / 2483 / 2464 / 152 |
| SK + MK | 8.241 / 148.66 / .577 / .354 / 1711 / 1737 / 133 | 7.208 / 111.33 / .596 / .351 / 2329 / 2334 / 214 | 7.211 / 102.31 / .667 / .336 / 2437 / 2448 / 157 |

Paired 90% fixture-bootstrap intervals for the change against the reference (`test_bootstrap.csv`):

| Comparison | Competition | MAE change | MSE change |
|---|---|---|---|
| S vs robust P3 | Six Nations 2025 | −0.009 [−0.025, +0.009] | −0.37 [−0.79, −0.02] |
| S vs robust P3 | Six Nations 2026 | +0.016 [−0.005, +0.037] | +0.19 [−0.16, +0.53] |
| S vs robust P3 | NCR 2026 | +0.004 [−0.005, +0.015] | – |
| SK vs robust P3 | Six Nations 2025 | **−0.030 [−0.056, −0.004]** | −0.61 [−1.09, −0.16] |
| SK vs robust P3 | Six Nations 2026 | +0.004 [−0.021, +0.028] | +0.34 [−0.42, +1.14] |
| SK vs robust P3 | NCR 2026 | +0.011 [−0.004, +0.027] | +0.36 [−0.18, +0.95] |
| SK + MK vs MK | Six Nations 2025 | −0.029 [−0.054, −0.003] | – |
| SK + MK vs MK | Six Nations 2026 | +0.005 | – |
| SK + MK vs MK | NCR 2026 | +0.009 | – |

API-observable bias on the Six Nations 2025–26 official slates (`test_bias_api_observable.csv`):

| Group | Robust P3 | S | SK |
|---|---:|---:|---:|
| Bench, all | −0.36 | +0.01 | −0.01 |
| Bench back-row | −1.22 | −0.84 | −0.84 |
| Bench back-three | −2.12 | −1.64 | −1.63 |
| Bench fly-half | −1.37 | −0.98 | −1.00 |
| Bench centre | −0.66 | −0.13 | −0.20 |
| Bench front row, locks, scrum-half | −0.1 to +0.7 | +0.2 to +1.1 | +0.2 to +1.1 |
| Start hooker | +2.78 | +2.43 | +2.50 |
| Start scrum-half | +2.28 | +2.12 | +1.99 |

Mean squared error on API-observable points:

| Group | Robust P3 | S | SK |
|---|---:|---:|---:|
| Bench | 34.03 | 33.86 | 33.87 |
| Starters | 96.75 | 96.52 | 96.45 |

The front-row, lock and scrum-half replacements move from roughly unbiased to slightly over-forecast; the change is level, not a ranking change.

**2025–26 blocks** (API-observable; same matches as the official Six Nations slates plus other tests; `blocks_2025_26_test.txt`):

- SK has lower deviance in 13 of 14 events and lower MSE in 13 of 14.
- Bench bias −0.45 → −0.10; bench MSE 38.69 → 38.33; starters' MSE 98.74 → 98.41.
- Pearson 0.5881 → 0.5888, with bench 0.274 → 0.276.

**Friendly-25** (raw statistics; `friendly25_metrics.txt`) is the counter-example:

| Metric | Robust P3 | S | SK |
|---|---:|---:|---:|
| Count-stat MAE | 0.65414 | 0.65572 | 0.65480 |
| Bench observable bias | +0.37 | +0.74 | +0.72 |
| Starters' MSE | 88.30 | 88.25 | 87.80 |
| Events with lower deviance (of 14) | – | 4 | 8 |

These friendlies' replacements were already over-forecast by the reference, so raising them does not help there.

### 2.3 Verdicts by the frozen rules

- **Bench back-row, back-three, centre and fly-half.** API bias shrinks on the official slates, all 2025–26 blocks and all development sets, and event MSE or deviance improves except on Friendly-25. **Fixed (level).**
  - The remaining official-label bench back-row gap (−2.46 under S) is official-only statistics (BS, scrums won, metres definition: about −1.3) plus season-specific minutes (about −0.6).
- **Starting hooker / scrum-half over-forecast.** Partly fixed by SK on blocks (+1.44 → +1.03). On the official slates the change is small (+2.78 → +2.50; scrum-half +2.28 → +1.99). This is mostly a level effect of two small samples (60 rows each).
- **Fantasy MAE.** Neutral (all |change| ≤ 0.03), except SK on Six Nations 2025 (−0.030, interval excludes 0).
- **Squad and super-sub points.** Noise by rule 3:
  - SK's +113 smoothed squad points in Six Nations 2025 comes from captain +50, super-sub +39 and other picks +24. It exceeds the 1% jitter SD (37) for that season only.
  - Six Nations 2026 and NCR show +4 and +10.
  - The 2023–24 development slates show −47 and −20.

## 3. Super-subs

### 3.1 Picks before and after (Six Nations 2025–26; realised tripled points, smoothed over 40 jitters in brackets)

| Round | Hindsight best | Robust P3 | S | SK | MK |
|---|---|---|---|---|---|
| 2025 R1 | Reffell 29 | Sheehan (9.9) 75 [75] | Sheehan 75 [75] | Sheehan 75 [75] | Sheehan 75 |
| 2025 R2 | Conan 30 | Sheehan (10.9) 30 [30] | Sheehan 30 [30] | Sheehan 30 [30] | Lucchesi 21 |
| 2025 R3 | Curry 19 | Ashman (9.6) 3 [3] | Ashman 3 [3] | Ashman 3 [3] | Ashman 3 |
| 2025 R4 | T. Williams 37 | Lake (9.7) 48 [52.5] | Lake 48 [84] | **M. Smith (10.6) 108 [91.5]** | Lake 48 |
| 2025 R5 | Cunningham-South 50 | Willis (11.6) 66 [66] | Willis 66 [66] | Willis 66 [66] | Willis 66 |
| 2026 R1 | Timoney 33 | Pollock (14.2) 9 [9] | Pollock 9 [9] | Pollock 9 [9] | Pollock 9 |
| 2026 R2 | Pollock 22 | Pollock (12.9) 66 [66] | Pollock 66 [66] | Pollock 66 [66] | Pollock 66 |
| 2026 R3 | Botham 27 | Griffin (9.2) 30 [37.6] | **M. Smith 54 [53.4]** | **M. Smith 54 [53.4]** | Turner 51 |
| 2026 R4 | T. Jordan 19 | Pollock (12.6) 0 [0] | Pollock 0 [0] | Pollock 0 [0] | Pollock 0 |
| 2026 R5 | D. Murray 26 | Pollock (10.1) 24 [25.5] | Pollock 24 [25.8] | Pollock 24 [24.6] | Pollock 24 |
| **Total (smoothed)** | 876 | 351 [365] | 375 [412] | 435 [418] | 363 |

The hindsight column shows untripled points; its total is tripled. The full table is in `test_supersub_picks.csv`.

The forecasts change the pick in only 2 of 10 rounds, both near-ties. S changes the pick in only 13–18% of block weekends. The hindsight-best replacement was a back-row type in 7 of 10 rounds, but always picking the top-forecast back-row loses −7 and −28 per 10 weekends on observable points (pre-2025 and 2025–26 blocks).

### 3.2 Expected versus realised, and near-ties

- Replacement forecasts sit in a narrow band (within-round SD 1.3–1.7) against a realised SD of 6.3–6.9.
- The forecast's within-round correlation with replacement points is 0.29 for the Six Nations 2025–26 official slates and 0.21 on development data.
- The calibration slope of replacement points on the forecast is 0.9–1.3, so the forecasts are not over-confident at the bench level.
- The top two forecasts are within 1 point in 7 of 10 official rounds, and within 0.5 in the median block weekend.
- Swapping one near-tie moves a season by 30–60 tripled points, the size of every difference in the table above.

### 3.3 Ceiling (`supersub_ceiling.csv`, `ceiling_variance.csv`)

**Method.** The persistent (predictable-in-principle) variance of replacement points is estimated from same-player covariance across different matches within 730 days, centred on the round mean. On Six Nations 2025–26 official points:

- persistent variance 7.4, 90% player-bootstrap interval [4.1, 11.2];
- total variance 47.8, so at most ~15% of replacement-point variance is predictable from player identity;
- the forecast already captures ~70% of it (explained variance 5.4).

On development (2023–24) the figures are 4.6 persistent, with ~31% captured. On observable points the predictable share is 10–14%.

Simulated expected tripled super-sub points per 10 rounds, with the missing persistent signal recovered in part:

| Scenario | Six Nations 2025–26 | Development 2023–24 |
|---|---:|---:|
| Random replacement | 228 | 224 |
| Current forecasts (expectation) | 405 | 318 |
| Realised by current forecasts | 345 (smoothed 365) | 291 |
| +25% / +50% of missing signal | 411 / 417 | 332 / 347 |
| Oracle of persistent means | **428** [405, 464] | **372** [318, 422] |
| Hindsight | 876 | 905 |

**Ceiling: about +24 tripled points per Six Nations season-pair** (at most about +60) from perfect player knowledge. Of the hindsight gap (about 530 realised, 471 against the current expectation), about 95% is match noise beyond any player-level forecast: who scores the late try.

The null for a super-sub difference, from 3%-perturbed equally informed copies (bench arg-max proxy), has SD 24 for Six Nations 2025 and 4.6 for 2026 (`test_supersub_null.csv`). SK's super-sub gains sit at about +2.9 SD for 2025 (2026 is unchanged on the proxy), but the pick changes in only one 2025 round. Upside or quantile rules cannot add expected points under linear scoring and were excluded on principle.

## 4. What remains

- **Official-only statistics** (BS, scrums won, POTM, official metres) explain most of the remaining back-row and starting-back gaps. That work belongs to the official-stats agent.
- **Season-specific bench minutes** (Six Nations 2025–26 back-rows +2.4 minutes) have no pre-lock signal in this data. Published bench-usage or split information would be new information.
- **Friendly-25** contradicts the bench raise. Friendly bench usage may differ, for example more rotation. Track it prospectively before using S or SK for friendlies.
- **Recommendation: freeze SK** (and SK + MK) beside H2 and MK for NCR GW4–7, and compare prospectively on MAE, MSE, bench bias and smoothed super-sub points.

## Reproduction

```bash
# status-aware (S) and shrunk (SK) forecasts at every lock (cache-only)
python -m research.position_supersub --runs RUNS --output OUT --components-fallback COMP --sets devcr blocks official friendly [--variant shrunk]
# fantasy, squad and super-sub metrics (dev or official); --mk adds matchup + kicking rows
python -m research.position_supersub_eval --runs RUNS --status OUT --set dev|official --mk --output EVAL
# raw-event and observable-points metrics on archived blocks or Friendly-25
python -m research.position_supersub_blocks --runs RUNS --status OUT --set blocks|friendly --output EVAL
# end-to-end opt-in engines
python -m model.unified.rolling_eval --native-categories --status-rates --output OUT
```

- `RUNS` is the scratch runs directory (`base/`, `devcr/`, `f25_ctx/`, `feature_cache/`, and `raw/` beside it).
- `COMP` holds the exported Six Nations 2026 R5 components, which are missing from `base/components`.
- Analysis scripts for the diagnosis tables are in `position_supersub_2026-10/scripts/`. They need scratch paths (`common.py`).

**Verification and side effects.**

- Reference numbers reproduce the ledger exactly: 8.251/1654/1651, 7.240/2356/2342, 7.238/2513/2510; MK 8.232/1706/1703.
- Tests: `tests/test_status_rates.py` (6 tests).
- No RapidAPI requests were made, `data/cache` is untouched, and no model was promoted.
