# Large computed-rubric benchmark, October 2026

## Outcome

The new benchmark has 535 slates (437 with squad decisions), 133,262 player rows and 39 monthly point-in-time refits of robust P3, covering every cached club and international match from July 2023 to September 2026. It scores 11 engines under three API-observable rubrics.

**Decisiveness.**

- It resolves accuracy differences of about **0.005–0.008 fantasy MAE** between raw-level engines. The 13 official rounds resolve about 0.03–0.06.
- It resolves squad-point differences of about **2–5 points per slate**. The official rounds resolve 15–26 points per round.
- As a demonstration, a copy of robust P3 with 3% random forecast noise is detected as worse, with the interval excluding zero (MAE +0.007, MSE +0.16).

**Main results**, all as paired per-slate differences against robust P3 with 90% intervals (widest of slate, block and lock bootstraps):

1. **The empirical baseline is clearly worse everywhere.**
   - MAE +0.42 and MSE +10.1; within-slate Spearman −0.046.
   - Smoothed squad points −22 per slate.
   - This holds in every competition and rubric, including the Six Nations slates (MAE +0.06).
2. **The fantasy-point blends lose here.** H1 (1 Oct candidate), H2 and SK+H2 are significantly worse than robust P3:
   - on the full benchmark: MAE +0.050 / +0.065 / +0.069, MSE +0.7 to +1.1, Spearman −0.004 to −0.005;
   - on out-of-sample internationals: H1 +0.059, H2 +0.041, SK+H2 +0.030 MAE.

   The gains they showed on the official slates do not transfer to observable-component labels. This is consistent with the round-2 finding that the blend is a level correction towards official-only points: the official Six Nations mean is 16.5 points against 13.4 computed here. Two further causes: on club slates the baseline has almost no player history, and its Six Nations slates (where H2 still gains −0.035 MAE) overlap the slates used for selection.
3. **On internationals the raw-level candidates are distinguishably better, and SK+MK is best.** The gain is consistent across all three rubrics and holds out of sample:

   | | International MAE (Six Nations rubric) | International MSE | Out-of-sample internationals MAE | Out-of-sample internationals MSE |
   |---|---|---|---|---|
   | SK+MK | −0.049 [−0.065, −0.033] | −0.75 [−1.07, −0.40] | −0.046 [−0.072, −0.022] | −0.92 [−1.40, −0.43] |
   | MK | −0.035 [−0.050, −0.019] | −0.33 | −0.036 [−0.060, −0.014] | −0.38 |
   | SK | −0.016 [−0.027, −0.005] | −0.48 [−0.74, −0.23] | −0.012 (n.s.) | −0.55 [−0.94, −0.20] |

   SK+MK also raises international Pearson by +0.005 and Spearman by +0.004. The NCR and SRP rubrics agree: SK+MK international MAE is −0.036 and −0.060.
4. **On club slates no candidate improves on robust P3.**
   - MK costs +0.005 MAE / +0.26 MSE and SK +0.009 MAE; SK+MK costs +0.013 MAE.
   - C2+K15 is the only engine that is at least neutral (−0.002 MAE).
   - On club slates the matchup step reduces to an Elo-edge term whose coefficients were fitted on internationals. The status-aware component's bench factors also do not help club forecasts.
5. **Squad points still separate almost nothing.**
   - Across 437 slates no candidate has significantly higher smoothed squad points than robust P3.
   - MK and SK+MK lose 2.5–3.0 points per slate (Six Nations rubric, significant, club-driven).
   - On internationals SK gains +4.6 [+0.4, +9.2] points per slate, which is marginal.
   - The null pair (two equally good noise copies) also crosses the 90% line in 3 of 12 decision cells. Single significant squad-point cells are therefore weak evidence; only consistent patterns across rubrics and scopes count.

**Bottom line.**

- Robust P3 is distinguishable from the empirical baseline on every metric.
- Candidates are now distinguishable on accuracy, with opposite signs by family:
  - the blends are worse on computed-rubric labels;
  - the raw-level SK+MK and MK are better on internationals and slightly worse on clubs.
- For the international fantasy games, **SK+MK (without the fantasy blend)** is the best-supported engine on this benchmark.
- The blend's official-slate gain should be re-examined against official labels before it is adopted; it does not hold on computed labels.

## Results

All tables: Six Nations-rubric unless stated; MAE and RMSE are means over slates; differences are engine minus robust P3, paired per slate.

### Accuracy, all 535 slates (133,262 players)

| Engine | MAE | RMSE | Bias | Pearson | Spearman | Cal. slope |
|---|---:|---:|---:|---:|---:|---:|
| Robust P3 (reference) | 6.290 | 8.839 | −0.65 | 0.6011 | 0.6424 | 1.018 |
| Empirical baseline | 6.708 | 9.391 | −0.20 | 0.5552 | 0.5966 | 0.867 |
| C2+K15 | 6.289 | 8.837 | −0.63 | 0.6012 | 0.6427 | 1.004 |
| H1 (1 Oct) | 6.340 | 8.876 | −0.52 | 0.5971 | 0.6387 | 1.006 |
| MK | 6.288 | 8.848 | −0.67 | 0.6000 | 0.6420 | 1.008 |
| H2 | 6.355 | 8.900 | −0.53 | 0.5948 | 0.6370 | 1.006 |
| SK | 6.295 | 8.835 | −0.64 | 0.6023 | 0.6435 | 1.058 |
| SK+MK | 6.293 | 8.845 | −0.65 | 0.6010 | 0.6431 | 1.047 |
| SK+H2 | 6.359 | 8.898 | −0.51 | 0.5953 | 0.6374 | 1.033 |
| Null copy A / B (3% noise) | 6.297 / 6.297 | 8.848 / 8.849 | −0.65 | 0.6001 / 0.6000 | 0.6414 / 0.6416 | 1.013 |

### Accuracy, 87 international slates (14,950 players)

| Engine | 6N MAE | 6N RMSE | 6N Pearson | NCR MAE | NCR RMSE | SRP MAE | SRP RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Robust P3 | 6.359 | 8.901 | 0.6032 | 6.199 | 8.721 | 8.453 | 12.387 |
| Empirical baseline | 6.629 | 9.153 | 0.5789 | 6.370 | 8.933 | 8.711 | 12.668 |
| C2+K15 | 6.360 | 8.906 | 0.6035 | 6.210 | 8.726 | 8.465 | 12.393 |
| H1 (1 Oct) | 6.396 | 8.929 | 0.6011 | 6.223 | 8.745 | 8.498 | 12.425 |
| MK | 6.323 | 8.882 | 0.6055 | 6.171 | 8.704 | 8.401 | 12.360 |
| H2 | 6.377 | 8.916 | 0.6020 | 6.199 | 8.734 | 8.457 | 12.402 |
| SK | 6.342 | 8.874 | 0.6057 | 6.190 | 8.705 | 8.441 | 12.363 |
| **SK+MK** | **6.309** | **8.859** | **0.6078** | **6.163** | **8.692** | **8.393** | **12.343** |
| SK+H2 | 6.363 | 8.894 | 0.6037 | 6.190 | 8.721 | 8.444 | 12.383 |

On the 15 Six Nations slates (2024–26), the ordering matches the official ledger: robust 6.093, MK 6.054, SK+MK 6.055, H2 6.057, SK+H2 6.057, H1 6.091, baseline 6.153. These slates were in-sample or used for selection.

### Consistency across rubric and family

Mean per-slate difference against robust P3. `*` marks a 90% interval that excludes zero. OOS is international slates outside the coefficient and selection samples (33 slates; 14 with decisions).

| Engine | MAE 6N all | 6N club | 6N intl | 6N intl OOS | NCR intl | SRP intl | MSE 6N intl OOS | Smoothed pts 6N all | 6N club | 6N intl |
|---|---|---|---|---|---|---|---|---|---|---|
| Empirical baseline | +0.419* | +0.448* | +0.270* | +0.365* | +0.171* | +0.259* | +6.12* | −22.1* | −23.7* | −10.7* |
| C2+K15 | −0.001 | −0.002 | +0.001 | +0.006 | +0.011* | +0.012* | +0.08 | −0.3 | +0.3 | −4.5* |
| H1 (1 Oct) | +0.050* | +0.053* | +0.037* | +0.059* | +0.024* | +0.045* | +0.69* | −0.3 | −0.4 | +0.5 |
| MK | −0.001 | +0.005* | −0.035* | −0.036* | −0.028* | −0.051* | −0.38 | −2.5* | −2.2* | −4.6 |
| H2 | +0.065* | +0.074* | +0.018 | +0.041* | +0.000 | +0.004 | +0.52 | −2.4 | −3.0 | +1.5 |
| SK | +0.005* | +0.009* | −0.016* | −0.012 | −0.009 | −0.012 | −0.55* | −0.3 | −1.0 | +4.6* |
| **SK+MK** | +0.003 | +0.013* | **−0.049*** | **−0.046*** | **−0.036*** | **−0.060*** | **−0.92*** | −3.0* | −3.1* | −2.3 |
| SK+H2 | +0.069* | +0.082* | +0.004 | +0.030* | −0.009 | −0.008 | +0.07 | −2.2 | −2.8 | +1.8 |
| Null copy A | +0.008* | +0.007* | +0.013* | +0.004 | +0.007* | +0.006 | +0.03 | −0.7 | −0.4 | −2.8 |
| Null pair (A − B) | +0.000 | +0.001 | −0.001 | −0.008 | −0.003 | −0.006 | −0.18 | +2.2* | +2.8* | −2.2 |

Per-competition matrices for MAE, MSE, Spearman and smoothed points are in `research/big_benchmark_2026-10/tables.md`.

- **Matchup step.** The international gain of MK appears in the pooled international weeks (−0.035 MAE), the Six Nations (−0.039) and the NCR (−0.017). Its club losses are small and widespread: Japan +0.011, Super Rugby +0.016.
- **Blends.** They lose most in Japan League One and the Currie Cup, where the baseline's positional/RugbyPass fallback is weakest. Currie Cup baseline RMSE is 14.3 against 9.5 for robust P3, because of a few extreme baseline forecasts.

### Decisions (Six Nations rubric, mean squad points per slate)

| Engine | Club realised / smoothed | International realised / smoothed | Smoothed regret club / intl |
|---|---:|---:|---:|
| Robust P3 | 398.3 / 398.4 | 402.3 / 402.1 | 327 / 276 |
| Empirical baseline | 375.7 / 374.7 | 390.7 / 391.3 | 351 / 287 |
| H1 | 398.1 / 398.1 | 403.6 / 402.6 | 328 / 276 |
| MK | 396.5 / 396.2 | 398.2 / 397.4 | 329 / 281 |
| H2 | 395.2 / 395.4 | 403.6 / 403.6 | 330 / 275 |
| SK | 397.4 / 397.4 | 406.6 / 406.7 | 328 / 271 |
| SK+MK | 396.3 / 395.4 | 399.5 / 399.8 | 330 / 278 |
| SK+H2 | 395.3 / 395.7 | 403.3 / 403.9 | 330 / 274 |

- Hindsight optima average 726 points (club) and 678 points (international). Every model-based engine captures 55–59% of them.
- Captain and super-sub choices are the knife edges. "Flat" squad points, which count every member once, cut the per-slate paired SD by about a quarter (for example the null pair: 27 → 22 points).

### Raw events (pooled, all slates; loss relative to robust P3)

SK lowers loss on most count events:

- tries 0.992;
- try assists 0.986;
- clean breaks 0.993;
- penalties conceded 0.993;
- tackle turnovers 0.993.

It raises metres loss (1.015). MK and C2+K15 improve penalty-goal loss (0.97) and slightly worsen metres and runs, because these pooled numbers are dominated by club rows, where only the Elo term applies. Details: `events_summary.csv`.

### Validation of the computed labels

On the ten official Six Nations 2025–26 rounds (1,362 players), the computed Six Nations-rubric points correlate with official points:

- r = 0.938 (Spearman 0.941);
- 2025: r = 0.943; 2026: r = 0.934.

Their mean is 13.4 against 16.5 official. The 3-point gap is the official-only components (POTM, scrums, lineout steals, 50-22s, kicks retained) and the metres scale.

Every rubric event was observed for every row in the evaluation window; no row was unscorable.

The SK reconstruction was checked on lock 2023-07 by refitting the tree:

- the robust P3 refit reproduces the saved forecasts exactly (max |Δmean| = 0);
- a directly built SK blend matches the reconstructed one to 1.4e-14.

The memoised empirical baseline equals the unmemoised per-slate baseline exactly on lock 2026-09.

## Power analysis

Paired per-slate standard deviations are measured on the benchmark itself; δ is the difference to detect at 80% power with a two-sided 5% test.

| Metric (Six Nations rubric) | Paired SD per slate | Design effect | Slates for δ | Benchmark MDE | Official SD per round | Official rounds for δ | Official MDE (13 rounds) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAE, δ = 0.02: raw-level candidates (MK, SK, SK+MK) | 0.044–0.064 | 1.9–2.3 | 73–182 | 0.007–0.012 (535 slates) | 0.037 (MK) | 27 | 0.029 |
| MAE, δ = 0.02: null pair | 0.042 | 1.1 | 37 | 0.005 | – | – | – |
| MAE, δ = 0.02: blends (H1, H2) | 0.136–0.170 | 2.2 | 800–1,250 | 0.024–0.031 | 0.048–0.077 | 45–117 | 0.037–0.060 |
| MAE, international only | 0.052–0.095 | 1.1–2.7 | 81–356 | 0.016–0.040 (87 slates) | – | – | – |
| Smoothed squad points, δ = 5 | 17–29 (null pair 27) | 1.0–1.6 | 90–405 (null 227) | 2.3–4.8 (437 slates) | 19–33 | 111–344 | 15–26 |
| Realised squad points, δ = 5 | 24–34 (null pair 33) | 1.0–1.4 | 200–530 | 3.4–5.5 | 22–42 | 155–565 | 17–33 |
| Flat smoothed squad points, δ = 5 | 14–21 (null pair 22) | 1.0–1.7 | 65–183 | 1.9–3.2 | – | – | – |
| Squad points, international only, δ = 5 | 18–28 | 1.0–1.5 | 104–340 | 7–13 (54 slates) | – | – | – |

**MAE.**

- The official 13 rounds can detect about 0.03 (raw-level changes) to 0.06 (blends); their design effect is ignored, so this is optimistic.
- The benchmark detects 0.005–0.012 for raw-level changes.
- Blends have far larger per-slate variance, because their effect changes sign between competitions. Their pooled mean is therefore less informative than per-family results.

**Squad points.**

- Detecting 5 points per round needs about 100–400 rounds. The official set has 13, which can only detect differences of 15–26 points per round (200–340 per 13-round season).
- The benchmark's 437 slates detect 2–5 points per slate on the full pool, but only 7–13 points on the 54 international decision slates.
- Smoothing and the flat variant remove about 30–55% of the realised-score variance. They are the right decision metrics for future comparisons.

**Recommendation.**

- Judge forecasting changes on paired MAE, MSE and within-slate correlation over the full benchmark and over its international slice. Require consistency across rubrics.
- Treat squad points as a guard-rail: a change should not lose more than about 3 points per slate. They should not be used as a selection criterion.
- When reading per-cell intervals, remember that roughly 1 in 10 null cells will cross the 90% line.

## Why a larger benchmark

The official benchmark has 13 rounds with official fantasy points (Six Nations 2025 and 2026, NCR 2026 GW1–3). The hill-climb reports showed that this is too few to separate the models now being compared:

- a 1% forecast jitter moves season squad totals by 22–37 points;
- perturbed copies of equally good models differ by 41–148 points per season;
- MAE differences of 0.02–0.05 are rarely significant.

Fantasy MAE also rewards medians on a skewed target, so a benchmark that only reports MAE can prefer a worse expected-value forecast.

This benchmark uses every cached match from July 2023 to September 2026. Fantasy points are computed from the store's API events under fixed rubrics. It reports both median-style (MAE) and mean-proper (MSE/RMSE) metrics, ranking metrics, decile calibration and squad decisions on synthetic slates, with uncertainty at the slate, competition-season and fit level.

## Design

### Data and point-in-time rule

- **Store.** The 22 September 2026 research store: 179,617 player-match-team rows and 3,899 fixtures, SHA-256 `23645c25…a62b`. Its prior-only training features were reused. No RapidAPI request was made and `data/cache` was not touched.
- **Grain.** Every table keeps the `(fixture_id, player_id, team)` grain; duplicate keys are an error.
- **History.** Each forecast uses only matches with kickoff + 3 h before its lock (`model.history.past_matches`). Each lock also asserts that no evaluated fixture entered training. Candidate features and playing positions are rebuilt from pre-lock history only (`build_frozen_feature_frames`).

### Locks, slates and blocks

| Unit | Definition | Count |
|---|---|---:|
| Lock | One robust P3 refit at 00:00 UTC on the 1st of each month, July 2023 – September 2026. Every slate whose first kickoff falls in that month is forecast from it. | 39 |
| Slate | A competition round (`source_round`) for club competitions, the Six Nations and the NCR. Other internationals (Rugby Championship, Pacific Nations Cup, World Cup, Lions tour, tour matches, ad-hoc tests) are pooled into one slate per ISO week, like a test-window game. Rescheduled fixtures more than 6 days after a round's first kickoff form a `_late` slate. | 535 |
| Decision slate | A slate with at least 3 fixtures, enough for a 16-player squad under a four-per-team cap. | 437 |
| Block | A competition season, or a pooled-international calendar window. Used as a bootstrap cluster. | 41 |

The 535 slates hold 2,897 fixtures and 133,262 player rows: 448 club slates (118,312 rows) and 87 international slates (14,950 rows). The frozen manifest is `research/big_benchmark_2026-10/manifest.json`; it lists every slate's fixtures, lock and flags plus the input hashes.

**Why monthly shared locks.** Refitting per round would need about 600 fits. One fit per competition season would leave forecast horizons of up to 9 months. Monthly locks need 39 fits (about 2.5–4.5 minutes each at two threads) and give horizons of 0–31 days. That is comparable to the official tournament-frozen slates, which are forecast up to six weeks ahead.

The trade-off is that a slate late in a month is forecast without the previous three weeks of results. This handicaps every engine equally.

### Rubrics (API-observable components only)

Labels and forecasts use the same observable components, so both sit on one scale. Players who did not play score zero.

| Rubric | Scored components | Official components missing |
|---|---|---|
| Six Nations | try 15 (forward) / 10 (back), try assist 4, conversion 2, penalty 3, drop goal 4, defender beaten 2, offload 2, floor(metres/10), tackle 1, breakdown steal 5 (API `tackle_turnover` proxy), penalty conceded −1, yellow −5, red −8 | 50-22 (7), lineout steal (7), scrum won (1), kick retained (2), player of the match (15) |
| NCR | try 12, try assist 5, conversion 2 / missed −1, penalty 3 / missed −1, drop goal 5, defender beaten 2, offload 2, clean break 3, tackle 1, missed tackle −1, turnover won 4, turnover conceded −1, penalty conceded −1, yellow −5, red −10 | lineout steal (5), lineout won (1; the API count is diagnostic-only and not forecast), interception (5), front-row scrum won (2), player of the match (15) |
| Super Rugby Pacific style | try 15, try assist 9, conversion 2 / missed −1, penalty 3 / missed −1, drop goal 3, line break 7 (API `clean_breaks`), floor(metres/10), tackle 1, missed tackle −1, turnover won 4 (`tackle_turnover`), defender beaten 2, offload 2, penalty conceded −1, yellow −5, red −10 | lineout steal (5) |

A row missing any rubric event is unscorable for that rubric and leaves both the metrics and the decision pool. In this window every rubric event was observed for every row, so no row was dropped.

Expected points are exact under each forecast distribution: E[floor(X/10)] for lognormal metres, linear means otherwise (`research/big_benchmark/rubrics.py`).

### Engines

Every engine derives from the same lock-time robust P3 fit, so differences between engines are paired.

| Engine | Definition |
|---|---|
| `p3_robust` (reference) | `RobustEmpiricalEventModel` + V4 GBDT, `EventWeightedBlend` with the existing event weights. This is the `rolling_eval` / `dev_six_nations_current_rubric` recipe. |
| `empirical_baseline` | `model.empirical_unified.project_candidates`, run per slate at the lock. Rubric configs are restricted to the observable components (front-row scrum bonus 0), and its player-of-the-match term is removed exactly (`strip_potm`). **It profiles international history only**, so on club slates most players fall back to positional or RugbyPass priors. |
| `c2k15` | Team-strength (Elo edge) calibration, then kicking concentration with γ = 1.5. |
| `h1_oct1` | 1 October candidate: 0.75 × `c2k15` + 0.25 × baseline. |
| `mk` | Matchup calibration (Elo edge + opponent conceded profile), then kicking γ = 1.5. |
| `h2` | Round-2 candidate: 0.7 × `mk` + 0.3 × baseline. |
| `sk` | Robust P3 with the status-aware, within-position-shrunk empirical component (`raw_benchmark.status_rates.ShrunkStatusEmpiricalEventModel`, merged from `af85e5a`). It keeps the same trees and blend weights.<br>Only the empirical component is refitted per lock. The tree is recovered exactly from the saved blend (`research/big_benchmark/status.py`). |
| `sk_mk`, `sk_h2` | SK, then matchup + kicking γ = 1.5; and 0.7 × that + 0.3 × baseline. |
| `null3_a`, `null3_b` | The reference times a fixed per-player factor exp(N(0, 0.03)). These are equally informed copies about as different from the reference as the candidates are. Their pair difference has expectation zero and measures pure evaluation noise. |

**Matchup on club slates.** The matchup opponent profile uses international history only. On club slates `mk` and `h2` therefore apply only their Elo-edge term and kicking.

**Sample flags.** The candidates' coefficients (team strength, matchup, γ) were fitted on international blocks that ended before 2025. Their weights were chosen with the official Six Nations 2025/26 and NCR 2026 rounds in view. The 54 slates in either set are flagged `in_sample`; every "out-of-sample" row excludes them.

### Metrics

- **Accuracy.** Computed per slate, then averaged with slates weighted equally, as the official ledger averages round MAEs:
  - MAE;
  - MSE and RMSE (mean-proper);
  - bias;
  - within-slate Pearson and Spearman correlation.

  Pooled player-weighted MAE/RMSE, calibration slope and decile calibration are in `accuracy_summary.csv` and `calibration_deciles.csv`.
- **Raw events.** Per-target loss for the three raw-level engines, from the existing `raw_benchmark.metrics.event_metrics` (`events_summary.csv`).
- **Decisions.** The pool is every scorable player in the slate's fixtures. The rules are the official Six Nations evaluation rules in `rolling_eval.evaluate`:
  - 15 fantasy starters by positional quota, plus one super-sub who must be a named replacement;
  - a captain who must be a named starter;
  - at most four players per team, no budget;
  - captain ×2, super-sub ×3 if he plays.

  Replacements with no recorded start take the conventional role of their bench jersey.

  For each slate, rubric and engine the benchmark records:
  - realised squad points;
  - smoothed squad points: the mean over 40 identical 1% multiplicative jitters, with draws shared across engines;
  - flat squad points: every member counts once, which removes the captain and super-sub knife edges;
  - the hindsight optimum and regret.

  The optimiser is a sparse MILP with provably dominated roles removed. A test checks that its objective equals `model.ncr_project.optimise` on random pools; it runs in about 6 ms per solve.
- **Uncertainty.** Paired differences against the reference are bootstrapped three ways: by slate, by block and by lock (4,000 draws). Tables show the widest 90% interval. A block or lock bootstrap counts only when it has at least 8 clusters, because fewer understate the spread.
- **Power.** The required number of slates is n = ((z₀.₉₇₅ + z₀.₈) · SD_d / δ)² × design effect. SD_d is the standard deviation of paired per-slate differences. The design effect is the squared ratio of the widest clustered interval to the slate interval. The official-only equivalent uses the 13 official rounds' paired SDs (`research/hillclimb_2026-10-02` ledger: robust P3, H1, MK, H2, empirical).

## Limitations

- **No prices for club slates.** Decisions use the Six Nations no-budget rules everywhere. Budget-constrained games (NCR, Super Rugby) would select differently.
- **API versus official statistics.**
  - Labels omit every official-only component (POTM, scrums, lineout steals, 50-22s, kicks retained, interceptions).
  - Breakdown steals and turnovers won use the API `tackle_turnover` count. Official 2025 metres are about 1.45× API metres.
  - The computed Six Nations rubric is checked against official points on the ten 2025–26 rounds (see Validation). It is not identical, so level effects that help official MAE (for example the empirical blend) need not help here.
- **Oracle teamsheets.** Candidates are the players who appeared on the final teamsheets, with their real starting status. Late withdrawals and bench usage are not forecast. The same holds for the official benchmark.
- **Shared fit per month.** Slates within a lock share one fit, and slates within a competition season share players. The lock and block bootstraps account for this; the slate bootstrap alone would overstate precision.
- **The empirical baseline is international-centric.** On club slates it is close to a positional prior. The blends (`h1_oct1`, `h2`) inherit this, so their club results test "blend with a weak baseline", not the blend as it would work with club history.
- **In-sample candidate coefficients** on internationals before 2025 (flagged and excluded from out-of-sample rows).
- **Club relevance.** Club slates measure the same forecasting skill on different players and teams. A gain there need not transfer to the international fantasy games, and the reverse also holds.

## Reproduction

```bash
# Inputs: a rolling_eval output directory with inputs/player_match.csv and inputs/training_features.pkl
export OMP_NUM_THREADS=2   # LightGBM is capped to this inside the fit stage
python -m research.big_benchmark --base RUNS/base --output OUT --stage manifest   # frozen manifest; refuses changed inputs
python -m research.big_benchmark --base RUNS/base --output OUT --stage fit        # 39 fits, resumable per lock
python -m research.big_benchmark --base RUNS/base --output OUT --stage sk         # status-aware empirical refits (cheap)
python -m research.big_benchmark --base RUNS/base --output OUT --stage score      # per-player points, resumable per lock
python -m research.big_benchmark --base RUNS/base --output OUT --stage decide --draws 40   # resumable per slate; --locks splits work across processes
python -m research.big_benchmark --base RUNS/base --output OUT --stage report \
    --official-rounds RUNS/h2_test/rounds.csv
# or --stage all; restrict with --locks YYYY-MM ..., --engines ..., --rubrics ...
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_THREAD_LIMIT=1 python -m pytest -q tests/test_big_benchmark.py
```

New engines plug in through `research/big_benchmark/score.py` (`ENGINES`, `raw_engines`). They are scored from the saved lock fits without refitting the reference. An engine that needs its own fit can save raw forecasts per lock beside `p3_robust.jsonl.gz`.

## Run record

- **Compute.**
  - 39 robust P3 fits at two LightGBM threads: 2.2 hours of fitting, 3.1 hours with features and baselines. `fits.json` lists every lock's cohort, latest training kickoff and timings.
  - 39 SK empirical refits took about 1–2 minutes each.
  - 14,421 squad decisions, each with 40 jitters plus two hindsight optima, came to about 600k MILP solves.
- **LightGBM threads.** The model code passes `n_jobs=-1`, which LightGBM 4 maps to every logical CPU regardless of `OMP_NUM_THREADS`. The fit stage caps it to `OMP_NUM_THREADS` (`fit.lightgbm_threads`). Without the cap, a fit on a shared machine took over 40 minutes.
- **Committed evidence.** `research/big_benchmark_2026-10/` holds:
  - the frozen manifest and run info;
  - fit summaries, accuracy, calibration, decision and raw-event summaries;
  - all paired differences (`differences.csv.gz`), the power table, official validation and rendered tables.

  Forecasts, player tables and per-slate decisions stay in the scratch output (275 MB).
- **No side effects.** No RapidAPI requests were made, `data/cache` is unchanged, production routing is unchanged and no model was promoted or retrained in the repository.
