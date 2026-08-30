# All-rugby hill-climb ledger — P3 unified event model

Branch `agents/allrugby-p3`. Objective: minimise the competition-balanced stable
raw score of Historical Raw Benchmark v1, pooled over **all** international rugby
from 2022-01-01 (21 exact-kickoff tournament holdout folds, both hemispheres).
Lower is better. Per-competition rubric scores are reported, never fitted.

Seed 17. Weight grid step 0.005. NCR GW4–7 was never read.

## Control

- Reconstructed frozen `p3_event_50` stable score: **0.8864704365**
- Frozen ledger value: **0.8864704365**
- Absolute difference `0.0e+00`; worst per-fold difference `2.2e-16`
- Reproduced: **True** across 21 folds

The frozen `p3_event_50` artifacts are self-contained, so the empirical and v4
component predictions were recovered from them directly and re-blended offline —
no engine was refitted. Re-blending at w=0.5 reproduces the frozen predictions and
the frozen per-fold metrics exactly. The frozen v1 ledger was never written to.

Two further control matches: the reconstructed rubric aggregates reproduce the
ledger's `p3_event_50` figures exactly (ncr MAE 6.1727 / spearman 0.5367 / capture
0.7187; six_nations 6.2767 / 0.6287 / 0.7689, 98 slates each), and the degenerate
`low_history` relative loss (848.68, from near-zero naive kicking losses) reproduces
the frozen ledger's value too. That cohort gates nothing.

## Precommitted acceptance rule

- **(a)** competition-balanced stable_score improves on 0.886470
- **(b)** paired-by-fold bootstrap difference excludes 0
- **(c)** neither north nor south cohort regresses by >2%
- **(d)** no tournament-family stable regression >2%
- **(e)** no extended-event loss regression >5% -- satisfied by construction: every candidate leaves the 8 extended events at the frozen 0.5 weight, so their predictions are unchanged
- **(f)** any newly fitted parameter is cross-fitted leave-one-fold-out and additionally checked on a strict temporal split

Ties and sub-threshold trials are rejected and logged.

## Trials

| candidate | params | stable_score | delta_vs_frozen | boot_p05 | boot_p95 | temporal_delta | accepted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1_global_weight | 1 | 0.886763 | 0.000293 | -0.000012 | 0.000710 | 0.000349 | False |
| C2_per_target_weight | 24 | 0.879476 | -0.006994 | -0.011579 | -0.002548 | -0.009538 | True |
| C3_per_loss_family_weight | 4 | 0.884520 | -0.001951 | -0.003221 | -0.000705 | -0.002730 | True |
| C4_density_parametric_weight | 2 | 0.886993 | 0.000523 | 0.000164 | 0.001033 | 0.000370 | False |
| C5_shrunk_per_target_weight | 25 | 0.879366 | -0.007105 | -0.011122 | -0.003251 | -0.005917 | True |
| C6_log_space_all_targets | 0 | 0.885688 | -0.000782 | -0.001540 | -0.000095 |  | True |
| C7_history_depth_weight | 3 | 0.887343 | 0.000872 | 0.000280 | 0.001649 | 0.000676 | False |
| C8_log_space_per_target_weight | 24 | 0.879517 | -0.006954 | -0.011805 | -0.002247 | -0.010105 | True |

Frozen baseline **0.886470**. `stable_score` for fitted candidates is
leave-one-fold-out cross-fitted: the weights applied to a fold were fitted without
it. `temporal_delta` is a second, stricter check — fit on the 10 earliest folds,
evaluate on the 11 latest.

### Rejected, and why

- **C1_global_weight** — The global 0.5 was chosen on 6N-2025 LORO, never on this target. Refitting one global v4 weight on the historical folds should move it toward the all-rugby optimum.
  - REJECTED: (a) stable_score 0.886763 did not improve on frozen 0.886470
  - REJECTED: (b) paired-by-fold bootstrap CI [-0.000012, 0.000710] includes 0
  - In-sample reference 0.886433; cross-fitted 0.886763.
- **C4_density_parametric_weight** — Encode the density hypothesis directly: w(target) = clip(a + b * positive_rate, 0, 1), monotone in event density.
  - REJECTED: (a) stable_score 0.886993 did not improve on frozen 0.886470
  - REJECTED: (b) paired-by-fold bootstrap CI [0.000164, 0.001033] includes 0
  - In-sample reference 0.886421; cross-fitted 0.886993.
- **C7_history_depth_weight** — The blend weight should vary by row, not by competition: the empirical prior should earn weight where a player has little career history for the GBDT to exploit. w(n) = clip(a + b*n/(n+k), 0, 1) on prior match count -- competition-independent, so it stays inside the one-model mandate.
  - REJECTED: (a) stable_score 0.887343 did not improve on frozen 0.886470
  - REJECTED: (b) paired-by-fold bootstrap CI [0.000280, 0.001649] includes 0
  - In-sample reference 0.886470; cross-fitted 0.887343.

Three independent negatives, all pointing the same way — **the blend weight does
not want to move globally, by event density, or by player history; it wants to move
by event type**:

- **C1** is the sharpest. The global weight curve bottoms at **w=0.52** (0.886433)
  against 0.886470 at w=0.50 — a gain of 0.000037. The 0.5 chosen on 6N-2025 LORO
  was already sitting at the all-rugby optimum, and *refitting it costs* accuracy
  (cross-fitted 0.886763) because the refit adds estimation variance to a flat curve.
- **C4** fails because density is the wrong axis: `metres` is dense (0.83) yet wants
  w≈0.09, while `red_cards` is ultra-sparse (0.003) yet wants w≈0.91.
- **C7** is the cleanest null of all: given the freedom to vary the weight with
  player career history, the all-folds fit chooses a=0.5, b=0 — exactly the constant
  frozen weight, reproducing 0.886470 in-sample to the digit. History depth carries
  no usable signal for this blend.

### Accepted

- **C2_per_target_weight** — cross-fitted 0.879476 (+0.789% vs frozen). Empirical priors win on sparse/attacking events and the GBDT wins on dense volume events, so one weight per target should beat a single global weight. Highest overfitting risk in the queue.
- **C3_per_loss_family_weight** — cross-fitted 0.884520 (+0.220% vs frozen). The metric mixes Poisson deviance, log-squared, log-loss and MAE. One weight per loss family captures the structural split at a fraction of the variance of 24 free weights.
- **C5_shrunk_per_target_weight** — cross-fitted 0.879366 (+0.801% vs frozen). Per-target weights shrunk toward the fitted global weight, with the shrinkage strength chosen by an inner leave-one-fold-out pass over the training folds only. Keeps C2's signal, discards its noise.
- **C6_log_space_all_targets** — cross-fitted 0.885688 (+0.088% vs frozen). The blend is on event MEANS, but the metric is Poisson deviance / log-squared, whose natural link is log. Blending every target on the log1p scale at the frozen 0.5 weight adds no fitted parameter.
- **C8_log_space_per_target_weight** — cross-fitted 0.879517 (+0.784% vs frozen). If the log link is the right space for these losses, the per-target weight should be fitted there too rather than in mean space.

All accepted candidates pass every gate against the frozen baseline. They are
alternative parameterisations of the same one-dimensional blend family rather than
composable deltas, so the hill-climb terminates at the best of them:
**C5, the shrunk per-target weight**, at **0.879366**.

## Incumbent: C5 — shrunk per-target blend weight

- Cross-fitted stable score **0.879366** vs frozen 0.886470
- Improvement **0.801%**
- Paired-by-fold bootstrap: mean -0.007105, 90% CI [-0.011122, -0.003251] — excludes 0
- Temporal holdout (fit 10 earliest folds, evaluate 11 latest): 0.887107 vs 0.893024, delta -0.005917, CI [-0.007824, -0.004136] — excludes 0
- Folds improved: 15/21
- Shrinkage λ by inner leave-one-fold-out: [0.05, 0.1, 0.15] across outer folds; 0.1 when fitted on all folds

### In-sample vs held-out statement

C5 fits 24 per-target weights plus one shrinkage strength (25 parameters).
**Nothing in the headline number is in-sample.** Every fold is scored with weights
fitted on the other 20 folds, and the shrinkage strength is chosen by a further
leave-one-fold-out pass *inside* those 20 — the held-out fold informs neither. For
reference the fully in-sample score is 0.877396; the honest
cross-fitted score is 0.879366. The gap
(+0.001970) is the overfitting that
cross-fitting removes and that a naive all-folds fit would have banked as a fake
win. The independent temporal split, where no held-out fold is even contemporaneous
with the fitting set, confirms the effect survives.

### Fitted weights (v4 share; 1 − w is the empirical share)

| target | weight_v4 | density | empirical | v4 |
| --- | --- | --- | --- | --- |
| metres | 0.0900 | 0.8254 | 0.8779 | 0.9585 |
| tries | 0.1600 | 0.1368 | 0.9639 | 1.0266 |
| try_assists | 0.1900 | 0.0949 | 0.9983 | 1.0510 |
| missed_conversion_goals | 0.2000 | 0.0258 | 0.9166 | 0.9571 |
| conversion_goals | 0.2050 | 0.0469 | 0.6757 | 0.7713 |
| clean_breaks | 0.2350 | 0.1883 | 0.9834 | 1.0122 |
| offload | 0.2500 | 0.2349 | 0.9462 | 0.9666 |
| defenders_beaten | 0.2700 | 0.4509 | 0.9382 | 0.9520 |
| penalties_conceded | 0.3650 | 0.3319 | 1.0174 | 1.0084 |
| turnovers_conceded | 0.3950 | 0.3986 | 1.0112 | 1.0065 |
| minutes | 0.4000 | 0.9829 | 0.9723 | 0.9838 |
| missed_tackles | 0.4200 | 0.5396 | 0.9696 | 0.9623 |
| bad_passes | 0.5550 | 0.2871 | 0.9855 | 0.9518 |
| rucks_won | 0.5850 | 0.8652 | 0.8573 | 0.7669 |
| yellow_cards | 0.5950 | 0.0255 | 1.0308 | 0.9961 |
| penalty_goals | 0.6200 | 0.0336 | 0.7325 | 0.7129 |
| tackles | 0.6200 | 0.9249 | 1.0392 | 0.9312 |
| missed_penalty_goals | 0.6250 | 0.0112 | 0.8470 | 0.8272 |
| runs | 0.6350 | 0.9157 | 1.0813 | 0.8964 |
| passes | 0.6400 | 0.8170 | 0.9280 | 0.8035 |
| drop_goal_missed | 0.6550 | 0.0027 | 0.8274 | 0.8116 |
| rucks_lost | 0.6600 | 0.1422 | 1.0219 | 0.9818 |
| drop_goals_converted | 0.8400 | 0.0011 | 0.8653 | 0.7302 |
| red_cards | 0.9100 | 0.0033 | 1.0499 | 0.8537 |

The structure is not noise — it is a clean split by *event type*:

- **Attacking / scoring events go to the empirical prior** (w≈0.09–0.27): metres,
  tries, try assists, conversions, clean breaks, offloads, defenders beaten.
- **Volume / workrate and discipline events go to the GBDT** (w≈0.55–0.66): passes,
  runs, tackles, rucks, bad passes, place-kicking attempts, cards.
- **Neutral, near the frozen 0.5**: minutes, missed tackles, turnovers and penalties
  conceded.

The v4 GBDT models minutes- and role-driven volume well but over-smooths rare
attacking upside, where a player's own empirical history carries more signal. The
two ultra-sparse outliers (`drop_goals_converted` 0.84, `red_cards` 0.91) rest on
very few events; shrinkage pulls them in and they move the mean-of-24 by little.

## Breakdown

### Per fold

| fold | tournament | hemisphere | stable_base | stable_cand | delta | pct |
| --- | --- | --- | --- | --- | --- | --- |
| six_nations_2022 | six_nations | north | 0.8842 | 0.8890 | 0.0048 | 0.5470 |
| summer_internationals_2022 | summer_internationals | mixed | 0.8568 | 0.8675 | 0.0107 | 1.2444 |
| rugby_championship_2022 | rugby_championship | south | 0.8982 | 0.9080 | 0.0098 | 1.0896 |
| autumn_internationals_2022 | autumn_internationals | mixed | 0.8523 | 0.8553 | 0.0030 | 0.3521 |
| six_nations_2023 | six_nations | north | 0.8450 | 0.8473 | 0.0024 | 0.2801 |
| rugby_championship_2023 | rugby_championship | south | 0.8731 | 0.8412 | -0.0319 | -3.6563 |
| summer_internationals_2023 | summer_internationals | mixed | 0.9123 | 0.9013 | -0.0110 | -1.2022 |
| rugby_world_cup_2023 | rugby_world_cup | mixed | 0.8603 | 0.8490 | -0.0113 | -1.3153 |
| six_nations_2024 | six_nations | north | 0.8837 | 0.8850 | 0.0013 | 0.1449 |
| summer_internationals_2024 | summer_internationals | mixed | 0.9266 | 0.9251 | -0.0015 | -0.1614 |
| rugby_championship_2024 | rugby_championship | south | 0.8990 | 0.8778 | -0.0212 | -2.3531 |
| pacific_nations_cup_2024 | pacific_nations_cup | south | 0.9188 | 0.9185 | -0.0003 | -0.0366 |
| autumn_internationals_2024 | autumn_internationals | mixed | 0.9119 | 0.9014 | -0.0105 | -1.1540 |
| six_nations_2025 | six_nations | north | 0.8461 | 0.8311 | -0.0150 | -1.7692 |
| british_irish_lions_2025 | british_irish_lions | south | 0.8549 | 0.8423 | -0.0126 | -1.4786 |
| summer_internationals_2025 | summer_internationals | mixed | 0.8950 | 0.8894 | -0.0055 | -0.6188 |
| rugby_championship_2025 | rugby_championship | south | 0.9306 | 0.8997 | -0.0309 | -3.3203 |
| pacific_nations_cup_2025 | pacific_nations_cup | south | 0.9615 | 0.9589 | -0.0026 | -0.2749 |
| autumn_internationals_2025 | autumn_internationals | mixed | 0.9008 | 0.8948 | -0.0061 | -0.6742 |
| six_nations_2026 | six_nations | north | 0.8690 | 0.8603 | -0.0087 | -1.0020 |
| nations_championship_2026 | nations_championship | mixed | 0.8357 | 0.8237 | -0.0119 | -1.4292 |

### Per tournament family

| tournament | frozen | C5 | pct |
| --- | --- | --- | --- |
| rugby_championship | 0.9002 | 0.8817 | -2.0603 |
| british_irish_lions | 0.8549 | 0.8423 | -1.4786 |
| nations_championship | 0.8357 | 0.8237 | -1.4292 |
| rugby_world_cup | 0.8603 | 0.8490 | -1.3153 |
| autumn_internationals | 0.8884 | 0.8838 | -0.5102 |
| six_nations | 0.8656 | 0.8625 | -0.3510 |
| summer_internationals | 0.8977 | 0.8958 | -0.2044 |
| pacific_nations_cup | 0.9402 | 0.9387 | -0.1584 |

### Hemisphere

| cohort | frozen | C5 | pct |
| --- | --- | --- | --- |
| north | 0.8736 | 0.8685 | -0.5837 |
| south | 0.8983 | 0.8862 | -1.3561 |

Every tournament family improves and both hemispheres improve — no gate (c) or (d)
regression anywhere. The gains are largest exactly where P3's edge over v1 was
thinnest: the south (−1.36% vs −0.58% north) and the Rugby Championship (−2.06%).
The 6 regressing folds are concentrated in 2022–early 2024, where the training
history behind both components is thinnest.

## Reconstructed rubrics — always reported

| engine | rubric | mae | mae_calibrated | spearman | mean_capture | slates |
| --- | --- | --- | --- | --- | --- | --- |
| C5_cross_fitted | ncr | 6.0669 | 6.0677 | 0.5415 | 0.7278 | 98 |
| C5_cross_fitted | six_nations | 6.1771 | 6.1936 | 0.6299 | 0.7709 | 98 |
| C5_deployed | ncr | 6.0600 | 6.0632 | 0.5417 | 0.7290 | 98 |
| C5_deployed | six_nations | 6.1728 | 6.1914 | 0.6300 | 0.7715 | 98 |
| p3_event_50_frozen | ncr | 6.1727 | 6.1305 | 0.5367 | 0.7187 | 98 |
| p3_event_50_frozen | six_nations | 6.2767 | 6.2478 | 0.6287 | 0.7689 | 98 |

`C5_cross_fitted` is the honest out-of-sample rule (weights fitted without the fold
being scored); `C5_deployed` is the all-folds fit that would actually ship.
`mae_calibrated` applies a leave-one-fold-out affine recalibration, so it is not
in-sample either.

Paired-by-slate bootstrap, C5 cross-fitted minus frozen P3 (98 slates):

| metric | rubric | mean | p05 | p95 | excludes_0 |
| --- | --- | --- | --- | --- | --- |
| mae | ncr | -0.105772 | -0.137525 | -0.077016 | True |
| mae | six_nations | -0.099672 | -0.133377 | -0.068020 | True |
| mean_capture | ncr | 0.009092 | 0.004643 | 0.013665 | True |
| mean_capture | six_nations | 0.001937 | -0.004383 | 0.007443 | False |
| spearman | ncr | 0.004717 | 0.002038 | 0.007487 | True |
| spearman | six_nations | 0.001172 | -0.001655 | 0.003964 | False |

**The raw-score gain does not cost rubric MAE — it improves it.** NCR MAE falls
6.1727 → 6.0669 and Six Nations MAE 6.2767 → 6.1771, both with bootstrap CIs
excluding 0. Capture and Spearman improve on both rubrics; significantly on NCR,
directionally but not significantly on Six Nations.

This resolves the tension flagged at the outset. `empirical_event` had better rubric
MAE (ncr 6.0608, capture 0.7279) than P3 despite a far worse raw score (0.939),
because the rubrics are dominated by tries, assists, conversions and metres —
precisely the events where the empirical prior beats the GBDT. The frozen global 0.5
was mis-weighting those rare high-value events. Routing them to the empirical
component while leaving volume events with the GBDT captures nearly all of the
empirical engine's rubric quality (ncr MAE 6.0669 vs 6.0608, capture 0.7278 vs
0.7279) at a raw score of 0.879 instead of 0.939. The component-aware blend does get
both.

One honest caveat: calibrated MAE narrows the gap. The frozen P3 carried a systematic
scale bias that affine recalibration corrects (ncr 6.1727 → 6.1305), whereas C5 is
already well calibrated so recalibration does not help it (6.0669 → 6.0677).
Comparing calibrated to calibrated, C5 still wins — ncr 6.0677 vs 6.1305,
six_nations 6.1936 vs 6.2478 — but by roughly 0.06 rather than 0.10. Part of the
raw-MAE gain is calibration that a recalibrated deployment would already capture.

## Verification

- The precomputed loss table is a shortcut, so it was checked against the unmodified
  `event_metrics` pipeline on all 21 folds: **max absolute per-fold difference 0.0**
  for both the frozen rule and C5. Pipeline stable score for C5 is 0.8793658777,
  identical to the table.
- Gate (e): extended-event predictions under C5 are **bit-identical** to frozen P3
  (max absolute difference 0.0 across all folds and all 8 extended events), because
  every candidate leaves extended events at w=0.5. No extended-event loss can regress.
- Fold hygiene is inherited unchanged from `folds.py`: exact-kickoff cutoffs, training
  rows strictly before, evaluation fixtures excluded from training, targets masked in
  candidate features.
- Determinism: SEED=17 throughout; reruns reproduce.

## Reproduce

```bash
python -m model.unified.raw_benchmark.allrugby_run      # control, diagnostics, trials
python -m model.unified.raw_benchmark.allrugby_history  # C7 (row-varying weight)
python model/unified/raw_benchmark/allrugby_verify.py   # pipeline-vs-table + gate (e)
python -m model.unified.raw_benchmark.allrugby_ledger   # render this file
```
Component predictions, the fold cache and the loss table are derived artifacts,
rebuilt on demand from the frozen P3 blends and excluded from git.

