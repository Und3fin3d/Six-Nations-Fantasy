# All-rugby hill-climb — Six Nations champion components

Objective: minimise the competition-balanced **stable raw score** of the
Historical Raw Benchmark v1 (lower is better; 1.0 = no better than the
training-only position x started x competition_level naive comparator).

Seed 17. The frozen `v1` ledger is untouched; every run here writes to
`data/unified/raw_benchmark/allrugby_sixnations/`.

## Control

Frozen champion partial stable score: **1.114404**
Reproduced by this harness: **1.114404** (delta +0.0e+00)

## Result

Final incumbent: **`c25_min100_shrink960`**

| basis | champion frozen | champion final | v1 | p3_event_50 |
| --- | --- | --- | --- | --- |
| like-for-like (13 targets) | 1.114404 | 1.031685 | 0.869291 | 0.847940 |
| full (24 targets) | n/a (13-target coverage) | 0.972910 | 0.874193 | 0.857527 |
| coverage (stable + minutes) | 13 | 24 | 24 | 24 |

Gap to v1 on the like-for-like basis: 0.245114 -> 0.162394 (**33.7% closed**).

## Reconstructed-rubric MAE (stable tier)

| trial | rubric | mae | spearman | mean_capture | slates |
| --- | --- | --- | --- | --- | --- |
| t0_baseline | six_nations | 6.136118 | 0.6571 | 0.8013 | 10 |
| c25_min100_shrink960 | ncr | 6.256319 | 0.5554 | 0.7667 | 10 |
| c25_min100_shrink960 | six_nations | 6.167289 | 0.6509 | 0.8046 | 10 |

The frozen champion produces **no NCR rubric rows at all**: it lacks heads for
NCR-scored events, so every row fails the rubric's completeness mask. Extending
coverage makes the champion NCR-scorable for the first time.

## Per-target relative loss (final)

| target | relative_loss |
| --- | --- |
| drop_goals_converted | 2.3202 |
| drop_goal_missed | 1.1858 |
| penalties_conceded | 1.1180 |
| try_assists | 1.0689 |
| offload | 1.0675 |
| tries | 1.0441 |
| yellow_cards | 1.0426 |
| rucks_lost | 1.0041 |
| turnovers_conceded | 1.0001 |
| minutes | 0.9976 |
| red_cards | 0.9940 |
| clean_breaks | 0.9733 |
| defenders_beaten | 0.9480 |
| tackles | 0.9412 |
| missed_tackles | 0.9381 |
| bad_passes | 0.9268 |
| runs | 0.8296 |
| missed_penalty_goals | 0.8157 |
| missed_conversion_goals | 0.8016 |
| metres | 0.7811 |
| rucks_won | 0.7666 |
| passes | 0.6961 |
| penalty_goals | 0.6595 |
| conversion_goals | 0.4291 |

## Per-fold / cohort / tournament

| fold | full stable score |
| --- | --- |
| six_nations_2025 | 0.881909 |
| six_nations_2026 | 1.063911 |

north cohort: 1.031685; south cohort: not evaluable (both folds are Six Nations, so the north/south and tournament-family guards cannot bind on this fold set).

## Trial ledger

### Step 1 — incumbent `t0_baseline`

| candidate | l4l score | full score | cov | bootstrap p05/p95 | verdict | reasons |
| --- | --- | --- | --- | --- | --- | --- |
| c1_calib_global | 1.101118 | 1.101118 | 13 | -0.01363/-0.01294 | ACCEPT | - |
| c2_calib_global_sparse_only | 1.112295 | 1.112295 | 13 | -0.00427/+0.00005 | REJECT | paired-by-fold bootstrap does not exclude 0 (p05=-0.004266, p95=0.000048) |
| c3_calib_position | 1.111578 | 1.111578 | 13 | -0.00410/-0.00155 | ACCEPT | - |
| c4_revive_dead_heads | 0.956272 | 0.956272 | 13 | -0.34421/+0.02794 | REJECT | paired-by-fold bootstrap does not exclude 0 (p05=-0.344209, p95=0.027943) |

Accepted: **`c1_calib_global`**

### Step 2 — incumbent `c1_calib_global`

| candidate | l4l score | full score | cov | bootstrap p05/p95 | verdict | reasons |
| --- | --- | --- | --- | --- | --- | --- |
| c6_revive_red_only | 1.047228 | 1.047228 | 13 | -0.05579/-0.05199 | ACCEPT | - |
| c7_revive_drop_only | 0.996976 | 0.996976 | 13 | -0.28842/+0.08014 | REJECT | paired-by-fold bootstrap does not exclude 0 (p05=-0.288421, p95=0.080137) |
| c8_revive_both | 0.943086 | 0.943086 | 13 | -0.34421/+0.02814 | REJECT | paired-by-fold bootstrap does not exclude 0 (p05=-0.344208, p95=0.028145) |
| c9_calib_position_plus_revive | 0.952588 | 0.952588 | 13 | -0.33573/+0.03867 | REJECT | paired-by-fold bootstrap does not exclude 0 (p05=-0.335727, p95=0.038667) |

Accepted: **`c6_revive_red_only`**

### Step 3 — incumbent `c6_revive_red_only`

| candidate | l4l score | full score | cov | bootstrap p05/p95 | verdict | reasons |
| --- | --- | --- | --- | --- | --- | --- |
| c11_coverage | 1.047228 | 0.983415 | 24 | -0.14236/-0.03959 | ACCEPT | - |
| c12_coverage_shrink120 | 1.052956 | 0.987486 | 24 | -0.14134/-0.03771 | REJECT | like-for-like stable score regressed while widening coverage (1.052956 vs 1.047228) |
| c13_coverage_shrink480 | 1.041792 | 0.980355 | 24 | -0.14332/-0.04045 | ACCEPT | - |
| c14_minutes_blend025 | 1.045273 | 1.045273 | 13 | -0.00217/-0.00174 | ACCEPT | - |
| c15_minutes_blend050 | 1.043702 | 1.043702 | 13 | -0.00402/-0.00303 | ACCEPT | - |
| c16_minutes_blend075 | 1.042667 | 1.042667 | 13 | -0.00541/-0.00371 | ACCEPT | - |
| c17_minutes_blend100 | 1.042054 | 1.042054 | 13 | -0.00644/-0.00391 | ACCEPT | - |

Accepted: **`c13_coverage_shrink480`**

### Step 4 — incumbent `c13_coverage_shrink480`

| candidate | l4l score | full score | cov | bootstrap p05/p95 | verdict | reasons |
| --- | --- | --- | --- | --- | --- | --- |
| c18_cov_min050 | 1.038268 | 0.975593 | 24 | -0.00401/-0.00303 | ACCEPT | - |
| c19_cov_min100 | 1.036623 | 0.973957 | 24 | -0.00643/-0.00391 | ACCEPT | - |
| c20_cov_min075 | 1.037235 | 0.974406 | 24 | -0.00541/-0.00371 | ACCEPT | - |
| c21_cov_position_calib | 1.052226 | 0.986751 | 24 | +0.00854/+0.01233 | REJECT | like-for-like stable score did not improve (1.052226 vs 1.041792); full stable score did not improve; paired-by-fold bootstrap does not exclude 0 (p05=0.008538, p95=0.012331) |
| c22_cov_revive_both | 0.937788 | 0.924019 | 24 | -0.28795/+0.07994 | REJECT | paired-by-fold bootstrap does not exclude 0 (p05=-0.287950, p95=0.079942) |
| c23_cov_shrink960 | 1.036845 | 0.979264 | 24 | -0.00561/-0.00428 | ACCEPT | - |
| c24_cov_calib_shrink5 | 1.041645 | 0.980275 | 24 | -0.00031/+0.00001 | REJECT | paired-by-fold bootstrap does not exclude 0 (p05=-0.000306, p95=0.000012) |

Accepted: **`c19_cov_min100`**

### Step 5 — incumbent `c19_cov_min100`

| candidate | l4l score | full score | cov | bootstrap p05/p95 | verdict | reasons |
| --- | --- | --- | --- | --- | --- | --- |
| c25_min100_shrink960 | 1.031685 | 0.972910 | 24 | -0.00560/-0.00428 | ACCEPT | - |
| c26_min100_shrink240 | 1.042054 | 0.977003 | 24 | +0.00475/+0.00611 | REJECT | like-for-like stable score did not improve (1.042054 vs 1.036623); full stable score did not improve; paired-by-fold bootstrap does not exclude 0 (p05=0.004747, p95=0.006115) |
| c27_min100_nocalib | 1.049950 | 0.981213 | 24 | +0.01294/+0.01372 | REJECT | like-for-like stable score did not improve (1.049950 vs 1.036623); full stable score did not improve; paired-by-fold bootstrap does not exclude 0 (p05=0.012939, p95=0.013715) |
| c28_min100_revive_both | 0.931180 | 0.916843 | 24 | -0.28838/+0.07750 | REJECT | paired-by-fold bootstrap does not exclude 0 (p05=-0.288380, p95=0.077495) |

Accepted: **`c25_min100_shrink960`**

### Step 6 — incumbent `c25_min100_shrink960`

| candidate | l4l score | full score | cov | bootstrap p05/p95 | verdict | reasons |
| --- | --- | --- | --- | --- | --- | --- |
| c29_shrink1920 | 1.027483 | 0.975147 | 24 | -0.00480/-0.00360 | REJECT | full stable score did not improve |
| c30_shrink3840 | 1.024226 | 0.981692 | 24 | -0.00855/-0.00636 | REJECT | full stable score did not improve |

No candidate accepted — hill climb converged.

## Hypotheses

| trial | hypothesis |
| --- | --- |
| c10_extra_heads_coverage | Exploratory: add heads for the 11 unmodelled stable events. |
| c11_coverage | Extend coverage 13 -> 24 targets with per-minute rate heads fitted on the all-rugby strictly-prior training frame. |
| c12_coverage_shrink120 | Lighter pooling for the added heads. |
| c13_coverage_shrink480 | Heavier pooling for the added heads. |
| c14_minutes_blend025 | The champion's minutes model was fitted on 6N rotation; blend it toward a strictly-prior stratum mean. |
| c15_minutes_blend050 | Half-weight the champion minutes head. |
| c16_minutes_blend075 | Quarter-weight the champion minutes head. |
| c17_minutes_blend100 | Discard the champion minutes head entirely. |
| c18_cov_min050 | Coverage plus half-weighted champion minutes. |
| c19_cov_min100 | Coverage plus fully replaced minutes. |
| c1_calib_global | Experiment 18's volume deflation is load-bearing for XV selection but hurts point-accurate raw prediction; rescale each head to the strictly-prior observed level. |
| c20_cov_min075 | Coverage plus quarter-weighted champion minutes. |
| c21_cov_position_calib | Coverage plus per-position calibration. |
| c22_cov_revive_both | Coverage plus both dead heads revived. |
| c23_cov_shrink960 | Coverage with heavier pooling still. |
| c24_cov_calib_shrink5 | Stronger shrinkage on the calibration ratio itself. |
| c25_min100_shrink960 | Best minutes setting with heavier head pooling. |
| c26_min100_shrink240 | Best minutes setting with lighter head pooling. |
| c27_min100_nocalib | Ablation: is the deflation correction still carrying weight? |
| c28_min100_revive_both | Best config plus drop-goal revival. |
| c29_shrink1920 | Push pooling further. |
| c2_calib_global_sparse_only | Restrict the deflation correction to the sparse-count heads. |
| c30_shrink3840 | Push pooling further still. |
| c3_calib_position | Deflation is position-specific, so calibrate per position. |
| c4_revive_dead_heads | drop_goals_converted and red_cards heads emit identically zero; replace them with a shrunk per-minute rate x minutes_hat. |
| c5_revive_plus_calib | Exploratory: dead-head revival stacked on global calibration. |
| c6_revive_red_only | Revive only red_cards, whose occurrences are present in both folds. |
| c7_revive_drop_only | Revive only drop_goals_converted, absent from the 2025 fold. |
| c8_revive_both | Revive both dead heads. |
| c9_calib_position_plus_revive | Position calibration stacked on full revival. |
| t0_baseline | Control: reproduce the frozen champion partial score exactly. |
