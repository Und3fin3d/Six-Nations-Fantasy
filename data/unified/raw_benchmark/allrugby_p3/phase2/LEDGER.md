# All-rugby hill-climb ledger — P3 phase 2

Phase 1 held the two components fixed and searched the blend rule; that family
terminated at C5 (shrunk per-target weight, cross-fitted **0.879366**). Phase 2
moves the search onto the components and onto axes orthogonal to the blend
weight. Same metric, same folds, same precommitted acceptance rule, with the
incumbent raised to C5. Seed 17. NCR GW4–7 was never read.

## Control

- Frozen `p3_event_50` reconstructed through the generalised code: **0.8864704365** vs 0.8864704365 (|Δ| 0.00e+00)
- Phase-1 C5 reconstructed: **0.8793658777** vs 0.8793658777 (|Δ| 1.23e-11)
- Folds: 21

The two-component pair grid is a special case of the simplex, so reproducing
both phase-1 numbers through the generalised code is the control for it.

## Acceptance rule (unchanged from phase 1, incumbent raised to C5)

- **(a)** stable_score improves on the incumbent
- **(b)** paired-by-fold bootstrap difference excludes 0
- **(c)** neither hemisphere regresses by more than 2%
- **(d)** no tournament-family stable regression greater than 2%
- **(e)** no extended-event loss regression greater than 5% vs frozen P3
- **(f)** every fitted parameter is cross-fitted leave-one-fold-out and checked
  again on a strict temporal split (fit 10 earliest folds, evaluate 11 latest)

## Trials

| candidate                     | components                                                                         | stable_score | vs_C5     | vs_frozen | boot_p05  | boot_p95  | temporal_delta | accepted |
| ----------------------------- | ---------------------------------------------------------------------------------- | ------------ | --------- | --------- | --------- | --------- | -------------- | -------- |
| E1_swap_t3_global             | v4 + empirical_t3_eb_off_ebk_head                                                  | 0.884505     | 0.005139  | -0.001965 | 0.000397  | 0.009891  | 0.009909       | False    |
| E2_swap_t3_shrunk             | v4 + empirical_t3_eb_off_ebk_head                                                  | 0.877462     | -0.001904 | -0.009009 | -0.004075 | 0.000424  | 0.004119       | False    |
| E3_three_empirical            | v4 + empirical + empirical_t3_eb_off_ebk_head                                      | 0.87301      | -0.006356 | -0.013461 | -0.007614 | -0.005028 | -0.023062      | True     |
| N1_naive_shrinkage            | v4 + empirical + naive                                                             | 0.875386     | -0.00398  | -0.011085 | -0.005668 | -0.002236 | -0.054265      | True     |
| N3_naive_capped               | v4 + empirical + naive                                                             | 0.875384     | -0.003982 | -0.011087 | -0.005669 | -0.002237 | -0.015443      | True     |
| N2_naive_plus_t3              | v4 + empirical_t3_eb_off_ebk_head + naive                                          | 0.875049     | -0.004317 | -0.011422 | -0.006446 | -0.001925 | -0.05159       | True     |
| E4_three_empirical_plus_naive | v4 + empirical + empirical_t3_eb_off_ebk_head + naive                              | 0.869656     | -0.00971  | -0.016814 | -0.011624 | -0.007434 | -0.0352        | True     |
| E5_four_empirical             | v4 + empirical + empirical_t3_eb_off_ebk_head + empirical_t4_no_signal_max         | 0.872858     | -0.006508 | -0.013613 | -0.008422 | -0.004524 | -0.024543      | True     |
| G1_started_weight             | v4 + empirical                                                                     | 0.878791     | -0.000575 | -0.00768  | -0.001347 | 0.00025   | 0.001104       | False    |
| G2_position_weight            | v4 + empirical                                                                     | 0.879175     | -0.000191 | -0.007296 | -0.000967 | 0.00059   | 0.001135       | False    |
| E6_all_components             | v4 + empirical + empirical_t3_eb_off_ebk_head + empirical_t4_no_signal_max + naive | 0.870826     | -0.00854  | -0.015645 | -0.010782 | -0.006163 | -0.025095      | True     |
| C9_calibrated_E4              | v4 + empirical + empirical_t3_eb_off_ebk_head + naive                              | 0.863402     | -0.015963 | -0.023068 | -0.019231 | -0.012896 | -0.010637      | True     |
| C10_calibrated_E4_wide        | v4 + empirical + empirical_t3_eb_off_ebk_head + naive                              | 0.863769     | -0.015597 | -0.022701 | -0.019442 | -0.012024 | -0.011007      | True     |

### `E1_swap_t3_global` — REJECTED

**Hypothesis.** The empirical component was hill-climbed separately to 0.904538 standalone (t3: EB-shrunk minutes, per-event EB shrinkage, median minutes head, no opponent multiplier). P3's blend was fitted against the OLD component, so simply substituting the better one at the frozen 0.5 should already move the blend.

Components: `v4`, `empirical_t3_eb_off_ebk_head`; 201 weight points

Cross-fitted stable score **0.884505** (+0.005139 vs C5, -0.001965 vs frozen P3). Paired bootstrap vs C5: mean +0.005139, 90% CI [+0.000397, +0.009891].

Temporal split: 0.893129 vs 0.883220 (Δ +0.009909, CI [+0.005619, +0.014489]).

Rejected because:

- (a) stable_score 0.884505 did not improve on 0.879366
- (b) bootstrap CI [0.000397, 0.009891] includes 0
- (d) tournament-family regression >2%: british_irish_lions
- (f) temporal split delta +0.009909 is not an improvement

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| six_nations           | 0.8625488831972475 | 0.8602028581673273 | -0.272  |
| summer_internationals | 0.8958380774256649 | 0.8948535486222282 | -0.1099 |
| autumn_internationals | 0.8838229519754811 | 0.8875041749652336 | 0.4165  |
| pacific_nations_cup   | 0.9386980234782767 | 0.9447412040478603 | 0.6438  |
| nations_championship  | 0.8237351715536874 | 0.8321820054746173 | 1.0254  |
| rugby_world_cup       | 0.8490328582554593 | 0.8612368998214627 | 1.4374  |
| rugby_championship    | 0.8816658262036159 | 0.8971305814903415 | 1.754   |
| british_irish_lions   | 0.8422904682466399 | 0.8602393753217313 | 2.131   |

### `E2_swap_t3_shrunk` — REJECTED

**Hypothesis.** Re-fit C5's shrunk per-target weight against the improved empirical component. If the component swap and the per-target routing are complementary the two gains should compose.

Components: `v4`, `empirical_t3_eb_off_ebk_head`; 201 weight points; shrinkage λ 0.1

Cross-fitted stable score **0.877462** (-0.001904 vs C5, -0.009009 vs frozen P3). Paired bootstrap vs C5: mean -0.001904, 90% CI [-0.004075, +0.000424].

Temporal split: 0.887339 vs 0.883220 (Δ +0.004119, CI [+0.000903, +0.007309]).

Rejected because:

- (b) bootstrap CI [-0.004075, 0.000424] includes 0
- (f) temporal split delta +0.004119 is not an improvement

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| six_nations           | 0.8625488831972475 | 0.856961572471134  | -0.6478 |
| rugby_championship    | 0.8816658262036159 | 0.8770698894379112 | -0.5213 |
| nations_championship  | 0.8237351715536874 | 0.8213752748368339 | -0.2865 |
| summer_internationals | 0.8958380774256649 | 0.8953050066276522 | -0.0595 |
| autumn_internationals | 0.8838229519754811 | 0.8838231592571516 | 0.0     |
| pacific_nations_cup   | 0.9386980234782767 | 0.9398424369783285 | 0.1219  |
| rugby_world_cup       | 0.8490328582554593 | 0.853292585754971  | 0.5017  |
| british_irish_lions   | 0.8422904682466399 | 0.8465679490910611 | 0.5078  |

### `E3_three_empirical` — ACCEPTED

**Hypothesis.** Keep BOTH empirical components. The frozen one and t3 differ mainly in minutes propagation, so a per-target simplex can take the frozen component where its noisier minutes happen to help and t3 elsewhere.

Components: `v4`, `empirical`, `empirical_t3_eb_off_ebk_head`; 861 weight points; shrinkage λ 0.05

Cross-fitted stable score **0.873010** (-0.006356 vs C5, -0.013461 vs frozen P3). Paired bootstrap vs C5: mean -0.006356, 90% CI [-0.007614, -0.005028].

Temporal split: 0.880635 vs 0.903697 (Δ -0.023062, CI [-0.027720, -0.018211]).

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| rugby_championship    | 0.8816658262036159 | 0.8728416504922826 | -1.0009 |
| rugby_world_cup       | 0.8490328582554593 | 0.8411448136850336 | -0.9291 |
| six_nations           | 0.8625488831972475 | 0.8555229589270986 | -0.8146 |
| autumn_internationals | 0.8838229519754811 | 0.8775485232682003 | -0.7099 |
| pacific_nations_cup   | 0.9386980234782767 | 0.9326202833659879 | -0.6475 |
| summer_internationals | 0.8958380774256649 | 0.8900849628014788 | -0.6422 |
| nations_championship  | 0.8237351715536874 | 0.8205266670784116 | -0.3895 |
| british_irish_lions   | 0.8422904682466399 | 0.8443283758246661 | 0.2419  |

### `N1_naive_shrinkage` — ACCEPTED

**Hypothesis.** Half the targets sit at or above 1.0 relative loss for BOTH components -- neither engine beats the training-only position x started stratum mean there. Adding that comparator as a third component lets the per-target fit shrink toward it exactly where the engines have no edge. Guard: this cannot be judged on the raw score alone, because the comparator is the metric's denominator and cannot rank players; the rubric report decides.

Components: `v4`, `empirical`, `naive`; 861 weight points; shrinkage λ 0.15

Cross-fitted stable score **0.875386** (-0.003980 vs C5, -0.011085 vs frozen P3). Paired bootstrap vs C5: mean -0.003980, 90% CI [-0.005668, -0.002236].

Temporal split: 0.885180 vs 0.939446 (Δ -0.054265, CI [-0.064689, -0.044582]).

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| pacific_nations_cup   | 0.9386980234782767 | 0.9255984125835043 | -1.3955 |
| autumn_internationals | 0.8838229519754811 | 0.8783842574504866 | -0.6154 |
| summer_internationals | 0.8958380774256649 | 0.8904085139582917 | -0.6061 |
| rugby_championship    | 0.8816658262036159 | 0.8775220154835481 | -0.47   |
| six_nations           | 0.8625488831972475 | 0.8605256577813278 | -0.2346 |
| rugby_world_cup       | 0.8490328582554593 | 0.848449958868786  | -0.0687 |
| nations_championship  | 0.8237351715536874 | 0.8232416050500598 | -0.0599 |
| british_irish_lions   | 0.8422904682466399 | 0.8507056704048805 | 0.9991  |

### `N3_naive_capped` — ACCEPTED

**Hypothesis.** N1 with the comparator share capped at 0.5 per target. A target driven fully to the stratum mean has no within-stratum ranking left, however good its raw loss looks; the cap keeps at least half the mass on a real engine. If N3 recovers most of N1's raw gain then the gain was shrinkage, not the metric's denominator being handed the answer.

Components: `v4`, `empirical`, `naive`; 651 weight points; shrinkage λ 0.15

Cross-fitted stable score **0.875384** (-0.003982 vs C5, -0.011087 vs frozen P3). Paired bootstrap vs C5: mean -0.003982, 90% CI [-0.005669, -0.002237].

Temporal split: 0.885196 vs 0.900639 (Δ -0.015443, CI [-0.021844, -0.008862]).

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| pacific_nations_cup   | 0.9386980234782767 | 0.9255984125835043 | -1.3955 |
| autumn_internationals | 0.8838229519754811 | 0.8783722497136299 | -0.6167 |
| summer_internationals | 0.8958380774256649 | 0.8904085139582917 | -0.6061 |
| rugby_championship    | 0.8816658262036159 | 0.8775220154835481 | -0.47   |
| six_nations           | 0.8625488831972475 | 0.8605256577813278 | -0.2346 |
| rugby_world_cup       | 0.8490328582554593 | 0.848449958868786  | -0.0687 |
| nations_championship  | 0.8237351715536874 | 0.8232416050500598 | -0.0599 |
| british_irish_lions   | 0.8422904682466399 | 0.8507056704048805 | 0.9991  |

### `N2_naive_plus_t3` — ACCEPTED

**Hypothesis.** N1's shrinkage and E2's better component are orthogonal; fit them together.

Components: `v4`, `empirical_t3_eb_off_ebk_head`, `naive`; 861 weight points; shrinkage λ 0.2

Cross-fitted stable score **0.875049** (-0.004317 vs C5, -0.011422 vs frozen P3). Paired bootstrap vs C5: mean -0.004317, 90% CI [-0.006446, -0.001925].

Temporal split: 0.885747 vs 0.937337 (Δ -0.051590, CI [-0.062136, -0.042007]).

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| pacific_nations_cup   | 0.9386980234782767 | 0.9278167901152692 | -1.1592 |
| six_nations           | 0.8625488831972475 | 0.8561978211576824 | -0.7363 |
| rugby_championship    | 0.8816658262036159 | 0.8765870156267075 | -0.576  |
| summer_internationals | 0.8958380774256649 | 0.8912235756880784 | -0.5151 |
| autumn_internationals | 0.8838229519754811 | 0.8800224052945559 | -0.43   |
| nations_championship  | 0.8237351715536874 | 0.8219179225012092 | -0.2206 |
| rugby_world_cup       | 0.8490328582554593 | 0.852662795568163  | 0.4275  |
| british_irish_lions   | 0.8422904682466399 | 0.8535074069890106 | 1.3317  |

### `E4_three_empirical_plus_naive` — ACCEPTED

**Hypothesis.** E3 (keep both empirical components) and N1 (shrink toward the stratum mean) were both accepted and are orthogonal: one adds a second real engine, the other adds a shrinkage direction. Fit all four together.

Components: `v4`, `empirical`, `empirical_t3_eb_off_ebk_head`, `naive`; 1771 weight points; shrinkage λ 0.05

Cross-fitted stable score **0.869656** (-0.009710 vs C5, -0.016814 vs frozen P3). Paired bootstrap vs C5: mean -0.009710, 90% CI [-0.011624, -0.007434].

Temporal split: 0.880271 vs 0.915471 (Δ -0.035200, CI [-0.044754, -0.025175]).

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| pacific_nations_cup   | 0.9386980234782767 | 0.9209875544714607 | -1.8867 |
| rugby_championship    | 0.8816658262036159 | 0.8685105509360445 | -1.4921 |
| autumn_internationals | 0.8838229519754811 | 0.8735570743705035 | -1.1615 |
| summer_internationals | 0.8958380774256649 | 0.885681169211523  | -1.1338 |
| six_nations           | 0.8625488831972475 | 0.8541381205463523 | -0.9751 |
| rugby_world_cup       | 0.8490328582554593 | 0.8412951232447514 | -0.9114 |
| nations_championship  | 0.8237351715536874 | 0.8187287020804156 | -0.6078 |
| british_irish_lions   | 0.8422904682466399 | 0.8526514184889754 | 1.2301  |

### `E5_four_empirical` — ACCEPTED

**Hypothesis.** t4 is the empirical hill-climb's best-scoring config, rejected there only because its bootstrap CI straddled its own incumbent. As a blend component it faces no such gate: the per-target simplex can take it where it helps.

Components: `v4`, `empirical`, `empirical_t3_eb_off_ebk_head`, `empirical_t4_no_signal_max`; 1771 weight points; shrinkage λ 0.05

Cross-fitted stable score **0.872858** (-0.006508 vs C5, -0.013613 vs frozen P3). Paired bootstrap vs C5: mean -0.006508, 90% CI [-0.008422, -0.004524].

Temporal split: 0.882923 vs 0.907466 (Δ -0.024543, CI [-0.028661, -0.020180]).

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| pacific_nations_cup   | 0.9386980234782767 | 0.926608948358505  | -1.2879 |
| rugby_championship    | 0.8816658262036159 | 0.8707650983664131 | -1.2364 |
| rugby_world_cup       | 0.8490328582554593 | 0.8398310937411542 | -1.0838 |
| autumn_internationals | 0.8838229519754811 | 0.8774582435110667 | -0.7201 |
| six_nations           | 0.8625488831972475 | 0.8574050341067082 | -0.5964 |
| summer_internationals | 0.8958380774256649 | 0.8911479075085704 | -0.5236 |
| nations_championship  | 0.8237351715536874 | 0.8205305588735369 | -0.389  |
| british_irish_lions   | 0.8422904682466399 | 0.8493837717376475 | 0.8421  |

### `G1_started_weight` — REJECTED

**Hypothesis.** The two engines disagree most about bench players: the empirical engine scales counts by a per-player minutes table while the GBDT models minutes from exposure features. Splitting the blend weight by starter/bench is one row attribute the phase-1 search never tried, and it stays competition-independent. Every benchmark loss is a mean of per-row terms, so group weights are additively separable and fitted exactly.

Components: `v4`, `empirical`; row groups: `bench`, `starter`; 201 weight points; shrinkage λ 0.3

Cross-fitted stable score **0.878791** (-0.000575 vs C5, -0.007680 vs frozen P3). Paired bootstrap vs C5: mean -0.000575, 90% CI [-0.001347, +0.000250].

Temporal split: 0.882725 vs 0.881622 (Δ +0.001104, CI [-0.000377, +0.002698]).

Rejected because:

- (b) bootstrap CI [-0.001347, 0.000250] includes 0
- (f) temporal split delta +0.001104 is not an improvement

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| british_irish_lions   | 0.8422904682466399 | 0.8392936468302653 | -0.3558 |
| rugby_world_cup       | 0.8490328582554593 | 0.846497915899937  | -0.2986 |
| rugby_championship    | 0.8816658262036159 | 0.8794839404449522 | -0.2475 |
| autumn_internationals | 0.8838229519754811 | 0.8830282382480404 | -0.0899 |
| summer_internationals | 0.8958380774256649 | 0.8953841102312968 | -0.0507 |
| nations_championship  | 0.8237351715536874 | 0.8236079839423288 | -0.0154 |
| six_nations           | 0.8625488831972475 | 0.8632429975522466 | 0.0805  |
| pacific_nations_cup   | 0.9386980234782767 | 0.9402188609493447 | 0.162   |

### `G2_position_weight` — REJECTED

**Hypothesis.** Event-type routing was phase 1's finding; position is the other structural axis. A hooker's rucks and a winger's metres are produced by different mechanisms, so the component that models them best may differ by position as well as by event.

Components: `v4`, `empirical`; row groups: `front_row`, `second_row`, `back_row`, `half_back`, `back`, `other`; 201 weight points; shrinkage λ 0.3

Cross-fitted stable score **0.879175** (-0.000191 vs C5, -0.007296 vs frozen P3). Paired bootstrap vs C5: mean -0.000191, 90% CI [-0.000967, +0.000590].

Temporal split: 0.882757 vs 0.881622 (Δ +0.001135, CI [+0.000121, +0.002157]).

Rejected because:

- (b) bootstrap CI [-0.000967, 0.000590] includes 0
- (f) temporal split delta +0.001135 is not an improvement

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| rugby_championship    | 0.8816658262036159 | 0.8798802793046071 | -0.2025 |
| british_irish_lions   | 0.8422904682466399 | 0.8406636024826722 | -0.1931 |
| nations_championship  | 0.8237351715536874 | 0.822566084627836  | -0.1419 |
| pacific_nations_cup   | 0.9386980234782767 | 0.9375772239284421 | -0.1194 |
| rugby_world_cup       | 0.8490328582554593 | 0.8491502849479208 | 0.0138  |
| autumn_internationals | 0.8838229519754811 | 0.8840263833982563 | 0.023   |
| summer_internationals | 0.8958380774256649 | 0.8962136206724116 | 0.0419  |
| six_nations           | 0.8625488831972475 | 0.8637359668479219 | 0.1376  |

### `E6_all_components` — ACCEPTED

**Hypothesis.** E4 (both empirical components plus stratum-mean shrinkage) and E5 (a fourth empirical variant) were both accepted. Give the per-target simplex every component at once and let it choose; a coarser 0.1 lattice keeps the parameter count honest at five components.

Components: `v4`, `empirical`, `empirical_t3_eb_off_ebk_head`, `empirical_t4_no_signal_max`, `naive`; 1001 weight points; shrinkage λ 0.0

Cross-fitted stable score **0.870826** (-0.008540 vs C5, -0.015645 vs frozen P3). Paired bootstrap vs C5: mean -0.008540, 90% CI [-0.010782, -0.006163].

Temporal split: 0.882496 vs 0.907591 (Δ -0.025095, CI [-0.029940, -0.020043]).

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| pacific_nations_cup   | 0.9386980234782767 | 0.9183109349350496 | -2.1718 |
| rugby_championship    | 0.8816658262036159 | 0.8685421045913675 | -1.4885 |
| rugby_world_cup       | 0.8490328582554593 | 0.8404000085615162 | -1.0168 |
| autumn_internationals | 0.8838229519754811 | 0.8749515333981256 | -1.0038 |
| summer_internationals | 0.8958380774256649 | 0.888260252901081  | -0.8459 |
| six_nations           | 0.8625488831972475 | 0.8567556090509229 | -0.6716 |
| nations_championship  | 0.8237351715536874 | 0.8239726214932892 | 0.0288  |
| british_irish_lions   | 0.8422904682466399 | 0.8505041883711776 | 0.9752  |

### `C9_calibrated_E4` — ACCEPTED

**Hypothesis.** Phase 1 found the frozen P3 carried a systematic scale bias that affine recalibration removed, and a mixture of two calibrated means is not itself unbiased under a Poisson deviance. One multiplicative scale per target is the smallest correction for that, and it is orthogonal to the blend weight. Both stages are cross-fitted inside the same outer loop.

Components: `v4`, `empirical`, `empirical_t3_eb_off_ebk_head`, `naive`; 1771 weight points; shrinkage λ 0.05

Cross-fitted stable score **0.863402** (-0.015963 vs C5, -0.023068 vs frozen P3). Paired bootstrap vs C5: mean -0.015963, 90% CI [-0.019231, -0.012896].

Temporal split: 0.870985 vs 0.881622 (Δ -0.010637, CI [-0.014076, -0.007298]).

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| pacific_nations_cup   | 0.9386980234782767 | 0.9066423634649576 | -3.4149 |
| british_irish_lions   | 0.8422904682466399 | 0.8195787412697215 | -2.6964 |
| rugby_championship    | 0.8816658262036159 | 0.8607900066408988 | -2.3678 |
| nations_championship  | 0.8237351715536874 | 0.8087484100131738 | -1.8194 |
| six_nations           | 0.8625488831972475 | 0.8502088379565326 | -1.4306 |
| summer_internationals | 0.8958380774256649 | 0.8833524681284176 | -1.3937 |
| autumn_internationals | 0.8838229519754811 | 0.8727560628755359 | -1.2522 |
| rugby_world_cup       | 0.8490328582554593 | 0.8439564771907854 | -0.5979 |

### `C10_calibrated_E4_wide` — ACCEPTED

**Hypothesis.** C9 drove six targets onto the 0.70 floor of its scale grid -- drop goals, metres, missed penalty goals, red cards -- so the grid, not the data, was choosing their scale. Widen it to 0.40-1.60 and let them settle. If the extra freedom only buys raw score while the rubric MAE worsens, the shrinkage has stopped being calibration and started being deflation.

Components: `v4`, `empirical`, `empirical_t3_eb_off_ebk_head`, `naive`; 1771 weight points; shrinkage λ 0.05

Cross-fitted stable score **0.863769** (-0.015597 vs C5, -0.022701 vs frozen P3). Paired bootstrap vs C5: mean -0.015597, 90% CI [-0.019442, -0.012024].

Temporal split: 0.870615 vs 0.881622 (Δ -0.011007, CI [-0.014699, -0.007250]).

Per tournament family (candidate vs incumbent, % change):

| tournament            | base               | candidate          | pct     |
| --------------------- | ------------------ | ------------------ | ------- |
| pacific_nations_cup   | 0.9386980234782767 | 0.9054945880256536 | -3.5372 |
| british_irish_lions   | 0.8422904682466399 | 0.816171191557897  | -3.101  |
| rugby_championship    | 0.8816658262036159 | 0.8608234131324386 | -2.364  |
| nations_championship  | 0.8237351715536874 | 0.8093653884164554 | -1.7445 |
| summer_internationals | 0.8958380774256649 | 0.8838831954000479 | -1.3345 |
| six_nations           | 0.8625488831972475 | 0.8514976967814535 | -1.2812 |
| autumn_internationals | 0.8838229519754811 | 0.8733886933806237 | -1.1806 |
| rugby_world_cup       | 0.8490328582554593 | 0.8461415074000233 | -0.3405 |

## Phase-2 incumbent: `C9_calibrated_E4`

- Cross-fitted stable score **0.863402** vs C5 0.879366 and frozen P3 0.886470
- Improvement **2.602%** on the frozen blend, **1.815%** on C5
- Paired-by-fold bootstrap vs C5: mean -0.015963, 90% CI [-0.019231, -0.012896]
- Temporal split: Δ -0.010637, CI [-0.014076, -0.007298]

Deployed weights (all-folds fit; the headline number is cross-fitted):

| target                  | v4   | empirical | empirical_t3_eb_off_ebk_head | naive | scale |
| ----------------------- | ---- | --------- | ---------------------------- | ----- | ----- |
| metres                  | 0.0  | 0.45      | 0.55                         | 0.0   | 0.7   |
| passes                  | 0.5  | 0.2       | 0.3                          | 0.0   | 1.04  |
| rucks_won               | 0.5  | 0.05      | 0.45                         | 0.0   | 1.03  |
| runs                    | 0.45 | 0.15      | 0.4                          | 0.0   | 1.01  |
| drop_goal_missed        | 0.65 | 0.35      | 0.0                          | 0.0   | 0.7   |
| drop_goals_converted    | 0.85 | 0.0       | 0.15                         | 0.0   | 0.7   |
| minutes                 | 0.0  | 0.0       | 0.95                         | 0.05  | 1.0   |
| penalty_goals           | 0.4  | 0.0       | 0.55                         | 0.05  | 0.87  |
| conversion_goals        | 0.15 | 0.8       | 0.0                          | 0.05  | 1.02  |
| bad_passes              | 0.45 | 0.2       | 0.25                         | 0.1   | 1.06  |
| missed_penalty_goals    | 0.55 | 0.0       | 0.35                         | 0.1   | 0.73  |
| red_cards               | 0.85 | 0.05      | 0.0                          | 0.1   | 0.75  |
| yellow_cards            | 0.25 | 0.0       | 0.6                          | 0.15  | 1.01  |
| offload                 | 0.05 | 0.3       | 0.5                          | 0.15  | 1.0   |
| defenders_beaten        | 0.1  | 0.4       | 0.35                         | 0.15  | 1.01  |
| missed_tackles          | 0.25 | 0.45      | 0.1                          | 0.2   | 1.02  |
| rucks_lost              | 0.55 | 0.25      | 0.0                          | 0.2   | 1.01  |
| tackles                 | 0.45 | 0.3       | 0.0                          | 0.25  | 1.03  |
| clean_breaks            | 0.1  | 0.6       | 0.0                          | 0.3   | 0.98  |
| tries                   | 0.05 | 0.65      | 0.0                          | 0.3   | 1.02  |
| missed_conversion_goals | 0.1  | 0.55      | 0.0                          | 0.35  | 0.93  |
| turnovers_conceded      | 0.05 | 0.0       | 0.55                         | 0.4   | 0.99  |
| try_assists             | 0.0  | 0.6       | 0.0                          | 0.4   | 1.08  |
| penalties_conceded      | 0.05 | 0.0       | 0.5                          | 0.45  | 0.92  |

## Champion decision — raw score versus rubric

| candidate                     | stable_score | ncr_mae | ncr_mae_cal | six_nations_mae | six_nations_mae_cal |
| ----------------------------- | ------------ | ------- | ----------- | --------------- | ------------------- |
| C9_calibrated_E4              | 0.863402     | 6.0737  | 6.066       | 6.0997          | 6.1907              |
| C10_calibrated_E4_wide        | 0.863769     | 6.0737  | 6.066       | 6.0932          | 6.1944              |
| E4_three_empirical_plus_naive | 0.869656     | 6.0254  | 6.0611      | 6.1075          | 6.1707              |

The raw-score optimum is **`C9_calibrated_E4`**, but the standing rule is that a raw gain which costs reconstructed rubric MAE is not a gain worth shipping — and this is exactly that case. Per-target calibration buys raw score by deflating the sparse, high-value targets, which is right under a Poisson deviance and wrong for fantasy points.

**`E4_three_empirical_plus_naive` is the champion.** It is better than `C9_calibrated_E4` on Nations Championship rubric MAE by 0.0484 with a paired-by-slate bootstrap CI excluding 0, better on Six Nations Spearman, and level on Six Nations MAE and capture. Widening the calibration grid (C10) does not rescue it, which closes the calibration axis.

`C9_calibrated_E4` is kept in the ledger as the raw-score optimum, not deleted: it is the right starting point if the metric ever becomes the deliverable.

