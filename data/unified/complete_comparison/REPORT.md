# Complete-pool and all-cache model comparison

P3 robust native remains the strongest shared raw-stat research reference among the four evaluated candidates. Its average stable-event loss is 0.80% below native 50/50 and 0.54% below native event-weighted, with exploratory paired intervals favouring robust. Fantasy selection remains competition-dependent: robust leads observed NCR team points, while the empirical baseline leads the fully scored Six Nations totals. This evidence does not establish universal deployment superiority.

## What was repaired

Each modern Six Nations round now starts with all six 23-player teamsheets. The 1,380 candidate appearances include the 38 appearances that the earlier label-based pool omitted. A missing score no longer removes a player from consideration.

Eight modern Brex scores were recovered by fixing the official name match. Twelve unused substitutes receive an explicitly identified zero only when the cached record shows no playing time, start, recorded event or card. This leaves 1,350 workbook scores, 12 inferred unused-bench zeros and 18 unknown played-match scores. Thus 1,362 of 1,380 appearances have usable outcomes. Unknown outcomes remain missing; they never become arbitrary zeros. A selected squad with an unknown outcome cannot receive a reported total.

The shared matcher also repairs the Brex alias in 2023 and separates ambiguous Williams identities using uniquely matching minutes. Three NCR player mappings now point to the correct cached identities. Bench roles now use an earlier recorded starting role, with a strict three-hour availability lag. This corrects the previous assumption that a substitute jersey number defines playing position. Missing role history remains Unknown.

## Data used

| Source or comparison | Coverage | Use |
| --- | ---: | --- |
| Permanent match cache | 3,898 files; 3,892 have completed teamsheets | Exact timestamps, both teams, player statistics and source availability |
| Complete research store | 179,295 distinct player-match-team rows | Club and international history, filtered before every model cutoff |
| Raw international evaluation | 17,216 appearances; 374 fixtures; 22 tournament-season blocks | Event prediction comparison, 2022–2026 |
| Modern Six Nations | 1,380 candidates; 1,362 usable scores; 10 rounds | Fantasy error and complete-pool squad diagnostics |
| NCR 2026 | 758 candidate observations; 3 rounds | Fantasy error and budget-constrained squad diagnostics |
| RugbyPass competition statistics | 8,159 rows | Historical aggregate priors, filtered by season availability |
| World Rugby rankings | 24,547 rows | Historical team-strength inputs |

These counts overlap and must not be added together. The cache scan found 3,131 player rows absent from the current canonical CSV population, including 15 whole Pacific Nations Cup fixtures. Moving the source cutoff from 19 to 22 September also restores 552 club appearances. Every late club match is after the evaluation cutoffs and therefore cannot enter the fitted models.

The six cache files with empty teamsheets cannot provide player observations. Their identities and every included fixture are recorded in [cache_inventory.csv](cache_inventory.csv). Direct enumeration confirmed exact agreement between all 179,295 eligible cached player keys and the prepared store. The added identity fields, opponents, scores, event values and availability were independently checked.

The raw blocks cover Six Nations, summer internationals, Rugby Championship, autumn internationals, Pacific Nations Cup, World Cup, British & Irish Lions, and Nations Championship. The [frozen study](study_manifest.json) lists every evaluation fixture. Each block freezes its training and player history before its first fixture, using the existing three-hour result-availability convention. Official models use their recorded round locks. All models use the same candidate pool within each comparison.

## Raw-stat results

| Candidate | Equal-block stable-event relative loss |
| --- | ---: |
| P3 robust native | **0.875623** |
| P3 native event-weighted | 0.880415 |
| P3 native 50/50 | 0.882717 |
| Empirical event model | 0.954117 |

P3 robust's loss is 0.80% lower than native 50/50, 0.54% lower than native event-weighted, and 8.23% lower than the empirical event model. These percentages compare the pooled relative-loss values; they are not percentage-point improvements in fantasy accuracy.

| Robust minus comparator | Mean loss difference | Exploratory paired 95% interval | Tournament-period wins |
| --- | ---: | ---: | ---: |
| Native 50/50 | −0.007094 | [−0.011771, −0.003007] | 16 / 22 |
| Native event-weighted | −0.004792 | [−0.006504, −0.003168] | 19 / 22 |
| Empirical event model | −0.078494 | [−0.090182, −0.066820] | 22 / 22 |

Every leave-one-block-out mean still favours robust against each comparator. Against native 50/50, robust wins seven of eight competition-family averages and loses the 2023 World Cup. Its other losing periods are Six Nations 2022 and 2024, summer internationals 2022 and 2025, and autumn internationals 2022. Against the weighted variant, it loses Rugby Championship 2023, autumn internationals 2024 and Nations Championship 2026. The last difference is very small: +0.0000149 relative loss.

The improvement is not explained solely by adding opponents outside the twelve NCR nations. Within those twelve nations, robust beats native 50/50 by −0.005011, with a paired 95% interval of [−0.009816, −0.000571]. For the other nations, the mean also favours robust, but that comparison's interval includes zero. This population split is a post-fit diagnostic using the same all-player event normalisation; it did not select or refit a model.

The bench comparison is clearer than the starter comparison. Robust beats native 50/50 on bench event loss in 21 of 22 blocks, with an interval of [−0.015020, −0.007517]. The starter, low-history and northern-nation intervals against native 50/50 include zero. Position-specific results also vary. No model wins every relevant subset.

The frozen raw candidates contain 648 Unknown-role appearances across 20 blocks, all substitutes without an available prior start at that block's cutoff. These appearances remain included. There are no Unknown roles in modern Six Nations 2025–2026 candidate pools. The [role coverage table](frozen_role_coverage.csv) distinguishes this from the 588 unknown roles that a globally prepared evaluation subset would have shown.

Minutes remain less conclusive. Robust improves on the empirical event model, but its minute-loss interval against the other P3 variants crosses zero: [−0.010887, +0.001654]. The main stable-event result therefore does not establish a separate minutes advantage. These loss measures assess point forecasts; they do not validate the models' predictive intervals or joint event simulations.

See [raw_blocks.csv](results/raw_blocks.csv), [raw_families.csv](results/raw_families.csv), [raw_cohort_intervals.csv](results/raw_cohort_intervals.csv) and [raw_population_intervals.csv](raw_population_intervals.csv) for the complete comparisons.

## Official fantasy results

Lower player-score MAE and higher team points are better. Each cell shows MAE / team points. MAE is the mean of round-level MAEs over known labels. Team totals require every selected outcome to be known.

| Candidate | NCR 2026, 3 rounds | Six Nations 2025, 5 rounds | Six Nations 2026, 5 rounds |
| --- | ---: | ---: | ---: |
| Empirical fantasy baseline | 8.401 / 1,578 | 7.168 / 2,410 | 7.351 / 2,552 |
| P3 robust native | 8.370 / 1,739 | 7.243 / 2,356 | 7.244 / 2,496 |
| P3 native 50/50 | 8.404 / 1,682 | 7.268 / 2,290 | 7.311 / unknown |
| P3 native event-weighted | 8.354 / 1,562 | 7.322 / 2,236 | 7.317 / 2,513 |
| Tournament-frozen native control | 8.404 / 1,685 | 7.185 / 2,361 | 7.272 / unknown |

P3 robust has the highest observed NCR team total. The empirical baseline has the highest fully observed Six Nations total in both seasons. P3 robust has the lowest Six Nations 2026 player-score error, while the weighted variant has the lowest NCR error. Lower average prediction error therefore does not guarantee better selected teams.

The native 50/50 and frozen controls both select Jamie Dobie in Six Nations 2026 round one. His official score is missing. Their season totals are respectively **2,450 + Dobie's score** and **2,380 + Dobie's score**. Dobie would need 102 or 172 points respectively to tie the empirical baseline. These are exact break-even calculations, not estimates of his score. The [missing-score sensitivity table](official_missing_score_sensitivity.csv) also compares the other fully scored candidates.

For P3 robust versus the empirical baseline, the estimated NCR gain is 53.7 team points per round, with an exploratory 95% paired interval from −31 to +191. Across the ten Six Nations rounds, the difference is −11.0 points per round, with an interval from −41.6 to +15.3. Both intervals include zero. The available rounds do not establish a reliable advantage over the empirical baseline in either competition.

One narrower team-point comparison now excludes zero: P3 robust beats native 50/50 by 10.0 points per round across their nine mutually scorable Six Nations rounds, with a 95% interval from +1.2 to +21.6. This excludes the round with Dobie's missing score and remains exploratory. It does not establish superiority over the empirical baseline or over all ten rounds.

See [official_seasons.csv](results/official_seasons.csv) for full precision and [official_paired_intervals.csv](results/official_paired_intervals.csv) for each comparison and paired denominator.

## How to read the evidence

The raw comparison uses the existing empirical event model and three P3 candidates: native 50/50, native event-weighted, and native robust. The official comparison also retains a tournament-frozen control. The empirical fantasy baseline differs from the empirical raw-event model. No model weights or hyperparameters were selected from this run.

The primary raw measure normalises each of 23 stable event losses against a simple training-only comparator. It weights events equally within each tournament-season block, then weights the 22 blocks equally. Lower values are better. Minutes and newly recorded events are separate diagnostics. Missing event observations and targets without any earlier training support are explicitly excluded from their relevant paired metric, with counts retained in the support ledger.

Paired 95% intervals use 10,000 tournament-block resamples with seed 20260922. Competition-family results, losing blocks, position, starter/bench, low-history and Unknown-role diagnostics accompany the pooled score. These are exploratory intervals. Blocks share teams and training history, and existing weights were developed through historical research. Repeatedly examined historical outcomes do not become independent prospective evidence after a broader rerun.

Official squad-point intervals use the much smaller set of observed rounds. An interval that contains zero allows both an improvement and a deterioration. It does not prove that two models are equivalent. Six Nations lacks archived prices, so its squads remain price-free teamsheet diagnostics. NCR retains the documented budget and lineup limitations, including retrospective inputs for GW3.

## What still limits confidence

Eighteen played Six Nations scores lack a trustworthy official label in the available archive. Public aggregate values were inspected but disagreed with many known workbook scores, so they were not substituted. Their discrepancies are recorded in [external_source_conflicts.csv](external_source_conflicts.csv). This limits player-error coverage even though the candidate pools are complete.

Older official scores use different game rubrics. They are not pooled with modern fantasy points. Club matches improve training history but do not supply independent international fantasy outcomes. NCR GW4–7 occur in November 2026 and have no completed outcomes at this study date.

A stronger deployment decision requires archived lock-time prices, eligibility and roles, plus predictions frozen before future rounds. Those rounds should compare realised legal-team points under the same optimiser, with a minimum useful gain and acceptable regressions specified before results arrive. Existing historical results remain regression checks. The broader raw study can establish consistency across the available archive; it cannot establish universal fantasy superiority.

## Reproducibility and verification

The all-cache workflow completed all 35 jobs successfully in [run 35737817588](https://github.com/Und3fin3d/Six-Nations-Fantasy/actions/runs/35737817588), from source `3d231176050e1c5692497dfb11db13ef7f5b2dc4`. It contains 13 official-slate jobs and 22 raw-block jobs. The numerical runtime, model source, frozen weights, input files and all cache hashes are recorded. The prepared store SHA-256 is `2b70331c0cf215def67831cd27c987362e2ec8330b2afb41c8d204f48ef17313`.

The earlier run 35732721254 used the restricted CSV population. It is a diagnostic revision and is not the final all-cache result. The label-only run 35731009242 was stopped after the role defect was identified. Neither supplies the final model ranking.

The full existing local suite currently reports 120 passes and one stale fixed-count assertion: the audit now finds 2,164 usable labels rather than 2,144. Updating that assertion requires the user's explicit test-edit approval. No test has been created, weakened, skipped or modified. All 51 focused existing history, raw benchmark and V4 tests pass. The complexity check reports the existing parse_cache_fixture and rolling runner findings; it reports no new helper complexity finding. Separate standards and specification reviews are recorded in [REVIEW.md](REVIEW.md).

Independent arithmetic checks reproduced all 65 official model-round records and all 39 saved raw-to-fantasy score files. They confirmed both deliberately unknown squad totals. The maximum raw-to-fantasy difference was 1.07e-14 points. All 36,400 raw model/target/cohort rows across 22 blocks matched direct loss, naive-denominator, MAE and observation-count calculations. The maximum event-loss difference was 2.14e-14; fixture absolute-error sums differed by at most 1.14e-13. The 19 primary raw and official paired summaries were also recalculated independently.

The original prediction packages are preserved in three [evidence bundles](evidence/manifest.json): official, raw 2022–2024 and raw 2025–2026. Each contains the unchanged workflow-produced evidence.tar.gz files, including predictions, candidate roles, input manifests, source commit, environment and test logs. Model binaries and the reconstructible prepared CSV are excluded. The outer bundles and every inner package have SHA-256 hashes. Extract the outer bundles into one directory, then extract each inner package into its containing job directory. Run the aggregation command in [PROTOCOL.md](PROTOCOL.md) against that directory. summary.json records the aggregation-source and frozen-study hashes. Unknown totals are JSON null rather than non-standard NaN values.

An independent replay extracted all 35 packages from the three archived bundles, verified every outer and inner hash, and regenerated all 11 summary files byte for byte. Final standards and specification reviews found no remaining supported finding in the result delta. The pending test-count approval is the remaining completion blocker.
