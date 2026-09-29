# Fantasy model methodology and research workflow review

Review date: 26 September 2026. Source: merged PR #24, `93f9670f38ed1699c0029192ac38227392b329fb`.

## Decision

Keep `p3_robust_native` as the shared raw-event research reference. The available evidence does not justify replacing both competition incumbents. This review found a scoring defect that changes the NCR model ranking, and concentrated historical differences in captain and super-sub decisions. Resolve scoring and evaluation contracts before further model search.

AI can complete source audits, diagnostic calculations and independent experiment implementations in parallel. The previous two-week calendar had no measured basis. The existing 35-job comparison completed in 24 minutes 53 seconds on its GitHub runners. Research should advance when its inputs and decision criteria are ready. Independent confirmation must still wait for new matches and forecasts recorded before their locks.

The work completed here includes a fresh archive replay, squad decomposition, calibration diagnostics, 65 forecast-interpolation selections and an NCR scoring intervention. These are retrospective analyses. No model was refitted, production route changed, rugby API quota used, permanent cache modified, or test created or changed.

## 1. The evidence was verified against the merged source

The task checkout initially contained `06f76c7`. It was clean and detached. It now contains `93f9670f`, which also matched fetched `origin/main`. Its tree hash, `57e82caa1b76a336c7a4dc0659dceb6423038f74`, matches the final PR #24 branch at `4798cd8`. The canonical dirty checkout was not changed.

All three outer evidence bundles and all 35 inner packages matched their recorded SHA-256 hashes. Running the existing aggregation program reproduced all 11 committed result files byte for byte. The replay record is in [archive_replay.json](../data/unified/methodology_review_2026-09-26/archive_replay.json).

The verified source population contains 179,295 player-match-team rows from 3,892 completed fixtures. Raw evaluation covers 17,216 appearances in 374 fixtures and 22 tournament-season blocks. Official fantasy evaluation contains only 13 rounds across three competition-seasons. These counts overlap; the training population is not the sample size for fantasy-team superiority.

The complete modern Six Nations pools contain 1,380 appearances: 1,350 workbook outcomes, 12 separately identified unused-bench zeros, and 18 unknown played outcomes. Candidate selection does not depend on having an outcome. The two unscorable P3 season totals remain unknown. Source: [complete comparison report](../data/unified/complete_comparison/REPORT.md), [protocol](../data/unified/complete_comparison/PROTOCOL.md), and [study manifest](../data/unified/complete_comparison/study_manifest.json).

## 2. The raw advantage is real within its stated benchmark

| Candidate | Equal-block stable-event relative loss |
|---|---:|
| P3 robust native | 0.875623 |
| P3 native weighted | 0.880415 |
| P3 native 50/50 | 0.882717 |
| Empirical event model | 0.954117 |

Robust improves the headline loss by 0.80% against native 50/50 and wins 16 of 22 blocks. The paired exploratory interval for the difference is −0.011771 to −0.003007. This is a small, consistent retrospective raw-event improvement. It is not an 0.80% expected increase in fantasy points.

Three comparisons must remain separate:

1. The raw control is `EmpiricalEventModel`. The official Six Nations control uses `empirical_unified.project_candidates`; NCR uses `ncr_project.build_projection`. Their minutes, priors and bonus heuristics differ. A raw win over `empirical_event` does not imply a fantasy win over `empirical_baseline`.
2. Raw blocks freeze history at tournament start. Most official P3 forecasts update at each round. Raw and official results therefore differ in population, comparator, update policy and objective.
3. The headline averages 23 event losses equally. It excludes minutes and extended scoring events. Eleven of those 23 events do not score directly under the current Six Nations adapter.

The official comparison also omits the complete current Six Nations incumbent. `gw_update.sh:223–224` runs `model.run`, which persists the promoted research configuration through `model/run.py:185–201`. That is a different pipeline from PR #24's empirical fantasy baseline. The next deployment comparison must include this full incumbent, its selection outputs and its update policy under matched information. Comparing only robust with a convenient baseline would leave the replacement question unanswered.

Removing the unscored events does not remove robust's raw advantage. The 12 scored stable events still favour robust against native 50/50, by −0.009473 relative loss on average. Unscored events account for 30.3% of the net headline gain. Drop goals and red cards together account for 55.1% of that net gain across all 23 events. These are arithmetic contributions in normalised loss units, not contributions to fantasy points.

Some high-value attacking predictions get worse:

| Event | Robust minus native 50/50 relative loss | Blocks where robust loses |
|---|---:|---:|
| Tries | +0.012159 | 18 / 22 |
| Try assists | +0.009428 | 16 / 22 |
| Conversions | +0.014100 | 15 / 22 |
| Metres | −0.024179 | 1 / 22 |

The exploratory event intervals exclude zero for these four comparisons, but were not adjusted for the many inspected comparisons. The pattern justifies examining attacking ranks near selection boundaries. It does not prove that these errors caused the observed captain losses.

Robust's raw-event gain is clearer on the bench than among starters. Against native 50/50, bench event loss improves in 21 of 22 blocks; the starter interval includes zero. Equal-block minutes MAE is 9.879 minutes for robust and 9.917 for native 50/50. The 0.038-minute overall difference is small; robust improves bench MAE by 0.230 minutes and worsens starter MAE by 0.065 minutes. These are marginal point-forecast checks, not validated playing-time distributions.

Sources: [raw scoring relevance](../data/unified/methodology_review_2026-09-26/raw_scoring_relevance.csv), [event differences](../data/unified/methodology_review_2026-09-26/raw_event_differences.csv), [minutes summary](../data/unified/methodology_review_2026-09-26/raw_minutes_summary.csv), and the original [target intervals](../data/unified/complete_comparison/results/raw_target_intervals.csv). Implementation: `research/raw_comparison.py:106`, `model/unified/rolling_eval.py:299`, `raw_benchmark/config.py:15`, and `raw_benchmark/metrics.py:30`.

## 3. Captain and super-sub decisions explain most of the recorded squad differences

The following decomposition exactly reproduces all 26 robust and empirical squad totals. XV means each ordinary selected player's base points. Captain extra is the additional copy of the captain's points. Super-sub is the full tripled contribution of the sixteenth player. Every selected XV player in this diagnostic was a recorded starter.

| Robust minus empirical | Ordinary XV | Captain extra | Super-sub | Total |
|---|---:|---:|---:|---:|
| NCR 2026, three rounds | +2 | +6 | +153 | +161 |
| Six Nations 2025 | −18 | −6 | −30 | −54 |
| Six Nations 2026 | +16 | −48 | −24 | −56 |
| Both Six Nations seasons | −2 | −54 | −54 | −110 |

NCR GW2 contributes +191 points. Robust selects Henry Pollock as super-sub, with 65 base points, instead of Gianmarco Lucchesi, with 15. The tripled difference is +150. The other two NCR rounds contribute −30 overall.

Six Nations 2026 round four contributes −109: −78 from the XV and −31 from the captain. The other nine Six Nations rounds sum to −1. Removing that round is a sensitivity check, not permission to discard it. The season decomposition and round concentration show different aspects of the result; the small aggregate XV difference does not mean every XV decision was sound.

The archived robust-minus-empirical squad intervals remain wide: NCR +53.67 points per round, interval [−31, +191]; Six Nations −11, interval [−41.6, +15.3]. They permit both improvement and deterioration. Robust does beat native 50/50 on nine mutually scorable Six Nations rounds, by +10 per round with interval [+1.22, +21.56]. That narrower result excludes the missing-score round and does not establish superiority over empirical.

The measured conclusion is that a few amplified decisions dominate these historical comparisons. Whether either model makes those decisions better on average remains unresolved. Sources: [season decomposition](../data/unified/methodology_review_2026-09-26/season_differences.csv), [round decomposition](../data/unified/methodology_review_2026-09-26/round_differences.csv), [selected contributions](../data/unified/methodology_review_2026-09-26/selected_players.csv), and [original paired intervals](../data/unified/complete_comparison/results/official_paired_intervals.csv).

## 4. A verified NCR scoring defect changes the comparison

The NCR adapter adds `2 * E[scrums_won]` without restricting the player's position. Its caller does not perform the allocation assumed by the scorer's comment. The event is learned from Six Nations workbook scrum labels, which include all forwards. Consequently, NCR receives positive scrum-point forecasts for second-row and back-row players, and smaller positive values for backs.

The repository's NCR rubric restricts these points to the front row. The current official game's public application bundle confirms the same rule and the +2 weight. Its footnote says: “Awarded only to front-row players on the field when the scrum is won.” This confirms the current contract; the original July publication has not been independently archived. See the [official rules](https://fantasy.nationschampionshiprugby.com/rules), [official application source](https://fantasy.nationschampionshiprugby.com/static-assets/fantasy/static/build/static/index.KJ99yB3e.js), and [captured source metadata](../data/unified/methodology_review_2026-09-26/rule_sources.json).

I replayed all three NCR rounds after removing only this term from non-Prop/Hooker P3 forecasts. Models, all other forecasts, pools, prices, constraints and official outcomes stayed fixed. All nine original P3 selections reproduced before the intervention.

| Candidate | Archived MAE / team points | Front-row-only scrum scoring: MAE / team points |
|---|---:|---:|
| Robust | 8.370498 / 1,739 | 8.251674 / 1,654 |
| Native 50/50 | 8.404418 / 1,682 | 8.266717 / 1,570 |
| Native weighted | 8.353762 / 1,562 | 8.201576 / 1,658 |
| Empirical baseline | 8.401067 / 1,578 | Unchanged |

The intervention improves MAE for all three P3 models. It lowers robust's selected-team total by 85 and raises weighted P3's by 96. Weighted then leads robust by four points. Correcting a scoring contract can reduce a lucky historical total. Correctness must not depend on whether that total improves.

This is a demonstrated cause of NCR comparison changes. It cannot explain the Six Nations gap. The corrected results remain three-round retrospective diagnostics, with the original NCR lineup and price limitations. They do not establish weighted P3 as the deployment winner. The original PR #24 artifacts remain unchanged and reproducible; their NCR fantasy interpretation now needs this qualification.

Relevant code: `model/unified/scoring.py:49–66`, `model/unified/ncr_gw_eval.py:60–71`, `model/unified/raw_benchmark/coverage.py:153–176`, and `data/ncr/README.md:23–29`. Derived evidence: [rounds](../data/unified/methodology_review_2026-09-26/front_row_rounds.csv), [summary](../data/unified/methodology_review_2026-09-26/front_row_summary.csv), and [calculation manifest](../data/unified/methodology_review_2026-09-26/front_row_manifest.json).

A separate Six Nations rule discrepancy remains unresolved. The official [2026 guide](https://www.sixnationsrugby.com/en/m6n/news/how-to-win-at-six-nations-fantasy-rugby-2026) lists five points for a drop goal; all inspected repository Six Nations adapters use four. The guide may differ from the game engine or a historical rule version. Verify the season-specific game configuration and workbook before changing it. This discrepancy is shared across adapters and has not been shown to cause the robust-versus-empirical gap.

## 5. Better average error is insufficient for the selection objective

The current optimiser maximises the sum of predicted fantasy points under position, budget, country and hemisphere constraints. A player matters when their forecast changes a feasible selection or a multiplier assignment. Errors on clearly unselectable players can improve MAE without changing a team.

The five fixed interpolation weights below mix empirical and robust forecasts before the unchanged optimiser. No weights were fitted. All 65 resulting squads have known outcomes, and both endpoints reproduce all 26 archived selections and their objective inputs.

| Robust weight | NCR 2026 | Six Nations 2025 | Six Nations 2026 |
|---|---:|---:|---:|
| 0 | 1,578 | 2,410 | 2,552 |
| 0.25 | 1,576 | 2,409 | 2,549 |
| 0.50 | 1,599 | 2,501 | 2,492 |
| 0.75 | 1,594 | 2,463 | 2,548 |
| 1 | 1,739 | 2,356 | 2,496 |

These results use the archived scoring convention, including the NCR defect identified above. They demonstrate decision sensitivity; they do not select a blend. In NCR GW2, the super-sub switches from Lucchesi to Pollock between weights 0.75 and 1, adding 150 realised sub points. In Six Nations 2025 round four, the corresponding late switch from Marcus Smith to Dewi Lake loses 60 sub points. A smooth forecast change produces a discrete decision change with a noisy realised payoff.

The calibration results also resist a simple explanation. In Six Nations 2026, robust's mean bench bias is approximately +0.014 points, versus empirical's −0.896. Robust also has higher pooled bench rank correlation, 0.380 versus 0.349. It still loses 24 super-sub points that season. In 2025, robust has worse bench MAE and rank correlation. These are descriptive pooled diagnostics; within-position feasible alternatives and lock-specific information require closer analysis.

A global positive affine recalibration cannot alter this optimiser's selections. Every feasible squad has total point coefficient 19: 15 XV points, one extra captain copy, and three super-sub copies. Transforming every forecast to `a + b*p`, with `b > 0`, transforms every squad objective to `19*a + b*objective`. Relevant calibration must change relative player, role or group expectations. This invariance depends on the current fixed captain/starter and super-sub/bench constraints.

MAE elicits a conditional median, while a risk-neutral points optimiser needs conditional means. Raw metres loss uses squared log error, so it does not directly assess the arithmetic mean needed for expected points. Retain MAE for continuity, but add mean bias and squared error, then evaluate the actual decisions. Matching the forecast functional to the loss is a standard requirement; see [Gneiting, Making and Evaluating Point Forecasts](https://arxiv.org/abs/0912.0902).

The most useful additional diagnostics are feasible swap margins, the predicted gap between the best and next-best legal teams, selected-set calibration, and regret against the same-constraint hindsight optimum. Hindsight regret is an upper-bound diagnostic, not an available strategy. Where labels are missing, report partial identification or explicitly assumed bounds; do not optimise a supposedly complete hindsight team on a silently reduced pool.

Sources: [interpolation rounds](../data/unified/methodology_review_2026-09-26/interpolation_rounds.csv), [season interpolation](../data/unified/methodology_review_2026-09-26/interpolation_seasons.csv), [status diagnostics](../data/unified/methodology_review_2026-09-26/season_error_by_status.csv), and `model/ncr_project.py:412–458`.

## 6. Minutes and uncertainty need an explicit decision contract

For a prospective super-sub, the useful forecast is the expectation of the actual scoring rule: `3*Y` when the player enters from the bench, `0.5*Y` when the player starts, and zero when unused. Entry probability, conditional minutes and event rates interact. Multiplying an unconditional score by three assumes the bench status is known and stable. The historical final-lineup exercise cannot measure late lineup risk.

Use a mixture over start, bench entry and non-appearance when those states are uncertain at lock. Estimate event production conditional on exposure, with shrinkage for scarce player-role histories. Do not multiply marginal expected minutes and marginal event rates without checking their dependence and units. Replacing a minutes marginal alone does not necessarily change event means or create a coherent simulator.

The existing official Six Nations conversion already computes `E[floor(metres/10)]` from its assumed lognormal distribution. It does not make the simple `floor(E[metres]/10)` error. Its distributional assumption still needs validation. Linear event scoring depends on means; the nonlinear metres term also depends on the distribution.

The P3 blend moment-matches event marginals. `RawPrediction.sample` samples events separately, and sampled minutes are not an exposure mechanism for those events. The headline raw losses use point means and do not validate dispersion or dependence. A mean-calibrated model can therefore have unreliable intervals or implausible joint samples.

For expected additive squad points under fixed constraints, dependence does not change the sum of expected values. Captain doubling does not itself justify choosing a higher-variance player. Dependence becomes necessary for probabilities of winning, tournament-rank utility, uncertainty intervals and risk constraints. A joint simulator should follow an explicit need for those decisions and pass marginal calibration, zero/entry probability, coverage and team-total consistency checks first.

Source: `model/unified/rolling_eval.py:179–198`, `model/unified/contracts.py`, `model/unified/raw_benchmark/blend.py`, and `model/unified/raw_benchmark/robust_empirical.py`.

## 7. Data, constraints and temporal validity limit the claims

| Issue | Current evidence | Required treatment |
|---|---|---|
| Prices and fantasy positions | Six Nations uses zero prices, infinite budget and reconstructed playing positions. | Retain the diagnostic label. Only archived lock-time prices and official eligibility can establish deployable legal-team performance. |
| Lineups | Six Nations uses final recorded teamsheets. NCR GW3 has retrospective prices and corrected final lineups. GW1 excludes New Zealand and France. | Compare the same pools, but do not claim an exact historical user decision replay. Archive future lock-time cohorts. |
| Missing outcomes | Eighteen played Six Nations outcomes remain unknown. Rolling/frozen 2026 totals contain Dobie's missing score. | Preserve missing labels and paired denominators. Their totals remain `2450 + Dobie` and `2380 + Dobie`, not zero-filled totals. |
| Source disagreement | The inspected public replacement source disagreed with 73 of 642 known workbook matches. | Recover missing labels through source reconciliation, not selective substitution. Audit matching error separately from predictive error. |
| Sparse scoring events | Headline raw loss excludes extended events; `scrums_won` has only one supported evaluation block, and several other workbook events have two. | Record event definition, source, season availability and coverage. Do not treat unobserved events as demonstrated zeros. |
| Raw-event meaning | Six Nations all-forward scrum labels entered NCR's front-row scoring term. Interceptions and other rubric terms lack full direct observation. | Separate rugby event definitions from competition-specific allocation. Audit omitted terms and proxies before attributing residual bias to model fit. |
| Temporal filtering | `past_matches` applies a three-hour lag to exact kickoffs and excludes the cutoff day for date-only history. | Retain strict prior-only features and training. A hash proves which revision ran, not when the value was first published. |
| Selection policy | The MILP fixes a P-status captain and B-status super-sub. Boosters are outside these comparisons. | Describe this policy restriction. Verify official permitted choices. Benchmark booster use separately if it becomes part of the objective. |
| Research reuse | Existing weights and earlier model decisions used these historical seasons. | Treat every present slice, interpolation and correction as development evidence. More resampling does not restore an untouched holdout. |

Unknown outcomes are not random by assumption. Players with identity problems, late changes or unusual roles may be more likely to lack labels. Report which candidate and policy selections are exposed. The existing complete-pool repair addresses selection exclusion; it does not eliminate outcome uncertainty.

Use paired rounds for squad outcomes and paired fixtures for player metrics. Retain tournament-season summaries and leave-one-block-out sensitivity. Do not bootstrap player rows as independent decisions. Only two modern Six Nations seasons and three NCR rounds are available, so a season-cluster interval cannot be made precise by algorithm choice.

Rolling-origin evaluation uses earlier observations for each forecast, as described by [Hyndman and Athanasopoulos](https://otexts.com/fpp3/tscv.html). Nested chronological selection reduces within-experiment tuning bias. It cannot erase prior human or AI inspection of the same outcomes. [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) explain why selection criteria themselves can be overfit.

## 8. The November workflow needs repair before a promotion decision

The current `gw_update.sh` retains the NCR incumbent and separately freezes the active shadow. [shadow_active.json](../data/unified/v3/shadow_active.json) identifies `p3_event_50` for GW4–7. Its raw-history path is tournament-frozen. A rolling robust model is a separate experiment.

Preserve the original candidate, artifacts and pooled single-look protocol. Correctness fixes must be recorded as new versions or explicit pre-outcome amendments, with the original record retained. A known scoring defect cannot become acceptable because a frozen experiment later passes. The review did not change this protected configuration.

Concrete preparation work is required:

1. Freeze the incumbent forecasts and actual selected decisions beside the shadow. The current shadow manifest hashes the candidate model and forecasts, but evaluation later reads mutable incumbent projection CSVs.
2. Freeze catalogue, prices, eligibility, roles, team sheets, lock time, mappings, source revision, history, scoring version, optimiser settings and prediction hashes. Verify complete coverage, unique keys and both write-once files before accepting a freeze.
3. Route evaluation to the declared experiment. The P3 decision specifies a 5% MAE / five-percentage-point mean-capture severe-failure veto. The generic v3 evaluator instead applies 2% / two-percentage-point gates and reads a failed v3 historical gate. Resolve the mismatch before outcomes, preserving the intended registered thresholds.
4. Enforce the prescribed GW4–7 pooled evaluation. The generic CLI permits arbitrary round subsets. Archive results as they arrive, but do not tune the protected candidate from intermediate comparative results.
5. Validate current, complete teamsheets. Snapshot code can report partial sheets; the freeze path has no explicit full-cohort readiness check. The shell's existing-file shortcut should verify both the forecast and its matching manifest.
6. Hydrate the declared Git LFS model and store objects before execution. They are pointer files in this checkout. Use the current Python 3.11 preflight and pinned dependencies. This review reused an existing compatible environment without hydrating or retraining those artifacts.

References: `gw_update.sh:144–185`, `model/unified/v3/shadow.py:124–151,228–237,276–335`, `model/unified/raw_benchmark/report.py:192–202`, `model/unified/v3/cli.py:119–129`, and `ncr_snapshot.py:149–155,193–205`.

The current NCR projection does call `past_matches` and filter ranking snapshots. The older blanket claim that this function lacks historical filtering is obsolete. Its remaining retrospective limitations concern other inputs and their publication times.

The Six Nations command currently invokes a historical training/backtest harness and its promoted configuration. Before a future season, verify a lock-time prediction entry point, season handling and feature-store readiness. A file or docstring still named `sealed_2026` does not make repeatedly inspected 2026 outcomes an untouched holdout.

## 9. Research should follow dependencies

The first row below has already produced a material finding in this review. Later rows are proposed research, not completed experiments or a scheduled background job.

| Work package | Start condition | Deliverable and decision rule |
|---|---|---|
| A. Scoring and data contracts | Available source and archived predictions. | Confirm season-specific rules, correct role allocation, reconcile labels and replay all affected candidates without refitting. Stop comparisons with unsupported scoring terms or mismatched populations. Preserve prior revisions. |
| B. Prospective capture and evaluator repair | Exact protocol and hydrated artifacts. Can run alongside A. | Demonstrate complete pre-lock capture for incumbent and challenger, immutable paired evidence and correct protocol routing. Reject incomplete, stale, post-lock or mismatched artifacts. |
| C. Decision-error ledger | A's corrected score revision. | Record legal swaps, forecast margins, captain/sub alternatives, position/country/budget binding constraints and per-event score contributions. Use common pools and unknown-label accounting. Select mechanisms from recurrent errors, not the largest single realised miss. |
| D. Small forecast corrections | C identifies a repeatable error and earlier development data support it. | Compare role-specific shrinkage/calibration, exposure/entry modelling and goal-kicking-role changes separately. Fit transformations only on earlier out-of-fold forecasts. Reject changes that only improve a descriptive pooled metric or one influential round. |
| E. Component complementarity | A–D supply aligned predictions and an explicit remaining error. | Compare robust, a fixed simple blend and a small regularised combination. Select within earlier chronological blocks. Reuse component forecasts when their hashes and cutoffs match. Do not choose the 2025 interpolation winner from this review. |
| F. Frozen challenger and future decisions | A candidate, update policy, objective and decision rules are fixed before lock. | Record future legal teams and predictions under the actual inputs. Evaluate only at the predeclared point. Keep the incumbent when evidence is incomplete or the new policy fails its criteria. |

Package recovery, input capture and source review can run concurrently. Independent implementations and historical outer folds can also run concurrently once the protocol is fixed. Each mechanism's interpretation depends on the corrected common benchmark. Candidate selection precedes prospective freezing, which precedes outcome analysis.

Use existing work before implementing replacements. Issue #21 describes a goal-kicking-role plus independent minutes package that was absent from the reviewed PRs. Recover its exact source and configuration, then compare robust alone, role only, minutes only and both on the same revision. If it cannot be recovered, record it as unavailable; a new implementation is a new candidate.

Decision-focused learning is a valid later candidate, but 13 official slates give many opportunities to overfit discrete team choices. Begin with simple regularised corrections and common-constraint evaluation. The [Smart Predict, then Optimize paper](https://arxiv.org/abs/1710.08005) establishes the distinction between prediction loss and decision loss; it does not establish that its training method will improve this rugby dataset.

## 10. Decisions and stopping rules

The primary objective for a new policy should be expected legal-squad points. Report each competition separately. Keep ordinary XV, captain extra and super-sub contributions as explanatory outcomes; they do not replace the total. Use actual deployed incumbents for the prospective comparison, alongside robust as a research reference.

Before future outcomes, register a minimum useful gain `delta`, an acceptable deterioration margin for each competition, a precision target, the evaluation point and a missing-data rule. Express margins in points per round, with their percentage equivalents. The existing frozen P3 veto is not replaced by these proposed rules.

For a superiority claim, require the lower paired interval to exceed the registered useful-gain threshold and satisfy the registered regression safeguards. For a non-inferiority or operational adoption decision, use the corresponding predeclared rule and describe the weaker claim explicitly. An inconclusive result remains inconclusive. Do not choose the claim or threshold after seeing the result.

Estimate attainable precision before committing to a small target. The observed Six Nations robust-minus-empirical round differences have a standard deviation of about 48.24 points. An illustrative independent, normal approximation for 80% power at a two-sided 5% level gives about 183 rounds to detect a true ten-point gain against zero. This is not a sample-size recommendation: ten reused rounds give an unstable variance estimate, the actual rounds are dependent, and the proposed corrected policy can have different variance. It demonstrates why four November rounds cannot guarantee a precise small-gain conclusion.

Use the following stopping rules during development:

- Stop a branch when its stated mechanism fails on the next chronological development block or violates its predeclared safeguard. Retain the result in the ledger.
- Stop tuning a component when gains disappear under common scoring, common candidates or removal of one influential round. Record that sensitivity; do not delete the round.
- Stop claiming calibrated uncertainty when coverage, entry probabilities or joint support fail. A useful mean predictor may remain eligible for mean-based research.
- Exclude invalid pre-lock evidence from confirmatory claims. Withhold complete totals and promotion when required outcomes are unknown. Continue registered capture and apply the predeclared missing-data rule.
- Reopen a rejected branch only for a new mechanism, materially different data or a demonstrated defect. Repeating the same search on the same audit outcomes adds selection pressure.

Historical nested evaluation can prioritise research now. Prospective NCR evidence supports an NCR conclusion. A universal claim also needs future Six Nations inputs and outcomes. A reversible operational decision under uncertainty is possible, but it must not be labelled independently established superiority.

## 11. Prior experiments and actual resource limits

Previous experiments constrain the next search:

- TeamShare improved a consulted historical raw guard but lost the recorded NCR selection, 1,410 versus 1,596. Its older Six Nations comparison also lost, 2,219 versus 2,626, despite lower MAE. Revisit it only with a concrete change to the failed decision mechanism or validated team/event consistency. Its distributional reliability remains unestablished.
- The broad NCR Bayesian search lost to the fixed grouped model on its separate audit; its retained decision was `retain_fixed_models`. More trials on that audit would consume its independence.
- The zero-shot TimesFM-3 probe was 2.7% worse than rolling Ridge in development and 9.1% worse on its then-sealed 2026 check. A larger model or new training setup needs a distinct hypothesis and license review, not an automatic rerun.
- PR #7's E4 component-complementarity idea remains worth assessing. Its chronological result used mismatched component-grid indices and is invalid. A corrected named-component interface and properly nested fits must precede any new performance claim.

These prior results were verified in the surviving original project reports. They use older evaluation revisions and are not directly comparable with PR #24 totals. Sources: [PR #7 archive](../data/unified/research_synthesis/pr7/REPORT.md), [research synthesis](../data/unified/research_synthesis/REPORT.md), the canonical `data/ncr/bayes_opt/REPORT.md`, `research/TIMESFM3_TEAMPLAY_PROBE.md`, and the surviving `55b5/6n/data/unified/agnostic_search` reports. The latter reports are outside this merged checkout and were read without importing their changes.

The archived comparison used 35 jobs, eight-way maximum job concurrency, two OpenMP threads and one OpenBLAS thread. Its measured wall time was 1,493 seconds. Summed runner time was 10,474 seconds, including setup and checks; summed comparison-step time was 7,826 seconds. Comparison steps lasted 51–386 seconds. These measurements describe those GitHub runners and existing models. They are not a runtime guarantee for the M1 laptop or a new model family.

`fit_comparison_models` already reuses one fitted V4 component for several P3 variants. Preserve this reuse. Hash features, folds, model parameters and component predictions before cache reuse. Parallelise independent fits and reads, while keeping permanent-cache writes and immutable forecast creation serial. Do not add paid API collection or routine retraining to a cache-only research replay.

The immediate work can proceed without waiting for November: scoring repair, metadata capture, protocol routing, package recovery and bounded historical mechanism experiments. Serial evaluation results determine which candidate to freeze. November supplies new lock-time inputs and outcomes for NCR GW4–7. AI can reduce implementation latency; it cannot create independent international matches.

## Verification and reproducibility

The [evidence directory](../data/unified/methodology_review_2026-09-26/README.md) contains the retained scripts, derived tables, manifests and reproduction commands. The original PR #24 bundles remain the source of forecasts. A fresh replay of all four retained analysis scripts reproduced all 18 retained CSV tables byte for byte. Original official selections were reproduced before any intervention. Separate numerical and methodology reviews found no unresolved substantive error in this report.

Five existing history/scoring tests passed during this review. Complexity checks on the retained analysis scripts reported no findings. No new tests were created and no existing tests were changed. The full suite was not rerun: the earlier recorded result remains 120 passes and one unchanged label-count assertion, which expects 2,144 rather than 2,164. That historical result is not represented as a fresh pass.

The host permitted the required repository reads, scratch writes and public network checks. No access restriction prevented the review. Outstanding work concerns scoring-version confirmation, production preparation and future evidence, rather than an imposed development calendar.
