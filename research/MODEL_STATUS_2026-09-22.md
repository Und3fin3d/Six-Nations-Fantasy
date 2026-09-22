# Model selection and evaluation decision, 22 September 2026

## Current complete-pool and all-cache result

The completed [all-cache comparison](../data/unified/complete_comparison/REPORT.md) retains `p3_robust_native` as the strongest shared raw-stat research reference among the four evaluated candidates. Its stable-event loss is 0.875623, compared with native 50/50 at 0.882717, native weighted at 0.880415 and the empirical event model at 0.954117. The 0.80% advantage over native 50/50 has an exploratory paired 95% difference interval of [−0.011771, −0.003007]. Robust wins 16 of 22 tournament-season blocks against that comparator; it does not win every tournament or playing-role group.

The research store now includes all 179,295 eligible cached player-match-team keys across 3,892 fixtures. The evaluation contains 17,216 appearances in 374 international fixtures and 22 tournament-season blocks. Every fit uses strictly earlier history. The complete Six Nations pools contain all 1,380 candidate appearances, with 1,350 workbook scores, 12 separately marked unused-bench zeros and 18 unknown played-match scores.

| Current complete-pool comparison | Empirical fantasy baseline MAE / team points | P3 robust native MAE / team points |
| --- | ---: | ---: |
| NCR 2026, GW1–3 | 8.401067 / 1,578 | 8.370498 / 1,739 |
| Six Nations 2025 | 7.168003 / 2,410 | 7.242949 / 2,356 |
| Six Nations 2026 | 7.350584 / 2,552 | 7.243760 / 2,496 |

Robust leads observed NCR team points. The empirical baseline leads the fully scored Six Nations totals. Their team-point uncertainty intervals include zero in both competitions. The native 50/50 and frozen controls select Jamie Dobie in Six Nations 2026 round one, where his score is missing; their full-season totals remain unknown. A nine-round complete-case comparison favours robust over native 50/50, but it excludes that missing-score round.

All 35 jobs in [run 35737817588](https://github.com/Und3fin3d/Six-Nations-Fantasy/actions/runs/35737817588) passed. Independent arithmetic reproduced 65 official model-round records, 39 raw-to-fantasy score files, 36,400 raw metric rows and 19 primary paired summaries. The full local test suite has 120 passes and one unchanged fixed-count assertion expecting 2,144 labels instead of the repaired 2,164. Tests remain unchanged at the user's request, so the fixed-count failure remains documented.

The evidence supports the shared research reference, with stronger historical coverage. It does not establish universal fantasy superiority or change production routing. Future promotion still requires genuinely prospective predictions and complete lock-time selection inputs. The [report](../data/unified/complete_comparison/REPORT.md), [protocol](../data/unified/complete_comparison/PROTOCOL.md) and [archived evidence](../data/unified/complete_comparison/evidence/manifest.json) contain the full boundaries and losing comparisons.

## Archived PR #7 / PR #20 merge review

The following sections preserve the earlier review and its then-current figures. The complete-pool results above supersede those historical comparison numbers.

PR #20 was merged as an incremental correctness and research change. Its `p3_robust_native` model remains the retained shared research reference. No evaluated model has established universal deployment superiority.

Issue #21 records a later retained candidate: corrected identities, the historical goal-kicking-role gate and an independent robust minutes head. That package is absent from PR #7 and PR #20. At the merge review, the remote `codex/unified-refinements-pr-20260920` branch pointed to the unchanged PR #20 head. Closing the documentation issue does not complete its outstanding research actions or publish that package.

### Comparable recorded results

These are recorded historical results, not fresh model fits performed for this review. Lower MAE and higher selected-team points are better. The empirical fantasy baseline and raw-event comparator are separate pipelines.

| Evaluation | Empirical comparator | PR #20 | Later retained package, issue #21 |
| --- | ---: | ---: | ---: |
| NCR 2026 GW1–3: fantasy MAE / team points | 8.401189 / 1578 | 8.364626 / 1740 | 8.371914 / 1740 |
| Six Nations 2025: fantasy MAE / team points | 7.139481 / 2410 | 7.231897 / 2416 | 7.231778 / 2416 |
| Six Nations 2026: fantasy MAE / team points | 7.389271 / 2548 | 7.254336 / 2550 | 7.253437 / 2550 |
| Friendly-15: count-stat MAE | 0.691210 | 0.668166 | 0.667837 |
| Friendly-15: metres MAE | 11.396364 | 11.153362 | 11.153362 |
| Friendly-15: minutes MAE | 10.466982 | 10.332761 | 10.210754 |

PR #20 uses the original player crosswalk. Its identity-corrected NCR reference is 8.374655 / 1740. Therefore, the two candidate columns are different evaluation revisions. Their NCR MAEs cannot isolate the effects of the role and minutes refinements. Friendly-15 values come from issue #21's later evaluation; PR #20 itself contains only the three tournament evaluations.

Both Six Nations team totals are price-free diagnostics on restricted labelled pools. The [independent audit](../data/unified/pr20_verification/REPORT.md) found 38 omitted teamsheet rows, including 26 players who took the field. Complete-pool superiority remains unverified. NCR has budget checks, but GW3 uses corrected final lineups and retrospective prices. The candidate loses Six Nations 2025 MAE. PR #20's NCR gain comes entirely from GW2; it loses GW1 and GW3. All three recorded round-bootstrap intervals for team-point gains include zero.

### Why PR #7 was not selected

E4 has a strong historical raw score of 0.869656. However, its pooled reconstructed fantasy scores do not compare complete deployed pipelines. Its own board reports NCR-rubric MAE of 6.3091 for the Nations Championship family, against empirical 6.2925. It also lacks a prediction-time entry point. It has no comparable complete four-part scoreboard. These findings do not support its title's claim of winning every competition.

The [PR #7 archive](../data/unified/research_synthesis/pr7/REPORT.md) preserves every phase-2 trial and its useful findings. The closure review also confirmed that its temporal comparison applies C5 weight indices to the wrong component table. The reported E4 temporal gain is invalid. The original raw means remain recorded evidence, without a verified chronological promotion result.

### Comprehensive measurement protocol

1. Recover the exact retained package and its evidence. Freeze candidate source, configuration, data hashes and identity mappings. Include the full NCR empirical pipeline, the full Six Nations incumbent, PR #20 and the retained package. Treat E4 as not evaluated until it has a reproducible prediction interface.
2. Preserve all four existing evaluations as regression benchmarks. Use identical player-match keys, labels and histories. Fail incomplete predictions, duplicate identities, missing rounds, missing statistics and mismatched model support. Keep genuine missing labels distinct from zero outcomes.
3. Evaluate chronological outer blocks using only earlier history. Select hyperparameters inside earlier development blocks, then freeze the selected configuration before each outer block. Refit every candidate at the same locks. Record source publication times where available and retain the strict three-hour convention where they are unavailable.
4. Obtain genuine archived prices, eligibility, positions, lineups and lock times. Until then, label affected Six Nations team results as diagnostics. Use the same deterministic optimiser and game rules for all candidates. Keep NCR GW4–7's existing frozen protocol unchanged.
5. Make realised legal-team points the primary decision measure. Report each competition-season separately. Decompose ordinary XV, captain and substitute contributions. Report fantasy MAE, position-aware ranking and selection regret against a hindsight legal optimum as secondary measures. The hindsight optimum is a diagnostic upper bound, not an available strategy.
6. Keep Friendly-15 count-stat, metres and minutes MAE separate. Retain per-statistic and per-fixture errors. Add event distribution scores and interval coverage before claiming reliable uncertainty or simulation. Inspect starter/bench, position, hemisphere, tournament and history-depth groups. Do not equate a separate minutes marginal with a coherent joint event simulator.
7. Estimate uncertainty from paired fixtures and rounds, not independent player rows. Account for tournament-season dependence when enough seasons exist. Report absolute effects, intervals and the largest losing slices. Predeclare a minimum useful gain and acceptable regression margins before new outcomes are seen. Use development variance to estimate the required number of future rounds; three and five rounds do not justify a universal claim.
8. Freeze predictions before new locks on suitably untouched future rounds. Preserve every candidate and failure in the ledger. Do not reuse repeatedly inspected outcomes as fresh confirmation. Promote one shared configuration only after it improves the primary decision measure and satisfies the predeclared cross-tournament safeguards.

The rolling-origin design follows [Forecasting: Principles and Practice](https://otexts.com/fpp3/tscv.html). Separating model selection from final evaluation addresses the selection bias described by [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html). The game-specific metrics and promotion criteria above are recommendations for this project.

### Review and verification

The reviewed PR #20 head was `73fb3ce9a010480ed2443a6bca1eefc395d55072`, against main `64485d2e374178b51108a298683b9ba43ab9628b`. Its model and test files exactly match the source from the recorded successful 121-test Actions run.

After the cache correction, all 121 existing tests passed locally in 314.02 seconds. The run used Python 3.11.15 and the pinned `requirements-model.txt` and `requirements-unified.txt` packages, with external pytest plugins disabled and `OMP_THREAD_LIMIT=2`. It included the genuine frozen LFS model and the neural path. There were 156 pandas fragmentation warnings. The environment preflight, CLI help, all nine input hashes and `git diff --check` also passed. An earlier attempt in the existing Python 3.12 environment stopped during collection because BeautifulSoup was missing; the complete pinned-environment run supersedes it. No full historical comparison was refitted for this merge review. The subsequent [independent audit](../data/unified/pr20_verification/REPORT.md) recalculated all 13 rounds and freshly fitted NCR GW2. That fit reproduced all four MAEs, squads and team totals, with expected-point differences at most 1.07e-14.

The independent specification review found one cache-provenance defect. The run manifest now hashes all nine previously omitted evaluation files before accepting cached models. These include fixtures, teams, crosswalk, official feeds, saved projections and Six Nations targets. The reviewer checked the full exercised data-loading call graph and accepted the correction. No model parameters, fitted artifacts, tests or production routing were changed during this review.

The complexity check reports existing high-complexity functions and the new evaluation runner. The runner remains a maintainability limitation. The cache correction preserves its current structure and does not suppress the finding.

#### Standards

The independent standards review found three nonblocking concerns in the original PR: its new `pathlib` import, added comments, and repeated categorical-fit keyword construction. The first two conflict with the current personal code rules. The third is a possible duplication smell. This merge review preserves the already-written path handling and comments; it adds neither. A broader style rewrite would change the previously verified source without improving the requested model comparison.

#### Specification

The review found one P2 cache-provenance defect, which the seven-line input-hashing correction resolves. The reviewer accepted that correction. No blocking specification finding remains. The later refinement package and universal-superiority claim remain outside this PR's implemented scope.

The review totals are three nonblocking standards findings and one resolved specification finding. The strongest standards concern is the original path import; the specification defect was incompatible evaluation-cache reuse.

### Sources

- [Issue #21: complete research ledger](https://github.com/Und3fin3d/Six-Nations-Fantasy/issues/21).
- [PR #20: selected incremental change](https://github.com/Und3fin3d/Six-Nations-Fantasy/pull/20).
- [PR #7: alternative E4 research](https://github.com/Und3fin3d/Six-Nations-Fantasy/pull/7).
- [PR #20 recorded comparison and tests](https://github.com/Und3fin3d/Six-Nations-Fantasy/actions/runs/35470275137).
- [Native-category results](../data/unified/native_categories/REPORT.md).
