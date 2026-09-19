# Friendly-15 and domain-weighting experiment

Base: PR #20, source branch at 73fb3ce9a010480ed2443a6bca1eefc395d55072. No production promotion or automatic PR creation.

## Fixed additional metric

Evaluate raw player-match statistics on exactly 15 international friendlies with no official fantasy labels in this repository. Choose the latest completed canonical fixtures in cache competition 30 before 2026-09-19, excluding fantasy-labelled fixtures. Selection uses fixture metadata, not outcomes, score margins or prediction errors. Pin fixture IDs, kickoffs, cache hashes and statistic names in fixtures.json; never silently replace fixtures on rerun.

The selected IDs, in chronological order, are: 8518300, 8518309, 8518390, 8518318, 8544553, 8518327, 8518336, 8518345, 8518354, 8518363, 8518372, 8546596, 8553454, 8546605, 8545138. These comprise 11 November 2025 and four August 2026 friendlies.

For each game and statistic, compute MAE over players with an observed label. Include observed zeroes and unused substitutes. Missing labels are not zeroes. Missing predictions, different player cohorts, duplicate keys or incompatible comparison support fail evaluation. Report all 24 per-stat MAEs (22 stable count statistics, metres and minutes). The count-stat headline averages the 22 count-stat MAEs equally, then the 15 games equally. Metres MAE and minutes MAE are separate headlines, not mixed into the count average. Entirely unobserved game/stat cells are disclosed and invalidate the affected headline, rather than being silently skipped.

All models refit at the same fixture kickoff proxy using earlier completed history only (the existing strictly-more-than-three-hour lag). Previous fixtures may enter later histories. Current labels are masked before feature construction. Historical teamsheets are oracle inputs. This measures raw rugby prediction; it does not fabricate fantasy labels or use fantasy weights in the friendly metric.

## Bounded next-model test

The canonical current history contains 153,962 club and 20,546 international player-match rows. Test whether club-dominated training is harming international generalisation. Keep one model and one setting across all tournament families. Keep the robust empirical component, native categories, existing P3 event weights, event heads and tree capacity fixed.

Control: PR #20 p3_robust_native with natural row weighting.
Candidate A: equal total club/international training weight (existing level_balanced scheme).
Candidate B: the same level balancing plus a 730-day historical half-life.

The tree loss and post-fit empirical-Bayes residual aggregates must use the same training weights; otherwise the residual adjustment could undo the weighting change. Natural weighting remains numerically unchanged and all existing defaults stay unchanged. No competition-ID feature, fantasy-points head, competition-specific coefficient, new blend search or neural architecture is introduced.

Choose between A and B only on a separate set of the latest 12 eligible friendlies before 2025: 8476315, 8472247, 8472337, 8476324, 8476333, 8476342, 8476351, 8476360, 8472346, 8476369, 8472256, 8476378. These IDs are disjoint from Friendly-15. Rank by lower count-stat MAE; eligibility requires improvement over the PR #20 control and no more than 2% regression in either minutes or metres MAE. If neither passes, record rejection; do not rename a failure as success or select on the 15 test games.

After development selection, freeze the chosen configuration and evaluate Friendly-15 plus the existing 13-round official fantasy comparison. The empirical raw-event engine is the friendly comparator; the complete corrected empirical fantasy pipeline remains the fantasy comparator. Do not describe them as identical estimators.

## Evidence and claims

An improvement claim requires lower fantasy MAE and higher selected-squad points against the empirical reference in each evaluated tournament-season, as well as lower Friendly-15 count-stat MAE without material minutes/metres regression. Report paired fixture/round bootstrap intervals, all individual games/rounds, and the complete ledger. A point estimate is not definitive superiority if intervals include zero.

Six Nations remains price-free unless genuine pre-lock price archives are found. NCR round 3 retains its retrospective lineup/price correction. Existing 2025 weights and historical results were used in previous research; the new metric is additional retrospective evidence, not a fresh prospective holdout. NCR GW4–7 remains untouched. No rugby API calls, source-cache refreshes, fitted binaries, incumbent switches or automatic PR creation.
