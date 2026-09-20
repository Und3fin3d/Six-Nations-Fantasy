# Fixed-seed averaging experiment

Base: Friendly-15 research at `1e27aa0b946fbfb03f01490d16728086724ed2fb`, including the PR #20 control. This is a separate research branch, not a production switch.

## Hypothesis and single candidate

The current tree learner uses row and feature subsampling. Test whether averaging three independently seeded fits improves prediction and squad stability without changing the information set or weakening the empirical comparator.

Candidate: `p3_seedbag_native`, an equal-weight predictive mixture of the complete PR #20 raw-event model fitted with tree seeds **17, 29, 43**. Seed 17 is the exact control. The robust empirical component, natural training weights, tree capacity, event blend weights, features, cutoffs and optimiser rules are unchanged. The same seeds and weights apply to every game and tournament. No seed is selected or dropped using outcomes. Distribution mixtures retain between-member variation; predictive variance is not divided by the number of members.

Only this candidate and the existing control are evaluated. There is no hyperparameter grid, adaptive stopping, post-result weight change, or candidate selection from Friendly-15/fantasy outcomes. These are already-examined historical benchmarks: results are exploratory retrospective evidence, not a newly untouched holdout. A successful point estimate will not be described as definitive superiority without appropriate uncertainty and coverage.

## Mandatory complete comparison

Every candidate report contains the empirical comparator, PR #20 `p3_robust_native`, and the candidate on:

1. Nations Championship 2026 rounds 1-3: fantasy-point MAE and selected-squad points.
2. Six Nations 2025 rounds 1-5: fantasy-point MAE and selected-squad points.
3. Six Nations 2026 rounds 1-5: fantasy-point MAE and selected-squad points.
4. The identical fixed Friendly-15 fixtures: count-stat MAE, metres MAE, minutes MAE, all per-stat errors and all individual games.

Use the original Friendly-15 fixture IDs, source hashes, complete cached teamsheets and missing-label rules. No game substitution, omitted failed round or synthetic fantasy labels. Use the full empirical fantasy pipeline for tournaments and label the friendly comparator separately as the empirical raw-event model.

The candidate must improve fantasy MAE and squad points against empirical in each tournament-season, and improve Friendly-15 count MAE without more than 2% regression in metres/minutes. Also report every comparison against the PR #20 control. Report paired whole-round/fixture 90% bootstrap intervals with seed 17 and 10,000 draws, not player-row confidence intervals. Stronger definitive-superiority claims require evidence beyond point estimates and remain constrained by missing budgets and reused test data.

Six Nations remains a price-free diagnostic. NCR GW3 retains the disclosed retrospectively corrected lineup/prices; GW1 excludes New Zealand and France. All histories use the same conservative completion cutoff, and current targets remain masked. Unversioned data and oracle teamsheets cannot prove original publication times. NCR GW4-7 is not used.

## Engineering and validation

Reuse the existing matched-history runner, scoring adapters and optimiser rather than creating a parallel evaluation pipeline. Add tests for equal-weight means, mixture variances, event support, full player/fixture/team alignment, identical single-member behavior, valid seeds and artifact round trips. Run the full suite before comparisons. Recompute each saved MAE and selected-squad total, verify complete job coverage and source/input hashes, and compare the two controls against preceding evidence. Do not promote a model or create a PR automatically. No rugby API calls, cache refreshes or fitted binaries in Git.
