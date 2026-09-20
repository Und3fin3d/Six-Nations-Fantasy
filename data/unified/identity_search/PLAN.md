# Native player-identity experiment

Base: Friendly-15 at `1e27aa0b946fbfb03f01490d16728086724ed2fb`, including the PR #20 control. This is a separate research branch; main, PR #20 and production remain unchanged.

## One prespecified candidate

PR #20 made categorical splits native but still replaces every player ID with the literal `pooled` and applies a post-fit multiplicative player residual adjustment. Test whether the now-native categorical representation can learn player identity in context instead of that global adjustment.

Candidate `p3_identity_native`: `pool_player_id=False`, `player_effects=False`, `native_categories=True`. Retain the robust empirical component, natural training weights, seed 17, tree capacity, P3 per-event blend weights, scorer, optimizer and all histories. This changes the representation and its corresponding personalization mechanism together; it is not an isolated claim about either switch separately. Unknown players retain the encoder fallback and empirical priors. No tournament identity, fantasy outcome head or competition-specific coefficient is introduced.

The control is exactly PR #20 `p3_robust_native` (pooled identity plus post-fit effects, native categories, seed 17). No search over these flags, no selection of lucky seeds, no post-result blend tuning. The three-seed experiment is a separate predeclared candidate, not a reason to change this one.

## All four required evaluations

Always show both empirical and PR #20 controls alongside this candidate on NCR 2026 rounds 1-3, Six Nations 2025 rounds 1-5, Six Nations 2026 rounds 1-5, and the identical fixed Friendly-15 fixture manifest. Fantasy metrics are player-point MAE and optimized squad points; Friendly-15 metrics are count-stat MAE, metres MAE and minutes MAE, with per-game/per-stat details retained. No omitted failed evaluation or substituted fixture.

The empirical fantasy pipeline remains the comparator for fantasy teams; the empirical raw-event engine is explicitly a different comparator on friendlies. Histories, cohorts and constraints are identical. Require lower fantasy MAE and higher squad points against empirical in every tournament-season, and lower Friendly-15 count-stat MAE without more than 2% minutes/metres regression. Report all deltas against PR #20 and paired whole-round/fixture 90% bootstrap intervals (10,000 draws, seed 17). No automatic promotion or PR.

These are previously inspected historical evaluations, not an untouched test set or a prospective proof. Six Nations squads remain price-free diagnostics; NCR GW3 retains retrospectively corrected lineup/prices and GW1 excludes New Zealand/France. Unversioned history and oracle teamsheets retain their original limitations. NCR GW4-7 is never read. No rugby API calls or cache refreshes.

## Validation

Reuse the canonical matched-history runner and the complete four-evaluation report/audit. Add regression tests for actual flag wiring, player-ID preservation versus pooling, default-control equivalence, invalid settings and frozen configuration. Run the full repository tests, then all 28 fixture/round jobs. Independently recalculate saved MAEs and team points and verify both controls match previous evidence. Retain source/config/input hashes and all regressions. No fitted binaries in Git.
