# Empirical-component timing ablation

Registered on 2026-09-26 before execution. This is one fixed ablation justified by the verified absent-entry timing-label defect. No parameter search is permitted.

## Scope and fixed inputs

Compare the empirical component of `p3_robust_native` fitted on the original canonical store with the same component fitted on the corrected minutes-availability store. Keep all archived V4 predictions fixed. This does not measure a full tree refit, a deployable model, or prospective superiority.

Original input SHA-256: `2b70331c0cf215def67831cd27c987362e2ec8330b2afb41c8d204f48ef17313`.

Corrected input SHA-256: `e4b20637e2b514d622bb35cc7c43762e0eddb4f3b4e50f1ca2e39155ae96e6be`.

Only minutes, minute availability and minute provenance differ, each for the same 1,224 identified rows. All event labels, identities, timestamps and positions are fixed. Preserve both inputs and all prior results.

Use the same 13 official slates: Six Nations 2025 rounds 1–5, Six Nations 2026 rounds 1–5, and NCR 2026 GW1–3. Use the established candidate pools, roles, cutoffs, common optimiser and corrected scoring from `research.decision_replay`. Six Nations remains price-free historical diagnosis; NCR retains the documented retrospective-input limits. Unknown outcomes remain unknown.

## Fitting and component recovery

For every slate, fit original and corrected `RobustEmpiricalEventModel` with their unchanged 420-day decay and four-appearance minutes prior. Fit `EmpiricalEventModel` on the original input for recovery verification. Admit only history returned by `past_matches`: exact kickoff plus strictly more than three hours before the lock; date-only histories exclude the cutoff day. Retain existing prior-only player-position handling. Do not use current or future outcomes, and do not alter candidate pools.

Read each fixed event weight `w` from the archived blend metadata and verify it against the existing weight configuration. Before accepting any result, require the saved robust-minus-weighted mean for every player/event to equal `(1-w) * (old_robust_empirical_mean - old_legacy_empirical_mean)` within absolute tolerance `1e-9`. Require identical prediction keys and event support. Verify minutes with the same identity. A recovery failure stops admission; diagnose it instead of approximating it.

## One candidate formula

For every event, set:

`new_blend_mean = archived_robust_mean + (1-w) * (new_empirical_mean - old_empirical_mean)`.

For metres and the separately reported minutes marginal, preserve the fixed tree contribution to the mixture second moment:

`new_second_moment = old_blend_variance + old_blend_mean**2 + (1-w) * ((new_empirical_variance + new_empirical_mean**2) - (old_empirical_variance + old_empirical_mean**2))`.

`new_blend_variance = new_second_moment - new_blend_mean**2`.

Use the existing lognormal approximation and nonlinear Six Nations metres expectation with these updated moments. Require finite nonnegative means and positive variances. Count and binary events use their updated means in deterministic scoring. No claim is made about recalibrated count tails, tail dependence, joint coherence, or a refitted tree. Do not infer an extra entry multiplier.

## Comparators, admission and stopping

Compare the candidate against the saved corrected `p3_robust_native`, actual incumbent, and fixed equal robust/incumbent blend from `/tmp/6n-decision-workflow/remedies/complete`. Reproduce comparator forecasts and common-optimiser outcomes before interpreting differences. Report each season/competition, all round totals, XV/captain/super-sub contributions, fantasy MAE, bias and RMSE, unknown selected outcomes, and paired round uncertainty.

Use the existing advancement criteria without amendment: positive Six Nations 2025 development total, nonnegative Six Nations 2026 audit total, and positive pooled mean advantage after every single Six Nations round removal. Shared research priority also requires nonnegative NCR total. Report admission against each comparator; improved performance against the robust reference alone does not establish superiority to the incumbent or fixed blend.

Use the existing descriptive 20,000-resample paired round bootstrap with seed 20260926 and 95% percentile intervals. These already-consulted historical results are not independent confirmation. Retain every outcome, including failure. Stop after this single ablation and its verification. No model promotion or protected NCR GW4–7 change is authorised by this protocol.
