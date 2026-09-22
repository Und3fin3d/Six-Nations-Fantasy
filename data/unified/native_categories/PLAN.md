# Native categorical splits — bounded experiment

This experiment follows the matched rolling-history evaluation. It does not promote a model or create a PR.

## Hypothesis

The encoder emits integer codes for player, position, team, opponent and competition level. UniversalGBDT concatenates those codes into a NumPy matrix and passes no categorical_feature argument to LightGBM. Its automatic mode therefore treats the codes as numeric. Test explicit categorical splits without changing the encoder, features, target definitions, tree settings, empirical priors or existing blend weights.

The change is flag-gated, disabled by default, and applies consistently to minutes, event regressors, binary heads and optional v4 hurdle heads. It introduces no competition identity and no fantasy-points target.

## Fixed candidates and evaluation

Refit at each of the same 13 locks used by rolling_eval: Six Nations 2025 rounds 1–5, Six Nations 2026 rounds 1–5 and NCR 2026 rounds 1–3. Evaluate native-category versions of the three existing shared blends (50/50, event-weighted, robust-empirical event-weighted). No search over parameters or event weights. Also retain the native-category tournament-frozen control to separate freshness from representation.

Compare with both the matching previous numerical-category models and the recomputed empirical baseline. Verify identical cohorts, outcomes, inputs and cutoffs; independently recompute MAE and squad totals from prediction and squad files.

A candidate must lower round-balanced player MAE and increase total squad points in each tournament-season to pass the observed-metric screen. Report every candidate and every failed slice. Do not choose per-tournament configurations. The same global candidate must pass everywhere. This retrospective screen alone cannot establish clear prospective superiority.

Six Nations remains a price-free diagnostic unless genuinely archived pre-lock prices are found. NCR GW3 retains its retrospective corrected-lineup/prices qualification. No PR may claim budget-feasible or prospective superiority from these results. NCR GW4–7 remains unread and untouched.

The old weights were selected using 2022–2025 and 2026 outcomes were already known; these are not new untouched holdouts. No acceptance threshold or hypothesis is revised after this run.

## Checks

New regression tests must verify category indices reach count, minutes and binary heads, both hurdle heads, default numerical-mode equivalence, unseen-category prediction and artifact round-trip. Preserve the original frozen artifact and production defaults.

Reference: LightGBM 4.6.0 LGBMRegressor.fit categorical_feature documentation and Advanced Topics / Categorical Feature Support.
