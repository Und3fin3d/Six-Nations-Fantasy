# Rolling-history comparison plan

Recorded before reading results from this comparison. This branch does not open a PR or alter deployed models.

## Fixed evaluation

Evaluate all five Six Nations rounds in 2025, all five in 2026, and Nations Championship GW1-3 in 2026. Every engine receives history strictly before the same round's UTC lock day. Date-only sources cannot establish availability during that day. Previous rounds remain eligible; the current round and future rounds are excluded. Rebuild history from permanent caches, not the older tournament-start store.

The full NCR empirical projection is the NCR control. The existing competition-independent port of that empirical machinery is the Six Nations control. These are distinct from the simplified raw empirical blend component. Use the same candidate cohort, outcome vector and team-selection constraints for all models in each round.

## Fixed candidates

- `p3_rolling`: freshly fitted v4 plus raw empirical at the original global 50/50 weight.
- `p3_weighted`: the existing canonical event-weight vector, unchanged.
- `p3_shrunk`: that same vector plus international-only empirical minutes, shrinkage toward position/status minutes with three prior matches, and exposure-weighted positional event priors.

All candidates include the correctness fixes: prior-day source filtering, conservative full-season aggregate availability, event-specific RugbyPass exposure, and once-per-fixture team form. No candidate fits an official fantasy-points head. No competition-specific model weights are introduced.

The existing weighted configuration was selected on 2022-2025. Therefore Six Nations 2025 is DEVELOPMENT evidence, not an independent test of that configuration. The 2026 datasets already existed in the repository and are RETROSPECTIVE diagnostics, not a new untouched prospective test. NCR GW4-7 remains unread and the frozen deployed shadow is unchanged.

## Decision

Report per-round results, then mean round MAE and total selected-team points for every competition/season. Both strictly lower MAE and strictly higher team points are required against the empirical control in every included competition/season. Missing rounds, missing models, changed cohorts or non-finite outputs invalidate an improvement claim. Do not drop a losing round or alter these criteria after seeing its result.

NCR uses the same budget, positions, nation and hemisphere limits, captain and bench-only super-sub optimisation for every engine. Its archived final feeds are retrospective inputs, not proof of original pre-lock prices or publication times.

The current Six Nations data support the existing positional-XV/captain/super-sub diagnostic, not a budget-valid team comparison. Historical prices and a complete eligible pool must be verified before claiming cross-tournament superiority for real fantasy squads or opening a promotion PR. The diagnostic cannot silently stand in for that evidence.

Keep raw predictions and execution logs in run artifacts; retain compact metrics, selected squads and source/cutoff manifests in the working branch. Do not overwrite the old benchmark or promote any candidate based only on lower raw-event loss.
