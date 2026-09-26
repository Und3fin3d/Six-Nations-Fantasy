# Role-policy continuation: active objective, no promotion

The fantasy-decision improvement objective is unfinished. This is a research checkpoint, not a completion report. The permission to finish with a workflow and negative results remains superseded. PR #25 stays draft; production and the protected November experiment are unchanged.

## Executed comparison

Registration: `DECISION_ROLE_POLICY_REGISTRATION_2026-09-26.json`, committed as `f0d1d25580f79b42922b7a0a164559201e6f70b7` before either new policy was scored. Execution source: `e3649c852ba3a33cba8a2c806211f89628ee7feb`. GitHub Actions run `36257541020`, job `108447071902`, completed successfully on 26 September 2026. Evidence is retained in `data/unified/decision_continuation_2026-09-26/run-36257541020/`, including complete pools, all selected identities, every round and component, solver certificates, checks, dependencies, source/input hashes and admissions. No model was refitted in this experiment.

Before any new policy was evaluated, all 52 archived control squads and their metrics reproduced. The new exact optimiser also reproduced all 52 point-head endpoints. The registered joint and lexicographic policies were then run on every complete candidate pool; independent constraint counts and solver residuals were retained. Unknown labels stayed in the pools. None of these policies happened to select an unknown outcome. Historical Six Nations prices remain unavailable: the Six Nations comparison is explicitly price-free and position/country constrained, not budget-verified.

## All three fantasy benchmarks

Each cell is realised squad points / mean round fantasy-point MAE. A role-only policy retains its named underlying point forecast; its MAE is not a new raw-model result.

| Policy / comparator | Six Nations 2025 | Six Nations 2026 | NCR 2026 GW1–3 |
|---|---:|---:|---:|
| Historical empirical baseline | 2410 / 7.168003 | 2552 / 7.349180 | 1578 / 8.401067 |
| Lock-rebuilt incumbent common point optimiser | 2260 / 7.441466 | 2539 / 7.304946 | 1578 / 8.401067 |
| Corrected robust P3 common point optimiser | 2356 / 7.242949 | 2496 / 7.243352 | 1654 / 8.251674 |
| Previously registered equal incumbent/P3 blend | 2311 / 7.239365 | 2569 / 7.201558 | 1576 / 8.288021 |
| Newly constrained full-head incumbent, sequential | 2435 / 7.441466 | 2599 / 7.304946 | 1578 / 8.401067 |
| Joint incumbent role-ranking policy | 2589 / 7.441466 | 2559 / 7.304946 | 1578 / 8.401067 |
| P3 XV utility with native captain/sub heads, joint | 2446 / 7.242949 | 2619 / 7.243352 | 1578 / 8.401067 |

**Fixed Friendly-15:** not re-evaluated in this selection-only run; no new raw forecasts were produced. The exact historical count-stat/metres/minutes values are not reassigned from an incompatible prior cohort. The fixed 15-fixture list and strict chronological limits remain required for the next full-model comparison. This missing re-evaluation is visible, not a claim that the full four-benchmark model objective has passed.

NCR was deliberately left on the incumbent route for the Six Nations-native role-head policies, as registered. Thus they are unchanged against the NCR incumbent and lose 76 points against robust P3 there. This is not a cross-competition improvement.

## Admission and interpretation

Neither candidate passes the registered comparison against both incumbent renderings.

- Joint native roles gain 154 points in 2025 but lose 40 in the 2026 historical audit against the full-head constrained incumbent. Removing its most favourable round also reverses the pooled direction.
- P3 role transfer gains 11 and 20 points respectively against that stronger incumbent. Its pooled 31-point gain includes a 141-point gain in Six Nations 2025 round five. Removing that round leaves a 110-point deficit, or -12.222 points per remaining round. Its improvement over the historical empirical baseline also fails the single-round-removal rule.
- The same P3 role transfer gains 90 and 123 against plain robust P3. This is useful mechanism evidence that the point-only replay discards valuable bonus-role information. It is not permission to choose the weaker comparator and stop.
- The newly constrained sequential incumbent itself beats the common point-head rendering by 175 and 60 points. This is a better comparator representation, not a retrospective claim that the original native illegal squads were valid.

All intervals remain descriptive on a small, reused historical sample. The two predeclared candidates add to, rather than erase, the earlier multiple-candidate search. There is no fresh holdout, confirmed future superiority or relaxation of the original admission criteria.

## Measured explanation gained

The original 108-of-110 bonus-role diagnosis described P3 versus the empirical comparator, not every comparison. Against the stronger full-head incumbent, ordinary XV ranking is also consequential. In 2025, P3 role transfer loses 116 ordinary XV points and 35 additional captain points while gaining 162 full super-sub points. In 2026 it loses 6 ordinary XV points, gains 23 captain and 3 sub points. These sums reconcile exactly to +11 and +20.

Joint native roles differ from the sequential comparator in only one 2025 round: changing the super-sub gives +162 points while the necessary XV changes cost 8. The resulting +154 season gain is wholly concentrated. In 2026, different country-slot allocations cost 43 ordinary XV points for only 3 additional sub points. Optimising arbitrary standardised ranking utilities does not establish a calibrated exchange rate between these contributions.

The recorded failure diagnostics also show that the earlier kicking adjustment changed 1,723 forecasts without changing any of the 13 selections. Earlier role calibration made no XV changes, changed three captains for -21 points and one sub for +27. Fixed position/status offsets cannot reorder players within that group and mostly cancel under fixed XV positional quotas. Global point blends and small minute-component changes therefore did not directly address within-role ranking and constrained marginal decisions.

## Next executable work selected from this evidence

1. Run `research/decision_opportunity_diagnostics.py` on the exact original and timing-corrected stores, original archived raw predictions and the now-reproduced complete role-policy pools. Retain all feasible one-player exchanges, captain-only alternatives, unknown outcomes, event-weighted residuals and unresolved scoring reconciliation. Separate observed duration from participation: a recorded zero with positive activity is not an unused bench appearance.
2. Use those results to choose and register the next full-model experiment. The unresolved alternatives are conditional duration/event opportunity, prior opponent/team attacking opportunity, and calibrated role utility rather than another global blend or re-tuned z-score coefficient. Refit the full corrected-history P3 control alongside the structural candidate; the previous fixed-tree timing ablation is not that control. Preserve all three tournaments and the fixed Friendly-15 comparison.
3. Investigate whether the constrained slot trade-offs are forecast-mean errors or rank-utility scale errors. Do not replace the declared comparison with a favourable policy subset or tune the two failed policies on the audit.

The existing broader hypothesis register in `DECISION_CONTINUATION_2026-09-26.md` remains active, including missing historical prices and publication provenance. None of these candidate failures disproves the conditional-opportunity or context hypotheses.

## Execution failures kept distinct from research failures

The first runtime recovery run `36256561519` reproduced the exact original store, then failed because the overlay invocation lacked `PYTHONPATH`. Run `36256779492` corrected the invocation and retained the exact verified stores. The connector rejected the 921,188,861-byte runtime artifact because its per-file maximum is 536,870,912 bytes; run `36257138991` split it without rebuilding or changing contents. The local execution host subsequently returned ClientError/TransportError during recovery. Computation moved to Actions instead of treating this as an external scientific blocker. No research candidate was scored in those failed recovery attempts.

No tests were added or changed, including temporary tests. Existing focused checks ran in the successful role-study workflow. Original artifacts, cache, source history, failed candidates, credentials and November registration are preserved. No rugby API calls were made. The Actions workflow retains results only on the same draft PR branch and does not merge or change production routing.
