# Consolidated all-rugby research findings

This report preserves verified findings from PRs #3, #4, and #5.
PR #6 supplies the canonical implementation and audit trail.
The superseded PR runners and large artifact trees are not merged.

## Decision

Merge the PR #6 event-weight search and retrospective checkpoint infrastructure.
Keep the active competition incumbents unchanged.
Keep `p3_event_50` frozen for the prospective NCR GW4-7 shadow.
Do not use any result in this report to retune that shadow.

## P3 event-weight finding from PR #4

PR #4 found that event type is the useful P3 blend axis.
Its cross-fitted C5 score was `0.879365878` against frozen P3 at `0.886470436`.
Its paired fold-bootstrap interval was `[-0.011121936, -0.003250625]`.

The PR #4 search and the sealed PR #6 search produced similar weights.
The 24 shared weights have correlation `0.971854`.
Their mean absolute difference is `0.040417`.
`p3_event_weight_comparison.csv` records every shared weight.

PR #4 used 2026 when it selected C5 from C1-C8.
Its clean-checkout command also omitted required component generation.
PR #6 therefore remains the canonical implementation.
PR #6 selects on 2022-2025 and uses 2026 only as confirmation.

## Empirical-engine findings from PR #5

The empirical control scored `0.939031042`.
Empirical-Bayes minutes shrinkage and removal of the matchup multiplier reached `0.909890483`.
Per-event player-history shrinkage reached `0.906895540`.
A separate median minutes target head reached `0.904537865`.
The no-player-signal variant had the best exploratory score at `0.900753906`.

These results identify useful mechanisms for a future empirical experiment.
No candidate passed the original gate against v1.
The final PR changed the acceptance comparison after trial results existed.
The empirical implementation and its 348 trial artifacts are therefore excluded.
`empirical_candidates.csv` preserves the verified trial scores and hypotheses.

## Six Nations champion findings from PR #3

The frozen champion scored `1.114404354` on 13 shared targets.
Calibration, red-card revival, pooled extra heads, and simpler minutes reached `1.031684792`.
Coverage extension produced a 24-target score of `0.972909992`.

The frozen champion emitted zero for `red_cards` and `drop_goals_converted`.
The available evidence supported red-card revival.
The evidence did not resolve drop-goal revival.
The simpler minutes estimate beat the champion minutes head on its home folds.

The evaluation contained only Six Nations 2025 and 2026.
It contained no south cohort and no second tournament family.
The final score remained worse than v1 at `0.874193257` and P3 at `0.857526977`.
The champion runner is therefore excluded.
`six_nations_champion.csv` preserves the comparable scores.

## Canonical PR #6 result

The PR #6 weighted P3 score is `0.877944912`.
Official NCR labels did not enter its weight search.
The candidate scored 1473 team points on retrospective NCR GW1-3.
The NCR incumbent scored 1595 points.

The guarded checkpoint reached 1630 points.
The checkpoint used official NCR GW1-3 and available 2026 raw labels.
It is a reachability result, not prospective validation.
It remains inactive.

## Next evidence

Run the frozen `p3_event_50` shadow on NCR GW4-7.
Evaluate all four rounds before a promotion decision.
Do not tune weights or mechanisms on those rounds.

If empirical-engine research continues, freeze the original gate first.
Add input hashes and clean-checkout generation before running trials.

If champion research continues, rebuild its feature stack on multiple tournament families.
Require north and south evaluation before accepting a universal champion variant.
