# Claude Code second opinion - latest

- Status: `completed`
- Created: `2026-06-30T06:22:03.694088+00:00`
- Evidence fingerprint: `c2e26aca28f2a2897bafc74e146195f335d16ee602867a423a6a0c983d718365`
- Authority: advisory only; numerical acceptance remains authoritative.

## Decision

- Verdict: `agree`
- Recommendation: `keep_incumbent`
- Confidence: `medium`

The opening-market PIT audit correctly terminates: delta000's aggregate value_team gain of +0.0031 barely clears the +0.003 threshold while exhausting the full MAE regression budget (+0.020), regressing selection (top15 -0.013, top30 -0.020), and failing three of four stability diagnostics (minutes_perturb 0/3, round_drop 1/4, team_drop 2/5). The monotone collapse of value_team as delta weight rises (δ=0: 0.717, δ=0.25: 0.712, δ=0.50: 0.712, δ=1.00: 0.699) closes the entire opening-market post-target family. Keeping the incumbent is well-justified. The one unresolved concern is capt_top1=0.0 across all five dev rounds with captain_upside_weight already at its tested ceiling — while statistically plausible at this sample size, no remaining captain lever exists within the current mechanism family.

## Supporting evidence

- delta000 value_team gain of +0.0031 meets the aggregate gate but simultaneously exhausts the MAE regression budget (+0.020) and regresses selection (top15 -0.013, top30 -0.020), yielding no Pareto improvement on the full metric set.
- Three of four stability diagnostics fail for delta000 (minutes_perturb 0/3, round_drop 1/4, team_drop 2/5; only position_drop 5/5 passes), confirming the gain is structurally fragile and confined to one or two specific rounds rather than a generalizable signal.
- The monotone decay of value_team across delta weights (0.717 → 0.712 → 0.712 → 0.699) rules out any regularised blend within the opening-odds post-target family; there is no weight at which a robust net gain exists.
- Incumbent round-team values span 0.577–0.858 and round-MAE spans 6.91–7.91, indicating high natural round variance that correctly demands the 3-of-5 robustness gate rather than aggregate metrics alone as the primary acceptance criterion.
- Closing-market invalidation for timestamp reasons is correct and removes the strongest potential market signal; prioritising PIT integrity over marginal aggregate gains is the right call given the unverified lag to the historical fantasy lock.

## Concerns

- capt_top1=0.0 across all five dev rounds: P(zero successes | p≈1/15) ≈ 0.70 so this is statistically plausible, but captain_upside_weight is already at its tested maximum (1.0) with all captain variant families exhausted — no remaining mechanism lever exists to improve captain accuracy under the current architecture.
- The delta000 aggregate value_team gain (+0.0031) clears the gate by only 0.0001 while failing three stability checks; this proximity means the accept/reject boundary is extremely sensitive to single-round outcomes in a five-round regime, and the gate may not reliably distinguish signal from noise at this margin.
- Round 4 (MAE=7.91, team_value=0.686) is a consistent outlier across all candidates; if this round has structurally unusual characteristics (high injury turnover, referee patterns, weather) not representable in five rounds, it may be systematically inflating stability-test rejection rates for borderline candidates.
- post_target_residual_clip=6.0 is permissive relative to the ~7-point average MAE; no clip sensitivity analysis has been reported within the resid030+delta010 incumbent framework, leaving open whether high-leverage single-match residuals are amplifying round-to-round variance on rounds 1 and 4.
- With over 200 tested configurations on five dev rounds, the marginal expected gain from further parameter sweeps within existing feature families is diminishing; the search surface is nearly exhausted and further gains likely require new data sources or structural model changes.

## What would reverse it

- Sealed-season evaluation shows capt_top1 > 0 in most rounds, indicating the five-round failure is sampling noise and the current captain mechanism is adequate for production.
- The opening-market timestamp lag to the historical fantasy lock is independently confirmed within acceptable tolerance, revalidating the opening-odds pipeline for re-evaluation of delta000 under a robustness design that addresses the 2-of-5 round distribution.
- Three or more additional dev rounds are added, reducing per-round influence below 12% and enabling meaningful re-evaluation of borderline candidates that currently fail the 3-of-5 gate by a single round.
- A new PIT-safe data source (e.g., verified pre-lock team selection announcements, betting exchange volume rather than price) introduces signal orthogonal to the current feature set.
- A structural competition change (scoring rule revision, squad constraint change, captaincy format revision) breaks the stationarity assumption of the dev-season residuals, requiring full recalibration of the residual scope and weight.

## Next experiments

1. **resid030_delta010_clip3** - The incumbent's post_target_residual_clip=6.0 is permissive enough that single-match outlier residuals propagate into the XV selection in high-MAE rounds (rounds 1 and 4: MAE 7.74 and 7.91 vs. mean 7.41). Tightening the clip to 3.0 within the exact resid030+delta010 framework — a combination not present in tested_candidate_names — may reduce round-MAE variance without meaningful aggregate regression, which would both improve stability and potentially lift value_team on the currently-problematic rounds. Minimum test: Set post_target_residual_clip=3.0, hold all other incumbent parameters fixed. Accept if max_round_mae decreases relative to incumbent (currently 7.91) and value_team does not regress by more than 0.003.
2. **resid030_delta010_group_minn5** - With five dev rounds and approximately six matches per round, team-position cells qualifying at min_n=3 may contain as few as three observations, producing high-variance residual estimates that look like signal in a few rounds but are not generalisable. Raising post_target_residual_group_min_n from 3 to 5 forces sparse team-position cells to contribute zero residual rather than a noise-amplified correction — a mechanism untested in the incumbent framework — and may stabilise round_team on rounds where team selection is unusual. Minimum test: Set post_target_residual_group_min_n=5, hold all other incumbent parameters fixed. Accept if value_team is flat or positive and round_team improves in at least 3 of 5 rounds without MAE regression exceeding 0.010.
3. **post_target_xgb_weight_zero** - The combiner already established LGBM dominance over XGB (combiner_lgbm_weight=1.0, combiner_xgb_weight=0.0), yet post-target processing retains XGB at weight=0.5. If LGBM dominance generalises to the post-target stage — a plausible structural extension not explicitly tested in the current incumbent framework — removing XGB from post-target (post_target_xgb_weight=0.0) would produce a fully LGBM-consistent pipeline and could reduce noise introduced by the weaker model at a high-leverage selection stage. Minimum test: Set post_target_xgb_weight=0.0, hold all other incumbent parameters fixed. Accept if value_team gain >= 0 and MAE does not regress by more than 0.010.
