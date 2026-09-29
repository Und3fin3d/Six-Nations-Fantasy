# Competition scoring contracts, 26 September 2026

The shared NCR scorer now awards scrum points only when the forecast position is `Prop` or `Hooker`. Six Nations forecasts with an explicit 2026 season now award five points per drop goal.

## Versions

| Version | Contract | Use |
| --- | --- | --- |
| `ncr_front_row_v2` | Two scrum points for `Prop` and `Hooker`; zero for other positions. | Default NCR scoring and corrected research. |
| `ncr_unallocated_scrums_v1` | Two scrum points for every supplied player scrum event. | Explicit historical replay and the protected original `p3_event_50` experiment. |
| `six_nations_2026_v2` | Five points per drop goal; other shared adapter weights unchanged. | Explicit 2026 scoring. |
| `six_nations_legacy_v1` | Four points per drop goal; existing shared adapter weights. | Historical replay, 2025, and calls without a season. |

`scorer_for(competition, season=..., version=...)` selects a contract. An explicit version takes precedence over the season. `rolling_eval.expected_points` accepts `season` and `scoring_version`. Its round manifests record the selected version.

NCR sample scoring requires a canonical position when a nonzero scrum event is present. `is_forward` cannot establish front-row eligibility. The explicit legacy version retains the original behaviour, including the original random samples.

## Evidence

The current [NCR application](https://fantasy.nationschampionshiprugby.com/static-assets/fantasy/static/build/static/index.KJ99yB3e.js) restricts the two-point scrum award to front-row players. The [methodology review](METHODOLOGY_REVIEW_2026-09-26.md) records that source and the retrospective allocation intervention. The original July rule publication remains unavailable.

The 2026 workbook provides primary recorded-score evidence for the Six Nations drop-goal weight. In [Sheet5, row 65](../2026/sixnationmatch.xlsx), D. Edwards has one try, four conversions, 25 metres, four defenders beaten, two tackles and one drop goal. The recorded total is 35:

`10 + 4×2 + floor(25/10) + 4×2 + 2 + 5 = 35`.

The old four-point rule produces 34. This is the only nonzero drop-goal row in the 2026 workbook. The 2025 workbook contains none, so its totals cannot determine the 2025 drop-goal weight. The [official 2026 guide](https://www.sixnationsrugby.com/en/m6n/news/how-to-win-at-six-nations-fantasy-rugby-2026) also lists five points. The current game's public rules endpoint returned HTTP 503 during verification. The workbook, rather than an inaccessible game configuration, supports this correction.

## Retained contracts

Existing forecast and result artifacts retain their recorded scoring interpretation. Recalculation under a corrected version creates separate results. No model was refitted by this repair.

The original Six Nations incumbent uses the frozen reconstruction scale in `build_targets.score_components` and `model.baselines.score_recon`. That trained-model contract still uses four drop-goal points. The shared raw-event scorer, rolling evaluation, season-aware comparison baselines, benchmarks, rubric features and diagnostic callers use the corrected 2026 rule when their season is known. The generic prediction CLI selects the season from each candidate date and records the scoring version. Evaluation with a competition name uses each row's season; an explicit scorer object preserves its selected contract. Generic calls without a season retain the named legacy version.

Raw-event loss weights and the historical dual-rubric search seed remain unchanged. Those weights define historical research objectives; they do not allocate individual NCR scrum points.

## Verification

[Verification evidence](../data/unified/scoring_contract_2026-09-26/verification.json) records source hashes and deterministic calculations. Nine archived NCR forecast files contain 2,274 forecasts. The explicit legacy scorer reproduced the original expected-point formula and sampled-score formula exactly. Corrected scores matched the original minus non-front-row scrum points, with maximum floating-point error `3.56e-15`.

The worksheet calculation produced 34 under the legacy Six Nations contract and 35 under the 2026 contract, matching the recorded 35 points. Archived research files, cached API responses and existing tests were not changed.

Ten archived robust forecast files supplied a further 1,380 Six Nations forecasts. Every 2025 forecast remained unchanged. Each 2026 change equalled one additional point per expected drop goal, with maximum floating-point error `1.06e-14`.

The complexity check found 12 existing findings across the expanded set of affected files. A separate check of their `HEAD` versions produced the same functions and complexity values. This repair added no complexity finding.

The existing focused suite completed with 87 passes and two failures. The existing NCR category test supplies a nonzero scrum event without a position; the corrected contract rejects that ambiguous role. The existing audit count test expects 2,144 labels, while the retained repaired data contains 2,164. Both tests remain unchanged. The run covered unified scoring, NCR evaluation, raw benchmarks, v3, v4, v5, the pooled stack and rolling history.

After the final generic CLI routing change, the model and pooled-stack tests produced 12 passes and the same missing-position failure. Complexity checks on the CLI and evaluation module had no findings. Direct calculation on recorded rows confirmed zero change for two 2025 drop goals and one additional point for each of two 2026 drop goals.
