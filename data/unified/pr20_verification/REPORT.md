# Independent verification of PR #20

The recorded arithmetic is correct. The native robust candidate improves the previous robust candidate within the evaluated cohorts. The evidence does not establish universal superiority or complete-pool Six Nations team superiority.

This audit checks merged source `48a8d2574d40ebb9def2335c55e5241ce5242df5` and the original Actions artifacts. It preserves the distinction between saved-result recalculation, source inspection and fresh model fitting.

## Recalculated results

Each cell contains fantasy MAE / selected-team points. MAE is the equal-weight mean of round MAEs. Team points include captain and substitute multipliers.

| Model | NCR 2026, three rounds | Six Nations 2025, five rounds | Six Nations 2026, five rounds |
| --- | ---: | ---: | ---: |
| Empirical baseline | 8.401189 / 1578 | 7.139481 / 2410 | 7.389271 / 2548 |
| Previous robust candidate | 8.368760 / 1665 | 7.243192 / 2267 | 7.326482 / 2505 |
| Native robust candidate | 8.364626 / 1740 | 7.231897 / 2416 | 7.254336 / 2550 |

The native candidate improves both reported metrics against the previous robust candidate in all three seasons. Against empirical, it improves MAE in two seasons and aggregate team points in all three restricted comparisons. Six Nations 2025 MAE regresses by 0.092415. The +6 and +2 Six Nations point leads remain small.

The NCR candidate scores 513, 694 and 533 against empirical 529, 493 and 556. Its aggregate gain comes from GW2. Its selected super-sub contributes 195 points in that round. The gain is not an every-round improvement.

## What was independently checked

- Recalculated all 65 native and 65 preceding model/round MAEs from saved player predictions and repository official labels.
- Recalculated all 130 selected-team scores from official points, minutes and status, with separate XV, captain and substitute contributions.
- Checked all 130 squads for unique players, position counts, nation limits, captain/substitute eligibility and cohort membership. Checked all 30 NCR squads for budget and hemisphere limits.
- Rescored all 78 saved rolling raw-prediction files using independent numerical calculations. The maximum difference from saved expected points was 1.42e-14.
- Reconstructed all 30 season/model summary rows. Maximum MAE difference was 1.78e-15; team-point totals matched exactly.
- Compared empirical predictions, candidate IDs and outcomes across the two runs. IDs and outcomes match exactly. Baseline prediction differences are at most 2.14e-14.
- Recomputed all six native-versus-empirical paired round-bootstrap intervals with seed 17 and 10,000 draws. They reproduce the report. All three team-point intervals include zero; the Six Nations 2025 MAE interval lies above zero.
- Inspected source and input manifests for all 13 native jobs. Native and preceding runs have equal data, cache, store and round-history hashes. Latest recorded training match timestamps precede locks by at least 5.58 hours.
- Inspected candidate masking, prior-only feature construction, chronological history admission, frozen controls and opt-in categorical handling. The current source differs from the recorded native source only in the reviewed evaluation-manifest correction.

`verification.json`, `round_metrics.csv`, `summary.csv`, `intervals.csv` and `baseline_parity.csv` preserve the calculations. The two original compressed Actions evidence packages are retained in `evidence/`; their hashes and origins are in `manifest.json`. These preserve the saved predictions instead of depending solely on expiring Actions storage.

## Fresh NCR GW2 reproduction

A new fit from corrected PR #20 source `599a51987eaac8830c6d1b20870c8c6c3b6747b7` reproduced the decisive NCR GW2 comparison. The isolated run rebuilt the original 174,508-row store. Its store and weight hashes match the original job. The round manifest also matches: 171,794 training rows, 3,782 fixtures and 276 earlier NCR rows. The only model-source difference is the reviewed evaluation-input manifest correction.

All four MAEs, selected player IDs, captain/substitute assignments and realised team totals match the original job. Fresh expected-point differences are at most 1.07e-14. `p3_robust_native` again has MAE 8.308258510134976 and 694 team points, against empirical 8.39324427480916 and 493. The rolling and weighted candidates reproduce 737 and 681 team points respectively. This confirms the recorded result; it does not remove its selection and coverage limits.

The run used Python 3.11.15 and the pinned numerical packages with `OMP_THREAD_LIMIT=2` and `OPENBLAS_NUM_THREADS=1`. Its fit-and-score stage took 730.9 seconds. `fresh_ncr_gw2.csv` records every candidate comparison. `fresh_ncr_gw2.json` records the command and provenance checks. `evidence/fresh-ncr-gw2-evidence.tar.gz` preserves the new predictions, squads, manifests and log, without fitted binaries. Only this round was freshly fitted; all 13 rounds were recalculated from their saved evidence.

## Newly identified limitation: incomplete Six Nations pools

`rolling_eval.official_slates` filters `model_targets.csv` on `official_pts.notna()` before constructing the comparison and optimiser pool. This excludes 38 of 1,380 teamsheet rows: 16 in 2025 and 22 in 2026. The actual totals are 674 and 668 candidates across the respective seasons. Individual rounds have 131–137 candidates instead of 138.

Twenty-six omitted rows have positive minutes. Twelve are unused substitutes. Therefore, the comparison includes available zero outcomes but does not include every unused substitute or every teamsheet player. `six_nations_missing_labels.csv` lists all omissions.

Some missing values reflect failed label joins. Juan Ignacio Brex is omitted in every 2025 round, while the official source contains Italy's J. Brex with 27, 5, 26, 11 and 19 points. Three 2026 rows also exist as N. Brex. Do not replace unmatched labels with zero. Resolve source identities and coverage first.

All models use the same restricted cohorts, so the saved comparisons remain internally comparable. However, the evidence does not establish that their rankings or +6/+2 team-point advantages survive complete-pool evaluation. A complete-pool replay must also obtain genuine historical prices before establishing budget-feasible superiority.

## Other limits that remain

NCR GW3 uses corrected final lineups and retrospective prices. GW1 excludes New Zealand and France for every candidate. The original NCR player crosswalk has the three identity errors documented in issue #21; the reported 8.364626 is the original, uncorrected reference. The corrected reference is 8.374655, and belongs to a different evaluation revision.

The empirical baseline is recomputed under the common history conventions. It is not a replay of saved production forecasts. Historical publication times are unavailable; the three-hour lag is an availability convention. Existing weights and repeatedly inspected 2025/2026 results prevent claims of untouched confirmation. The small numbers of rounds limit uncertainty estimates. No bootstrap interval corrects for all prior selection or repeated searches.

Friendly-15 and the later retained role/minutes package are outside PR #20. This audit does not newly verify their reported outcomes. No NCR GW4–7 outcomes were accessed, and no production model was promoted.

## Correct conclusion

Retain the history corrections and opt-in native categorical model as an incremental research improvement. Retain `p3_robust_native` as the shared research reference. Qualify every Six Nations team comparison as both price-free and restricted to matched official labels. Repair label coverage before claiming complete-pool improvement. The original decision not to promote a universal champion remains correct.
