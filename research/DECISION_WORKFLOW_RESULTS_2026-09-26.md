# Fantasy decision workflow: results and operating boundary

The registered comparisons do not support replacing either competition's incumbent with a new shared candidate. Correctness repairs and immutable forecast capture are supported. Robust P3 remains a raw-event research reference. Its small average raw-stat gain does not establish better fantasy decisions.

The complete diagnosis is in [the methodology review](METHODOLOGY_REVIEW_2026-09-26.md). The [protocol](FANTASY_DECISION_PROTOCOL_2026-09-26.md) and [formula registration](DECISION_REMEDIES_2026-09-26.json) preceded these new candidate results. All historical outcomes had already been consulted. Neither 2026 nor the later chronological audit is an untouched holdout.

## Why the gains do not reliably reach squads

The raw headline averages 23 event losses. Eleven events do not score in Six Nations and account for 30.3% of the net robust-versus-rolling improvement. The scored events also improve in aggregate, so irrelevant events do not explain the whole result. Tries, assists and conversions regress while some other events improve. Those attacking events matter heavily for high-scoring selections.

The decision is also discontinuous. Fifteen ordinary places, one extra captain contribution and a three-times super-sub contribution amplify different errors. Against the historical empirical comparator, the two Six Nations seasons lose 110 squad points: ordinary XV −2, captain extra −54, super-sub −54. The corrected NCR scrum rule changes several super-sub choices and reduces robust's historical total from 1,739 to 1,654. The original apparent bonus-role success partly depended on an invalid scoring allocation.

A better marginal minutes prediction need not alter an event forecast. The raw model samples minutes and events separately; changing only minutes leaves deterministic expected fantasy points unchanged. Applying a second entry multiplier to an already unconditional event mean would count non-entry twice. Timing labels also need an availability audit before treating every recorded zero as an unused substitute.

Finally, the original official comparator was not the complete deployed Six Nations model. The new replay includes its point head, separate decision heads, and actual positional selection policy. It exposes why comparator names, features, constraints and policies must be stated separately.

## The actual Six Nations incumbent

All 1,380 candidate fixture/player/team keys align with the deployed configuration. No unmatched-player fallback was needed. Eighteen official outcomes remain unknown across the complete pools. They were masked before fitting and selection, then restored only for evaluation.

Two replays retain the same promoted model configuration. `native_store` excludes future round rows before fitting and season-wide normalisation, while retaining the materialised historical features. `lock_rebuilt` also reconstructs match-derived player form, own-team, opponent and team-play features using history available before the whole round lock. This prevents Saturday and Sunday fixtures from using earlier same-round results.

| Common optimiser and pool | Six Nations 2025 | Six Nations 2026 |
| --- | ---: | ---: |
| Incumbent, native feature store | 2,250 | 2,529 |
| Incumbent, features frozen at lock | 2,260 | 2,539 |
| Historical empirical comparator | 2,410 | 2,552 |
| Corrected robust P3 | 2,356 | 2,496 |

Robust gains 96 points in 2025 and loses 43 in 2026 against the lock-rebuilt incumbent. Across ten rounds its descriptive mean difference is +5.3 points, with a paired round-bootstrap interval of −17.5 to +29.0. Removing its largest positive round reverses the pooled direction. This does not establish superiority or non-inferiority.

The incumbent's full positional policy produces 2,626 and 2,686 points before the separate captain-head repair. Those totals are not legal-squad comparisons. The policy exceeds four players per country in three of five 2025 rounds and every 2026 round. It also uses 52 position assignments that differ from the common historical pool. The report now states its positional-diagnostic boundary and selects complete forecast pools without consulting observed points.

The captain audit found another path for later-season information: prior variance used latent historical targets fitted on later official labels. The production repair uses known modern official outcomes, otherwise deterministic reconstruction. Replayed native-policy totals fall to 2,600 and 2,660. All point, component, selector and super-sub heads remain identical. Country violations remain, and the point-head comparison above is unchanged.

Historical Six Nations prices are unavailable. All common-optimiser Six Nations results are price-free diagnostics with positional and country constraints. Historical aggregate publication times and bio/role mappings are not fully versioned. Rebuilding a match-time cutoff does not prove that every source was published before that cutoff. The current Six Nations production loader is also tied to historical seasons; these replay results do not claim a general live 2027 deployment.

## Registered remedies

Every candidate uses the same full pool and optimiser. Fitted adjustments use only earlier official slates in the same competition. Six Nations 2026 can use earlier 2025 data. NCR starts with its declared fallback because it has no earlier same-rubric calibration slates. Unknown selected outcomes make a total unknown.

| Forecast or remedy | Six Nations 2025 | Six Nations 2026 | NCR GW1–3 |
| --- | ---: | ---: | ---: |
| Actual incumbent, common optimiser | 2,260 | 2,539 | 1,578 |
| Corrected robust P3 | 2,356 | 2,496 | 1,654 |
| Corrected weighted P3 | 2,236 | 2,513 | 1,658 |
| Equal robust/incumbent point blend | 2,311 | 2,569 | 1,576 |
| Prior-only regularised robust/incumbent blend | 2,433 | 2,503 | 1,597 |
| Equal robust/empirical point blend | 2,501 | 2,492 | 1,576 |
| Prior-only regularised robust/empirical blend | 2,447 | 2,570 | 1,597 |
| Prior-only position/status calibration of robust | 2,319 | 2,539 | 1,654 |
| Historical kicking-role adjustment of robust | 2,356 | 2,496 | 1,654 |

The equal incumbent blend gains 51 points in development and 30 in the later audit. Its pooled gain disappears when one round is removed: the remaining nine-round difference is −2 points. It therefore fails the registered influence criterion. Its descriptive mean is +8.1 points per round, with an interval of −21.9 to +38.9.

The regularised empirical blend gains 37 points in development and 18 in the later audit. Removing its strongest round also reverses the pooled result. The regularised actual-incumbent blend has the largest pooled increase, 137 points, but loses 36 in the later audit. Choosing it from the pooled total would ignore the declared audit rule.

Role calibration reduces NCR MAE from 8.252 to 8.167 without changing any NCR squad total. It loses 37 Six Nations development points relative to robust and gains 43 in the later audit. Its relative ranking correction does not provide a stable decision gain.

The kicking adjustment slightly improves MAE and changes no selected squad, captain or super-sub across all thirteen slates. Its historical formula was recovered from Issue 21, but the original package was unavailable. This implementation executes that formula on the current comparison population. It does not reproduce the unavailable source package.

The corrected rolling control selects one player with an unknown Six Nations 2026 outcome. Its season total remains unknown. The comparison does not delete that player or replace the outcome with zero.

No initial registered candidate meets all advancement criteria. No model is promoted. The fixed equal blend remains an informative simple control, and robust P3 remains the established raw-event reference. Neither is labelled a validated fantasy-squad winner.

## Timing labels and the empirical-component comparison

The cache audit found 1,224 substitutes with positive on-field activity but no recorded entry time. Their inferred zero-minute durations were incorrectly marked observed. The repair marks their minutes unknown and preserves every raw event value, availability mask and candidate key. It leaves 85 legitimate late-entry zeros and 48 separate unresolved timing anomalies unchanged.

A [separately registered comparison](TIMING_REMEDY_PROTOCOL_2026-09-26.md) then measured the repair's effect on the empirical component. It completed 39 chronological empirical fits across all thirteen slates. Archived tree predictions stayed fixed. Each slate first reproduced the archived component identities, then applied the corrected empirical means and metres/minutes mixture moments. This isolates one component; it does not measure a full tree refit or provide a deployable replacement model.

| Empirical-component timing repair | Six Nations 2025 | Six Nations 2026 | NCR GW1–3 |
| --- | ---: | ---: | ---: |
| Realised squad points | 2,356 | 2,496 | 1,654 |
| Difference from corrected robust P3 | 0 | 0 | 0 |

Every selected player, captain and super-sub matched robust across all thirteen slates. Fantasy MAE improved slightly in 2025 and worsened slightly in 2026 and NCR. The candidate failed the registered admission criteria against robust, the actual incumbent and the fixed equal blend. The label repair is supported by the recorded match events. This component-only experiment does not show a squad-selection benefit.

The first attempt used an older archive with a different input hash. The component-recovery guard rejected it before scoring. The retained retry used the registered archive, formula and tolerance. Its 67,450 component cells passed recovery checks. All 624 comparator selection rows and all 39 comparator metric rows matched the earlier run. The archive reader now checks the canonical input hash before accepting predictions. Both attempts are retained.

## Scoring and evidence capture

The [scoring contract](SCORING_CONTRACT_2026-09-26.md) names corrected and legacy versions. NCR awards scrum points only to props and hookers. Six Nations 2026 awards five points for a drop goal; the workbook's recorded total verifies that weight. Historical 2025 and generic seasonless calls retain the explicit legacy rule where evidence is incomplete.

The original P3 November GW4–7 experiment keeps its exact model, history, legacy scoring and registered severe-failure veto. Its configuration is unchanged. Capture now freezes both candidate and incumbent forecasts, both selected squads, source data, source code, configuration, constraints and hashes. Live capture requires complete current teamsheets and completion before lock. Stale, incomplete, mismatched and partial records cannot enter evaluation.

Outcomes are written once, only after a fresh official snapshot shows all six matches completed and the final scheduled kickoff is more than 24 hours old. The evaluator reads the frozen incumbent and outcome records. It requires exactly all four registered rounds and preserves the original 5% MAE / 5 percentage-point mean-capture veto. Passing that veto cannot deploy a scorer with a known correctness defect.

The historical GW2 rehearsal produced 262 forecasts and verified 116 frozen files. Repeated forecasts were byte-identical. The rehearsal is explicitly ineligible for prospective evaluation. Real November capture and evaluation cannot run until the actual pre-lock inputs and later outcomes exist.

## Verification and reproduction

The existing full suite completed with 119 passes and two failures after the production repairs. One existing assertion expects 2,144 labels, while the data has 2,164. This failure predates this change. The other supplies an NCR scrum event without a position, which the corrected API rejects. Tests were not added or changed.

Direct scoring calculations reproduced the legacy NCR formula on 2,274 forecasts exactly. Corrected scores differ only by the removed non-front-row scrum term. Another 1,380 forecasts verified unchanged 2025 scoring and the extra expected drop-goal point in 2026. All 39 raw forecast files in the new remedy run reproduce their archived expected points under explicit legacy versions before any correction.

The registered remedy run completed all thirteen slates with source and input hashes unchanged between start and finish. A replay from the final source reproduced all fourteen result tables byte-for-byte, including forecasts, selected squads and admission decisions. The original input also rebuilt byte-for-byte from the pinned base and permanent cache. The first remedy attempt stopped at a missing concurrent incumbent artifact; it is retained as incomplete and contributes no selected result.

Independent standards and requirements reviews checked the source and evidence. The captain replay now validates all thirteen original data/configuration hashes before and after execution. Its ten corrected slate files reproduce the pinned repair exactly. The complexity check covered all 38 changed Python files. It reports nineteen existing findings, with none new or increased against the base. Every completed remedy, fitted weight, selection, unknown outcome and failed admission is retained.

See [the evidence index](../data/unified/decision_workflow_2026-09-26/README.md) for commands, manifests, per-round results and verification. Bootstrap intervals describe a small, correlated historical sample. They are not confirmatory evidence.
