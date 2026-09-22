# Complete-pool model comparison

Protocol frozen before the new comparison fits on 22 September 2026.

## Data repair

The modern Six Nations teamsheets contain 1,380 appearances. The earlier comparison used 1,342 labelled appearances as its candidate pool. The repaired pool contains all 1,380 appearances, with 1,362 known outcomes and 18 unknown outcomes. An unknown outcome never becomes zero. MAE uses the known outcomes and reports its denominator. A selected squad with an unknown outcome has no reported total.

Eight modern Brex appearances match the official workbook through an Italy-specific name alias. Twelve unused substitutes have zero recorded minutes, no start, and no recorded scoring events or cards. Their zero labels have a separate source marker. Four earlier Brex appearances are also recovered. Minutes distinguish Tomos Williams and Teddy Williams in the official workbook; three previously incorrect Teddy scores change. The source remains the official workbook for those matches.

The public Oval Office 2026 table disagrees with the official workbook on 73 of 642 matched scores. It is retained as conflicting evidence and does not supply missing labels. The official fantasy catalogue returned HTTP 503 with site_active=False. Missing labels remain listed in missing_labels.csv.

Three NCR crosswalk corrections use exact names from cached teamsheets dated before NCR GW1: Kane James 243167 maps to 366690, Jaco Williams 252481 maps to 380118, and Leonel Oviedo 183329 maps to 169500. Duplicate fantasy or rugby identities now stop candidate preparation.

## Evaluation population

The canonical store contains 175,612 distinct player-match-team rows from 3,865 fixtures. Club and earlier international matches supply training history. They do not constitute independent official fantasy outcomes. Source counts and date ranges are recorded separately in source_inventory.csv.

The official-score comparison covers ten complete Six Nations teamsheets from 2025–2026 and NCR GW1–3. The Six Nations optimization remains an oracle-teamsheet exercise without archived prices. NCR retains its existing lock and budget rules. Scores from 2023 use an older rubric and are not pooled with modern official points. NCR GW4–7 occur in November 2026 and have no outcomes at the evaluation date.

The raw-event comparison covers 360 distinct international fixtures in 21 tournament-season blocks from 2022–2026. Each block freezes training and player history before its first fixture. The existing three-hour result-availability rule applies. No evaluation fixture enters that block's training set. Later blocks can use earlier completed matches. No club-only evaluation block is counted as international validation.

## Fixed candidates

The candidates are the existing empirical model, the native-category P3 50/50 blend, the existing event-weighted native blend, and p3_robust_native. The official comparison also retains the existing saved research references. No weights or hyperparameters are selected from these new results. The raw empirical event comparator differs from the official fantasy empirical baseline.

Existing weights were developed with historical research. Therefore this is a retrospective comparison with broader coverage, not an untouched prospective holdout. Repeated inspection of these seasons limits promotion claims. Production routing remains unchanged.

## Measures and uncertainty

For official scores, report player MAE, complete selected-squad totals, missing-label counts, and paired differences by slate and competition-season. Compare every model on the same observed player labels. Do not combine a partial squad total with complete totals. Report Six Nations and NCR separately because their rules and selection conditions differ.

For raw outcomes, report minutes and each available event separately, including the existing naive-relative loss and cohorts. Raw event units are not added together. Report equal-weight tournament-season summaries of relative loss, event-level differences, and losing competition slices. Match-level error sums and denominators support paired uncertainty calculations without treating player rows as independent matches.

Use paired resampling of tournament-season blocks for the main raw comparison, with 10,000 draws and seed 20260922. Also show competition-family consistency and leave-one-block-out sensitivity. Use paired slate resampling for official results, explicitly labelled exploratory because there are only 13 slates across three competition-seasons. Report 95% intervals and practical effect sizes. An interval containing zero permits both improvement and deterioration. It does not demonstrate equivalence.

Broader raw-event agreement can strengthen the research choice. It cannot establish superiority in official fantasy team points when labels, prices, or independent seasons remain limited. A definitive promotion decision requires frozen prospective forecasts and complete official outcomes.

## Reproduction

The workflow complete-model-comparison.yml runs 13 official-slate jobs and 21 raw-block jobs. Each job records source and data hashes, package versions, cutoffs, predictions, and existing test results. Model binaries are excluded from uploaded evidence. No paid API calls are required.

Run an official slate with `python -m model.unified.rolling_eval --native-categories --competitions six_nations --round-job six_nations_2025_r1 --output OUTPUT`.

Run a raw block with `python -m research.raw_comparison --fold six_nations_2022 --output OUTPUT`.

Before any successful fit, source inspection identified newly recorded events without earlier international training support: fifty_22, lineout_steals and potm in Six Nations 2023; scrums_won in 2025; kicks_retained in 2026. The raw comparison records each model's forecast support separately. Paired metrics use only observed rows with forecasts from every candidate. Unexpected missing forecasts for a target with prior international support stop the job. Unsupported targets are not replaced by zero.

Aggregate all 34 unpacked evidence directories with `python -m research.summarise_complete_comparison --evidence DIRECTORY --output OUTPUT`. The aggregator rejects duplicate folds, overlapping evaluation fixtures, missing jobs, differing source/store hashes, or unequal stable-target coverage. Its main raw summary weights the 23 stable events equally within each block and the 21 blocks equally. Minutes and extended events remain separate diagnostics. All intervals are exploratory and are not adjusted for multiple comparisons. Tournament blocks can share teams and training history, so bootstrap intervals do not remove all historical dependence.

NCR's existing historical limits also remain: GW1 excludes New Zealand and France; GW3 uses corrected final lineups and retrospective prices. The empirical fantasy baseline is a recomputation under common history conventions, not saved production forecasts. This study does not repair unavailable historical publication times or prices.

## Playing-role correction before the final fits

The first incomplete run exposed substitute jersey numbers being treated as playing positions. Pollock's number 21 became Scrum-half despite earlier recorded starts at number 8. This changes model features, event priors, try weights and squad selection. The unfinished run 35731009242 was stopped and is not the final comparison.

The corrected store uses the current jersey for starting players. For substitutes, it uses the latest recorded starting role that was available more than three hours before the match. Raw-block candidates use only that block's training history. This removes future-derived global modal positions and does not assume a fixed five-forward/three-back bench. All 480 modern Six Nations substitute appearances have prior-start support, including at their tournament-start cutoff. Their inferred playing roles are not claimed to reproduce an unavailable historical fantasy catalogue.

Across the full store, 51,133 substitute appearances have prior-start support; 10,073 have none and receive the explicit Unknown position. No unknown-role Six Nations slate is accepted. The existing binary is_forward feature is false for Unknown; raw-event cohorts expose Unknown separately, and no official fantasy result uses an unknown role. NCR candidates retain their fantasy-catalogue positions. Every model is refitted because the corrected training roles affect NCR as well as Six Nations.

The correction changes 143 modern Six Nations substitute roles and 38 forward/back classifications versus the previous jersey mapping. Historical differences are inference changes, not independently verified official position changes. position_changes.csv records the changed rows and their source classification.

All official models share the same current-slate playing roles and selection constraints. The official tournament-frozen control freezes fitted parameters and form history, while accepting the current teamsheet and the same pre-slate role metadata as its competitors. Its prediction roles and pool roles match. It is not a forecast made before the future teamsheet was known.
