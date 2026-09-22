# Independent review

Initial pinned change: 086b474..12fbeee. Reviews used separate fresh-context agents.

## Standards

One documented violation: two superseded name-key helpers and their imports remained after the shared matcher replaced their calls. Removed the dead implementations.

Two heuristic findings: ambiguous collection names in the raw comparison and similar manifest construction in two runners. Renamed the collections by purpose. Kept the runner-specific manifests because their contracts differ; both validate the actual source, store and weights, and the aggregator requires the shared source hashes to agree.

No new code comments or pathlib imports were found. Complexity checks still identify existing large functions. The new helper and aggregation functions have no complexity findings.

## Spec

One confirmed blocker: three historical Six Nations blocks contain events with no earlier international training labels. Those predictions are unavailable. Added an explicit forecast-support ledger and common comparison mask. Missing predictions for historically supported events still raise an error. No unknown outcome is replaced by zero.

The initial workflow omitted pytest and used an incorrect pip cache path. Both are fixed. No fits completed in those failed setup runs. Aggregation was pending at the initial review and is now implemented separately.

No additional supported temporal-leakage defect was found in the changed evaluation paths. The review used source inspection and a read-only scan of all 21 folds. Reviewers created or changed no tests.

Initial findings: Standards one documented violation and two heuristics; Spec one run-blocking forecast-support defect. Supported defects were repaired before using comparison results.

Follow-up review confirmed the support fix and paired calculations. It identified missing cross-job checks for frozen weights, official evaluation inputs and raw-driver hashes; the aggregator now validates those plus package versions. It also identified UK-English naming and duplicate season aggregation. The new module uses summarise/favour, and both runners use the same season-summary function. A scan of all 483 stable event/block combinations found no naive-loss denominator near the numerical floor. End-to-end result aggregation still requires the completed remote artifacts.

The role-repair delta passed standards review. Scientific review confirmed strict pre-match role lookup and all ten rolling Six Nations pool/candidate alignments. It found 14 role mismatches in later official frozen controls. Those controls now receive the same current-slate role metadata and selection constraints as every official competitor; their fitted parameters and form history remain frozen. The protocol distinguishes this from raw-block role history. A fresh Six Nations 2023 pilot completed all four candidates, all 92 stable target/model rows, and explicit unsupported-target reporting. No test was added or changed.

## Complete cache population review

The final restricted-source audit found a specification defect: 1,817 cached opponent appearances were absent from its evaluation population. A whole-cache inventory also found 15 complete Pacific Nations Cup fixtures missing from the CSV population. These findings required a new population and fresh fits. The restricted-source run 35732721254 is retained as a diagnostic revision only.

### Standards

The cache-population delta had one documented violation: importing the aggregation module read the study manifest into hidden global state. The command-line entry point now loads the study and passes it explicitly to the reusable functions. One heuristic finding identified repeated cutoff literals. The store builder now binds the cutoff once and passes it to both stages. No new tests, code comments or pathlib imports were added.

### Spec

Independent direct inspection confirmed exact coverage of all 179,295 eligible cached keys, with no missing, extra or duplicate keys. All 3,131 cache-added rows match their source identity, opponent, jersey, starter flag, timestamp, direct-event values and availability. The six excluded fixtures have empty teamsheets. Historical cutoffs and prior-role rules remain strict. Unknown playing roles remain explicit and included; final diagnostics must report their frozen-candidate counts. No supported correctness finding remains in the population repair.

Population-delta findings: Standards one resolved documented violation and one resolved heuristic; Spec no remaining defect. Fresh all-cache results remain required.
