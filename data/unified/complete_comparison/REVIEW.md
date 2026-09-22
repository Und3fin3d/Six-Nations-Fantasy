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
