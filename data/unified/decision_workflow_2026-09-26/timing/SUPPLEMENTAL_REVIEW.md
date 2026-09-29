# Supplemental independent review

This document preserves the initial supplementary pass. A final independent Astra review at ultra effort found no blockers and reproduced all 2,138 candidate scores within `1.42e-14`. It also verified all thirteen squads and admission calculations. See [the final review record](../verification/timing-astra-review.json). The original initial review remains unchanged inside the timing archive.

Reviewer: delegated read-only `timing_math_review`, gpt-5.6-sol, high reasoning effort.

Reviewed timing module SHA-256: `57493b4f486d151b36510e59eeabe4a55cd350ffce20bb0b6a787e1fc1472559`.

Reviewed protocol SHA-256: `3021e3876b0c11d6113a6c0ea9d6b1fb6a67bbf124a8600ed280135972f2e49e`.

The reviewer found no blocker in the component algebra, mixture moments, prior-only fitting and positions, corrected scoring, full candidate cohorts, unknown-outcome treatment, or separate comparator admission. The fixed-tree limitation agrees with the source and protocol.

One non-blocking finding concerned the comparator replay check at timing_ablation.py lines 140–148. It compares numerical metrics but does not itself compare selected IDs, named bonus roles, statuses, or unknown-selected counts. A tied optimiser result could therefore pass the numerical check with different selected exposure.

Resolution for this run: direct post-run verification compared all 624 comparator squad rows and all 39 comparator metric rows against the saved complete remedy run. Selected IDs, names, teams, positions, statuses, actuals, captain/sub flags, cutoffs, denominators, unknown counts and named bonus roles all matched. `verification.json` records these checks. No active source or results were changed to resolve the finding.

The reviewer performed no edits, tests, fits, model runs, network calls, or memory writes.
