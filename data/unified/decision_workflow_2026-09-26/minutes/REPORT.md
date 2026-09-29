# Bench-entry and minutes-availability diagnosis

This report records the repair-only stage. The subsequent [timing comparison](../timing/REPORT.md) fitted the empirical component and reselected all thirteen squads. It found no squad changes. The original repair-stage report remains in the evidence archive.

The separate entry-multiplier branch is closed. Empirical event means already multiply per-minute rates by an unconditional minutes estimate. Its minutes history includes observed zero-minute bench rows. V4 fits direct event totals on all available rows, including zero-minute bench rows. The weighted blend averages event distributions separately from its minutes marginal. Another entry multiplier would therefore apply entry risk twice without a supported conditional-event forecast.

The source audit identified a separate timing-availability defect. `ingest_6n.player_minutes` returns zero for every substitute with no entry event. The corrected-store parser previously marked every such inferred duration observed, even when the same cache recorded on-field activity.

## Measured support

The immutable canonical input has 179,295 rows across 3,892 fixtures. Its SHA-256 is `2b70331c0cf215def67831cd27c987362e2ec8330b2afb41c8d204f48ef17313`.

- All 62,488 bench rows had observed minutes. There were 4,012 zero-minute bench rows, including 527 international rows.
- Stable match-stat targets were observed for 3,880 of those 4,012 rows, including 511 of 527 international rows. V4 therefore does admit zero-minute examples.
- Of the zero-minute bench rows, 1,317 had a positive observed stable on-field event, excluding cards.
- Exactly 1,224 rows had positive activity and no recorded substitution entry. This includes 211 international rows. These zero durations lack timing support.
- There were 85 rows with recorded entry at minute 80 or later. Those durations can legitimately truncate to zero and remain unchanged.
- There were 48 zero-duration rows with recorded entry before minute 80. Of these, 47 had positive activity. Eleven had an off timestamp before their first on timestamp; 27 had multiple entry events; ten had one entry and equal on/off timestamps. These rows remain unresolved and unchanged. They are not all on/off inversions.

Examples of absent-entry contradictions include Cian Prendergast in fixture 8518327, Emmanuel Meafou in 8523943, Fin Smith in 8523979, and Joaquin Oviedo in 8518363. Their caches contain 7, 7, 4, and 6 tackles respectively, but no substitute entry record.

## Repair

Only `model/unified/raw_benchmark/coverage.py` changed. A bench player with no recorded entry and positive observed stable on-field activity receives missing minutes, false minute availability, and `unobserved_entry_time` provenance. Recorded entries, starters, and all raw event values remain unchanged. The fixture coverage ledger now reports observed derived-label counts accurately. The parser's existing player and coverage stages were separated to satisfy the complexity check.

`ingest_6n.py`, permanent cache, archived canonical data, saved models, and archived results remain unchanged. No duration was estimated. No entry multiplier was added.

## Evidence and verification

`timing_v1/minutes_overlay.csv` records all 1,224 changes at `(fixture_id, player_id, team)` grain. `timing_v1/player_match.csv` is the separate corrected research input. `timing_v1/manifest.json` records source, input and output hashes. The corrected input hash is `e4b20637e2b514d622bb35cc7c43762e0eddb4f3b4e50f1ca2e39155ae96e6be`.

The builder reparsed all 3,892 cached fixtures. Every directly cached raw event value and availability mask matched the original canonical input. Cache hashes and the original canonical hash remained unchanged. Independent CSV readback found changes only in minutes, minute availability and minute provenance, each for exactly 1,224 rows. All triple keys and their order remained identical, with no duplicates. All 85 late-entry zeros and all 48 unresolved pre-80-entry zeros remained unchanged.

The existing raw-benchmark and rolling-history suites passed: 38 tests. The changed source and reproduction script passed the complexity check. `git diff --check` passed. No tests were created or modified.

No model was fitted and no squad was reselected using the corrected input. Model gains remain untested. The prior numerical results retain their original source and input hashes. Any future corrected-data comparison must identify this new input and compare complete matched cohorts separately from those archived results.

## Reproduction

From the repository root, select a new output directory; the builder refuses to overwrite an existing output directory.

```sh
PYTHONPATH=. /tmp/6n-pr-decision/venv/bin/python /tmp/6n-decision-workflow/bench/build_timing_overlay.py --input /tmp/6n-complete-labels/all-cache-final-prepared/inputs/player_match.csv --output /tmp/6n-decision-workflow/bench/timing_v1_reproduced
```

`zero_minute_classification.csv` contains the complete zero-duration diagnosis. `event_support.csv` contains per-event availability counts. `unresolved_pre80_zero_minutes.csv` preserves the separate unresolved cases. `readback_verification.json` records the final semantic comparison.
