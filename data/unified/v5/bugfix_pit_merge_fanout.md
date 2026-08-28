# Bugfix report — `test_shrunk_features_are_point_in_time` merge fan-out

## Symptom

Pinned env (numpy 1.24.3 / pandas 2.1.0 / py3.11): 11 passed, 1 skipped, 1 failed.
`test_shrunk_features_are_point_in_time` failed at `assert len(merged) == len(prefix)`
with 31 != 30; the extra merged row showed fixture `vet-finternational-*` for player
`vet`.

## Root cause (test-fixture bug, not a model bug)

`_history_frame()` builds fixture ids as `f"{player}-f{level}-{j}"` with `j`
restarting at 0 inside every `add()` call. The final call
`add("vet", 1, "international", 0.0, start="2024-06-01")` therefore emitted
fixture_id `vet-finternational-0`, colliding with the first vet row from
`add("vet", 12, "international", 1.0)` (j=0, dated 2022-01-01). The key
`(fixture_id, player_id, team)` = `("vet-finternational-0", "vet", "T")` appeared
twice in the full frame (rows 0 and 57) and once in the 30-row prefix, so the
test's inner merge fanned that single prefix row out to two rows: 30 + 1 = 31.
The merged tail row being `vet-finternational-11` is consistent with pandas'
left-order-preserving inner join (duplicate inserted at result position 3,
shifting all later rows by one).

Evidence that `add_shrunk_features` is innocent: it performs
`sort_values(...).copy()`, column-wise assignments, and a final
`drop(columns=temp_cols)` — row count in equals row count out; it can neither
duplicate nor drop rows. The point-in-time property itself was never implicated:
the failure tripped on the row-count guard *before* any value comparison.

## Why "make the key unique", not "dedupe rows"

The two colliding rows are semantically *different matches* (2022-01-01 vs
2024-06-01) that accidentally received the same synthetic id — the fixture
author's intent is clearly "vet's 13th international" (12 prior internationals
plus one recent out-of-window match, used by
`test_shrinkage_direction_prior_vs_observed`). Deleting a row would corrupt the
history the other tests rely on. A model-side dedupe was also rejected:
`ShrunkFormGBDT.predict_frame` relies on 1:1 row preservation for the
concat-then-select scoring pattern, and silent dedupe would mask genuine data
bugs that the canonical store already resolves upstream via `source_priority`.
Key uniqueness is a data contract; the fixture must honour it.

## Fix

1. `tests/test_unified_v5.py::_history_frame`: a per-`(player, level)` sequence
   counter (`seq`) makes fixture ids unique by construction across `add()`
   calls. Only one id changes: the final vet row becomes
   `vet-finternational-12`. No test references history-frame fixture ids
   directly (they filter by `player_id`/dates), and `fit_tables` never reads
   `fixture_id`, so no other assertion is affected (dual calibration
   arithmetic: 640 dual club minutes -> cal 0.92, unchanged).
2. `test_shrunk_features_are_point_in_time`: added an explicit fixture-contract
   guard, `assert not frame.duplicated(subset=key).any()`. This strengthens the
   test — a future key collision fails at the fixture, with a message pointing
   at the real cause, instead of surfacing as a misleading PIT violation. The
   original row-count and per-column equality assertions are untouched.

## Secondary fix — fragmentation PerformanceWarnings in `shrinkage.py`

`add_shrunk_features` previously issued ~201 single-column inserts (8 scratch +
4 scratch/event + 2 features/event over 31 rate events + 7 history/slot),
tripping pandas' "DataFrame is highly fragmented" PerformanceWarning. Columns
are now assembled in dicts and attached via two batched `pd.concat` calls
(scratch block, feature block) through a small `_concat_block` helper that
drops same-named columns first, preserving the old overwrite semantics.
Values, dtypes, column order, and row count are identical; all Series placed in
the blocks share `df.index` (feature arrays are positional numpy), so
concat alignment is 1:1.

## Verification (adversarial self-review, no code execution available)

- Uniqueness: 58 full-frame keys enumerated — all distinct; 30 prefix keys
  match the full frame 1:1 -> `len(merged) == 30`.
- PIT equality: features are `shift(1)` + causal ewm within `player_id`;
  players' dates are distinct, so full-frame-only rows (dual club 6-7 dated
  after dual's prefix rows, vet's 2024 row, bg/new players) cannot change any
  prefix row's aggregates; tables are prefix-fit and shared between both calls.
- Determinism: sort key `(date, fixture_id, team, player_id)` gives identical
  per-player ordering in prefix and full frames.
- No behavioural change to `fit_tables`, `shrunk_feature_columns`, the encoder
  feature list, or serialized artifacts.

Expected result on re-run: 12 passed, 1 skipped, no PerformanceWarnings from
`shrinkage.py`.
