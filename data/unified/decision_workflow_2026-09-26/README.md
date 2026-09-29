# Decision workflow evidence

This directory supports [the results report](../../../research/DECISION_WORKFLOW_RESULTS_2026-09-26.md). The original PR #24 archives and November P3 configuration remain unchanged.

## Contents

- `remedies/`: all thirteen rounds, chronological fitted weights, paired comparisons, failed admission decisions and source/input manifests. The full archive also contains every forecast, selected squad, role correction and kicking adjustment, plus 94 exact execution-source files.
- `incumbent/`: deployed-configuration comparison, policy legality, separate captain repair and representative final-source verification. The full archive retains all component forecasts and source snapshots. The captain archive preserves its separate corrected policy outputs.
- `minutes/`: the 1,224-row timing-availability overlay, rule/source manifest and reproducible builder. The archive additionally retains the fixture-level coverage ledger. Neither the original canonical store nor any match cache was overwritten.
- `timing/`: the registered empirical-component comparison, all thirteen round results, failed admission decisions, component-recovery checks and exact selection verification. Its archive also retains detailed component calculations and the rejected wrong-archive attempt.
- `rehearsal/`: historical GW2 forecast, both decisions and manifest. The original rehearsal verified all 116 input/output files. Large input copies are not duplicated here; the exact original model/history are existing Git LFS objects. These retained summaries are not a prospective capture and cannot pass its live validator.
- `verification/`: existing-suite output, final-source replay hashes, unchanged protected configuration, complexity comparison, checked-source hashes, and independent review findings with their resolutions.
- `bundles.json`: sizes and SHA-256 values for the compressed evidence. Archives contain historical research outputs, not fitted replacement models.

The pinned numerical runtime was Python 3.11.15, NumPy 1.24.3, pandas 2.1.0 and SciPy 1.11.2. Use the repository model environment. Install the declared model/unified dependencies if needed. No command below calls a rugby API.

## Reproduce the original canonical input

The minute-availability repair intentionally changes a new canonical build. Reconstruct the original comparison input at its pinned base when reproducing the historical remedies:

```sh
REPO="$PWD"
RUN=/tmp/6n-decision-reproduction
PY=/tmp/6n-pr-decision/venv/bin/python
mkdir -p "$RUN"
GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach "$RUN/base" 93f9670f38ed1699c0029192ac38227392b329fb
git -C "$RUN/base" lfs pull --include='data/ncr/club_player_match.csv' --exclude=''
(cd "$RUN/base" && "$PY" -m model.unified.rolling_eval --prepare-only --output "$RUN/original")
shasum -a 256 "$RUN/original/inputs/player_match.csv"
```

This cache-only rebuild was executed during verification. It reproduced 179,295 rows, 3,892 fixtures and SHA-256 `2b70331c0cf215def67831cd27c987362e2ec8330b2afb41c8d204f48ef17313` exactly. Use a new scratch directory on another run.

Extract the original PR #24 archives using [the methodology evidence instructions](../methodology_review_2026-09-26/README.md). Set `ORIGINAL_EVIDENCE` to that extracted job directory.

## Reproduce comparisons

Extract the new evidence bundles into scratch storage. `tar -xzf BUNDLE -C "$RUN"` creates the named `remedies`, `incumbent`, `captain_repair` and `minutes` directories. The timing archive creates `empirical_timing_recovered`. Verify bundle hashes against `bundles.json` before use.

```sh
"$PY" -m research.decision_replay \
  --store "$RUN/original/inputs/player_match.csv" \
  --evidence "$ORIGINAL_EVIDENCE" \
  --incumbent "$RUN/incumbent" \
  --output "$RUN/remedy-replay"
"$PY" -m research.decision_summary --run "$RUN/remedy-replay"
```

This uses the retained incumbent point heads and reruns the registered fantasy remedies. It does not refit tree models. The archived source tree records the exact implementation used for the retained run; subsequent correctness repairs are identified separately. The summary command can also reproduce paired statistics directly from extracted remedy forecasts and metrics.

A fresh incumbent fit is available through `python -m research.incumbent_replay --store ... --evidence ... --output ...`. Current source includes the captain variance repair. It therefore has a different captain-policy contract from the original compatibility run. Its point heads remain unchanged under the promoted configuration. One representative final-source fit reproduced the separate corrected captain replay within `3.56e-15`; an all-slate final-source refit is not claimed.

Reproduce the isolated captain correction from saved component heads with:

```sh
"$PY" -m research.incumbent_captain_replay --original "$RUN/incumbent" --output "$RUN/captain-replay"
```

Rebuild the corrected minute-availability overlay with:

```sh
"$PY" "$REPO/data/unified/decision_workflow_2026-09-26/minutes/build_timing_overlay.py" \
  --input "$RUN/original/inputs/player_match.csv" --output "$RUN/corrected-minutes"
```

The builder checks all 3,892 cached fixture hashes and all raw event values/availability masks. It produces a new store with SHA-256 `e4b20637e2b514d622bb35cc7c43762e0eddb4f3b4e50f1ca2e39155ae96e6be`. The original store remains unchanged.

Reproduce the empirical-component timing comparison with the registered original archive and extracted remedy comparison:

```sh
"$PY" -m research.timing_ablation \
  --old-store "$RUN/original/inputs/player_match.csv" \
  --new-store "$RUN/corrected-minutes/player_match.csv" \
  --evidence "$ORIGINAL_EVIDENCE" \
  --comparison "$RUN/remedies" \
  --output "$RUN/timing-replay"
```

This performs 39 empirical fits and keeps every archived tree forecast fixed. It checks component recovery before scoring. All thirteen squad totals matched corrected robust P3; no timing candidate met the registered admission criteria.

## Protected P3 workflow

The normal NCR update invokes the configured shadow and verifies an existing capture instead of overwriting it. Its source receipt comes from `ncr_snapshot.py`; live capture requires the current complete teamsheets and completion before lock.

An explicit historical rehearsal uses a separate output directory:

```sh
"$PY" -m model.unified.v3.cli shadow-rehearse --help
```

After real prospective captures exist, `shadow-outcomes` archives completed official results once. `shadow-evaluate` accepts exactly GW4–7 and requires all four complete immutable cohorts. These commands preserve the original P3 veto. Historical rehearsals and partial rounds cannot enter it. No live November record or prospective result has been fabricated.
