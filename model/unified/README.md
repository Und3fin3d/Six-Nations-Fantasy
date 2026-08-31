# Unified rugby event model

This module replaces competition-specific fantasy-point learning with one model
of rugby. It predicts raw match events and minutes; deterministic scoring
adapters convert the same `RawPrediction` into Nations Championship or Six
Nations points.

## Active model: P3 event blend

The selected unified candidate is `p3_event_50`: a fixed global 50/50 blend of
the raw-event empirical engine and v4 GBDT. P3 had the best competition-balanced
stable score in Historical Raw Benchmark v1 (`0.886470`, 2.36% better than v1)
and was the only challenger to pass every frozen promotion gate.

P3 artifacts are self-contained: each serialized blend contains its fitted
empirical and v4 components. The separate component artifacts and the failed
v1-neural, GBDT-v3 and v5 artifacts have therefore been retired. Their source,
fold manifests and aggregate benchmark metrics remain so the selection decision
can still be audited. The latest v1 GBDT and competition-specific incumbents are
retained strictly as rollback models.

Do not retune P3's global 0.5 blend weight on NCR GW4-7. Those rounds are the
predefined live severe-failure veto, not another development set.

### Local event-weight hill climb

The research-only `p3_event_weighted` candidate gives each stable raw-event
head an optional competition-independent empirical/v4 weight. It optimises on
2022-2024 and uses 2025 as a soft selection guard. The selector does not use
2026. The existing 2026 and official NCR results are retrospective diagnostics.
Both fantasy adapters remain secondary diagnostics.

```bash
/tmp/6n-model-pinned/bin/python -m model.unified.cli raw-benchmark p3-hillclimb
```

The command writes its config, ledger, source manifest, raw metrics and report to
`data/unified/p3_hillclimb/`. It does not change the active P3 artifact or the
NCR and Six Nations incumbents.

### Retrospective NCR checkpoint search

The checkpoint search starts from the global event-weight hill-climb config. It
uses one competition-independent raw-event weight vector. It rejects candidates
that fail the frozen development, 2025, or per-fold raw-event guards. The
per-fold guard includes available extended-event and 2026 raw labels. It then
uses official NCR GW1-3 team points as a retrospective stopping checkpoint.

```bash
/tmp/6n-model-pinned/bin/python -m model.unified.cli raw-benchmark p3-checkpoint
```

The command writes the full trial ledger, selected config, source manifest, raw
guard evidence, direct official NCR verification and report to
`data/unified/p3_checkpoint/`. The result is retrospective optimisation because
the search uses 2026 raw labels and official fantasy results. The model form is
competition-independent, but NCR supplies competition-specific selection
feedback. The command does not promote the candidate or change an incumbent.

### Consolidated research findings

`data/unified/research_synthesis/REPORT.md` preserves the verified findings from
the superseded Six Nations champion, cross-fitted P3, and empirical-engine PRs.
It retains their useful diagnostics without merging their parallel runners or
large trial trees. PR #6 remains the canonical event-weight implementation.

## Legacy v1 workflow

```bash
/tmp/6n-model-pinned/bin/python -m model.unified.cli build-data
/tmp/6n-model-pinned/bin/python -m model.unified.cli train \
  --engine gbdt --asof 2026-07-04

# Rollback prediction: candidates.csv contains future player-fixture rows only.
/tmp/6n-model-pinned/bin/python -m model.unified.cli predict \
  --engine gbdt --model data/unified/models/unified_gbdt_2026-07-11.pkl \
  --competition ncr --candidates candidates.csv \
  --output data/unified/shadow_predictions.jsonl
```

Every `--asof` cutoff is exclusive. Features are built with shifted histories,
validation is chronological, duplicate source rows are merged at the
`(fixture_id, player_id, team)` grain, and unavailable labels remain masked.

The GBDT artifact is the retained universal rollback control. The neural model
was evaluated and retired after materially underperforming on the historical raw
benchmark. There are no competition-specific or direct fantasy-points heads in
the active unified candidate.

Keep the NCR and Six Nations incumbents as rollback adapters until P3 clears the
frozen NCR GW4-7 severe-failure veto.

## Retired experiments

The sections below document prior experiments. They are not active deployment
candidates and their fitted binaries have been removed where they are not a P3
dependency or rollback artifact.

### v2 — ranking-first stack

v1 predicts events and *refuses* to learn the event->points mapping, so the NCR
incumbent (which is tuned on ranking) wins. v2 keeps v1's raw-event GBDT as a
frozen **stage 1** and adds a pooled **stage 2** that learns fantasy production
directly:

```
stage-1 UniversalGBDT  ->  event expectations + deterministic expected points (s1_exp_points)
                         ->  stage-2 features (+ scoring-rubric vector, no competition id)
stage-2 RankStack       ->  LGBMRanker (ordering) + Huber LGBMRegressor (points)
                         ->  expected points, blended toward s1_exp_points
```

Labels are official fantasy points pooled across Six Nations and NCR
(`labels.py`), normalised to a within-round percentile for the ranker. The
regressor predicts points directly and is anchored by the already
competition-scaled `s1_exp_points`, so it needs no per-competition calibrator
and stays in-range for a competition it never trained on (the NCR GW1 transfer
case). The scoring rules enter only as the rubric vector (`rubric.py`); there is
deliberately no competition-identity feature.

```bash
# honest cross-competition benchmark (writes data/unified/benchmark_v2.md)
/tmp/6n-model-pinned/bin/python -m model.unified.benchmark_v2

# deployable stack for shadow prediction (Six Nations 2026 held out)
/tmp/6n-model-pinned/bin/python -m model.unified.cli train-stack \
  --stage1 data/unified/models/unified_gbdt_2026-07-11.pkl --holdout-2026
```

Evaluation is out-of-sample by construction: stage-1 is re-cut per leg
(2026-02-01 for the sealed Six Nations season, 2026-07-04/11 for NCR GW1/GW2),
NCR is scored leave-one-gameweek-out, and cohort parity is guaranteed by scoring
exactly the incumbent projection cohort. Promotion still needs the prospective
November shadow rounds.

### v3 — leak-free unified supermodel

v3 returns to the intended contract: one competition-independent model predicts
minutes and raw rugby events, and the existing deterministic adapters convert
those predictions to Six Nations or NCR fantasy points. It removes the learned
fantasy-points rank stack from promotion decisions.

```bash
# Data, scorer, zero-minute and point-in-time feature audit.
/tmp/6n-model-pinned/bin/python -m model.unified.cli v3 audit

# Bounded rolling-origin searches (8 baselines, 12 GBDT-v3s, exactly 20 neural configurations).
/tmp/6n-model-pinned/bin/python -m model.unified.cli v3 tune --engine all

# Fit an immutable pre-lock artifact and make it the current GW shadow model.
/tmp/6n-model-pinned/bin/python -m model.unified.cli v3 train \
  --engine gbdt_v3 --cutoff 2026-11-07T00:00:00Z \
  --output data/unified/v3/models/gbdt_v3_ncr_gw4.pkl --activate-shadow

# Official-points architecture selection and retrospective references.
/tmp/6n-model-pinned/bin/python -m model.unified.cli v3 benchmark

# Manual write-once shadow freeze (normally called by gw_update.sh).
/tmp/6n-model-pinned/bin/python -m model.unified.cli v3 shadow \
  --gw 4 --engine p3_event_50 \
  --model data/unified/raw_benchmark/v1/models/p3_event_50/nations_championship_2026.pkl

# After GW7 labels arrive, apply the combined prospective promotion gates.
/tmp/6n-model-pinned/bin/python -m model.unified.cli v3 shadow-evaluate \
  --engine gbdt_v3
```

All timestamps, audit results, bounded-search results, model manifests,
benchmarks and immutable shadow files live beneath `data/unified/v3/`. A fold
trains only on matches strictly before its round lock. Six Nations 2026 and NCR
GW1–2 are retrospective references; Six Nations 2025 is the untouched
architecture-selection season; NCR GW4–7 are the prospective promotion test.

### v4 — hierarchical EB player effects (P3 component)

v4 keeps the v1 direct-totals architecture and attacks the diagnosed NCR
failure (thin/mixed-history players regressing to positional means): the
`player_id` categorical is pooled away and replaced with post-hoc
empirical-Bayes multiplicative per-player effects (moment-matched K, training
frame only), plus level-split club/international form features. EB rate
*features* (variant A) and hurdle heads (P2) were tested and killed by
precommitted rules on the 6N-2025 LORO selection layer; NCR GW1–2 and 6N 2026
were fully quarantined from all admission decisions.

```bash
# admission experiments (writes data/unified/v4/p1_*, p2_*)
/tmp/6n-model-pinned/bin/python -m model.unified.v4.experiments A B
/tmp/6n-model-pinned/bin/python -m model.unified.v4.p2

# final report + display-only diagnostics (data/unified/v4/REPORT.md)
/tmp/6n-model-pinned/bin/python -m model.unified.v4.final_report

# November GW4-7: fit pre-lock, activate as an additional shadow engine
/tmp/6n-model-pinned/bin/python -m model.unified.v4.assemble fit \
  --config B --cutoff <GW_LOCK_UTC> \
  --output data/unified/v4/models/gbdt_v4_ncr_gw<N>.pkl
/tmp/6n-model-pinned/bin/python -m model.unified.v4.assemble activate \
  --model data/unified/v4/models/gbdt_v4_ncr_gw<N>.pkl --target-gw <N>
```

Plan, adversarial-review ledger and frozen gates: `data/unified/v4/RESEARCH_PLAN.md`.
Promotion uses the two-tier gate: 2026 only via decisive superiority on pooled
prospective GW4–7 (90% CI excluding zero on MAE and mean capture vs the NCR
incumbent while holding 6N); non-inferiority is deferred until the clean
prospective sample is powered (~20+ GWs). Incumbents stay deployed throughout.

## Historical raw benchmark v1

The stable/extended benchmark evaluates the competition-independent event seam
directly on fixed historical tournament holdouts. It rebuilds label availability
from the permanent match JSON cache: an absent `match_stats` key is unavailable,
not a zero. The legacy canonical store and all frozen v1-v5 artifacts remain
untouched.

```bash
# Write-once corrected store, source coverage ledger and event catalog.
/tmp/6n-model-pinned/bin/python -m model.unified.cli raw-benchmark audit

# Fit all raw candidates on 2022-2026 tournament blocks.
/tmp/6n-model-pinned/bin/python -m model.unified.cli raw-benchmark run \
  --engines v1 gbdt_v3 v4 v5_t empirical_event p3_event_50

# Aggregate stable/extended event metrics and reconstructed observable rubrics.
/tmp/6n-model-pinned/bin/python -m model.unified.cli raw-benchmark report

# Run the raw-eligible v1 neural challenger without rewriting the frozen core report.
/tmp/6n-model-pinned/bin/python -m model.unified.cli raw-benchmark run \
  --engines v1_neural

# Run the champion's limited native Six Nations event-head diagnostic, then report challengers.
/tmp/6n-model-pinned/bin/python -m model.unified.cli raw-benchmark champion-diagnostic
/tmp/6n-model-pinned/bin/python -m model.unified.cli raw-benchmark challengers-report
```

Artifacts live beneath `data/unified/raw_benchmark/v1/`. Historical teamsheets
are oracle lineup inputs. Historical raw performance is the primary selection
gate; NCR GW4-7 is retained only as a severe-failure confirmation veto.
The v2 rank stack remains points-only and therefore cannot enter raw-event loss;
its independent raw input is already represented by v1. The Six Nations champion
is reported only on the 12 stable event heads plus minutes that it genuinely emits.
