# Phase 3 / Reviewer A — leakage, circularity, power, complexity, falsifiers

Attacks on `RESEARCH_PLAN.md` (initial). Each item is a specific defect with a
demanded remedy; the plan author must accept, rebut with file evidence, or
escalate to the user.

## A1. Fold leakage through the EB global tables (accepted-class: serious)

The plan says calibration factors `cal_e`, position priors, and slot-minute
priors are computed on "the training slice only", but the implementation
sketch computes per-row ewm aggregates on the whole frame and *tables* on an
unspecified slice. If any table is ever fit on data at or after a fold's lock
(including the evaluation season's earlier rounds in a different fold), the
features leak. Remedy: tables must be fit **per fold** inside the training
window, stored in the fold artifact's manifest (hash or payload), and the
append-invariance test must cover the *composition* (tables+rows), not just
the row-wise aggregates. The current test plan only covers row-wise
invariance.

## A2. Circular selection on three folds (accepted-class: serious)

Workstream 3 (my own) condemned v3 for ~45 quasi-comparisons on three folds
whose year spread is ~13%, then produced a plan that runs: 5-rung ablation
ladder + K/c sensitivity + Family C oracle + E2 guarded-rate branch — all on
the same 2022–2024 folds. "Successive elimination" with per-rung paired tests
is still multiple comparisons against v1 on shared folds; with n=3 fold means,
no rung difference below ~2–3% is distinguishable from fold noise. Remedy:
(a) cap rung decisions with a precommitted effect-size threshold, not
"any improvement"; (b) require same-direction in ≥2/3 folds *and* pooled
paired-bootstrap p05 > 0; (c) the rung winner is still only a *candidate* —
6N 2025 remains the selection layer; (d) count every fold comparison in the
ledger so the multiplicity is visible.

## A3. The low-history proxy slice is mis-specified (accepted-class: decisive)

The demonstrated scar (NCR low_history +0.59) is dominated by players with
**zero or near-zero international rows but rich club rows** (the NCR pool's
debutants and SH fringe). The plan's E3 slice "career_matches<5 on
international folds" mixes in 6N debutants (URC-heavy but already capped) and
misses the exact mechanism. Worse: a player with zero prior international rows
has no international *validation* rows in 2022–24 folds unless their first cap
falls inside the fold — the slice exists but is thin and era-biased. Remedy:
define the E3 slice as (i) zero prior international rows with ≥3 prior club
rows, evaluated on their first international appearances, plus (ii) <5
international rows; report slice n per fold; if any fold slice has n<30, say
so and treat that fold's slice as uninformative rather than averaging it in.

## A4. Underpowered Family C oracle (accepted-class: moderate)

A hierarchical fit on 2023–24 internationals for tries/tackles cannot produce
a K posterior tight enough to "contradict K=220" — try rates have ~1–2
positives per 10 player-matches; the shrinkage posterior will be wide by
construction. Spending a day to produce an inconclusive interval is fine only
if the plan stops pretending it is a validation gate. Remedy: reclassify the
oracle as *report-only*; the binding check on K/c is the precommitted
sensitivity ablation K×{0.5, 2}, c×{0, 0.55, 1} on folds.

## A5. Missing falsifiers (accepted-class: moderate)

- No falsifier for "club→test calibration is real": if `cal_e ≈ 1` with tight
  intervals for the volume events, the level-split machinery is dead weight.
  Demand: E3 must report `cal_e` with bootstrap CIs per event and drop the
  level split if the high-coverage events are all consistent with 1.
- No falsifier for the mask-hurts case beyond E1's 2pp threshold: masking
  also shrinks the predicted event set (contract surface). Demand: report the
  raw-deviance and calibration movement under masking, not just capture/MAE.
- No decision rule for *partial* success: if E1 explains ~3pp and E3 gains
  ~0.5% low-history, does the candidate proceed? Demand a written rule before
  E0 runs: proceed only if the candidate is projected (folds + E1 arithmetic)
  to clear every retrospective gate that failed in v3, with the arithmetic
  shown in the ledger.

## A6. potm attach oversold (accepted-class: minor)

`ncr_potm.csv` has 6 rows (GW2 only). Attaching them does not "make potm an
observable NCR event" in any way that moves the model: the potm head trains on
6N labels; six positives change nothing. Its real value is diagnostic fidelity
(store reconstruction for NCR rows) and a coverage asymmetry between GW1 (no
file rows) and GW2. Remedy: recharacterise as an audit improvement; do not
list it as a model change; extend the file only if new data arrives through
existing pipelines (no new collection).

## A7. Retrospective NCR rows leak lineup realism into evaluation — by design, but name it

The benchmark's NCR evaluation rows come from the post-match store, where
`started`/`jersey` are the actual lineups (public pre-lock, so PIT-legal), but
the *shadow* path derives them from fantasy status P/B (`shadow.py::
ncr_candidates`). Any drift between "actual lineup flags" and "status-derived
flags" makes retrospective NCR metrics systematically kinder than prospective
ones. Remedy: quantify on GW1–2 — compare store `started` vs status P/B
agreement; report the delta; if agreement <95%, expect prospective metrics to
degrade and say so in the shadow report template.

## A8. Guard threshold provenance (accepted-class: minor)

The "≥5 positives" guard and the 20-minute rate floor (E2) are new magic
numbers without provenance. Remedy: pick them once, record them in the ledger
as a-priori choices, and do not iterate them against folds — or fold them into
the (already criticised) multiplicity budget explicitly.

## A9. WhatReviewer A could not break

- Exclusive-lock folds, write-once manifests, sha256-chained artifacts,
  cohort parity, median-fill disclosure, and the untouched safe-overall gate:
  all carried over correctly.
- The E1 scorer-ablation rescore is legitimate counterfactual analysis because
  GBDT event heads are independent — zeroing one event's mean cannot
  contaminate another head's output; conclusion validity holds.
- Prohibited-feature list is complete w.r.t. every prior PIT violation on
  record (weather, closing odds, post-lock info).
