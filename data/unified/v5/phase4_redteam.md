# Phase 4 — final red-team gate on the v5 plan

Six questions, answered against `RESEARCH_PLAN.md` (final, post-review).
Verdicts: PASS / PASS-WITH-CONDITION / FAIL.

## Q1. Is the plan temporally leak-free? — PASS-WITH-CONDITION (recorded)

Checked every information path:

- **Fold harness**: exclusive locks, evaluation fixtures removed from training,
  write-once manifests (`model/unified/v3/harness.py`) — unchanged, sound.
- **EB tables**: after L1, all global tables (cal_e, position priors, slot
  minutes, global means) are fit per fold on the fold's training window and
  hashed into manifests; composition-level append-invariance is a test
  requirement, not a hope.
- **Row features**: all shifted groupby transforms; v1's append-invariance
  property is preserved by construction and re-tested for the new columns.
- **POTM attach**: post-match actuals; used for store reconstruction fidelity
  and (post-lock) as training labels only. Evaluation rows never expose it to
  features (features are shifted histories). GW1 has no POTM rows — a coverage
  asymmetry, disclosed as diagnostic-only (L6).
- **Lineup flags**: store `started`/`jersey` are actual, pre-lock-public
  lineups; the residual retrospective-vs-shadow realism gap is measured by
  E0c (L7), not ignored.
- **Residual finding R1 (recorded, mitigated)**: E1 chooses masking using
  NCR GW1–2 rows — the same rows the retrospective gate reads. That makes the
  retrospective NCR layer **corroborative rather than decisive** for the
  masking component. Mitigations: (i) the masking decision is structural
  (position-level attribution bias of the feed, E0a) and independently
  prior-ARTed by the incumbent's published choice to drop the same event;
  (ii) the contamination is symmetric — the incumbent's own components were
  tuned on GW1–2 as well, which is precisely why the protocol already makes
  GW4–7 mandatory for everyone; (iii) the prospective layer is untouched by
  E1. No remedy stronger than disclosure exists without destroying the
  retrospective layer for both competitors; the plan discloses it (§8, §13).

## Q2. Can the first experiments falsify the central hypothesis? — PASS

Central hypothesis: *attribution bias + missing shrinkage priors explain the
NCR gap, and repairing them yields safe-overall non-inferiority without
deviance regression.* The first three experiments are each capable of killing
it before any model is built: E1 (<2pp attribution recovery → S1 stop), E3/K1
(EB gain <1% low-history observable MAE in ≥2/3 informative folds → EB layer
dropped to terminal rung T), E0a (position-uniform reconstruction bias →
attribution theory weak). The plan spends ~1 CPU-day before its first model
fit and can end the cycle there.

## Q3. Fair comparison vs v1 and both specialists? — PASS

- v1 re-run as A0 in identical folds; ladder decisions are paired against it.
- Specialists enter only as frozen pre-lock artifacts
  (`model_predictions_2026.csv`, `ncr_gw{1,2}_projections.csv`), same as v3.
- Cohort parity and median-fill disclosure retained; matched-only metrics
  reported alongside (workstream 3, §3).
- The 6N 2025 selection layer actually *favours* the 6N incumbent (it was
  tuned on the 2025 backtest, `RESEARCH_GOAL.md`) — conservative for the
  candidate, disclosed here for completeness.
- Gate asymmetry: none. The safe-overall rule is identical for every engine.

## Q4. Sample size adequate for every learned component? — PASS

- Tree heads: the v3 search-validated regime (31 × 180 × 23, min_child 35,
  ~200k rows); unchanged.
- EB layer: <150 closed-form parameters, fold-fit, with explicit shrinkage
  fallbacks (cal_e→1, priors→global, K fixed); dual-history rare events are
  shrunk rather than estimated freely.
- Guards prevent degenerate rare-event heads (≥20 valid, ≥5 positives,
  a-priori fixed, L8).
- Thin components are explicitly *not* learned: per-player dispersions,
  potm-from-6-rows, any neural parameter (F10).
- Slices used for decisions carry an n≥30 informativeness floor (L3).

## Q5. Promotion and stopping rules precommitted? — PASS

Unchanged safe-overall gate (≤2% MAE, ≤2pp capture at every N, subgroup
≤0.5, positive pooled utility, prospective GW4–7). Kill switch K1, ladder
decision rules, partial-success rule, S1–S6, and shadow ownership (baseline
until retrospective passes) are all written into the plan text. No rule was
weakened anywhere in this cycle; two rules (ladder thresholds, comparison
counting) were strengthened.

## Q6. Does a failed result leave the repo and incumbents safe? — PASS

- v5 is additive: no edits to `contracts.py`, `scoring.py`, v1, v3, or any
  incumbent; production routing untouched.
- Both specialists remain deployment incumbents throughout; the v1 baseline
  keeps shadowing GW3–GW7 unless the full retrospective gate passes.
- v5 artifacts live under `data/unified/v5/` and are quarantined on failure;
  S4 permits at most one corrective cycle under the existing narrow-miss
  rule, then the cycle closes with specialists retained.

## Residual risks accepted by the gate

- R1 (above): masking decision is informed by the retrospective layer;
  disclosed, mitigated structurally, and ultimately adjudicated by GW4–7.
- R2: K/c constants are incumbent-inherited and GW1–2-contaminated; bounded
  by the sensitivity rung and kill switch K1.
- R3: ewm halflife 4 matches is unsearched v1 inheritance; flagged taste;
  deferred to a future cycle with its own multiplicity budget.
- R4: prospective GW4–7 power detects non-inferiority regressions of the
  observed size, not small superiority; the plan asks for exactly the former.

## Verdict

**GO.** The plan is temporally leak-free subject to disclosed condition R1;
its first experiments can falsify the central hypothesis; comparisons are
fair; every learned component has an explicit sample-size argument; promotion
and stopping rules are precommitted and unweakened; failure leaves the
repository and both incumbents untouched. Proceed to implementation of
`model/unified/v5/` and `tests/test_unified_v5.py` exactly as specified in
§4 and §11, with E0/E1/E2/E3 results to be appended to this ledger by the
operator before E4 is run.
