# Phase 1 / Workstream 3 — Evaluation adversary (v5 cycle)

Brief: attempt to falsify the entire unified-model goal; audit whether the
current comparison can distinguish model quality from sparse NCR noise;
challenge cutoffs, cohorts, comparators, capture variance, weighting,
observable-points construction, contamination, multiple testing, and the
feasibility of promotion before more NCR labels exist. Propose stopping rules.

## 1. Can the current comparison distinguish quality from noise?

**NCR layer: partially.** Two gameweeks × ~265 rows. Top-10 capture is 10
players; the GW2 actual top-10 includes a 65-point bench explosion (Pollock)
that alone is ~14% of the capture denominator (~457 pts). The paired bootstrap
MAE CI for candidate-minus-incumbent is **[-0.17, +0.56] — includes zero**
(`decision.json`), yet the deterministic 2% gate failed on the +0.20 point
estimate. Verdict: the NCR **MAE** gate at two rounds cannot separate the
models; the NCR **capture** gaps (-6.3/-10.2/-9.3/-4.2pp) share sign across
both GWs and all four Ns, which is unlikely under pure noise — treat ranking
gap as real, MAE gap as unresolved.

**6N layer: yes, adequately.** 5 rounds × ~134 rows in both 2025 and 2026;
baseline-vs-incumbent MAE differences (-0.14, -0.11) are small but the
unified-is-competitive-on-6N conclusion is consistent across 10 folds and two
engines (`fold_metrics.csv`).

**Reconstruction floor contaminates everything NCR.** The scorer contract
itself has NCR reconstruction MAE 6.34 (GW1) / 4.38 (GW2) with bias -5.1/-3.5
(`audit/scorer_reconstruction.csv`). A 2% MAE gate at 9.0 pts is 0.18 pts —
**~30× smaller than the floor**. Part of what the NCR MAE gate measures is
whose *attribution assumptions* better match the platform's, not who knows
more rugby. This must be reported alongside gates (it does not weaken them).

## 2. Contamination audit

- **NCR GW1–2 are doubly contaminated**: they influenced unified development
  (they are declared retrospective references) *and* incumbent development
  (bench-slot priors came from a "GW1 audit" — `model/ncr_project.py`
  comments). The frozen projection CSVs used by the benchmark are the honest
  pre-lock artifacts, so the *comparison* is fair, but any v5 tuning against
  GW1–2 NCR metrics would destroy the layer's remaining value. **Rule: v5
  model decisions may use rolling folds + 6N 2025 only.**
- **6N 2025 is single-use already**: 20+ configs were searched against folds
  that include 6N 2025-era matches, and the selection policy then read 6N
  2025. It remains the selection layer; do not also use it for ablation
  tuning (folds exist for that).
- **Multiple testing in the v3 cycle**: 8 baseline + 12 GBDT + 20 neural +
  blend + 4 feature-admission tests + engine selection, all on three folds
  whose year-to-year deviance spread is ~13%. Best-vs-runner-up objective gaps
  were 0.1–0.6% (`baseline_summary.csv`, `gbdt_summary.csv`) — config "wins"
  at that scale are noise. v5 must precommit a *small* ablation ladder with
  paired fold tests and treat the final config as frozen before the selection
  benchmark is touched.
- **Cutoff asymmetry**: 6N folds lock at the round's first match date
  (conservative); NCR folds use exact fixture locks (`harness.py`). Fine for
  leakage, but 6N training thus excludes same-round earlier matches while NCR
  includes same-GW earlier matches — a mild, declared asymmetry; keep, since
  it is conservative in both directions of the two competitions' schedules.

## 3. Cohort challenges

- **Label cohort = incumbent projection cohort** (`labels.py::_ncr_rows`):
  confirmed matchday 23s only; the official pool is 445–466
  (`data/ncr/ncr_rank_evaluation.md`). Top-100 official scorers absent from
  *both* pools (Piccardo 57, Ntamack 56 in GW2) are invisible to the
  benchmark. Capture numbers are therefore flattered relative to full-pool
  reality, for every model equally. Acceptable for model comparison;
  unacceptable to quote externally as absolute capture.
- **Median fill couples scores to the crosswalk**: unmatched NCR rows
  (~15–20/GW, `predictions.csv` `matched=False`) receive the candidate's own
  position median, so a model's score partially reflects its own smoothing on
  rows where it has *no information*. Both models suffer equally; still,
  report matched-only metrics as a diagnostic.
- **Subgroup gates are the sharpest instrument**: low_history (+0.59) fired
  with a clear mechanism (F12). Keep bench/low-history subgroup gates; add
  **position-bias** (mean predicted-minus-official by position, NCR) as a
  precommitted *diagnostic* (not a gate), because F7 says the next failure
  will be position-shaped.

## 4. Metric-weighting challenges

- `scoring_importance()` weights raw deviance by |fantasy weight| (tries 15,
  potm 15, metres 0.1). potm is **unobservable in NCR** yet carries max
  weight; metres is unscored in NCR. Deviance remains a development metric
  only; promotion is official-points-only — already true; keep it that way.
- Observable-points construction (`metrics.py`) is sound but capped at 0.859
  Spearman in NCR GW1 — using it for tuning is legitimate per the audit gate
  (≥0.85), marginal in NCR.

## 5. Feasibility of promotion before more NCR labels exist

- The decisive layer is **prospective GW4–7** (~4 rounds × ~270 rows, frozen
  pre-lock shadows, `shadow.py::evaluate_prospective_shadows`). One GW3
  baseline shadow already exists (`data/unified/v3/shadow/`).
- Power: ~1,080 rows cannot prove a <2% MAE difference, but the gates are
  non-inferiority gates — they only need to *detect regressions* of the size
  seen retrospectively (≥2pp capture). That is achievable; superiority is not
  required anywhere.
- **Promotion before GW4–7 labels would be unjustified under any v5 result**:
  the retrospective NCR layer is two noisy, contaminated rounds. The plan
  precommits: no GW4–7, no promotion — unified stays shadow-only.

## 6. Is the unified goal falsified?

No. The goal survives scrutiny because: (i) 6N parity/superiority is
demonstrated across 10 folds; (ii) the NCR MAE gap is inside bootstrap noise;
(iii) the NCR ranking gap has two concrete, fixable mechanisms
(scorer-attribution bias F7, low-history priors F12) with cheap falsification
experiments; (iv) the specialist incumbents remain deployed, so the downside
of a failed cycle is zero production risk. What *is* close to falsified:
(a) exposure/rate as the target structure (F2/F3; one last cheap test E2);
(b) neural multi-task at this data scale (F10); (c) any unified model that
ignores attribution quality and shrinkage (F7/F12).

## 7. Proposed stopping rules (precommitted)

- **S1**: E1 shows scorer-attribution explains <2pp of the NCR capture gap →
  the raw-event contract loses its main excuse; stop new model work, fall back
  to v1+masking, hand the residual gap to the user as a structural finding.
- **S2**: E3 shows shrunk-EB features give no low-history slice gain on dev
  folds → abandon the EB line; candidate is v1+guards only.
- **S3**: E4 (6N 2025 selection replay) regresses vs v1 by >1% MAE or >2pp any
  capture → revert to v1 baseline; do not run the full benchmark.
- **S4**: Full retrospective safe-overall gate fails *after* E0–E4 fixes → at
  most one corrective cycle under the existing narrow-miss rule; otherwise
  retain specialists, unified remains shadow-only, cycle closes.
- **S5**: GW4–7 labels never materialise → retain specialists; no promotion is
  possible by construction.
- **S6**: Any point-in-time violation discovered at any stage → stop, audit,
  and quarantine every artifact derived from the contaminated fold.

## 8. What the adversary could not kill

- The rolling-origin, exclusive-lock fold harness with write-once manifests
  (`harness.py`) — keep verbatim for v5.
- Cohort parity with the incumbent projection cohort — keep.
- The safe-overall promotion rule — keep *unchanged*; v5's job is to pass it
  honestly, not to move it.
