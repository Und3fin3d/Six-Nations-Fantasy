# v5 corrective cycle — diagnosis, decision, falsifiers

_Date: post-benchmark corrective cycle (the plan's single allowed cycle, stop
rule S4 / narrow-miss rule). Inputs: `benchmark/REPORT.md`,
`benchmark/predictions_v5.csv` vs `data/unified/v3/benchmark/predictions.csv`
(columns compatible; `engine` distinguishes models), frozen per-fold metrics
in `data/unified/v3/benchmark/fold_metrics.csv`, plan §4/§8/§10._

## 1. What the benchmark said

- v5 ≡ v1 on MAE/Spearman/Pearson everywhere (all bootstrap CIs span 0).
- **K1 fired**: low-history slice 11.67 vs baseline 11.66 — the EB layer's
  raison d'être produced 0.0% gain.
- **NCR top-10 capture gate failed hard**: v5 44.6% vs baseline 56.4%,
  incumbent 62.7% (gate: ≥ incumbent − 2pp).
- 6N side fine (7.265 selection MAE, best-in-table; 6N-2026 top-10 74.2%).

## 2. Diagnosis — why capture collapsed while MAE stayed level

`top_n_capture` is **points-weighted**: the actual points scored by the
model's *predicted* top-N, divided by the actual top-N's total. It is decided
at the extreme top of a ~270-row cohort; MAE is bulk-dominated and Spearman is
insensitive to ~10 rows. The two can decouple exactly as observed.

I recomputed per-fold capture from the prediction CSVs; my method reproduces
the frozen numbers exactly (baseline GW1 52.6% / GW2 60.2% → 56.4%; incumbent
62.6/62.9 → 62.7%), then applied it to v5.

### 2.1 The row-level mechanism (NCR GW1, baseline 52.6% → v5 ≈ 44.9%)

v5's predicted top-10 (actual points in parentheses):

> Kolbe (42), Ashman H (16), Marx H (40), Taylor H (25), Savea (43),
> Wainwright (26), Lake H (5), Hanekom (19), Wiese (34), Sheehan H (16)
> → 266/593 = **44.9%**

Baseline's top-10 on the same rows is also hooker-heavy (4 hookers) but keeps
**Jac Morgan** at #10 with 27.12 predicted (actual 62). v5 moves:

- Jac Morgan 27.12 → 25.81 (rank 10 → ~13) — *out*;
- Sheehan 26.58 → 27.18 — *in*.

One boundary swap (Sheehan in, Morgan out) costs 46 actual points = **−7.7pp
capture**, the entire GW1 gap. The same ±1–2 pt pattern repeats across the
boundary zone (25–31 predicted pts): matched-row deltas v5 − baseline are
**hookers +0.3…+1.8** (Ikanivere +1.8, George +1.3, di Bartolomeo +1.1,
Aumua +0.9, Sheehan +0.6, Nasser +0.6, Lamothe +0.3) and **try-scoring backs
−0.3…−1.3** (J.Morgan −1.3, Attissogbe −1.2, Wainiqolo −1.1, Ravutaumada −1.0,
Rowe −0.85, Jalibert −0.8, W.Jordan −0.2). GW1's actual top-10 is all
backs/loose forwards (80, 74, 62, 60, 55, 55, 54, 53, 50, 50).

### 2.2 Two stacked causes

**(a) The hooker-credited `lineouts_won` artifact (shared with v1/gbdt_v3).**
The NCR scorer pays 1 pt per `lineouts_won`; the feed credits hookers with
lineout wins (plan F7), so every unified model predicts ~6–12 phantom points
per starting hooker. Predicted hookers (~24–35 pts) crowd the top-10 boundary
while the actual top-10 contains ~0 hookers. The incumbent **documents and
drops the artifact** — its hooker projections are 14–23 — and leads capture
by ~6pp over every unified variant. This is the *structural* blocker the plan
flagged, now measured at row level: it fully accounts for baseline's
56.4% → incumbent 62.7% gap.

**(b) The EB + slot layers make the boundary strictly worse (v5-only).**
Shrunk rates deflate club-heavy recent form by `cal_e` (club→test) and pull
toward position priors that *themselves embed the lineout artifact* (hooker
priors inflated, back-three priors diluted); slot features add a second
position-prior minutes channel. Net: the model's top end compresses toward
position averages — forwards up, hot backs down — exactly the wrong direction
for a points-weighted top-10 instrument, and exactly invisible to MAE.

**Dose-response confirmation.** The buggy-harness run (dtype split ⇒ every
eval player predicted as a debutant ⇒ priors dominate) scored GW1 32.0% /
GW2 43.5%. Fixed-harness v5 (real history, shrunk layer active) ≈ 44.9% GW1.
Baseline (no shrunk layer) 52.6%. More prior weight ⇒ monotonically worse
capture. The layer is not merely inert (aggregate metrics); at the instrument
that gates promotion it is **harmful**.

### 2.3 Why MAE didn't move

Swapping predictions within the 24–32 pt band changes almost no absolute
errors (hookers overshoot by ~8–20 pts either way; backs undershoot by
~30–60 either way). Capture, being points-weighted at the top, is the only
mainstream metric that sees it.

### 2.4 Harness/provenance note for the operator

`fold_metrics_v5.csv` on disk contains only the 10 6N rows; the two NCR
per-fold metric rows are absent (the checkpoint was overwritten by a later
6N-only invocation). The REPORT's 44.6% NCR average is consistent with the
fixed-harness GW1 I recompute (44.9%) but the on-disk `predictions_v5.csv`
GW2 rows imply ≈ 61.5%, so the exact GW2 per-fold provenance is unrecoverable
from the artifacts alone. This does not change the diagnosis (every reading
puts v5 ≤ baseline on NCR capture and far below the incumbent), but the NCR
per-fold rows should be rebuilt from a single clean run on the rerun.

## 3. Decision (per the precommitted ladder)

Options considered:

- **(c) Targeted EB fix.** Rejected. No crash-level defect exists in
  `shrinkage.py` (PIT-safety, sorting, table fitting all verified against the
  artifacts). The layer's failure is conceptual, with *two* independent
  strikes: K1 (zero low-history gain) and the top-boundary compression above.
  Repairing it would mean re-tuning K/c/halflife — explicitly out of scope for
  this cycle, and the plan's K1 rule already prescribes the remedy.
- **(a) Masking only, keep EB.** Rejected: leaves the K1 violation standing
  and keeps the artifact-carrying priors in the model.
- **(b) Fire K1 → terminal rung T = v1 + guards + masking.** **Adopted.**

Implementation (all config-gated, reversible for ablation forensics):

- `V5Config.use_eb_features` (new) = **False**; `use_slot_minutes` = **False**
  → EB shrunk-rate and slot layers skipped (`model.py` also skips
  `fit_tables`); the model reduces to v1's feature set + the rare-event
  guards (A1).
- `V5Config.mask_attribution_corrupted` = **True** → `lineouts_won` is
  excluded from fitting and prediction; the deterministic scorers read it as
  zero (NCR-neutral-to-positive, 6N-inert since the 6N scorer never reads
  it). This is the E1 masking decision, now taken on direct row-level
  evidence: de-hookering the predicted top-10 is worth an estimated
  **+5…+10pp** capture (§4), far above S1's 2pp stop threshold.
- Scope note: `scrums_won` (front-row allocation, NCR +2) and `potm`
  (0% audit coverage) remain documented risks (F7) but expanding the mask
  beyond the precommitted event is a new experiment, not this cycle.
- `benchmark.py` untouched: default config now *is* terminal rung T.

## 4. Predicted effect on the failing gates

1. **NCR top-10 capture**: masking removes ~6–12 predicted points from every
   hooker. On GW1-2026 arithmetic that moves 3–4 hookers out of the predicted
   top-10 (boundary ~25–28) in favour of actual top-10 backs: worth ≈
   +7…+12pp vs v5's 44.9%, landing ≈ 55–62%. Killing the EB layer removes
   the additional anti-back nudges (the Morgan/Sheehan swap). Expected
   outcome: **≥ baseline's 56.4%, plausibly approaching the incumbent gate
   (60.7%) — not assured on two noisy folds.**
2. **NCR MAE**: hooker overshoot shrinks (predictions ~30 → ~22 vs actuals
   ~15–25); direction of the incumbent's 8.90 vs baseline 9.10. Gate (≤ +2%)
   holds comfortably either way.
3. **6N**: `lineouts_won` is unscored there; guards touch only degenerate
   heads. Expect v1 parity within MC noise (the v1-through-harness control
   reproduces 7.24/7.27).
4. **low_history slice**: no shrinkage layer → no mechanism for the v3-style
   +0.59 regression; expect baseline parity (11.66), gate ≤ +0.5 holds.
5. **Pooled utility**: hinges on the capture recovery above.

## 5. Falsifiers (what result would kill this decision)

- **F1**: T's NCR top-10 capture < baseline's 56.4% + 2pp on the rerun → the
  attribution thesis (or its implementation) is wrong → stop, retain v1 +
  specialists (S1/S4).
- **F2**: T's NCR MAE > baseline + 2% → masking is destroying real signal
  (the platform really credits feed attribution *and* the model's hooker
  means were calibrated to it) → stop, retain specialists.
- **F3**: 6N metrics for T deviate from v1 beyond MC noise → implementation
  defect; quarantine the change.
- **F4**: capture recovers but low_history regresses > +0.5 vs baseline →
  gate still fails; no further cycle exists — retain specialists.

## 6. Honest bottom line

Even if T performs exactly as predicted, the remaining gap to the NCR
incumbent (62.7%) may not fully close: the incumbent also carries calibrated
club/test priors and competition-specific knowledge that v1+guards+masking
deliberately lacks. The plan's promotion bar (capture ≥ incumbent − 2pp at
*every* N, plus positive pooled utility, plus the GW4–7 prospective shadow)
remains untouched. If the T rerun misses it, the honest outcome stands:
**retain v1 as universal baseline and both specialists; v5's contribution is
the measured obituary of the EB feature layer (K1) and the first direct,
row-level quantification of the scorer-attribution blocker (worth ~6pp of NCR
top-10 capture to whoever owns the incumbent feed contract).**
