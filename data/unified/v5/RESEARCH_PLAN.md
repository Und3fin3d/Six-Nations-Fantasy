# Unified rugby model v5 — research and execution plan

_Status: FINAL after Phase 3 cross-examination and Phase 4 red-team. The
adversarial-review ledger (§13) records every criticism and its resolution;
plan sections amended by review are marked `[R]`._

## 1. Executive recommendation

**Build Family A: a shrunk-form direct-totals GBDT ("v1 + empirical Bayes")
with rare-event guards and an attribution-quality data fix — but through a
precommitted kill-switch ladder whose terminal, lowest-complexity rung
(masked v1 + guards) is the candidate if the EB layer cannot prove itself on
dev folds. Promotion only through the unchanged safe-overall gates including
the prospective NCR GW4–GW7 shadow.** Confidence: **moderate** that v5 reaches
honest NCR non-inferiority; **low-moderate** that it beats the NCR incumbent
on capture. v3's failure decomposes into (a) rare-event engineering in the
rate layer, (b) position-corrupted scorer attribution in NCR, and (c) missing
shrinkage priors for low-history players. v5 deletes (a), measures (b) before
touching it, and makes (c) earn its place or die.

## 2. Causal diagnosis of v3 (Phase 1, workstream 1)

1. **Direct totals is the right target parameterisation for this store.** v1
   fits one Poisson/tweedie objective per event on all rows (zero-minute
   included); v3 multiplies three separately fit quantities
   (`p_play × E[min|play] × rate/80`) and the product amplifies error exactly
   where Poisson deviance is most sensitive (F2, F4).
2. **v3's +6.7% deviance regression is a 2022-fold, rare-event artifact**:
   tackle_turnover 1.978 vs 0.925, tackle_try_saver 0.605 vs 0.271, drop-goal
   events 2–4× worse; 2023/2024 are parity (±1%); metres *improved* (F2, F3).
   Mechanism: played-only rate fitting, near-zero-minute rate denominators,
   and 99.5% caps on mostly-zero rate distributions (F4). E2 (one day) is the
   final falsifier before the rate layer's obituary is sealed [R-B4].
3. **v3's small official-MAE edge is hurdle minutes/appearance** — modular,
   and not requiring the rate layer (F11).
4. **The actual promotion blocker is NCR ranking**, driven by (i)
   position-correlated scorer-attribution bias — hooker-credited
   `lineouts_won +1` (the incumbent documents and drops this artifact),
   model-predicted-but-unobservable `scrums_won`/`potm` (F7; audit coverage
   0%) — and (ii) **low-history players** (career<5: +0.59 MAE) where the
   incumbent shrinks toward calibrated club/test priors and the unified models
   have level-blind ewm form plus a bare player_id categorical (F12, F6).
5. **NCR MAE gates sit on a 4.4–6.3 reconstruction floor** (bias -5.1 GW1;
   observable Spearman 0.859): a 2% MAE gate (0.18 pts) is ~30× below that
   floor; capture gates are the informative NCR instrument (F8). The NCR MAE
   gap itself is inside bootstrap noise (CI [-0.17, +0.56]).

## 3. Hypothesis tree

```
Unified NCR gap (capture -4..-10pp; MAE +2.2% [CI spans 0]; low_history +0.59)
├── H1 scorer-attribution bias (F7)           → E0a, E1 → global attribution masking (if E1 ≥ 2pp)
├── H2 missing shrinkage priors, low history  → E3      → Family A shrunk-EB features (kill-switch K1)
│   └── H2a level-blind club/test form (F12)  → E3      → level-split calibrated form (falsifier: cal_e CI)
├── H3 rare-event parameterisation fragility  → E2      → delete rate layer; rare-event guards
├── H4 residual model-quality gap             → only if E1+E3 insufficient → S1: stop, retain specialists
└── H5 evaluation noise (bootstrap CI)        → prospective GW4–7 decides; never tune on NCR GW1–2
```

## 4. Model family and fallback `[R-A1, A6, A8, B1, B4]`

**Candidate ladder (Family A), each rung gated by §8's decision rules:**

- **A0 = v1 exact re-run** (control).
- **A1 = v1 + rare-event guards**: count head requires ≥20 valid rows and ≥5
  positives, else a stored level-stratified global mean; thresholds fixed a
  priori here and never iterated against folds (ledger L8).
- **A2 = A1 + shrunk EB rate features**: per player×event, level-split
  `(wm_i·rate_i + 0.55·wm_c·cal_e·rate_c + 220·prior_{pos,e}) / (wm_i + 0.55·wm_c + 220)`,
  ewm halflife 4 matches (provenance: v1 default, flagged taste, not searched
  this cycle), `cal_e` global club→test factor per event from dual-history
  players, priors = position×event per-80 means shrunk to the global mean.
  **All global tables (cal_e, priors, slot minutes, global means) are fit per
  fold on that fold's training window only and recorded in the fold manifest
  payload** [R-A1]. K=220, c=0.55 are incumbent-inherited and contaminated by
  NCR GW1–2 (ledger L9); the sensitivity rung K×{0.5, 2}, c×{0, 0.55, 1} is
  the binding check.
- **A3 = A2 + slot minutes features** (p(play|squad), starter-rate, minutes
  when starting/benched + position-slot priors).
- **A4 = A3 + attribution masking** (`lineouts_won` globally, only if E0a/E1
  support it). POTM attach (`ncr_potm.csv`) is **diagnostic-only** — it
  improves store reconstruction fidelity, not the model (6 GW2 positives)
  [R-A6].
- **Terminal rung T = A1 + masking** (the "subtraction" candidate,
  Reviewer B's AP-B): **K1 kill-switch — if A2's low-history-slice gain over
  A1 is <1% observable-points MAE in ≥2/3 informative folds (slice n≥30), the
  EB layer is dropped and T is the candidate.**

**Fallbacks:** (i) T above. (ii) Guarded exposure/rate (Family B) exists only
as the E2 falsifier; it can be resurrected as a candidate **only if** E2 shows
>5% 2022 deviance recovery with no 2023/24 regression **and** it beats v1 on
the A3 low-history slices — otherwise the branch is dead [R-B4]. (iii) Family
C hierarchical-Bayes oracle is **report-only**, not a validation gate [R-A4].

## 5. Raw-to-fantasy conversion boundary (exact; unchanged)

`RawPrediction.events[event].mean` + `minutes` →
`CompetitionScorer.score_prediction` (seeded Monte-Carlo) → points. Left of
`RawPrediction`: competition-independent. Right: deterministic,
competition-specific. **Boundary changes: none.** Attribution-masked events
are absent from `events` and score as zero via the existing `_event` default —
a global data/model-exclusion decision recorded in each artifact manifest.

## 6. Features `[R: provenance table added]`

Admitted (PIT-safe; shifted or fold-training-slice-only): v1 base set
(career_matches, days_since_last, recent_minutes, recent_start_rate,
team_recent_margin, jersey, started, is_forward; categorical player_id,
position, team, opponent, competition_level); shrunk level-split rates +
weights; slot minutes priors; `wr_` and `rolecert_` blocks (v3-admitted).
Prohibited: weather (archived observed, not pre-lock forecast); style (failed
v3 admission); market/odds (prior PIT violation); competition identity,
competition-specific points/ranks/calibrators/blend weights; official points
as training targets; RugbyPass-derived priors (not PIT-available for 6N rows).

Provenance flags (Review B3): direct totals = evidence (F2–F4); K/c =
incumbent inheritance, contaminated, sensitivity-checked; halflife 4 = v1
default, taste, not searched; 31-event contract = taste (non-scoring events
are diagnostics-only; future simplification candidate); bernoulli heads
excluded from rate features = taste, immaterial.

## 7. Data sufficiency and parameter counting `[R-A3, A4]`

- Store ~200k rows; internationals from 2019; club from ~2023 (F9). Labels:
  1,874 (6N 2025–26: 1,342; NCR GW1–2: 505; match rate 98.6%).
- Tree budget ≈ 31 heads × 180 × 23 leaves on ~200k rows (search-validated).
  EB layer: <150 closed-form global parameters, fold-fit, none gradient-fit.
- Low-history target definition [R-A3]: slice (i) zero prior international
  rows with ≥3 prior club rows, evaluated on first international appearances;
  slice (ii) <5 prior international rows. Slice n reported per fold; folds
  with n<30 are uninformative for the slice and excluded from the kill-switch
  denominator.
- Family C oracle: 2023–24 internationals, tries/tackles only, report-only —
  its posterior is expected to be wide and must not gate anything.

## 8. Evaluation protocol (frozen) `[R-A1, A2, A3, A5, B2]`

Layers (unchanged): **dev** = rolling folds 2022/2023/2024, international
validation, strict exclusive locks; **selection** = 6N 2025 official points;
**retrospective reference** = 6N 2026 + NCR GW1–2 (never tuned against);
**prospective** = NCR GW4–7 frozen pre-lock shadows. **The v1 baseline owns
all GW3–GW7 shadow slots until a v5 candidate has passed the full
retrospective gate** [R-B2]. Artifacts: write-once manifests, sha256 chains,
fold manifests now include the EB table payloads [R-A1]. v5 outputs under
`data/unified/v5/`; v3 artifacts never regenerated.

Comparisons (cohort parity throughout): v1 baseline (universal control), both
specialist incumbents, and the ladder A0–A4/T. **Ladder decision rules**
[R-A2]: a rung beats its predecessor only if (a) pooled paired-bootstrap p05
of the fold objective delta > 0, (b) same-direction in ≥2/3 folds, and (c)
the low-history slice does not regress; rungs failing are dropped, and every
fold comparison is counted in the ledger (visible multiplicity). The kill
switch K1 (§4) is evaluated once, after A2's fold run. The ladder survivor is
the *single* candidate run on 6N 2025 (E4), then the full benchmark, then
shadow.

**Partial-success rule** [R-A5]: after E1 and E3, the candidate proceeds to E4
only if ledger arithmetic (E1 capture recovery + E3 slice gains mapped through
the v3 failure list) can clear every gate that failed in v3 — NCR capture ≥
incumbent − 2pp at all N, low_history ≤ +0.5, MAE ≤ +2%, pooled utility > 0 —
with the arithmetic written into the benchmark report. Otherwise the cycle
stops at S1.

**Falsifiers added** [R-A5]: E3 reports `cal_e` with bootstrap CIs per event —
if the high-coverage events are all consistent with 1, the level split is
dropped; masking decisions report raw-deviance and calibration movement, not
just capture/MAE; E0c audits store-`started` vs status-P/B agreement on NCR
GW1–2 (retrospective rows use actual lineup flags; shadows derive them from
fantasy status) and the agreement rate is reported with the shadow template
[R-A7].

Primary metrics: official-points MAE, tie-aware top-N capture (10/25/50/100),
Spearman; raw-event deviance and calibration are development diagnostics.
Promotion margins: **unchanged** safe-overall (≤2% MAE, ≤2pp capture at every
N, both competitions, subgroup ≤0.5, positive pooled utility, prospective
GW4–7 gate). Uncertainty: paired bootstrap CIs for MAE **and** capture, both
competitions; matched-only metrics alongside median-filled; NCR position-bias
table as a precommitted diagnostic (not a gate).

## 9. Compute budget and elimination

E0a/E0b/E0c/E1: hours, pandas-only. E2: one fold × 12 configs (~1 day, hard
cap). E3: ≤5 rungs × 3 folds (~2 days). Family C oracle: ≤1 day. E4: ~half
day. Full benchmark: ~1 day. Total ≈ 5 CPU-days. Any stage exceeding 2× its
budget is paused and reported, not extended.

## 10. Stop conditions (precommitted)

S1 attribution explains <2pp of NCR capture gap (E1) → stop new model work,
report the structural finding; S2/K1 EB layer below threshold → candidate =
terminal rung T; S3 E4 regresses vs v1 (>1% MAE or >2pp any capture) → revert
to v1, no full benchmark; S4 retrospective gate fails after fixes → one
corrective cycle only under the existing narrow-miss rule, else retain
specialists; S5 GW4–7 labels missing → no promotion possible; S6 any PIT
violation → stop and quarantine derived artifacts.

## 11. Implementation, tests, migration, rollback `[R-A1]`

- `model/unified/v5/`: `config.py` (V5Config, frozen a-priori constants);
  `shrinkage.py` (`ShrunkTables` fit per training window + row-wise PIT-safe
  shrunk features); `model.py` (`ShrunkFormGBDT`: v1-shaped
  fit/predict_frame/save/load, guards, event exclusion); `data_quality.py`
  (attribution masking helper + POTM attach, config-gated); `metrics.py`
  (position-bias, low-history slice helpers). No edits to `contracts.py`,
  `scoring.py`, v1, or v3 artifacts.
- `tests/test_unified_v5.py`: **composition-level append-invariance** (tables
  fit on a prefix + features on extended frame ≡ features computed from the
  prefix alone) [R-A1]; shrinkage direction (low history → prior, high
  history → observed); club calibration bounds/direction; rare-event guard
  fallback; masking ⇒ event absent from prediction and scorer yields zero;
  POTM attach semantics; determinism + pickle round-trip; RawPrediction
  validity under both scorers; unseen-player fallback; a store smoke test
  skipped unless the canonical store exists.
- Migration: none — v5 is additive; v1 remains universal baseline; both
  specialists remain deployment incumbents; production routing untouched.
  Rollback: retain specialists (status quo); v5 artifacts quarantined under
  `data/unified/v5/`.

## 12. Risks

- K/c inherited from incumbent-tuned-on-GW1 (contaminated) → sensitivity rung;
  kill switch K1 bounds the damage.
- Attribution masking could hurt if the platform really credits the feed
  attribution → E0a/E1 decide before masking is admitted; masking is
  config-gated and manifest-recorded.
- Fold multiplicity across the ladder → capped rungs, effect-size thresholds,
  comparison counting (ledger L2).
- 6N 2025 reuse → exactly one candidate reaches it.
- GW4–7 power suffices for non-inferiority detection, not superiority — the
  plan asks exactly that.

## 13. Adversarial-review ledger

**Reviewer A** (`phase3_review_a.md`):
- L1 (A1, fold leakage via EB tables): **accepted** — tables fit per fold,
  recorded in manifests; composition-level append-invariance test added (§8,
  §11).
- L2 (A2, circular selection on 3 folds): **accepted** — precommitted
  effect-size + ≥2/3-fold + bootstrap-p05 rules; comparison counting (§8).
- L3 (A3, low-history slice mis-specified): **accepted** — slice redefined;
  n≥30 informativeness threshold (§7).
- L4 (A4, underpowered Family C oracle): **accepted** — oracle demoted to
  report-only (§4, §7).
- L5 (A5, missing falsifiers): **accepted** — cal_e CI falsifier; masking
  deviance/calibration reporting; partial-success rule (§8, §10).
- L6 (A6, potm oversold): **accepted** — recharacterised as diagnostic-only
  (§4).
- L7 (A7, lineup-flag realism): **accepted** — E0c agreement audit added;
  shadow-template disclosure (§8).
- L8 (A8, guard-threshold provenance): **accepted** — ≥20 valid/≥5 positives
  fixed a priori here; not iterated (§4).
- A9 (unbroken items): noted — harness, cohort parity, median-fill
  disclosure, gate integrity, E1 validity, prohibited list all stand.

**Reviewer B** (`phase3_review_b.md`):
- L9 (B1, "subtraction not addition"): **partially accepted** — AP-B becomes
  the explicit terminal rung T with kill-switch threshold; rejected as the
  *sole* plan because the low_history +0.59 gate failure is demonstrated and
  masking cannot repair it (`decision.json`, F12).
- L10 (B2, "wait for GW4–7"): **rejected** — v5 consumes no prospective
  rounds; baseline keeps shadowing; the accepted part (v5 touches no shadow
  slot before passing retrospective) is adopted (§8).
- L11 (B3, taste audit): **accepted** — provenance flags added (§6);
  non-scoring-event simplification logged as future work.
- L12 (B4, E2 branch discipline): **accepted** — resurrection requires
  low-history superiority, not just deviance recovery (§4).

## 14. Unresolved questions for the user

1. If E1 shows the NCR capture gap is mostly scorer-attribution, is global
   `lineouts_won` masking acceptable, or should the scorer contract be
   versioned instead (bigger change, same evidence)?
2. Is the Family C report-only oracle worth its ~1 day, or is the K/c
   sensitivity rung sufficient?
3. Should matched-only NCR metrics carry any decision weight, or remain
   diagnostics only?
