# Unified Rugby Supermodel v4 — Research Plan (adversarial protocol)

Status: FINAL — passed Phase 4 red-team gate as DELIVER-WITH-CORRECTIONS; all four
corrections applied below (ledger #19–22).

## Protocol ledger
- [x] Phase 1: failure diagnostician / model researcher / evaluation adversary (independent)
- [x] Phase 2: primary draft plan
- [x] Phase 3: Reviewer A (plan only) + Reviewer B (plan + phase-1 reports) — plan revised
- [x] Phase 4: red-team gate (fresh context) — DELIVER-WITH-CORRECTIONS, applied
- [ ] Unresolved user decisions surfaced
- [ ] ExitPlanMode

---

## Context

Three unified-model attempts (v1 raw-event GBDT, v2 learned rank-stack, v3 exposure/rate GBDT)
all beat the 6N incumbent but failed the NCR promotion gate. User directive: move on from v3 and
"finally make the unified model that works". This plan is the product of the adversarial handoff
protocol (`/tmp/unified-rugby-v4-research-handoff.md` — to be copied into the repo): three
independent Phase-1 researchers, two independent cross-reviewers, one red-team gate.

## 1. Executive recommendation (confidence: medium on mechanism; low on 2026 promotability)

Build **v4 = the v1 direct-totals `UniversalGBDT` (architecture unchanged) + two structural
fixes aimed at the verified failure mode** (thin/mixed-history players' sparse high-value events
regress to positional means, scrambling NCR top-N ranking):

- **P1 — hierarchical empirical-Bayes shrinkage** of per-player per-80 rates: player-specific
  expanding in-store rate (club+intl, minutes-weighted) shrunk toward position×level expanding
  prior, **K fitted per event by variance moment-matching, not pinned**. Tested in two variants,
  picked on 6N-2025: (a) as input features; (b) as post-hoc out-of-fold multiplicative player
  effects (no `player_id` categorical). Level-separated form features (club vs intl EWM +
  counts) ride along in the same refit.
- **P2 — hurdle heads** for the sparse events that make the top-N (tries 86.7% zero, assists
  91.3%, breaks, turnovers): P(≥1) classifier × positive-count regressor, **moment-matched back
  into the existing `EventDistribution`** so `contracts.py`/`scoring.py` are untouched.

Before any modelling, **P0 repairs the ruler**: error decomposition (hard gate), POTM label
backfill + POTM-aware development proxy for NCR, incumbent-POTM-ablation check, and a
decision-relevant non-inferiority margin derived from picker (`value_xv`) sensitivity.

Promotion evidence is **prospective-only** (NCR GW4–7 pooled, burn-once). The full gate (MAE
AND capture) is retained — NOT weakened — with an honest two-tier outcome: 2026 promotion only
via decisive superiority; non-inferiority promotion deferred until the pooled clean sample is
adequate (~20+ GWs, realistically 2027). Neural is excluded (20/20 configs regressed, 3 failures).

## 2. Evidence-backed diagnosis (Phase 1, verified independently by both reviewers)

- **Ranking gap, not level gap**: NCR Pearson incumbent 0.561 vs baseline 0.461; identical
  dispersion (pred-std/actual-std ≈ 0.48 both); MAE near-tied (8.90 vs 9.10, bootstrap CI spans 0).
- **Concentrated in GW1 cold-start**: top-10 capture 52.6% vs 62.6% (GW1) → 60.2% vs 62.9% (GW2).
- **Mechanism**: regression toward positional means — over-predicts hookers (+3.3), under-predicts
  back-three (−5.1)/centres (−4.2). Roigard (74 pts) ranked 108th by baseline, 1st by incumbent.
  NCR player pool: median 7 intl fixtures, 35.6% ≤3 — thin-history is the norm; flat `player_id`
  categorical + equal-weight club/intl EWM borrows no strength (the incumbent's EB shrinkage,
  `ncr_project.py:315,336-338`, is exactly this fix).
- **Ruled out**: team-sheet asymmetry (`started`/`jersey` are prediction-time features), WR/matchup
  signal (residual↔margin ≈ 0; v3 admission delta −0.00035 = noise), bench mispricing, compression.
- **v3 lost on surrogate-objective mismatch** (p_play×minutes×capped-rate never optimizes count
  deviance; +6.7% deviance swamped −0.7% MAE gain). Exposure info was already in the features.
- **The development proxy is POTM-blind for NCR** (potm +15 = biggest event, coverage 0.0; NCR
  reconstruction bias −5.1/−3.5 masked by 6N-weighted headline) — tuning on it steers away from
  the exact top-10 stars the gate fails on.
- **The evaluation itself is broken**: current conjunctive gate fails a truly-equal model ~73% of
  the time on 2 GWs (capture-diff sampling SD 8.6pp vs 2pp margin). NCR GW1–2 + 6N 2026 are
  burned (steered three iterations). Powering a 2pp capture gate needs >100 GWs; 5pp ~50 GWs;
  2% MAE ~24 GWs. **No non-inferiority promotion is honestly decidable in 2026.**

## 3. Options considered and rejected

| Option | Verdict | Why |
|---|---|---|
| Neural (any form) | REJECT | 20/20 configs >3% worse at rung one; 3rd failure |
| Learned rank-stack / per-competition calibrators / router | REJECT (constraint) | v2 sweep drove learned weight to zero; violates one-model mandate |
| v3 exposure/rate targets redux | REJECT | Structural surrogate mismatch; exposure already an input |
| Rubric pseudo-labels on club rows | REJECT | Raw-event heads already train on those rows |
| International_x4 weighting; sample-weight time-decay | REJECT | Already in v3 search space; `natural`/`null` won (best_baseline.json, best_gbdt.json) |
| WR/opponent conditioning as new admission | REJECT | Contradicts diagnosis (residual↔margin ≈ 0); `team_recent_margin` already a base feature |
| External RugbyPass JSON snapshots as training features | REJECT | Current-season aggregates = hindsight for historical rows; the store's own club rows provide the same player-specific prior PIT-safely |
| POTM Bernoulli head | CONDITIONAL | Only if P0's incumbent-POTM-ablation shows material edge AND labels backfillable; incumbent's POTM is a ≤2pt deterministic bump, likely small |
| Pure wait-for-November (build nothing) | REJECT (user directive) | But P0 alone is the fallback if P1/P2 fail their kills |
| GW3 scramble (lock 2026-07-18 06:40 UTC) | REJECT | <24h vs 1–2 day build; rushing wastes the burn-once slot; +1 GW is marginal (both reviewers) |

## 4. Chosen hypothesis and fallback

**Central hypothesis (falsifiable):** the unified GBDT's NCR top-N deficit is caused by
unshrunk thin/mixed-history rate estimates and an under-sharpened sparse-event tail. Imposing
(P1) hierarchical shrinkage and (P2) hurdle tail structure — with no architecture, target, or
seam change — improves tail calibration and thin-history ranking on 6N-2025 + the repaired
proxy, and directionally closes the NCR gap on prospective GW4–7.

**"Thin history" is re-operationalized (red-team correction #2)**: thin := **<5 international
matches in-store at fixture date** (a new PIT feature), NOT total `career_matches` — by the old
measure 6N-2025 has ZERO thin rows (min 67 median career), so the draft's kill cohort was empty,
and the v3 gate's `low_history<5` NCR guard fired on just 7 rows (noise). The intl-exposure
cohort definition and its minimum n (≥30, else the thin-history read is declared untestable on
that fold) are precommitted here; the same definition replaces the promotion guard's cohort.

**Kill criteria are conjunctive and on the failing metric**: P1 must improve rank metrics
(Spearman + mean capture) on 6N-2025 LORO without degrading overall MAE, AND improve the
thin-intl cohort where n≥30 (if the cohort is too small on 6N-2025, that criterion is recorded
as untestable and P1 is admitted only on the overall metrics — stated plainly, since the
thin-history mechanism then rests entirely on the single prospective NCR look). P2 must improve
P(≥1) reliability AND p90-vs-realized AND not regress capture. Both dead → v4 = P0 only
(repaired evaluation), freeze until November; that is the honest floor.

**Fallback ordering** within P1 is empirical, not taste: variant (a) features vs (b) post-hoc
effects chosen on 6N-2025. (Reviewer B's evidence — trees under-exploited engineered features in
v3 — predicts (b); the data decides.)

## 5. Staged plan (P0 → P1 → P2 → assemble)

**P0 — repair the ruler (hard gate, ~1 day, burns nothing):**
1. Error decomposition on NCR baseline predictions (already-produced artifacts, read-only):
   minutes vs per-80-rate vs context error, split by history depth and forward/back.
   **Branch rule: minutes-dominated → stop the tail/rate program entirely; the fix is a
   different, smaller job (report to user).**
2. Full NCR rubric-coverage audit (red-team correction #1): POTM is NOT in the feeds — the only
   source is hand-scraped `data/ncr/ncr_potm.csv` (6 rows, GW2 only; GW1 missing; GW4–7 will
   need the same scraping). POTM omission is only ~7% of the −5.08/row NCR reconstruction bias;
   the audit must quantify ALL unobserved scored events (interceptions +5, lineout_steals +5,
   scrums_won +2, all coverage 0.0) and store-vs-`cur_gd_points` discrepancies, then repair the
   proxy for whatever is actually recoverable. Ablate POTM out of the *incumbent's* output to
   measure how much of its top-10 edge survives. A POTM head stays CONDITIONAL and could only
   train on 6N (NCR has no store POTM); the backfill must never touch frozen `official_pts` or
   the evaluation cohort — it may populate a shifted raw event only.
3. Derive the MAE non-inferiority margin from picker/`value_xv` sensitivity with a PINNED
   protocol (red-team correction #4): fixed 2025 squad, predeclared perturbation grid (uniform
   MAE-equivalent noise injected into projections at σ ∈ {0.1..1.0}), margin = smallest σ that
   changes ≥1 `value_xv` XV pick in ≥50% of 200 seeded reps. Computed once in P0, then frozen.
4. Copy the handoff into `data/unified/v4/HANDOFF.md`; write `data/unified/v4/RESEARCH_PLAN.md`
   (this plan, required output of the protocol).

**P1 — hierarchical EB shrinkage (one refit per variant + LORO):** as §1. All features
`.shift(1)`/expanding, PIT-safe; priors from strictly-prior rows; K fitted by within/between
player variance moment-matching per event. Unit tests: 0-history → prior; ∞-history → player
rate; no-competition-id guard; jersey train/serve-skew test (real jerseys in training vs
`START_JERSEY`/`BENCH_JERSEY` synthesis at inference, `v3/shadow.py:73`).

**P2 — hurdle heads (half day):** tries, try_assists, clean_breaks, tackle_turnover. Moment-match
mixture → NB mean/dispersion; regression guard: v4 flags off ⇒ bit-identical v3 baseline.

**Assemble:** admitted components → `gbdt_v4` engine through the existing v3 harness (lock folds,
hash-checked immutable artifacts, write-once shadows). Admission decisions: ≤3 (P1 variant, P1
admit, P2 admit), all on 6N-2025 LORO + repaired proxy. **NCR GW1–2 and 6N 2026 are fully
quarantined — no looks, not even directional** (both reviewers; supersedes the draft's
"direction-only budget").

## 6. Frozen evaluation protocol and promotion gates (pre-registered)

- **Selection layer**: 6N 2025 LORO (official points) + repaired POTM-aware proxy. Components
  that help 6N but are neutral-or-negative on the repaired proxy are flagged, not auto-admitted
  (addresses the 6N-selection/NCR-target mismatch: gbdt_v3 had *higher* 6N utility yet worse NCR).
- **Quarantined**: NCR GW1–2, 6N 2026 — diagnostics in reports only, never inputs to decisions.
- **Prospective read**: NCR GW4–7 (Nov 6/13/21/27), shadows frozen write-once pre-lock via the
  existing `gw_update.sh` + `v3 shadow` infra, **pooled, single look**, no per-GW peeking, no
  corrective cycle. GW3 is skipped for v4 (baseline's GW3 shadow, already frozen, still gets
  scored as v3 evidence).
- **Promotion gate — unchanged in content, honest about power** (user approval required, §10):
  - The safe-overall rule is retained: MAE within margin AND top-10/25/50/100 capture within
    margin on BOTH competitions, positive pooled utility.
  - **Tier 1 (2026, pooled GW4–7)**: promotion only via *decisive superiority* — v4 beats the
    NCR incumbent with the paired-bootstrap 90% CI excluding zero on MAE **and** mean capture,
    while holding the 6N side. **Honest reachability (red-team correction #3): at n≈1,050 the
    CI half-widths are ≈0.26 MAE and ≈6pp capture, so tier 1 requires beating the incumbent by
    ≳0.26 MAE and ~10pp capture — a v4 that merely CLOSES the 6.3pp gap lands at parity and is
    NOT promotable in 2026. Tier 1 is a safety valve for an unexpectedly dominant model, not the
    expected path; tier 2 (2027) is the realistic promotion route.**
  - **Tier 2 (non-inferiority)**: decided only when the pooled clean prospective sample reaches
    the pre-computed power threshold (~20+ GWs; realistically during NCR 2027). Until then a
    "PASS-pending-power" is reported, incumbents stay deployed.
  - Every table reports bootstrap CIs; MAE always reported (raw + calibrated) per house rule.
- **Failure**: any tier-1 miss in November → incumbents retained; no fourth iteration against
  the same label pool; v4 (if it passed its 6N kills) continues shadowing into 2027 for tier 2.

## 7. Compute / data budget

- Pinned env `/tmp/6n-model-pinned/bin/python` (rebuild from requirements-model.txt if wiped).
- P0 ≈ ½–1 day analysis; P1 ≈ 2–4 refits + LORO (~90s–5min each, deterministic multithread);
  P2 ≈ half day. Total ≤ ~3h compute, ~2–3 days effort. Hard cap: exceeding this without passing
  kills ⇒ freeze until November.
- No new/paid data. POTM backfill only from feeds already in `data/ncr/`. RugbyPass JSON
  snapshots explicitly excluded from training features (hindsight).

## 8. Risks, stopping rules, rollback

- **S1 (mis-aim)**: P0 error decomposition minutes-dominated → stop, report, rethink.
- **S2 (no traction)**: P1 AND P2 fail conjunctive kills on 6N-2025 → v4 = P0 only; freeze.
- **S3 (quarantine)**: any decision traceable to NCR GW1–2 / 6N 2026 voids the run.
- **S4 (November)**: tier-1 miss → incumbents stay; no corrective cycle; tier-2 clock continues.
- **Risk — proxy repair fails** (POTM not backfillable / repaired proxy still misranks stars):
  proxy keeps its current role but is never a sole admission criterion; noted in report.
- **Risk — incumbent hindsight inflation**: its GW1–2 numbers were produced in-window; treat its
  edge as an upper bound (favours v4 slightly in November — one more reason superiority-only).
- **Risk — shadow-freeze discipline** (red team Q1): `freeze_shadow` reads `player_match.csv`
  live; GW4–7 PIT safety requires the store to contain no post-lock rows at freeze time. Add an
  assertion (max store date < lock) to the freeze path rather than relying on operator timing.
- **Risk — fitted K leakage** (red team Q1): K must be re-fit inside each fold's strict
  training frame, never globally; enforced by a test.
- **Rollback**: v4 additive + flag-gated; flags-off = bit-identical baseline (regression-tested);
  production routing (`model/run.py`, `model/ncr_project.py`, `gw_update.sh`) untouched;
  incumbents deployed throughout; artifacts write-once under `data/unified/v4/`.

## 9. Adversarial-review ledger

| # | Source | Criticism | Resolution |
|---|---|---|---|
| 1 | Diagnostician | Team-sheet asymmetry (prior sessions' assumption) | ACCEPT: ruled out; removed from hypothesis set |
| 2 | Adversary | Gate fails truly-equal model ~73% | ACCEPT: two-tier gate (§6); surfaced to user |
| 3 | Adversary | GW1–2 + 6N-2026 burned | ACCEPT (strengthened by A4/B4): full quarantine, S3 |
| 4 | Adversary | Proxy POTM-blind, steers away from failure mode | ACCEPT: P0.2 repair + incumbent ablation |
| 5 | Researcher | Pseudo-labels redundant | ACCEPT: dropped |
| 6 | Rev A (FATAL 1) | MAE-primary gate = metric-swap; hypothesis unfalsifiable; motivated reasoning | ACCEPT: capture restored to the gate; 2026 path = superiority only; non-inferiority deferred to powered sample |
| 7 | Rev A (FATAL 2) | n≈1,300 MAE non-inferiority is a coin flip; plan overstated power | ACCEPT: power stated; tier-2 waits for adequacy |
| 8 | Rev A (S3) / Rev B10 | GW3 scramble unrealistic (<24h) and hazardous | ACCEPT: GW3 dropped for v4 |
| 9 | Rev A (S4) / Rev B4,B12 | 6N-2025 utility poor NCR proxy; ≤6 budget a fig leaf; disjunctive kills cosmetic | ACCEPT: quarantine; ≤3 conjunctive admissions; repaired proxy added to selection; mismatch flagged in §6 |
| 10 | Rev A (S5) / Rev B6,B7 | E1 weighting + E3 decay + E4 WR already rejected by v3 search; grab-bag | ACCEPT: dropped (kept only level-split form inside P1) |
| 11 | Rev A (S6) | EB prior mis-specified: incumbent anchors on player-specific rp rates, not position means | ACCEPT: prior = player-specific in-store expanding rate shrunk to position×level; rp snapshots excluded as hindsight |
| 12 | Rev B (fact 1) | Redesigned MAE gate already FAILS on current data (+0.195 > 0.178) | ACCEPT: stated plainly; v4 must genuinely improve NCR, not re-grade |
| 13 | Rev B (ledger 1) | K=220 pinned is taste | ACCEPT: K fitted by moment-matching |
| 14 | Rev B (ledger 2/8) | Family C and hurdle under-weighted vs evidence | ACCEPT: hurdle promoted to structural bet; C = co-equal P1 variant |
| 15 | Rev B (ledger 5) | 2% margin arbitrary | ACCEPT: derived from picker sensitivity (P0.3) |
| 16 | Rev B (ledger 9) | POTM head presented as clearly-good | ACCEPT: conditional on ablation evidence |
| 17 | Rev B (ledger 11) | "Medium-high confidence" unearned | ACCEPT: downgraded (§1) |
| 18 | Rev A (verdict) | Ranking-primary gate demanded | PARTIAL: capture restored as co-primary in the retained gate; pure ranking-primary rejected because MAE guard-rail is still needed against level drift — both must hold (file-backed: gate content unchanged from v3, only the power tiers added) |
| 19 | Red team (Q-material) | P0.2 "backfill POTM from feeds" is FALSE — feeds carry no POTM; only hand-scraped ncr_potm.csv (GW2 only); POTM ≈7% of NCR proxy bias; interceptions/steals/scrums + store-official discrepancies are the bigger drivers | ACCEPT: P0.2 rewritten as full rubric-coverage audit; POTM head conditional, 6N-trainable only; backfill firewalled from official_pts/cohort |
| 20 | Red team (Q2/Q4) | Thin-history kill cohort EMPTY on 6N-2025 under `career_matches`; v3 low_history guard fired on 7 rows (noise); mechanism defined by intl exposure but never operationalized that way | ACCEPT: thin := <5 intl matches in-store (PIT feature); precommitted cohort + min n; guard cohort replaced; untestable-on-6N case stated plainly |
| 21 | Red team (Q5) | Tier-1 presented as a live 2026 path but requires ≳0.26 MAE / ~10pp capture superiority at n≈1,050 — gap-closing v4 = parity, not promotable | ACCEPT: reachability computed and stated in §6; tier 2/2027 named the realistic route |
| 22 | Red team (Q5) | Picker-sensitivity margin under-specified = post-hoc degree of freedom | ACCEPT: pinned protocol in P0.3 (fixed squad, predeclared σ grid, 200 seeded reps, frozen once computed) |
| 23 | Red team (unverified) | Per-GW capture split (52.6/62.6 → 60.2/62.9) and "median 7 intl fixtures / 35.6% ≤3" could not be independently reproduced from artifacts | NOTED: treated as diagnostician-reported, to be re-verified as part of P0.1 before any reliance |

## 10. User decisions (RESOLVED 2026-07-17)

1. **Promotion policy**: user chose the **two-tier gate**. 2026 promotion only via decisive
   superiority on pooled GW4–7; realistic non-inferiority promotion decision in 2027 with a
   powered sample; full MAE+capture gate retained.
2. **If P1 and P2 both die on 6N-2025**: user chose **allow one more idea** — a single further
   structural attempt is authorized under the same quarantine rules (S3 unchanged, admission
   budget +1, still no looks at NCR GW1–2 / 6N 2026) before the freeze-until-November floor
   applies. S2 is amended accordingly.

## 11. Execution record (2026-07-17, decisions frozen)

All three admission decisions were spent on the 6N-2025 LORO selection layer, exactly as
precommitted in §5; quarantine S3 held (no NCR GW1–2 / 6N 2026 reads before admission freeze).

- **Decision 1 — P1 variant**: **B wins** (pooled `player_id` + post-hoc EB player effects +
  level-split form). A (EB features into the tree) failed its kill: Spearman 0.642 and mean
  capture both below baseline — trees under-exploited the engineered features, as Reviewer B
  predicted from the v3 evidence. Artifacts: `p1_folds_A_B.csv`, `p1_verdicts_A_B.json`.
- **Decision 2 — P1 admit**: **ADMITTED**. B vs frozen baseline: Spearman 0.653 vs 0.647,
  top-10 capture 69.3% vs 66.2%, mean capture 80.9% vs 80.3%, MAE 7.272 vs 7.267 (holds at
  ≤0.5%), thin-intl (<5 intl matches) MAE 5.91 vs 5.97 at pooled n=69 (≥30 → testable).
- **Decision 3 — P2 hurdle**: **NOT admitted**. Conjunctive kill: p90-coverage gap improved
  (0.050 vs 0.063) and capture held (80.97% vs 80.93%), but P(≥1) Brier regressed (0.0951 vs
  0.0923). Artifacts: `p2_folds.csv`, `p2_tails.csv`, `p2_verdict.json`.
- **Final configuration: `gbdt_v4[B]`.** The §10.2 "one more idea" allowance was NOT needed
  (P1 admitted). Deployable path: `model/unified/v4/assemble.py` fit/activate; shadow freezing
  via `model/unified/v3/shadow.py` (knows `gbdt_v4`) + `gw_update.sh` step 6 (multi-engine).
- Display-only diagnostics + paired-bootstrap CIs + November runbook: `REPORT.md` /
  `final_diagnostics.csv` / `final_bootstrap.json` (produced after this freeze; per §6 they
  cannot revise any admission).

## 12. P3 — §10.2 allowance spent (2026-07-17): global blend with the empirical unified

Prompted by the NCR-machinery-on-6N replay (`model/ncr_on_6n.py`): the incumbent's empirical
engine, generalized into a competition-independent model (`model/empirical_unified.py` — one
engine, only rubric weight dicts differ, strict PIT, driven by the unified store), posted
6N-2025 MAE 7.142 (best ever) and NCR top-10 capture 70.2% (display-only; beats even the
deployed incumbent's in-window 62.7%). It and gbdt_v4[B] win on complementary axes.

**P3 = one GLOBAL blend weight** (competition-independent by construction), swept on 6N-2025
LORO only (`model/unified/v4/p3_blend.py`): w_v4=0.5 chosen; blend MAE 7.054 / Spearman 0.663 /
mean capture 81.1% — beats BOTH components on all three → **ADMITTED** under the precommitted
rule. Quarantine held (weight chosen before any quarantined-set look).

Display-only reads (`p3_blend_display_folds.csv`): 6N-2026 MAE 7.195/Spearman 0.687; NCR MAE
**8.743** — first unified point estimate to beat the incumbent's 8.90 (paired bootstrap
blend−incumbent MAE −0.16, 90% CI −0.40…+0.08; top-10 −2.7pp, CI ±14pp = noise). **Final v4
candidate for the GW4–7 shadows: 0.5·gbdt_v4[B] + 0.5·empirical_unified.** Follow-up before
November: a predict-time (pre-lock squad) entry point for `empirical_unified.project_round`
so `gw_update.sh` can freeze the blended shadow alongside both components.

## Verification (post-implementation)

- `pytest tests/test_unified_v4.py tests/test_unified_v3.py tests/test_unified_stack.py` green;
  PIT-shift tests for every new feature; shrinkage limits; hurdle moment-match; jersey-skew test.
- Flags-off regression: v4 harness reproduces v3 baseline numbers bit-for-bit.
- P0 artifacts: error-decomposition table; POTM ablation result; derived margin memo.
- Final report: v4 vs baseline vs incumbents (MAE raw+calibrated, Spearman, capture, bootstrap
  CIs) on 6N-2025 selection; quarantined sets shown as "diagnostic display only"; GW4–7 shadow
  freeze wired into `gw_update.sh` and verified by an overwrite-refusal test.
