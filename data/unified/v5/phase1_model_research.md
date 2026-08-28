# Phase 1 / Workstream 2 — Model researcher (v5 cycle)

Brief: propose 2–4 substantially different unified model families; at least one
a disciplined extension of the strong v1 baseline; at least one may rethink the
target representation. No fashionable complexity without data-to-parameter and
validation arguments. Evidence IDs refer to `phase1_failure_diagnosis.md`.

## Family A — Shrunk-form direct-totals GBDT ("v1 + empirical Bayes"), RECOMMENDED

**Idea.** Keep v1's winning target structure exactly — direct per-event match
totals fit on all rows (zero-minute rows included), Poisson / tweedie /
bernoulli heads, minutes as a direct target, deterministic scorers downstream
(`model/unified/gbdt.py`). Change only the **feature layer** and **guards**:

1. **Hierarchical shrunk rate features** replace level-blind ewm per-80:
   per player × event, ewm-weighted event totals and minutes are split by
   level (international vs club), combined as
   `(wm_i·rate_i + c·wm_c·cal_e·rate_c + K·prior_{pos,event}) / (wm_i + c·wm_c + K)`
   with K = 220 weighted minutes (the incumbent's validated constant,
   `model/ncr_project.py`), c = 0.55 club confidence, and `cal_e` one global
   club→test calibration factor per event estimated on dual-history players in
   the training slice only. Priors are position×event per-80 means, themselves
   shrunk toward the global event mean. This is exactly the mechanism the NCR
   incumbent wins low-history players with (F12), transplanted into the
   unified contract with **zero competition-specific parameters**.
2. **Slot-aware minutes/appearance features**: shifted ewm p(play|in squad),
   starter-rate, mean minutes when starting / when benched, with jersey-slot
   position priors as fallback (incumbent's bench-slot insight, learned
   globally). Fed to the *direct* minutes/event heads — no hurdle
   multiplication (F4).
3. **Rare-event guards**: a count head is fit only with ≥20 valid rows **and**
   ≥5 positives; otherwise a stored shrunk global mean is emitted (kills the
   F3/F4 degenerate regime in every family, including v1).
4. **Attribution-quality masking (data, not scorer)**: events with
   demonstrated position-corrupted feed attribution (candidate:
   `lineouts_won`, pending E0a/E1) are masked `available__=False` globally at
   the canonical-store/feature layer. Predictions then omit the event and the
   unchanged deterministic scorer treats it as zero
   (`scoring.py::_event` default) — reproducing the incumbent's documented
   choice inside the contract, without touching `scoring.py` and with no
   6N effect (the 6N scorer never used `lineouts_won`). NCR POTM labels from
   the existing `data/ncr/ncr_potm.csv` are attached (one winner per fixture
   ⇒ all other rows in covered fixtures are true zeros), turning potm into an
   observable NCR event instead of an extrapolated one (F7).
5. **Level-blindness fix**: club and international form are separate features
   with the global calibration factor, so a URC 100-metre game no longer
   equals a Test 100-metre game (F12); `competition_level` stays categorical.

**Parameter budget.** v1 GBDT ≈ 31 targets × 180 trees × ≤23 leaves on ~200k
rows — already the search winner. The EB layer adds per event: 1 calibration
factor + 1 prior table (8 positions), all closed-form, plus 2 shared constants
(K, c) fixed a priori from the incumbent and re-validated on folds. <150 new
global parameters, none gradient-fit. Feature count grows from ~70 to ~95
numerics — safe for the same tree budget.

**Why this is the disciplined v1 extension the brief asks for**: identical
targets, identical contract, identical search-winning tree config
(`best_baseline.json`: natural weighting, 180 trees, 23 leaves); changes are
confined to shrinkage features + guards, each gated by its own ablation (E3).

**Falsifiability.** E3 measures low-history slice deviance/MAE on 2022–24
international folds. If shrunk features do not beat v1's level-blind ewm on
the low-history slice, the family is dead in one step.

## Family B — Guarded exposure/rate ("v3.1"), FALLBACK ONLY

**Idea.** Keep v3's appearance × conditional-minutes × per-80-rate structure
but repair the F4 mechanics: rate denominator floored at 20 minutes, caps from
positives-only quantiles, min-positives guard with global-rate fallback, and
event means re-scaled post-hoc so `p_play·E[min|play]·rate/80` matches the
marginal totals on training (closure of the product identity).

**Evidence for**: v3's hurdle minutes genuinely helped observable MAE (F11);
E2 is cheap. **Evidence against being the main line**: even a perfect fix only
restores deviance parity (2023/2024 were already parity, F2) and does nothing
for the NCR position-bias (F7) or low-history (F12) gaps, which are the actual
promotion blockers (F6). Pursued only as the E2 experiment; promoted to the
candidate model **only if** Family A fails E3 *and* E2 shows >5% deviance
recovery on 2022 with no 2023/24 regression.

## Family C — Partial-pooling Bayesian event model (validation oracle, not the build)

**Idea.** Per event, a Gamma–Poisson (or log-normal) hierarchical GLM:
`log rate ~ player effect + position effect + team/opponent effects + level
offset + form`, minutes as offset; full empirical-Bayes partial pooling.

**Why not the build**: (a) ~200k rows × 31 events × non-linear interactions
(position×opponent×level×recency) that GBDT captures for free would need a
large custom sampler/VI stack — weeks of work with real divergence risk;
(b) its marginals for veterans would likely be *worse* than GBDT because the
interaction structure is hand-specified; (c) v1 already wins without it.
**Why keep it on the table**: a per-event hierarchical fit on a *small* slice
(e.g., 2023–24 internationals, tries and tackles only) is the cleanest way to
*validate Family A's K and cal_e constants* independently — a one-day oracle,
not a candidate. If the oracle's shrinkage posterior contradicts K=220/c=0.55
materially, A's constants are re-set before E3 conclusions are trusted.

## Family D — Rethought target: scoring-component bundles (documented, rejected for v5)

**Idea.** Replace 31 raw events with ~6 latent scoring components (attacking
volume, goal-kicking, defence, discipline, set-piece, minutes) predicted
jointly, then decoded deterministically to events/points.

**Rejected**: breaks the `RawPrediction` per-event contract unless a learned
decoder is added (a second learned layer with weak identifiability — exactly
the v2 lesson); official-points evaluation still flows through the per-event
scorers, so the bundle adds an information bottleneck with no measurable
upside; and F7 says the binding problem is attribution/shrinkage, which a
bundle cannot express. Kept in the ledger as the considered-and-rejected
"rethink the target" option.

## Uncertainty / calibration posture (all families)

Keep v1's residual-moment dispersion per event (global NB dispersion /
lognormal sigma) — no per-player dispersion learning (data-to-parameter
indefensible at ~30 internationals/player). Calibration diagnostics (PIT of
predicted points vs official by position) are reported in evaluation; no
competition-specific recalibration is permitted anywhere.

## Ranked build order

1. **A-lite**: Family A items 3 (guards) + 1 (shrunk rates) → E3.
2. **A-full**: + item 2 (slot minutes) + item 4 (masking/potm, E0/E1-gated)
   + item 5 → E4 (6N 2025 selection replay) → full benchmark.
3. **B**: only via the E2 branch condition.
4. **C-oracle**: one-day constant validation, parallel to A-lite.

## Explicit non-goals

- No competition identity features, points heads, calibrators, or blend
  weights (constraint).
- No new external data; RugbyPass priors are incumbent-internal and are **not**
  ported (they would violate "one global model learns from the canonical
  store" spirit and are unavailable for 6N rows in PIT form).
- No re-opening of weather/style/market features (Phase 1, §E).
- No neural track in v5 (F10); the blend concept is dead with it.
