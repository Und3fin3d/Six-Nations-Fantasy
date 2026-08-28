# Phase 3 / Reviewer B — the strongest alternative plan, and a taste audit

## 1. The strongest alternative plan: "subtraction, not addition"

**AP-B.** Ship **v1 + guards + attribution masking only** (plan's A1/A4 merged)
as the single v5 candidate. Skip the entire EB feature layer. Rationale:

1. The v3 decision's NCR numbers say the MAE gap is bootstrap-noise (CI
   [-0.17, +0.56]) and the capture gap is the real deficit. Phase 1's own
   row-level evidence (F7) attributes a large share of that capture gap to
   position-correlated scorer bias — which masking addresses directly,
   cheaply, and with a measurable pre-read (E1).
2. The EB layer's benefit is mechanism-plausibility, not measurement. Its
   proving ground (thin, era-biased first-cap slices on 2022–24 folds — see
   Reviewer A3) is the weakest evaluation surface in the plan. Shipping it
   adds ~25 features, 3 constants, and a calibration table to a candidate
   that must pass a *safe-overall* gate whose history (v3) punished every
   added degree of freedom.
3. Every prospective gameweek is irreplaceable evidence. The candidate most
   likely to *pass* GW4–7 is the one with the fewest ways to be wrong in a
   new subgroup. Subtraction minimises subgroup risk; addition redistributes
   it.
4. If the masked v1 fails GW4–7 on low_history again, the EB layer is still
   there, now justified by *prospective* evidence rather than fold-proxy
   hope. The existing narrow-miss rule even allows one corrective cycle.

**Where AP-B is genuinely stronger**: multiplicity (one candidate, no ladder),
power (decision rests on E1 + selection benchmark + prospective, not on 3
noisy folds), and risk shape (nothing new to regress).

**Where AP-B is weaker (evidence, not taste)**: the NCR low_history +0.59
subgroup gate (`decision.json`) is a *demonstrated* failure that masking
cannot repair — masked or not, a debutant's prediction is still an
uncalibrated level-blind ewm over club rows, while the incumbent shrinks
toward calibrated priors (F12). GW4–7 squads will contain debutants again.
AP-B's answer — "fix it after it fails prospectively" — spends the only
prospective rounds of the year to learn what the retrospective subgroup gate
already told us. The correct synthesis is not AP-B vs the plan; it is the
plan's ladder with a **precommitted kill-switch threshold**: if E3's
low-history-slice gain is <1% observable-MAE in ≥2/3 informative folds, the
EB layer is dropped and the candidate *is* AP-B. The plan already gestures at
this (S2) but leaves the threshold undefined — define it.

## 2. Second alternative: "wait for GW4–7 before designing v5"

Collect baseline shadows through GW3–GW7, then design v5 with 4–6 prospective
NCR rounds in hand. Rejected: v5 development consumes no prospective rounds
(shadows freeze regardless of which engine is active — GW3 already ran with
the baseline), and the prospective layer's decision value is identical whether
v5 exists or not. Waiting only delays the earliest honest promotion. The one
valid kernel: **do not let v5 touch GW3–7 shadow slots until it has passed the
full retrospective gate** — the baseline keeps shadowing until then.

## 3. Taste-vs-evidence audit of the initial plan

| Decision | Basis | Verdict |
|---|---|---|
| Direct totals over exposure/rate | F2/F3/F4 (2022 collapse is engineering; 2023/24 parity) | Evidence — moderately strong |
| Kill the rate layer without E2's verdict | E2 not yet run | **Taste-leaning**; keep E2 as the one-day falsifier before the obituary is final |
| K=220, c=0.55 shrinkage constants | Incumbent's values, tuned partly on NCR GW1–2 | **Contaminated inheritance**; acceptable only with the K×{0.5,2} sensitivity rung |
| ewm halflife 4 matches | v1 default; never searched | **Taste**; note it, do not search it this cycle (multiplicity budget) |
| Keeping all 31 events incl. non-scoring (runs, passes, rucks) | Contract fidelity; deviance diagnostics only | **Taste**; dropping them would not change points at all — flag as future simplification, keep for now |
| Bernoulli heads excluded from rate features | potm is per-match, cards rare | Taste, immaterial |
| Masking `lineouts_won` globally vs an NCR-only handling | Global masking is the only competition-independent option; scorer is frozen | Evidence (constraint) |
| Selection = 6N 2025 alone; NCR GW1–2 reference-only | v3 protocol | Evidence (protocol) |
| 2% MAE / 2pp capture / +0.5 subgroup margins | Existing safe-overall rule | Untouchable per constraints |
| E2 spends a day on a design 2023/24 already condemned | Diagnostic closure value | Weak taste — capped at one day, acceptable |

## 4. Demands on the plan author

B1. Fold AP-B into the plan as the **explicit terminal rung of the ladder**
    with the kill-switch threshold written down (EB gain <1% low-history
    observable-MAE in ≥2/3 informative folds ⇒ candidate = masked v1+guards).
B2. State that the baseline engine continues to own GW3–GW7 shadow slots until
    a v5 candidate passes the full retrospective gate.
B3. Mark K/c/halflife/provenance rows in the plan with their verdicts above so
    future cycles don't re-litigate them as if they were measured.
B4. Commit that E2's result cannot resurrect the rate layer unless it *also*
    beats v1 on low-history slices — otherwise the "fallback" branch is dead
    code that exists only to be argued about.
