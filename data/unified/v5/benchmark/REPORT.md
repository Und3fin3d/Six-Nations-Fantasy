# v5 (ShrunkFormGBDT, Kimi K3 implementation) — benchmark vs frozen v3 comparison

Run 2026-07-17, fixed harness (`model/unified/v5/benchmark.py`), mc=800,
same folds/cohorts/seeds as the frozen v3 benchmark
(`data/unified/v3/benchmark/`). v5 config: all defaults
(EB shrinkage on, slot minutes on, guards on, masking off).

Harness note: the first run showed a spurious +1.5 MAE regression caused by a
str-vs-int key dtype mismatch that split every eval player's history at
predict time (see `_cohort_predictions` comment). v1-through-harness control
reproduces the frozen baseline (7.24 vs 7.27 fold 1), and fixed-harness v5
fold 1 = 7.37.

## Six Nations 2025 (selection)

| Model | MAE | Spearman | Pearson | Top10 | Top25 | Top50 | Top100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (v1) | 7.267 | 0.647 | 0.595 | 66.2% | 75.7% | 84.4% | 94.7% |
| gbdt_v3 | 7.266 | 0.648 | 0.595 | 68.4% | 74.0% | 84.7% | 94.5% |
| incumbent | 7.407 | 0.643 | 0.588 | 67.7% | 76.7% | 85.2% | 94.2% |
| **v5** | **7.265** | 0.646 | 0.596 | 68.6% | 75.7% | 84.2% | 94.9% |

## Six Nations 2026 (retrospective reference)

| Model | MAE | Spearman | Pearson | Top10 | Top25 | Top50 | Top100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (v1) | 7.191 | 0.686 | 0.667 | 70.7% | 78.4% | 87.2% | 95.2% |
| gbdt_v3 | 7.141 | 0.689 | 0.669 | 69.4% | 79.2% | 86.2% | 95.0% |
| incumbent | 7.305 | 0.666 | 0.637 | 71.5% | 77.1% | 83.5% | 95.2% |
| **v5** | 7.188 | 0.683 | 0.666 | **74.2%** | 76.8% | 87.5% | 95.2% |

## NCR GW1–2 (retrospective reference)

| Model | MAE | Spearman | Pearson | Top10 | Top25 | Top50 | Top100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (v1) | 9.097 | 0.522 | 0.468 | 56.4% | 61.3% | 64.9% | 76.2% |
| gbdt_v3 | 9.011 | 0.536 | 0.478 | 51.2% | 60.1% | 66.1% | 76.2% |
| gbdt_v4 (other session, display-only) | 8.97 | 0.532 | — | 55.5% | 60.3% | 67.7% | 77.6% |
| incumbent | 8.901 | 0.581 | 0.559 | 62.7% | 71.5% | 74.2% | 80.4% |
| **v5** | 9.065 | 0.524 | 0.468 | **44.6%** | 58.7% | 65.8% | 75.9% |

## Paired |error| differences (retrospective, 90% bootstrap CI)

- NCR: v5 − baseline −0.033 [−0.114, +0.044]; v5 − incumbent +0.163 [−0.200, +0.518] (n=532)
- 6N:  v5 − baseline −0.004 [−0.071, +0.065]; v5 − incumbent −0.117 [−0.317, +0.072] (n=668)

## Subgroups (retrospective NCR)

| Slice | n | v5 | baseline | incumbent |
|---|---:|---:|---:|---:|
| low_history (career<5) | 34 | 11.67 | 11.66 | **11.06** |
| bench (status B) | 180 | 6.18 | 6.28 | **6.06** |

## Verdict against the plan's own gates (`data/unified/v5/RESEARCH_PLAN.md`)

1. **v5 ≡ v1 statistically.** Every MAE/Spearman/Pearson difference vs baseline
   is inside bootstrap noise on both competitions. The EB shrunk-rate + slot
   minutes features neither help nor hurt aggregate accuracy.
2. **Kill-switch K1 fires.** The EB layer was built to fix the low-history
   slice; it delivers zero gain there (11.67 vs baseline's 11.66, incumbent
   11.06 still ahead). Per the plan, the EB layer has not earned its place.
3. **NCR capture gate fails.** Top-10 capture 44.6% vs incumbent 62.7%
   (−18pp; gate allows −2pp) and below baseline's 56.4%. Same blocker as
   gbdt_v3, unresolved.
4. **6N side is fine but not better.** v5 matches baseline and beats the 6N
   incumbent on MAE, with the best 6N-2026 top-10 capture (74.2%) — a mild
   positive, inside noise.
5. **No promotion case.** v5 does not clear the retrospective safe-overall
   gate; v1 remains the universal baseline and both specialists remain
   incumbents. The attribution-masking and E-ladder experiments (E0a/E1/E2/E3)
   the plan precommitted were bypassed by building the full default config
   directly; the negative result here is consistent with the plan's own
   prediction that untested layers should be assumed inert until proven.

---

# Corrective cycle result — terminal rung T (v1 + guards + `lineouts_won` masking, EB/slot off)

Rerun 2026-07-17 evening, same protocol. EB-full artifacts archived as
`fold_metrics_v5_ebfull.csv` / `predictions_v5_ebfull.csv`.

## NCR GW1–2 (retrospective)

| Model | MAE | Spearman | Pearson | Top10 | Top25 | Top50 | Top100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (v1) | 9.097 | 0.522 | 0.468 | 56.4% | 61.3% | 64.9% | 76.2% |
| gbdt_v3 | 9.011 | 0.536 | 0.478 | 51.2% | 60.1% | 66.1% | 76.2% |
| v5 EB-full (pre-corrective) | 9.065 | 0.524 | 0.468 | 44.6% | 58.7% | 65.8% | 75.9% |
| **v5-T (corrective)** | **8.961** | 0.532 | 0.488 | 53.8% | 60.3% | 65.5% | **77.4%** |
| incumbent | 8.901 | 0.581 | 0.559 | 62.7% | 71.5% | 74.2% | 80.4% |

6N: v5-T ≡ v1 within MC noise on both layers (selection 7.262 vs 7.267;
retro 7.200 vs 7.191), as predicted (masking is 6N-inert).

Paired |error| (retrospective, 90% CI):
- NCR: v5-T − baseline **−0.138 [−0.268, −0.009]** — the first unified variant
  to beat v1 on NCR MAE with a CI excluding zero. v5-T − incumbent +0.058
  [−0.274, +0.383] (inside the ≤+2% gate).
- 6N: v5-T − baseline +0.009 [−0.016, +0.034] (inert, F3 pass).
- low_history: 11.32 (vs baseline 11.66, incumbent 11.06) — F4 pass.

## Falsifier ledger (precommitted in `corrective_cycle.md` §5)

- **F1 — FIRES.** NCR top-10 capture 53.8% < baseline 56.4% + 2pp. Masking
  recovered +9.2pp over EB-full (44.6 → 53.8) but did not clear the baseline
  band, let alone the incumbent gate (62.7 − 2pp). Per the precommitment:
  stop; no further cycles.
- F2 pass (MAE improved, significantly vs baseline).
- F3 pass (6N inert). F4 pass (low-history improved).

## Final decision

**No promotion. Retain v1 as universal baseline and both specialists as
deployment incumbents** (stop rules S1/S4). The cycle's durable findings:

1. The EB shrunk-rate/slot feature layer is measurably harmful at the
   points-weighted top-N instrument (−9pp NCR top-10) and inert everywhere
   else — K1 confirmed twice (aggregate + row level).
2. `lineouts_won` masking alone is worth +9pp NCR top-10 capture and a
   statistically significant NCR MAE gain at zero 6N cost — but roughly half
   the capture deficit vs the incumbent is *not* attribution: the incumbent's
   remaining edge (calibrated club/test priors, competition knowledge) is
   still ~9pp of capture and unaddressed by any unified variant to date.
3. v5-T is the strongest unified NCR MAE to date (8.961; cf. gbdt_v4 8.97,
   gbdt_v3 9.01, v1 9.10) and could inform the v4 line's Nov GW4–7 shadow
   programme, but it does not meet the frozen promotion gates.
