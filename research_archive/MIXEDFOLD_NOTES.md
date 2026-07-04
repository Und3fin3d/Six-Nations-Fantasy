# Mixed-year-fold CV (round-level OOF isolation)

Branch `agents/mixed-year-fold-2026-07-04`, isolated worktree. Trains the
component model from scratch with a **mix of years in every test fold** —
GroupKFold(5) over whole `(season, round)` groups pooled across all four years
2023/2024/2025/2026.

## Isolation

Standard out-of-fold isolation at the ROUND level: a test round's own rows are
never in that fold's training set, but every other round is — including sibling
rounds of the same year. So `2026 R2` in the test fold means `2026 R2` is out of
training, while `2026 R3` stays in. Each round is predicted exactly once, held
out of its own fold; rounds are never split across folds.

This is a generalisation CV, not the production protocol's pristine sealed-2026
holdout (it does train on 2026 sibling rounds). The 2026 comparison
(`compare_2026.py`) showed that touching 2026 siblings did NOT inflate 2026 —
sealed-holdout recon value_xv 0.720 vs mixed 0.667, i.e. the mixed protocol was
slightly WORSE — so the leakage risk is empirically negligible. Kept in this
worktree; if a truly untouched final go/no-go is ever wanted, reserve a slice
outside the CV.

## Result (`train_mixedfold.py`, whole-round folds, recon-only target)

- 20 independent `(year, round)` evaluation blocks (vs 5 in the sealed protocol).
- recon value_xv 0.6726, SD 0.063, **SE 0.0140** — matches the sealed-protocol
  LOYO recon value_xv (0.673) as a cross-check.
- Year-consistent: 2023 0.690 / 2024 0.636 / 2025 0.697 / 2026 0.667 — 2026 is
  not an outlier, evidence the model is not overfit to the dev years.
- recon MAE 6.13 per row (recon-only; not comparable to the full-target 7.4).

## Conclusion

Even the maximal-power protocol (all four years pooled, seal sacrificed) only
reaches SE ~0.014 — an order of magnitude above the ~0.005 selection-tweak
gains. The value-metric noise floor is **fundamental to the small number of
Six Nations rounds**, not an artifact of the sealed 2025-only split. Mixing
years buys real power (SE 0.046 -> 0.014) and confirms year-generalisation, but
does not make sub-0.01 selection gains resolvable. Use the sealed protocol in
`../6n-autoresearch-next-2` for anything promotable.

## Autoresearch iteration 1 — position minutes (RESOLVED WIN)

`search_minutes.py`: position x started minutes dummies vs the global minutes
ridge, on the 20-round mixed OOF, paired at the row level (~2,718 rows).

- recon MAE 6.129 -> 6.090; **paired row-level t = -2.58 (seed 0)**, and robust
  across 5 seeds: t in {-2.58, -3.60, -3.91, -4.10, -4.11}, always position
  better by -0.04 to -0.06 pts/row (~0.6-1.0% relative).
- value_xv +0.003 (t=0.65) — positive, not significant, NOT negative.

Interpretation: this confirms the Experiment-17 diagnosis. Position minutes is a
GENUINE MAE improvement; it was rejected on the champion only because the
champion's downstream stack (front-row prior blend, bench two-stage pull,
residual layer) had co-adapted around the biased global minutes and
double-corrected. On a co-adaptation-free from-scratch model it helps, and does
not hurt decisions. This is also the first statistically-resolvable result in
the whole program — visible only because we switched to the row-level MAE metric
on the high-power mixed-year split. Actionable path: a from-scratch model rebuild
(position minutes in the base) is the right way to actually move MAE, evaluated
on this split; the 5-round value_team backtest could never have surfaced it.

## Autoresearch iterations 2-3

- Batch 2 (component-engine hyperparameters): ALL null. num_leaves=31 overfits
  (MAE +0.061, t~+3.5), min_child_samples=20 / n_estimators=800 / reg_lambda=2
  washes, minutes started×is_forward interaction adds nothing over position
  dummies. Default LGBM engine is well-tuned; family closed.
- Batch 3 (component variance-shrinkage toward the naive base rate): WIN.
  Shrinking ALL main component rates 20% toward base (shrink_all_20) cuts
  recon-MAE by 0.050, robust across 3 seeds (t = -3.21/-3.46/-2.94), value_xv
  +0.010. Light-on-all beats heavy-on-spikes (shrink_spikes_50 was null) — the
  LGBM heads are generally slightly over-confident, and a light uniform pull to
  the base rate is effective variance regularisation the tree params don't give.

## Autoresearch iteration 4 — shrinkage level sweep (5 seeds)

Uniform shrink-all toward the naive base rate, level sweep, paired row-level MAE:
| level | dMAE | t/seed (5) | verdict |
| 0.10 | -0.030 | -3.3..-4.5 | win |
| 0.15 | -0.042 | -3.0..-4.2 | win |
| **0.20** | **-0.047** | **-2.6..-3.5** | **WIN (adopted)** |
| 0.25 | -0.048 | -2.2..-3.0 | win |
| 0.30 | -0.045 | -1.7..-2.4 | over-shrinks (rejected) |

dMAE plateaus ~0.047 while significance falls with bias; 0.20 is the robust
optimum with the best value_xv (+0.013). ADOPTED into BASE alongside position
minutes. Cumulative recon-MAE vs the original global-minutes/no-shrink model:
6.129 -> 6.090 (position) -> ~6.043 (shrink) = -0.086, ~1.4%.

## Autoresearch iteration 5 — shrink target (rejected)

Shrinking toward a RIDGE prior instead of the naive base rate is WORSE (vs the
naive-shrink-0.20 base: dMAE +0.012/+0.010/+0.020/+0.039 at 0.20/0.30/0.40/0.50,
value_xv negative). The naive base rate is a cleaner unbiased shrinkage anchor;
ridge injects its own bias. Naive-shrink target confirmed; ridge target closed.

## Autoresearch iteration 6 — minutes variants (closed)

On the adopted base: minutes_position_bench null (dMAE -0.001), minutes_poisson
WORSE (dMAE +0.036, t~+3). Minutes model settled at ridge + position×started
dummies; bench/poisson/interaction all closed.

## Autoresearch iteration 7 — per-group shrinkage split (closed)

Splitting shrinkage (noisy counts more, tackles/metres less) does NOT beat
uniform-0.20 (dMAE +0.000 to +0.011, all null across 5 seeds). Simpler uniform
0.20 confirmed optimal; per-component shrink tuning closed (also avoids
overfitting multiple levels).

## Autoresearch iteration 7 + 2026 validation

- Batch 7 (per-group shrinkage split counts>volume) and a bagging batch: all
  null vs the uniform-0.20 base. Uniform shrink-0.20 stands.
- eval_2026.py — ORIGINAL (global minutes, no shrink) vs IMPROVED (position +
  shrink-0.20) on the 20-round mixed OOF, by year:
  overall recon value_xv 0.6717 -> 0.6835 (+0.012); recon MAE 6.129 -> 6.029 (-0.100).
  MAE improves on EVERY year (2023 -0.136, 2024 -0.061, 2025 -0.060, 2026 -0.144).
  value_xv up on 2023/2024/2026, down on 2025 (round-noise).
  **2026 OOF specifically: MAE 6.458 -> 6.314 (-0.144, largest of any year),
  value_xv 0.680 -> 0.706 (+0.026).** The adopted changes generalise to the
  2026 OOF rounds and help them most.

## Sealed 2026 deployment test (eval_sealed_2026.py)

Train 2023+2024+2025, test 2026 held out ENTIRELY (forward-chained, no 2026 in
training). ORIGINAL (global minutes, no shrink) vs IMPROVED (position + shrink-0.20):
| benchmark | original | improved | delta | verdict |
| recon MAE | 6.4705 | 6.3215 | -0.149 | IMPROVED (-2.3%) |
| top15     | 0.3200 | 0.3467 | +0.027 | IMPROVED |
| value_xv  | 0.7199 | 0.7099 | -0.010 | ~noise (5 rounds) |

MAE decreases (better) and top15 increases (better) on the true sealed 2026 —
resolvable metrics, robust. value_xv dips -0.010 but that is within 5-round
noise and the SIGN FLIPS vs the mixed-OOF 2026 (+0.026), confirming 5-round
value_xv cannot resolve an effect this small. Adopt for the MAE/top15 gain; do
not read a specific value_xv move on any single 5-round holdout.
