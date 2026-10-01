# Hill-climb report, 1 October 2026

## Outcome

The best candidate found improves fantasy-point MAE in all three fantasy competitions and improves Friendly-25 count-stat MAE. Its selected-squad points are unchanged within the measured noise floor: they are not a demonstrated improvement.

The candidate, `p3_hillclimb_2026_10` (`model/unified/hillclimb.py`), starts from robust P3 and applies three steps:

1. **Team-strength event calibration.** Each event mean is multiplied by `exp(beta_e * elo_edge / 400)`. The edge comes from a pre-lock margin-of-victory Elo (`model/unified/v4/context.py`). The `beta_e` coefficients come from team-level Poisson fits on the 14 archived international blocks before 2025.
2. **Goal-kicking concentration.** Each team keeps its expected kicking attempts, but they are reallocated in proportion to `attempts ** 1.5`. The exponent was chosen on the same pre-2025 blocks (`model/unified/kicking.py`).
3. **Blend with the empirical fantasy baseline:** `0.75 * model points + 0.25 * empirical points`.

It also fixes an NCR scoring defect. Scrum points now go only to props and hookers, as the official rules require (`model/unified/ncr_gw_eval.py`). Every model in this report is scored under that corrected rule.

| Benchmark | Robust P3 reference | Candidate | Empirical baseline |
|---|---:|---:|---:|
| NCR 2026: MAE | 8.251 | **8.222** | 8.401 |
| NCR 2026: squad points (smoothed) | 1654 (1651) | 1611 (1630) | 1578 (1584) |
| Six Nations 2025: MAE | 7.240 | **7.194** | 7.167 |
| Six Nations 2025: squad points (smoothed) | 2356 (2342) | 2333 (2351) | 2410 (2419) |
| Six Nations 2026: MAE | 7.238 | **7.218** | 7.351 |
| Six Nations 2026: squad points (smoothed) | 2513 (2510) | 2533 (2519) | 2552 (2556) |
| Friendly-25 count-stat MAE | 0.65414 | **0.65322** | 0.68068 (raw comparator) |
| Friendly-25 metres / minutes MAE | 10.857 / 9.899 | **10.826** / 9.899 | 11.068 / 10.000 |

**MAE evidence.** These are paired 90% fixture-bootstrap intervals for the candidate's MAE change against the reference:

| Competition | MAE change | 90% interval | Rounds where MAE improves |
|---|---:|---|---|
| NCR 2026 | −0.029 | [−0.058, +0.002] | 3 of 3 |
| Six Nations 2025 | −0.047 | [−0.074, −0.019] | 5 of 5 |
| Six Nations 2026 | −0.021 | [−0.062, +0.023] | 3 of 5 |
| Friendly-25 count-stat | −0.0009 | [−0.0032, +0.0016] | 16 of 25 games |

The blend weight was not fixed before the evaluation results were seen. This is disclosed under "Selection history" below. MAE improves in all three competitions for every weight from 0.6 to 0.9, so the result does not depend on the exact value of 0.75.

**Squad points.** Realised totals change by −43, −23 and +20. Smoothed totals change by −22, +9 and +9. Both are inside the noise floor described in the next section. Prospective rounds are needed before claiming a decision gain.

## Why squad points cannot separate these models

The reference's own forecasts were perturbed with independent multiplicative noise and the unchanged optimiser was re-run (40 draws):

| Forecast noise | NCR 2026 points SD | Six Nations 2025 SD | Six Nations 2026 SD |
|---|---:|---:|---:|
| 1% | 26 | 37 | 22 |
| 2% | 35 | 57 | 89 |
| 5% | 82 | 65 | 125 |

A forecast change of 1–2% moves season totals by 20–90 points, mostly through one captain or super-sub choice. Every recorded squad-point difference in this project is of that size. Some examples:

- The context run changed forecasts with a correlation of 0.9995 to the reference, yet "gained" 111 points in 2025.
- One NCR super-sub decided 150 points.

"Smoothed" points are the mean over 40 identical 1% jitters. Smoothing removes knife-edge tie-break luck while still using only realised outcomes.

## Findings

1. **NCR scoring defect.** The adapter credited expected scrums to every forward. With the official front-row-only rule, robust P3's NCR MAE falls from 8.370 to 8.251.
   - The reference's NCR squad lead depended on this bug. Pollock was picked as super-sub on 1.22 expected scrums (×2 points, ×3 multiplier).
   - Its corrected total is 1654, not 1726. Every comparison here uses the corrected rule.
2. **Team strength is mis-sized per event.** Team-level residuals correlate with the pre-match Elo edge. Strong teams score more tries than forecast (r=+0.16) but make fewer metres, runs and defenders beaten than forecast (r≈−0.2). This happens because the empirical component applies one world-ranking multiplier to every attacking event.
   - The fitted coefficients are stable when refitted at later locks (tries 0.16–0.175, metres −0.13 to −0.14).
   - On held-out 2025–26 blocks they improve team-level Poisson loss for every examined event. Team metres MAE falls from 102.3 to 98.8.
   - Adding the same signals as tree features did nothing: forecasts moved only in the fourth decimal.
3. **Kicking is smeared across players.** The top predicted kicker receives 67% of the team's predicted attempts but actually takes 88%. The second and third candidates receive 22% and 7% predicted, against 10% and 1.5% actual.
   - Concentrating with γ=1.5 cuts held-out kicking-points MAE by 12%.
   - On its own it improves Friendly-25 count-stat MAE in 22 of 25 games; that interval excludes zero.
4. **MAE favours medians on a skewed target.** Several fixes that make the expected value more accurate worsen fantasy MAE:
   - official metres being 1.45× API metres;
   - official breakdown steals versus the API tackle-turnover proxy;
   - an expected-points player-of-the-match term.

   MAE is minimised by the median, and raising the mean of a rare high-value event moves forecasts away from it. Median or quantile forecasts were therefore excluded on principle. On friendly count stats they would, for example, predict zero tries for everyone.
5. **Tree capacity and variance are not the bottleneck.**
   - More trees, more leaves or a slower learning rate did not beat the default on the Six Nations 2024 development block.
   - Perturbed refits correlate at 0.9995, so seed-bagging cannot help.
6. **The 2023 development season is confounded for MAE.** The 2023 official rubric averages 24.5 points per player against forecasts of about 13–14. Its MAE therefore rewards whichever model predicts higher, which pointed to a weight of 0.5. Scale-free 2023 criteria (Pearson and Spearman) favour robust alone, a weight of 1.0.

## Candidate ledger

Scoring is identical for every row: corrected NCR scrums, the same pools, the same optimiser, and the same jitter draws. Cells are MAE / realised squad points / smoothed squad points. Blends reuse their base model's raw forecasts, so their friendly column repeats that model's result.

| Candidate | NCR 2026 | Six Nations 2025 | Six Nations 2026 | Friendly-25 count / metres / minutes |
|---|---|---|---|---|
| p3_robust_native (reference) | 8.251 / 1654 / 1651 | 7.240 / 2356 / 2342 | 7.238 / 2513 / 2510 | 0.65414 / 10.857 / 9.899 |
| empirical_baseline | 8.401 / 1578 / 1584 | 7.167 / 2410 / 2419 | 7.351 / 2552 / 2556 | 0.68068 / 11.068 / 10.000 |
| **FINAL K15+C2, w=0.75** | **8.222** / 1611 / 1630 | **7.194** / 2333 / 2351 | **7.218** / 2533 / 2519 | **0.65322** / 10.826 / 9.899 |
| C2 team strength (registered) | 8.202 / 1714 / 1689 | 7.245 / 2290 / 2311 | 7.253 / 2514 / 2512 | 0.65396 / 10.826 / 9.899 |
| K15 kicking | 8.253 / 1654 / 1651 | 7.263 / 2313 / 2320 | 7.212 / 2499 / 2476 | 0.65337 / 10.857 / 9.899 |
| K15+C2 | 8.203 / 1714 / 1689 | 7.264 / 2316 / 2319 | 7.226 / 2454 / 2472 | 0.65322 / 10.826 / 9.899 |
| C2s (dev-selected events) | 8.281 / 1645 / 1651 | 7.235 / 2281 / 2271 | 7.246 / 2502 / 2502 | 0.65378 / 10.826 / 9.899 |
| K15+C2s | 8.284 / 1645 / 1647 | 7.238 / 2326 / 2312 | 7.221 / 2442 / 2442 | 0.65303 / 10.826 / 9.899 |
| C1 blend w=0.5 (protocol rule) | 8.288 / 1572 / 1582 | 7.143 / 2415 / 2410 | 7.241 / 2519 / 2534 | unchanged |
| C3 C2 + blend w=0.5 (protocol rule) | 8.260 / 1562 / 1585 | 7.137 / 2479 / 2437 | 7.245 / 2519 / 2529 | 0.65396 / 10.826 / 9.899 |
| Context features in trees | 8.275 / 1646 / 1642 | 7.254 / 2467 / 2472 | 7.248 / 2536 / 2510 | 0.65427 / 10.854 / 9.895 |
| Blend w=0.8 (test-peeked) | 8.260 / 1664 / 1641 | 7.191 / 2388 / 2382 | 7.230 / 2583 / 2536 | unchanged |
| K15+C2 blend w=0.8 (test-informed) | 8.217 / 1611 / 1641 | 7.205 / 2355 / 2342 | 7.217 / 2536 / 2504 | 0.65322 / 10.826 / 9.899 |
| C2 blend w=0.8 (test-informed) | 8.216 / 1611 / 1641 | 7.192 / 2413 / 2380 | 7.240 / 2531 / 2525 | 0.65396 / 10.826 / 9.899 |

Further candidates were rejected before the corrected NCR rule was introduced. Their NCR cells used the legacy scorer:

- Per-event P3 weights retuned on 2022–24 blocks for fantasy MAE or MSE: Six Nations 2025 MAE improved to 7.20–7.21, but NCR worsened and Six Nations 2026 was flat.
- Metres ×1.2–1.5: worse Six Nations MAE.
- Breakdown-steal remap: 2025 worse, 2026 better. Official position rates also shift between seasons; back-three rates were 0.126 in 2023 and 0.035 in 2025.
- Simulated-median forecasts: worse.
- Expected-points POTM heuristic: Six Nations MAE +0.13.
- Capacity variants: no development gain.

The K15+C2s blend makes NCR MAE worse at every weight from 0.6 to 1.0 (+0.033 to +0.045).

## Selection history (read before relying on the candidate)

1. `HILLCLIMB_PROTOCOL_2026-10-01.md` was committed before the 2023 development results were read. It fixed C1–C3 and the 2023 MAE rule for the blend weight.
2. Under that rule, all three protocol candidates fail. C1 and C3 lose NCR MAE and NCR points under the corrected scorer, and C2 loses Six Nations MAE.
3. Before the protocol, the official-slate blend grid for robust P3 had already been inspected. The protocol disclosed this.
4. The 2023 rule then turned out to be confounded (finding 6). Kicking concentration (γ chosen on development blocks) and the NCR fix came after the protocol.
5. The final weight is the midpoint of the interval bracketed by the two 2023 criteria, 0.5 and 1.0. It was chosen after the evaluation seasons had been inspected. The weight-insensitivity table below is the main safeguard against selection on noise, but it is not a substitute for untouched data.

MAE change against the reference, by blend weight:

| Weight | NCR 2026 | Six Nations 2025 | Six Nations 2026 |
|---:|---:|---:|---:|
| 0.60 | −0.008 | −0.079 | −0.015 |
| 0.70 | −0.023 | −0.058 | −0.020 |
| 0.75 | −0.029 | −0.047 | −0.021 |
| 0.80 | −0.034 | −0.035 | −0.021 |
| 0.85 | −0.039 | −0.022 | −0.021 |
| 0.90 | −0.043 | −0.008 | −0.020 |
| 1.00 | −0.049 | +0.023 | −0.012 |

Smoothed squad points stay within ±25 of the reference across weights 0.7–0.9 in every competition.

**Required confirmation.** Freeze `p3_hillclimb_2026_10` before the NCR GW4–7 locks in November 2026 and evaluate it prospectively against robust P3. Use the corrected scorer for both models.

## Benchmarks and reproduction

- **Reference.** `p3_robust_native`, refitted locally on the 22 September research store (179k rows). It reproduces the archived PR #24 slate results to within the later data refresh, for example Six Nations 2025 R1 MAE 6.897 against 6.891.
- **Friendly-25.** The latest 25 completed cached friendlies before 22 September 2026 (`data/unified/friendly25/fixtures.json`), chosen with the frozen Friendly-15 rule. The cohort is Friendly-15 plus the 10 preceding November 2025 tests. Both complete teamsheets are scored: 1,150 player rows.
  - All fixtures on one UTC day share that day's first kickoff as the lock.
  - Under this lock, robust P3 scores 0.65414 on all 25 games and 0.66414 on the Friendly-15 subset.

Commands:

```bash
# Reference slates; add --hillclimb to score the candidate alongside robust P3.
python -m model.unified.rolling_eval --native-categories --hillclimb --output OUT
# Friendly-25 reference forecasts (also fits a candidate variant).
python -m research.friendly_experiment --base OUT --output F25 --reference
# 2023 development slates.
python -m research.dev_six_nations_2023 --base OUT --output DEV
# Full ledger from saved forecasts.
python -m research.hillclimb_ledger --runs RUNS --transforms research/hillclimb_2026-10-01/candidate_transforms.json \
    --blends research/hillclimb_2026-10-01/candidate_blends.json --output LEDGER
```

Evidence in `research/hillclimb_2026-10-01/`:

- season and round results;
- every Friendly-25 game/stat cell;
- bootstrap intervals;
- 2023 development metrics;
- the capacity probe;
- the frozen and rolling team-strength coefficients;
- the candidate specifications.

No RapidAPI requests were made, the permanent cache is unchanged, and no model was promoted.

**Verification.**

- A fresh cache-only rebuild reproduced the research store byte for byte (identical SHA-256).
- A fresh full refit of Six Nations 2026 R3 through `rolling_eval --hillclimb` reproduced this ledger exactly: robust 7.423671 MAE / 463 points; candidate 7.428887 / 483.

The full suite gives 145 passed and 1 failed, run on the pinned stack with the LFS artifacts pulled. The failure is the pre-existing fixed-count assertion: it expects 2,144 labels and the data holds 2,164 (documented in PR #24). New tests cover:

- context and team-strength calibration;
- kicking concentration;
- the candidate module;
- the NCR front-row rule;
- the Friendly cohort and evaluation helpers.
