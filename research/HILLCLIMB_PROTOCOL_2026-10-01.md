# Hill-climb selection protocol (frozen before 2023 development results)

Written 1 October 2026, after the Six Nations 2023 development fits were launched and before any of their results were read.

## Reference and benchmarks

Reference: `p3_robust_native` refitted locally on the 22 September research store (`model.unified.rolling_eval --native-categories`). Every candidate is compared with this local reference and the empirical fantasy baseline on all four benchmarks:

1. Six Nations 2025 rounds 1–5: mean round fantasy MAE and selected-squad points.
2. Six Nations 2026 rounds 1–5: the same.
3. NCR 2026 GW1–3: the same.
4. Friendly-25 (the latest 25 completed cached friendlies before 22 September 2026; Friendly-15 is its latest 15): count-stat MAE, with metres and minutes MAE reported separately.

A "real improvement" improves MAE in all three fantasy competitions and Friendly-25 count-stat MAE, with no competition losing squad points against the reference.

## Already-measured candidates, recorded as rejected

- Pre-match team context (Elo, venue, points form) as tree features: worse MAE in both Six Nations seasons and Friendly-25.
- Tree capacity (360 trees, 63 leaves, 500 trees at learning rate 0.02): no development gain on Six Nations 2024.
- Per-event blend weights re-tuned on 2022–24 blocks for fantasy MAE or MSE: mixed on official slates.
- Official-stat remaps (metres x1.4; breakdown steals): mixed; MAE penalises raising means on a skewed target.
- Median/quantile point forecasts: worse, and excluded on principle because MAE on counts rewards predicting the median (for example zero tries) rather than a better model.

## Disclosure: the blend grid was already seen

Before this protocol, the official-slate grid for C1 was inspected: `w = 0.8` improved all six fantasy numbers against the reference, `w = 0.7` lost 118 NCR points through one super-sub change, and `w = 0.5` had the lowest MAE in two competitions. Choosing `w` from that grid would be selection on the test seasons. C1 therefore counts only if the 2023 rule below independently selects a weight, and it is reported at that weight whatever the result.

## Candidates fixed now

**C1: points blend.** `w * robust + (1 - w) * empirical_baseline` for fantasy points. `w` is chosen from {0.5, 0.6, 0.7, 0.8, 0.9, 1.0} by the lowest mean round MAE on Six Nations 2023 rounds 1–4 (development; strictly earlier history per round; older rubric). Ties within 0.005 go to the larger `w`. Raw-event forecasts are unchanged, so C1 cannot change Friendly-25.

**C2: team-strength event calibration.** Each robust event mean is multiplied by `exp(beta_e * edge / 400)`, where `edge` is the pre-lock Elo edge (`model/unified/v4/context.py`) and `beta_e` is a team-level Poisson fit on the 14 archived raw blocks before 2025 only (`beta_all.json`, already fitted). Already seen: held-out 2025–26 blocks improve team-level Poisson loss for every examined event and count-stat MAE (0.6646 to 0.6632), but not player fantasy MAE.

**C3: C2 followed by C1.** `w` is re-chosen by the same 2023 rule using calibrated robust forecasts.

No other weight, coefficient or candidate will be selected from the 2025, 2026, NCR or Friendly-25 results. All three candidates are reported in full, including failures.
