# Combining the October fixes with robust P3

Question: do the parallel agents' fixes improve our best unified model, robust P3?

- **SK** (agent B): status-aware, within-position-shrunk empirical component.
- **BP** (agent A): Six Nations official-stat adapter for breakdown steals and player of the match.

Each was also tested on top of MK (matchup calibration plus kicking concentration) and H2 (MK plus the 0.7/0.3 empirical blend). Every layer uses the settings frozen in its own protocol; nothing was tuned here. Command: `python -m research.champion_combo_eval --set {dev,official,friendly}`. Evidence is in `research/champion_combo_2026-10/`.

## Results

Each cell is MAE / MSE / smoothed squad points. Friendly-25 is raw count-stat MAE; BP and the blend do not apply there.

| Engine | NCR 2026 | Six Nations 2025 | Six Nations 2026 | Friendly-25 |
|---|---|---|---|---|
| robust (reference) | 8.251 / 147.4 / 1651 | 7.240 / 111.0 / 2342 | 7.239 / 101.6 / 2510 | 0.65414 |
| robust+BP | 8.251 / 147.4 / 1651 | 7.270 / 110.7 / 2330 | 7.252 / 100.9 / 2586 | 0.65414 |
| SK | 8.263 / 147.7 / 1662 | 7.210 / 110.4 / 2455 | 7.243 / 101.9 / 2514 | 0.65480 |
| SK+BP | 8.263 / 147.7 / 1662 | 7.250 / 110.3 / 2376 | 7.249 / 101.1 / 2568 | 0.65480 |
| MK | 8.232 / 148.3 / 1703 | 7.237 / 111.8 / 2306 | 7.205 / 101.8 / 2464 | 0.65308 |
| SK+MK+BP | 8.241 / 148.7 / 1737 | 7.250 / 111.3 / 2333 | 7.220 / 101.5 / 2543 | 0.65364 |
| H2 | 8.251 / 148.0 / 1626 | 7.157 / 109.9 / 2327 | 7.205 / 102.5 / 2509 | 0.65308 |
| SK+H2+BP | 8.252 / 148.2 / 1653 | 7.157 / 109.2 / 2348 | 7.203 / 101.8 / 2511 | 0.65364 |
| empirical baseline | 8.401 / 150.3 / 1584 | 7.167 / 108.9 / 2419 | 7.351 / 107.4 / 2556 | 0.68068 (raw comparator) |

Development set (Six Nations 2023 rounds 2–4; round 1 is excluded because its official rows are shifted):

| Engine | MAE | MSE |
|---|---:|---:|
| robust | 7.145 | 104.8 |
| robust+BP | 7.112 | 104.1 |
| SK+BP | 7.055 | 104.6 |
| SK+H2+BP | 7.078 | 104.2 |

## Paired fixture bootstrap (90% intervals)

| Comparison | NCR 2026 | Six Nations 2025 | Six Nations 2026 |
|---|---|---|---|
| robust+BP vs robust, MAE | 0 | +0.030 [−0.015, +0.074] | +0.014 [−0.043, +0.067] |
| robust+BP vs robust, MSE | 0 | −0.28 [−1.50, +0.87] | −0.72 [−2.61, +1.18] |
| SK vs robust, MAE | +0.011 [−0.004, +0.027] | **−0.030 [−0.058, −0.006]** | +0.004 [−0.021, +0.028] |
| SK+H2+BP vs robust, MAE | +0.001 [−0.033, +0.035] | **−0.084 [−0.114, −0.050]** | **−0.035 [−0.067, −0.004]** |
| SK+H2+BP vs robust, MSE | +0.78 [−0.53, +2.11] | **−1.83 [−2.54, −1.09]** | +0.22 [−0.88, +1.36] |
| SK+H2+BP vs H2, MAE | +0.001 [−0.008, +0.010] | 0.000 [−0.031, +0.035] | −0.002 [−0.034, +0.033] |
| SK+H2+BP vs H2, MSE | +0.20 [−0.18, +0.59] | **−0.72 [−1.44, −0.02]** | −0.69 [−1.97, +0.63] |

## Conclusions

- **BP on robust P3.** It lowers Six Nations squared error in both seasons and on development, without significance on test, and raises MAE slightly. This is the mean-versus-median trade-off: breakdown steals and player of the match raise back-row means on a skewed target. On its own it is not a clear champion improvement.
- **SK on robust P3.**
  - It significantly improves Six Nations 2025 (MAE −0.030, MSE −0.62).
  - It is flat on 2026 and NCR.
  - It worsens Friendly-25 count MAE (0.65414 → 0.65480), because friendly replacements were already over-forecast.
- **SK+H2+BP is the strongest stack.**
  - Against robust P3: MAE is significantly better in both Six Nations seasons (−0.084 and −0.035) and flat in NCR. It is the first candidate whose 2026 interval excludes zero.
  - Against H2: MAE is identical, but Six Nations squared error is lower (−0.72 and −0.69).
  - Squad points: smoothed totals change by +2, +6 and +1 against robust P3, which is within the noise floor.
  - Friendly-25: count MAE is 0.65364, better than robust P3 but worse than MK alone (0.65308), because SK slightly hurts friendlies.
- **Verdict.** The fixes are real component-level corrections but add little at the fantasy-point level. The best way to use them is inside the H2 stack. Freeze robust P3, H2 and SK+H2+BP before NCR GW4–7 and compare them prospectively.
