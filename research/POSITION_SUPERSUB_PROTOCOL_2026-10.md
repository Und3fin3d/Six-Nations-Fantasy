# Positional-bias and super-sub protocol (frozen before the official-slate test results)

Written 1 October 2026, before any status-aware forecast was scored on Six Nations 2025, Six Nations 2026, NCR 2026 or Friendly-25. The forecasts for those sets have been generated (`research/position_supersub.py`) but not scored.

## Disclosure

- The diagnosis tables (`official_component_bias.csv`, `api_component_bias.csv`, `bias_by_strength.csv`, `ceiling_variance.csv`, `supersub_ceiling.csv`) use the **reference** robust P3 forecasts on the evaluation seasons. They motivated the question but select no parameter.
- `blocks_metrics_by_era.txt` prints the archived 2025–26 blocks next to the pre-2025 ones. Those blocks contain the Six Nations 2025/2026 matches, scored on API-observable events from tournament-start forecasts. They were printed in the same run as the development blocks. The candidate has no tuned parameter, so they cannot have changed a choice; they are reported as test evidence and labelled as already seen.
- Six Nations 2023 round 1 official rows look shifted by one player (coordinator warning). Development results are reported with and without that round; no choice depends on it.

## Candidate S: status-aware empirical rates (`model/unified/raw_benchmark/status_rates.py`)

Robust P3 with the robust empirical component replaced by `StatusAwareEmpiricalEventModel`; tree component, blend weights and scorer unchanged.

- At each lock, a within-player multiplicative model `E[events] = rate_player × factor[position, started] × minutes` is fitted to all recency-weighted (half-life 420 days, as in the empirical model) pre-lock rows, club and international pooled.
- Bench factors are relative to the same position's starter factor, shrunk towards the all-position bench ratio with 20 pseudo-events. The value 20 was set a priori, not tuned.
- Training events are converted to starter-equivalent counts before the unchanged robust fit. Replacement forecasts multiply the starter rate by the bench factor.
- Goal kicking, cards and the sparse official-only events are never adjusted.
- There is no competition identifier, fantasy label or evaluation-season input.

Secondary row: S + MK (matchup calibration + kicking γ = 1.5, the best-supported round-2 raw change), against MK alone.

**Super-sub decision rule: unchanged** (the optimiser's arg-max of expected points). Fantasy scoring is linear in realised points, so an upside, quantile or variance rule cannot raise expected points. No such rule is a candidate.

## Development evidence (all selection-relevant numbers)

Pre-2025 archived blocks (14 tournaments, API-observable truth, `blocks_metrics_by_era.txt`, section `pre`):

- Bench forecast/actual ratio for robust P3 → S:
  - tries 0.80 → 0.92;
  - metres 0.81 → 0.93;
  - defenders beaten 0.87 → 0.97;
  - tackles 0.93 → 0.95;
  - try assists 0.86 → 0.97.
- Starter ratios move towards 1.0, for example tries 1.10 → 1.08.
- Poisson deviance (all rows) is lower in 13 of 14 events, and MSE is lower in 12 of 14.
- Count-stat MAE is 0.63949 → 0.64018, slightly worse, as expected for raising the means of skewed counts.
- Observable fantasy points, bench: bias −0.53 → −0.15, MSE 33.83 → 33.52, MAE 3.977 → 4.043, within-weekend Pearson 0.219 → 0.225.
- Observable fantasy points, starters: bias +0.26 → +0.09, MSE 92.44 → 92.32, MAE 7.152 → 7.116.

Six Nations 2023–24 current-rubric slates (`dev_seasons.csv`, `dev_bias_*.txt`):

| Metric | Robust P3 | S |
|---|---|---|
| MAE, 2023 / 2024 | 7.182 / 5.844 | 7.163 / 5.847 |
| MSE, 2023 / 2024 | 104.10 / 65.54 | 103.92 / 65.54 |
| Bench bias, excluding 2023 R1 | −1.51 | −1.16 |
| Starter bias, excluding 2023 R1 | −2.51 | −2.65 |
| Smoothed super-sub points, 2023 / 2024 | 164.6 / 103.4 | 159.2 / 102.7 |

Starters move down because the labels include unforecast official-only points. S is not a fix for those; that is the official-stats work. The Pearson correlation is unchanged.

Super-sub on 69 pre-2025 block weekends (observable points): S − reference = −3.9 per 10 weekends, 90% interval [−30, +18]. The pick is the same in 87% of weekends.

## Test evaluation (fixed now)

Score robust P3, S, MK and S+MK on the 13 official slates, plus Friendly-25 for S (raw stats). Report:

- fantasy MAE, MSE, Pearson, Spearman and bench Pearson;
- paired 90% fixture-bootstrap intervals for the MAE and MSE changes;
- realised and smoothed squad points, and realised and smoothed super-sub points;
- bias by status × position on official labels, and on API-observable points;
- Friendly-25 count-stat MAE, plus event ratio, MSE and deviance by status;
- the super-sub pick table for Six Nations 2025–26.

**Verdict rules, fixed in advance:**

1. A positional bias counts as **fixed by S** if, on the official slates, the API-observable bias of the bench group shrinks and the event-level evidence (MSE or deviance) does not get worse.
2. Fantasy MAE moving by less than ±0.02 counts as neutral.
3. Super-sub and squad differences count as **noise** unless they exceed the 1% jitter null (SD 22–37 per season) and agree in sign across seasons.
4. The super-sub ceiling is reported from the variance decomposition (`supersub_ceiling.csv`): realistic expected gain against the hindsight gap.
