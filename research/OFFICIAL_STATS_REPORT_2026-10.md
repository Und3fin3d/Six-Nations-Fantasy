# Six Nations official-only statistics: fixable or noise? (October 2026)

Protocol: `research/OFFICIAL_STATS_PROTOCOL_2026-10.md`, committed (4f48c3a) before any test-season fantasy scoring. The adapter is `model/unified/official_stats.py` (opt-in research module; production routing unchanged). The runners are `research/official_stats_eval.py` and `research/official_stats_summary.py`, and the evidence is in `research/official_stats_2026-10/`.

## Verdict

| Stat (pts) | Problem found | Pre-lock forecast, before → after (2025 / 2026) | Ceiling | Verdict |
|---|---|---|---|---|
| **Breakdown steals** (5) | The model forecasts API `tackle_turnover`. It correlates 0.03–0.11 with official BS per player-match and 0.06–0.29 per player-season, so the proxy is wrong. RugbyPass/Opta `turnovers_won` correlates 0.76–0.87 per player-season. | Correlation 0.19 / 0.20 → **0.40 / 0.36**. MSE 3.81 / 3.72 → **3.33 / 3.40** (−13% / −9%). | Poisson floor: max R² 0.20–0.26 even with minutes known (max r ≈ 0.45–0.51). Split-half player reliability 0.56. The new forecast sits at its own simulated noise floor (r 0.32–0.36 if it were the truth). | **Fixable, fixed.** The component improves strongly in both seasons. Fantasy-level effect is small (below). |
| **Player of the match** (15) | 41/42 winners came from the winning team and 41/42 were starters. The model's POTM term ignores both: corr 0.04, and match log-loss is worse than uniform (3.97 vs 3.83). Bench players were credited 0.1–0.2 pts. | P(team wins) × within-team softmax(0.1·xp): corr 0.04 / 0.05 → **0.17 / 0.17**. MSE 5.05 / 5.01 → **4.73 / 4.78**. Match log-loss 3.97 → 3.08, AUC 0.66 → 0.83. | If these probabilities were the truth, expected corr = 0.21 ± 0.04. Explainable POTM variance is 0.21 of 4.79 pts² (4%). Knowing the winner in advance only reaches log-loss 3.07. Realised points (ex post) reach 2.16. | **Discrimination fixed, now noise-limited.** About 96% of POTM variance is unforecastable before lock. |
| **Metres** (1 per 10 official m) | Definitional, not ability. In 2023 and 2025, official metres ≈ API metres + 2.3–2.9 m per carry (R² 0.94 vs 0.83 for scale-only), so props were 3.8×, back-three 1.4×. **In 2026, official ≈ 1.05 × API** (per-carry term ≈ 0, R² 0.94). The definition changed. The pooled "1.45×" mixes two regimes. | Carry-aware mapping refitted on the latest official rounds. 2025: MSE 8.31 → **6.21**, corr 0.646 → 0.655. 2026: MSE 2.95 → 3.46, corr 0.683 → 0.636. This is entirely 2026 round 1, where the 2025 mapping doubled forecasts. Rounds 2–5 are unchanged, because official ≈ API. | API-metres forecast corr 0.65–0.68 is at the model's own lognormal noise level (0.60–0.62 simulated). Official metres/80 split-half reliability is 0.70. | **Level/regime problem; discrimination is noise-limited.** The mapping fixes the 2025 level but cannot anticipate a definition change. Not recommended for fantasy MAE. |
| Scrums won (1) | Team scrum counts are not forecastable: team-match SW correlates −0.09 with the forecast, SD 14.7 pts per pack. Within a team, share follows minutes (r 0.73). There is a level under-forecast of about 13%. | – | – | Noise-limited (team scrum count). |
| Lineout steals / 50-22 / kicks retained | Already trained on official labels. Corr 0.15 / 0.06 / 0.49, with tiny means except kicks retained. | – | – | Noise-limited / immaterial. |

**Bottom line.** Breakdown steals were genuinely mis-specified, and POTM ignored who wins. Both are now forecast about as well as pre-lock information allows. Metres are a definitional level issue whose 2026 regime matches the API.

Together these components are small against total fantasy error. Breakdown steals plus POTM (**BP**) improves fantasy MSE in both seasons, but the change is not significant. Correlation moves by about 0.001, MAE is unchanged within noise, and squad points move within the noise floor. So the official-only stats are mostly a noise ceiling at the fantasy level, with breakdown steals the only material forecasting defect, now fixed.

## Fantasy-level results (Six Nations, robust P3 base, 10 rounds)

The rows below are from the final run, with the RugbyPass key-ambiguity fix described under Caveats. The frozen run differs by ≤0.001 MAE (`frozen_run/`).

| Engine | 2025 MAE / MSE / Pearson / Spearman | 2025 points (smoothed) | 2026 MAE / MSE / Pearson / Spearman | 2026 points (smoothed) |
|---|---|---|---|---|
| robust (reference) | 7.240 / 111.01 / 0.5944 / 0.6500 | 2356 (2342) | 7.239 / 101.61 / 0.6637 / 0.6963 | 2513 (2510) |
| + B | 7.256 / 110.90 / 0.5942 / 0.6510 | 2294 (2289) | **7.232** / 100.79 / **0.6659** / **0.6973** | 2582 (2567) |
| + P | 7.243 / 110.68 / 0.5963 / 0.6522 | 2330 (2346) | 7.249 / 101.51 / 0.6625 / 0.6962 | 2503 (2496) |
| + BP (definition-stable) | 7.262 / 110.68 / **0.5959** / **0.6516** | 2324 (2327) | 7.248 / 100.82 / **0.6643** / 0.6963 | 2599 (2585) |
| + BMP (primary) | 7.438 / 110.35 / **0.5976** / **0.6525** | 2324 (2346) | 7.287 / 100.51 / 0.6634 / 0.6958 | 2599 (2589) |
| MK (round-2 raw) | 7.237 / 111.75 / 0.5932 / 0.6475 | 2310 (2306) | 7.205 / 101.80 / 0.6659 / 0.7012 | 2483 (2464) |
| MK + BP | 7.259 / 111.45 / 0.5940 / 0.6482 | 2334 (2326) | 7.216 / 101.07 / 0.6656 / 0.7005 | 2498 (2516) |
| MK + BMP | 7.401 / 110.48 / 0.5959 / 0.6496 | 2326 (2326) | 7.248 / 100.58 / 0.6648 / 0.6996 | 2535 (2543) |

Paired 90% fixture bootstrap against the same base without the adapter (`keyfix_run/bootstrap.csv`):

| Change against robust | 2025 MAE | 2025 MSE | 2026 MAE | 2026 MSE |
|---|---|---|---|---|
| B | +0.016 [−0.010, +0.043] | −0.11 [−0.81, +0.59] | −0.007 [−0.042, +0.030] | −0.81 [−1.95, +0.36] |
| P | +0.003 [−0.021, +0.029] | −0.32 [−1.04, +0.38] | +0.011 [−0.021, +0.040] | −0.10 [−1.15, +0.89] |
| BP | +0.022 [−0.022, +0.067] | −0.33 [−1.43, +0.79] | +0.010 [−0.046, +0.061] | −0.79 [−2.67, +1.05] |
| BMP | **+0.197 [+0.072, +0.320]** | −0.65 [−3.84, +2.38] | +0.049 [−0.016, +0.110] | −1.10 [−3.23, +0.91] |

**The protocol's fantasy-level criterion is not met.** It required lower MSE and higher correlation in both seasons, with MAE not worse beyond noise:

- **BMP** lowers MSE in both seasons. Pearson rises in 2025 and falls slightly in 2026. MAE is significantly worse in 2025. This is the metres level effect: raising a skewed, under-forecast component moves forecasts away from the median.
- **BP** lowers MSE and raises Pearson in both seasons, and MAE is within noise. The correlation gain (+0.0015 / +0.0006) is far too small to claim.

**The development check agrees.** On 2023 rounds 1–4 (`dev2023_seasons.csv`), BMP gives MAE −0.034, MSE −2.6 and Spearman +0.004. P alone gives MAE +0.016 and Pearson −0.001.

**Why the component gains barely move the total.** Breakdown-steal points have an SD of about 2 pts against a total SD of 13. Removing a third of breakdown-steal error variance cuts total MSE by about 0.5%. The total residual is dominated by tries, tackles, defenders beaten and metres.

**NCR and Friendly-25 are unaffected by construction.** The adapter is applied only to Six Nations scoring. The raw events, the NCR adapter and the friendly count stats are untouched.

## Effects by position (for combining with the positional-bias work)

Expected component points, robust P3 → +BMP, against actual. 2025–26, 1,350 rows (`keyfix_run/components_by_position.csv`).

| Role | Breakdown steals: old → new (actual) | POTM: old → new (actual) | Metres: old → new (actual) | Total bias: robust → +B → +BP → +BMP |
|---|---|---|---|---|
| Back-row, starter | 0.92 → **2.00** (1.80) | 0.35 → 0.72 (0.59) | 1.78 → 3.00 (2.79) | −1.86 → −0.77 → −0.42 → +0.81 |
| **Back-row, bench** | 0.32 → **0.62** (0.86) | 0.10 → 0.01 (0.00) | 0.48 → 0.89 (1.06) | −2.84 → −2.54 → −2.63 → −2.23 |
| Back-three, starter | 0.86 → **0.36** (0.11) | 0.58 → 0.61 (0.60) | 4.32 → 5.48 (5.59) | −1.66 → −2.16 → −2.14 → −0.97 |
| Centre, starter | 0.91 → 0.75 (0.76) | 0.41 → 0.41 (0.38) | 2.49 → 3.58 (4.01) | −2.84 → −3.01 → −3.01 → −1.91 |
| Fly-half, starter | 0.68 → 0.39 (0.17) | 0.82 → 0.75 (1.00) | 2.08 → 2.98 (3.13) | −2.81 → −3.09 → −3.15 → −2.26 |
| Scrum-half, starter | 0.68 → 0.38 (0.33) | 0.79 → 0.53 (1.00) | 1.86 → 2.61 (1.60) | +1.70 → +1.40 → +1.15 → +1.89 |
| Hooker, starter | 0.53 → 0.84 (0.76) | 0.09 → 0.40 (0.25) | 1.00 → 1.85 (1.22) | +1.66 → +1.97 → +2.29 → +3.13 |
| Lock, starter | 0.78 → 1.00 (0.96) | 0.15 → 0.33 (0.38) | 0.73 → 1.59 (1.13) | −0.61 → −0.38 → −0.19 → +0.66 |
| Prop, starter | 0.35 → 0.33 (0.38) | 0.07 → 0.18 (0.12) | 0.37 → 1.04 (0.99) | −1.20 → −1.22 → −1.11 → −0.45 |
| Any bench back / half | 0.20–0.31 → 0.09–0.26 (0.00–0.22) | 0.11–0.20 → 0.00 (0.00) | – | small |

Breakdown steals explain about 0.5 of the 2.8-point bench back-row under-forecast; the rest is elsewhere. They move back-three, fly-half and scrum-half starters down by 0.3–0.5. The POTM change removes POTM credit from every bench player (about 0.1–0.2 each), which matters for super-sub ranking.

Within positions, breakdown-steal correlation rises for back-row starters (0.09 → 0.28), back-row bench (0.07 → 0.25) and lock starters (0.29 → 0.60).

## Decisions (captain and super-sub)

Robust → +BP/BMP changes 3–4 of 10 captains and 1 super-sub (`keyfix_run/decisions.csv`):

| Round | Robust captain or super-sub | +BP or +BMP choice | Effect |
|---|---|---|---|
| 2025 R1 | Dupont captain (31) | Ramos (37) | +6 |
| 2025 R4 | van der Merwe captain (28) | Willis (40) | +12 |
| 2025 R5, BP only | Ramos captain (36) | Penaud (10) | −26 |
| 2026 R1 | Dupont captain (30) | Earl (50) | +20 |
| 2026 R3 | Griffin super-sub (10) | Bayliss (19) | +27 |

The breakdown-steal fix promotes back-rowers (Willis, Earl, Bayliss) in tight decisions.

Smoothed squad points against robust:

| Variant | 2025 | 2026 |
|---|---:|---:|
| BP | −16 | +75 |
| BMP | +3 | +79 |
| B | −54 | +57 |

All are within the documented noise floor (perturbed-copy SD 43 and 148), so no decision gain is claimed.

## Method notes

**Breakdown steals.** The forecast is E[BS] = λ · E[minutes] / 80, with λ = (own official BS + 12·prior) / (own official 80s + 12). The prior is the official position rate × the player's relative RugbyPass turnovers-won rate (shrunk with 10 pseudo-80s), over seasons admitted by `past_seasons`.

- The shrinkages were chosen on 2023 rounds 2–4, with RugbyPass features from seasons completed before the 2025 lock (`bs_dev`).
- Official breakdown-steal definitions look stable. Back-row rates per 80 were 0.42 / 0.54 / 0.35, and RugbyPass agreement holds in both 2025 and 2026.
- The one exception is back-three. Its 2023 rate (0.13; 0.11 without round 1) is out of line with 2025–26 (0.01–0.04).
- Coverage: RugbyPass matched 98–99% of 2025–26 pool rows.

**Independent confirmation.** The parallel data-sources agent (`research/STAT_DATA_SOURCES_2026-10.md`, `research/stat_sources/bs_value.py`) independently found the same:

- the API `tackle_turnover` is the wrong stat (r 0.11 on England player-matches);
- Opta turnovers-won agrees (r 0.75);
- prior-season RugbyPass turnovers-won per 80 predicts official BS per 80 at r = 0.52, against 0.085 for the API rate.

That is the signal this adapter uses.

**Disagreement with that agent on metres.** Its "official ≈ API / 0.68, keep a learned scale" is a pooled figure. Per season the scale is 1.66 (2023), 1.79 (2025) and 1.05 (2026), and the 2023/2025 gap is a per-carry term, so a single learned scale is wrong for 2026.

**POTM.** Win probability is 1/(1+10^(−edge/400)) from the pre-lock margin Elo. β = 0.1 and bench weight 0.05 were chosen on the 12 matches of 2023. The favourite won 74% of matches.

**Metres mapping.** (a, b) are fitted by least squares on the current season's completed official rounds, or on the latest official season before round-1 results exist. Per round they were:

- 2025: a = 0.91–0.93, b = 2.3–2.9;
- 2026 rounds 2–5: a = 1.06–1.09, b ≈ −0.1.

## Sensitivity: 2023 round 1 excluded (suspected row shift)

The fantasy-sources agent reports that official 2023 round-1 rows are shifted by one player for every team except Wales. The data agree:

- 2023 official metres correlate 0.83 with API metres over rounds 1–4, against 0.94 over rounds 2–4.
- On rounds 2–4 the per-carry coefficient is 2.77, against 2.32 with round 1, which brings it closer to 2025's 2.9.

The development selections and the test were re-run without those rows (`sensitivity_no_2023r1/`). No conclusion changes:

- **POTM β.** On the 9 round 2–4 matches, the log-loss is flat between 0.1 and 0.2 (2.816 / 2.786 / 2.821). The best value moves to 0.15. β = 0.1 is kept, since it is within 0.03 of the best.
- **Breakdown-steal shrinkage.** On rounds 3–4, with history from round 2 onward, k_rp = 10 and k_rp = 20 tie (deviance 0.5592 / 0.5577), and k_off between 6 and 12 is still best. All variants beat the model (0.6025) and the position-only forecast (0.5838).
- **Development fantasy level, rounds 2–4.** Against robust: B gives MAE −0.039 and MSE −0.79; BMP gives MAE −0.036 and MSE −3.25; P gives MAE −0.002 and Pearson −0.001.
- **Test without 2023 round-1 history** (2025 / 2026):

| Variant | MAE | MSE |
|---|---|---|
| B | 7.264 / 7.236 | 110.94 / 100.86 |
| BP | 7.270 / 7.252 | 110.73 / 100.89 |
| BMP | 7.450 / 7.290 | 110.26 / 100.58 |

  These differ from the main run by at most 0.013 MAE.

## Caveats and selection history

- The diagnosis looked at 2025–26 at the component level before the protocol: correlations, definitions, predictor comparisons. This is disclosed in the protocol. Fantasy-level numbers were first computed after it.
- **Post-protocol fix.** The frozen code collapsed RugbyPass rows that shared a name key (`h|thomas`, `t|williams`, …) before checking for ambiguity, so two players' rows could mix. The final code drops keys whose slugs carry different statistics. The effect is ≤0.001 MAE (2025 only), and the frozen-run results are kept in `frozen_run/`.
- 2024 has no official per-stat file. Development evidence for official components is therefore only 2023 rounds 1–4. POTM has 12 development matches.
- The 2026 metres definition change could not have been known at the 2026 round-1 lock. Any metres mapping takes that risk.
- The breakdown-steal level is slightly high (0.69 vs 0.63 pts in 2025; 0.64 vs 0.57 in 2026) because the position prior pools the higher 2023 rates.

## Recommendation

- Keep the adapter opt-in.
- If anything is adopted, take **BP**, the breakdown-steal and POTM components:
  - they correct real mis-specification;
  - they improve MSE in both seasons;
  - they are neutral on MAE;
  - they fix structural absurdities: back-three credited 0.86 pts of steals against 0.11 actual, and bench players credited POTM.
- Do not adopt the metres mapping for MAE-scored work. The 2026 official metres already match the API definition.
- No further gain is available from these stats without new pre-lock information. The remaining error is Poisson or Bernoulli noise.

## Reproduction

```bash
export OMP_NUM_THREADS=1
python -m research.official_stats_eval --runs RUNS --set dev --cache CACHE --no-smooth --output DEV
python -m research.official_stats_eval --runs RUNS --set official --cache CACHE --output TEST
python -m research.official_stats_summary --input TEST --output TEST/summary
# Diagnostics (need RUNS/base inputs + saved forecasts; set OFFICIAL_STATS_SCRATCH to the scratch root)
python research/official_stats_2026-10/diagnostics/build_join.py   # then bs_study, bs_an, potm_study, metres_study, ceiling ...
```

`RUNS` is the rolling-evaluation scratch directory (`base/`, `devcr/`). The diagnostics output is `research/official_stats_2026-10/diagnostics_output.txt`.

**Tests.** `tests/test_official_stats.py` has 8 tests. Focused run: 19 passed with the kicking and matchup tests.

No RapidAPI requests were made, `data/cache` is unchanged, and no model was retrained or promoted.
