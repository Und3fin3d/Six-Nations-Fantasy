# Official-statistics protocol (frozen before any fantasy-level test scoring)

Committed before `research/official_stats_eval.py --set official` was run. It fixes the adapter, the variants and the success criteria for the Six Nations 2025/2026 evaluation slates.

## What was seen before freezing

The diagnosis phase looked at 2025–26 official data at the component level. This is disclosed so the test can be read correctly:

- Season-level correlations of official breakdown steals (BS) with the API `tackle_turnover` (0.06–0.29) and with RugbyPass `turnovers_won` (0.76–0.87).
- The official metres definitions by season: 2023 and 2025 ≈ API metres + 2.3–2.9 m per carry, 2026 ≈ 1.05 × API metres.
- POTM facts: 41/42 winners came from the winning team and 41/42 were starters.
- Component-level forecast comparisons on 2025–26 for BS predictors, POTM probability forms and the metres mapping (`agentA` scratch studies).

No fantasy-level (total points) adapter result on 2025, 2026 or NCR had been computed when this file was committed.

## Frozen adapter (`model/unified/official_stats.py`, `OfficialStatConfig` defaults)

**Breakdown steals.** Expected BS = λ · E[minutes] / 80, where:

- λ = (own official BS + 12 · prior) / (own official 80s + 12);
- prior = official position rate × RugbyPass relative turnovers-won rate.

Each part is computed as follows:

- **RugbyPass relative rate:** (tw + 10 · μ_pos) / (m/80 + 10) / μ_pos, over seasons admitted by `past_seasons`.
- **Position rate:** shrunk to the overall rate with 400 minutes.

Shrinkages 10 and 12 were chosen on 2023 rounds 2–4. For those rows the RugbyPass features came from seasons completed before the 2025 lock, so no 2025–26 outcome was used. Alternatives were k_rp ∈ {5, 10, 20} and k_off ∈ {3, 6, 12, 24}.

**Metres.** Official mean = a · E[API metres] + b · E[carries], with the coefficient of variation kept. (a, b) are fitted by least squares on the current season's completed official rounds. Before round 1 results exist, they are fitted on the latest official season instead.

**POTM.** p = P(team wins) × softmax(0.1 · xp) within the team, with bench weight 0.05, renormalised to 1 per match.

- P(team wins) comes from the pre-lock margin Elo.
- xp = expected Six Nations points excluding POTM, after the other adjustments.
- β = 0.1 was the best 2023 match log-loss among {0, 0.05, 0.1, 0.15, 0.2, 0.3}.

## Variants scored

- **Bases:** robust P3 (`robust`) and the round-2 MK raw candidate (`mk`).
- **Components:** `none`, `B`, `M`, `P`, `BP`, `BMP`.
- **Primary candidate:** `robust:BMP`. `BP` is pre-registered as the definition-stable alternative, because the metres rule is exposed to the 2026 definition change at 2026 round 1.

## Metrics and criteria

**Component level** (per stat, against official per-player statistics):

- mean prediction against mean actual;
- MSE and Pearson correlation;
- by position × starter/bench.

**Fantasy level** (2025 and 2026 separately):

- MAE, MSE, bias, within-round Pearson and Spearman;
- realised and smoothed squad points (40 × 1% jitter);
- captain and super-sub changes.

**Uncertainty.** Paired 90% fixture-bootstrap intervals (fixtures resampled within rounds) for the MAE and MSE changes against the same base without the adapter.

**Verdict per stat:**

- **Fixable** if its component forecast improves both MSE and correlation in both test seasons.
- **Noise-limited** otherwise, or if the improvement is a small share of its error. In that case the remaining ceiling is reported.

A fantasy-level claim requires:

- lower MSE and higher within-round correlation in both seasons;
- MAE not worse by more than the bootstrap noise.

Squad points are reported, never used to certify.
