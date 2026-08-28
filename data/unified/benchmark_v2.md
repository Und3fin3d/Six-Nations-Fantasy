# Unified supermodel v2 — ranking-first stacked benchmark

One stage-1 rugby-event model feeds one pooled stage-2 ranker+regressor conditioned on each competition's scoring rubric (no competition identity feature). Every number below is out-of-sample for the learned stack; metrics are computed within each round and averaged. See `model/unified/benchmark_v2.py` for the protocol.

## Six Nations (2026 sealed holdout)

| Model | MAE | Spearman | Top10 cap | Top25 cap | Top50 cap | Top100 cap |
|---|---:|---:|---:|---:|---:|---:|
| 6N champion | 7.30 | 0.666 | 0.715 | 0.771 | 0.835 | 0.952 |
| Unified v1 stage-1 | 7.24 | 0.690 | 0.729 | 0.789 | 0.860 | 0.951 |
| Unified v2 stack | 7.66 | 0.670 | 0.718 | 0.760 | 0.848 | 0.951 |

## Nations Championship (leave-one-gameweek-out)

| Model | MAE | Spearman | Top10 cap | Top25 cap | Top50 cap | Top100 cap |
|---|---:|---:|---:|---:|---:|---:|
| NCR incumbent | 8.90 | 0.581 | 0.627 | 0.715 | 0.742 | 0.804 |
| Unified v1 stage-1 | 9.11 | 0.516 | 0.539 | 0.636 | 0.653 | 0.760 |
| Unified v2 stack | 9.27 | 0.520 | 0.613 | 0.632 | 0.693 | 0.783 |

Cohort parity is by construction (incumbent projection cohort). Stage-1 feature match rate per gameweek: gw1 97% of 270, gw2 92% of 262.

## Promotion gate

- Six Nations: FAIL — MAE 7.66 regressed >2% vs 7.30; mean capture gain +0.001 below +3pp
- NCR: FAIL — MAE 9.27 regressed >2% vs 8.90; top_25_capture 0.632 regressed >2pp vs 0.715; top_50_capture 0.693 regressed >2pp vs 0.742; top_100_capture 0.783 regressed >2pp vs 0.804; mean capture gain -0.042 below +3pp

**Decision: DO NOT PROMOTE.** Production routing is unchanged regardless; promotion also requires the prospective November shadow rounds (GW4–7).

## Caveats (read before quoting these numbers)

- The 6N headline is a small-sample edge, not a proven win. Across the 5 sealed 2026 rounds, stage-1 beats the 6N champion on MAE in 3/5 rounds and on Spearman in 3/5; no significance test is applied. Read it as "competitive with, plausibly better than" the champion, contingent on `data/model_predictions_2026.csv` being a genuine pre-round forecast.
- The stack-vs-stage-1 comparison is confounded. Stage-2 trains on stage-1 features that were fit *including* those same training rows (non-OOF), and on only ~674 6N labelled rows (2024 has no fantasy points; the 2023 rubric era is excluded). "Learned stack does not beat deterministic stage-1" is therefore a working hypothesis, not a clean test of the v2 thesis — out-of-fold stage-1 features and hyperparameter tuning are the outstanding work.
- Stage-1 artifacts train only through the first 85% of pre-cutoff dates (the CLI reserves, then discards, a validation tail), so they saw *less* recent data than the `asof` label implies — a conservative bias for the stage-1 result.

