# Unified models — full old-vs-new comparison (consistent methodology)

Every row below is computed on the **same cohorts** with the **same tie-aware,
per-round metric code** (`model/unified/benchmark_v2.py`), so models are directly
comparable — unlike the original v1 `benchmark.md`, which used a 514-row NCR
cohort for the unified models and an **in-sample** Six Nations evaluation.
Ranking metrics are means across rounds; "capture" = actual points of the
predicted top N ÷ hindsight-optimal top N.

## NCR (GW1–GW2, 532-row parity cohort, leave-one-gameweek-out for learned models)

| Model | MAE | Spearman | Top 10 capture | Top 25 capture | Top 50 capture | Top 100 capture |
|---|---:|---:|---:|---:|---:|---:|
| **NCR incumbent** | **8.90** | **0.581** | 62.7% | **71.5%** | **74.2%** | **80.4%** |
| Champion transplant | 9.10 | 0.520 | 46.0% | 65.5% | 68.1% | 75.3% |
| Unified v1 GBDT (stage-1, honest) | 9.11 | 0.516 | 53.9% | 63.6% | 65.3% | 76.0% |
| Unified v1 neural (honest) | 9.49 | 0.489 | 35.9% | 53.4% | 64.0% | 76.4% |
| Unified v2 stack | 9.27 | 0.520 | 61.2% | 63.1% | 69.3% | 78.3% |

### NCR MAE by completed round

All models now share the identical 532 player-rounds (missing crosswalk players
receive position-prior predictions instead of being dropped).

| Model | GW1 MAE | GW2 MAE | Two-round weighted MAE | Player-rounds |
|---|---:|---:|---:|---:|
| NCR incumbent | 9.39 | 8.42 | **8.91** | 532 |
| Champion transplant | 9.21 | 9.00 | **9.10** | 532 |
| Unified v1 GBDT (stage-1, honest) | 9.84 | 8.38 | **9.12** | 532 |
| Unified v1 neural (honest) | 10.37 | 8.61 | **9.50** | 532 |
| Unified v2 stack | 10.06 | 8.48 | **9.28** | 532 |

The old report's "Unified GBDT 9.01 / neural 9.38 on 514 rows" flattered the
unified models: restoring the 18 hard-to-map players (who now get position-prior
forecasts) moves them to 9.12 / 9.50 on the full cohort.

## Six Nations (sealed 2026 season, per-round averaged)

Stage-1 and the v2 stack were cut at 2026-02-01 and never see 2026. The
original v1 numbers (GBDT 7.23, neural 7.73) came from a model **trained through
July 2026 and evaluated on Feb–Mar 2026** — in-sample, invalid, and shown only
struck through for the record.

| Model | MAE | Spearman | Top 10 capture | Top 25 capture | Top 50 capture | Top 100 capture |
|---|---:|---:|---:|---:|---:|---:|
| 6N champion | 7.30 | 0.666 | 71.5% | 77.1% | 83.5% | **95.2%** |
| **Unified v1 GBDT (stage-1, honest)** | **7.24** | **0.690** | **72.9%** | **78.9%** | **86.0%** | 95.1% |
| Unified v2 stack | 7.66 | 0.670 | 71.8% | 76.0% | 84.8% | 95.1% |
| ~~Unified GBDT (v1 report, in-sample)~~ | ~~7.23~~ | ~~0.696~~ | — | — | — | — |
| ~~Unified neural (v1 report, in-sample)~~ | ~~7.73~~ | ~~0.628~~ | — | — | — | — |
| Unified v1 neural (honest) | n/a\* | n/a\* | — | — | — | — |

\* An honest 6N neural number needs a neural retrain at the 2026-02-01 cutoff
(the only saved artifact is cut 2026-07-04, which saw the 2026 season). The
neural line was dropped from the v2 critical path after losing to the GBDT
everywhere, so the retrain was not spent.

Six Nations capture percentages are not comparable to the original v1 report's
(60.2% etc.): v1 pooled all 690 season rows into one ranking, this report ranks
within each round (matching how teams are actually picked) and averages.

Caveats: the stage-1-vs-champion 6N edge is a 3-of-5-round result with no
significance test; the v2-stack-vs-stage-1 comparison is confounded by non-OOF
stage-1 training features and ~674 labelled 6N rows. See `benchmark_v2.md`.

Sources: `benchmark_v2_six_nations.csv`, `benchmark_v2_ncr.csv`,
`benchmark_v2_ncr_extra.csv` (champion transplant + honest neural rows).
