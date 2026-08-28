# Unified head-to-head: stage-1 vs v2 stack (confounders removed)

Stage-2 trains on OUT-OF-FOLD stage-1 features (pre-2025 artifact for the
2025 training season); all stage-1 artifacts are full-window refits (no
15% tail discard). Stack config chosen on 2025 leave-one-round-out dev
only: **blend_s1=1.0, score=points_only** (gate: dev MAE within 2% of
stage-1). Incumbent rows are unchanged from benchmark_v2_full.md.

## Dev sweep (6N 2025 LORO, top configs)

```
 blend_s1      score_mode  dev_mae  dev_spearman  dev_mean_capture
      NaN stage1_baseline 7.108199      0.662584          0.813386
     1.00     points_only 7.108199      0.662584          0.813386
     1.00     rank+points 7.108199      0.660640          0.813284
     0.75     rank+points 7.151150      0.659438          0.811662
     0.75     points_only 7.151150      0.661689          0.808346
     0.50     points_only 7.287279      0.657997          0.807374
     0.35     points_only 7.403569      0.654240          0.806236
     0.50     rank+points 7.287279      0.657084          0.801750
```

## Six Nations 2026 (sealed holdout, per-round averaged)

| Model | MAE | Spearman | Top10 cap | Top25 cap | Top50 cap | Top100 cap |
|---|---:|---:|---:|---:|---:|---:|
| 6N champion (incumbent) | 7.30 | 0.666 | 71.5% | 77.1% | 83.5% | 95.2% |
| stage-1 full | 7.21 | 0.692 | 72.3% | 78.8% | 86.5% | 95.2% |
| v2 stack (OOF+tuned) | 7.21 | 0.692 | 72.3% | 78.8% | 86.5% | 95.2% |

## NCR (leave-one-gameweek-out, 532-row parity cohort)

| Model | MAE | Spearman | Top10 cap | Top25 cap | Top50 cap | Top100 cap |
|---|---:|---:|---:|---:|---:|---:|
| NCR incumbent | 8.90 | 0.581 | 62.7% | 71.5% | 74.2% | 80.4% |
| stage-1 full | 9.10 | 0.527 | 52.5% | 62.0% | 65.4% | 75.7% |
| v2 stack (OOF+tuned) | 9.10 | 0.527 | 52.5% | 62.0% | 65.4% | 75.7% |

Same-cohort per-round metrics; small samples (5 + 2 rounds), no
significance tests. Source CSVs: stack_experiments_dev.csv,
stack_experiments_results.csv.
