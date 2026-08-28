# NCR GW1-3 unified-model evaluation

> **GW3 lineup warning:** the original GW3 fantasy projection feed was carryover/stale (126 P/B mismatches). Its GW3 player-ranking section is superseded pending a full all-engine rerun. Use `TEAM_REPORT.md` for the corrected final-lineup P3 versus incumbent team comparison.

Official fantasy points are the outcome. P3, v1 and v5 use one frozen training cutoff at the first GW1 kickoff (`2026-07-04T07:10:00+00:00`); no NCR GW1-3 result enters their training history. The NCR incumbent uses each saved pre-lock weekly projection.

Unified predictions are raw-event expectations converted with the NCR adapter. The historical source does not provide player interceptions or trustworthy lineouts-won, so those terms are absent rather than imputed.

## Three-round average

Each top-N cell is **actual-rank overlap / actual-point capture**.

| Model | MAE | Spearman | Top 10 | Top 25 | Top 50 | Top 100 |
|---|---:|---:|---:|---:|---:|---:|
| P3 | 9.39 | 0.433 | 13.3% / 43.9% | 24.0% / 53.6% | 38.1% / 61.5% | 53.9% / 70.0% |
| v1 GBDT | 9.75 | 0.360 | 0.0% / 31.2% | 15.6% / 42.5% | 29.2% / 52.6% | 49.7% / 65.0% |
| v5 terminal | 9.75 | 0.360 | 0.0% / 31.2% | 15.6% / 42.5% | 29.2% / 52.6% | 49.7% / 65.0% |
| NCR incumbent | 8.89 | 0.503 | 26.7% / 60.5% | 36.0% / 67.5% | 48.3% / 72.1% | 60.6% / 76.4% |

## Round by round

### GW1 (270 players)

| Model | MAE | Spearman | Top 10 overlap/capture | Top 25 overlap/capture | Top 50 overlap/capture | Top 100 overlap/capture |
|---|---:|---:|---:|---:|---:|---:|
| P3 | 8.72 | 0.556 | 10.0% / 39.7% | 32.0% / 57.1% | 42.4% / 67.5% | 56.7% / 74.3% |
| v1 GBDT | 9.19 | 0.477 | 0.0% / 34.5% | 22.7% / 47.3% | 31.2% / 54.4% | 51.7% / 69.8% |
| v5 terminal | 9.19 | 0.477 | 0.0% / 34.5% | 22.7% / 47.3% | 31.2% / 54.4% | 51.7% / 69.8% |
| NCR incumbent | 8.25 | 0.630 | 30.0% / 60.4% | 40.0% / 67.2% | 48.4% / 72.2% | 68.0% / 86.4% |

### GW2 (262 players)

| Model | MAE | Spearman | Top 10 overlap/capture | Top 25 overlap/capture | Top 50 overlap/capture | Top 100 overlap/capture |
|---|---:|---:|---:|---:|---:|---:|
| P3 | 8.55 | 0.477 | 10.0% / 47.1% | 20.0% / 64.4% | 40.0% / 67.8% | 56.6% / 72.9% |
| v1 GBDT | 9.16 | 0.381 | 0.0% / 28.7% | 8.0% / 44.2% | 26.0% / 54.2% | 52.6% / 68.5% |
| v5 terminal | 9.16 | 0.381 | 0.0% / 28.7% | 8.0% / 44.2% | 26.0% / 54.2% | 52.6% / 68.5% |
| NCR incumbent | 8.42 | 0.547 | 20.0% / 62.9% | 36.0% / 75.3% | 50.0% / 77.2% | 61.6% / 77.0% |

### GW3 (270 players)

| Model | MAE | Spearman | Top 10 overlap/capture | Top 25 overlap/capture | Top 50 overlap/capture | Top 100 overlap/capture |
|---|---:|---:|---:|---:|---:|---:|
| P3 | 10.89 | 0.266 | 20.0% / 44.9% | 20.0% / 39.4% | 32.0% / 49.1% | 48.6% / 63.0% |
| v1 GBDT | 10.91 | 0.222 | 0.0% / 30.4% | 16.0% / 35.9% | 30.4% / 49.2% | 44.8% / 56.7% |
| v5 terminal | 10.91 | 0.222 | 0.0% / 30.4% | 16.0% / 35.9% | 30.4% / 49.2% | 44.8% / 56.7% |
| NCR incumbent | 10.00 | 0.333 | 30.0% / 58.3% | 32.0% / 60.0% | 46.4% / 66.8% | 52.2% / 65.6% |

## Verdict

P3 is the strongest unified model in every headline metric, but it does not beat the NCR incumbent on these three official rounds. Its mean MAE is 5.6% worse (9.39 versus 8.89), and its top-10/25/50/100 point capture trails by 16.6/13.9/10.6/6.4 percentage points.

v1 and v5 are exactly prediction-identical here. This is expected from v5's terminal rung: its EB and slot-minute layers are disabled and its remaining tree settings match v1; the masked `lineouts_won` event is not part of this benchmark's eligible target list.

## Interpretation limits

- GW3 official points were fetched after the round, but the model cutoff and candidate teamsheets are the saved pre-lock versions.
- P3, v1 and v5 are deliberately frozen before GW1 for the entire tournament block. This is stricter than retraining after each gameweek.
- MAE compares observable-event forecasts with full official points, so missing NCR rubric events can penalise all three unified models' point levels. Rank and capture metrics are less sensitive to that common level gap.
