# Unified rugby model — first cross-competition benchmark

All unified forecasts come from the same raw-event model. NCR and Six Nations points are produced afterward by deterministic rule adapters.

## NCR

| Model | MAE | Spearman | Top 10 capture | Top 25 capture | Top 50 capture | Top 100 capture |
|---|---:|---:|---:|---:|---:|---:|
| Champion | 9.10 | 0.520 | 45.2% | 64.2% | 66.9% | 73.9% |
| NCR | 8.91 | 0.581 | 61.6% | 70.1% | 72.9% | 78.8% |
| Unified GBDT | 9.01 | 0.524 | 55.9% | 57.7% | 63.3% | 73.4% |
| Unified neural | 9.38 | 0.500 | 33.8% | 52.6% | 63.3% | 74.7% |

### MAE by completed round

MAE is calculated against official platform points. NCR and champion share one common cohort; unified coverage is shown separately.

| Model | GW1 MAE | GW2 MAE | Two-round weighted MAE | Player-rounds |
|---|---:|---:|---:|---:|
| NCR | 9.39 | 8.42 | **8.91** | 532 |
| Champion | 9.21 | 9.00 | **9.10** | 532 |
| Unified GBDT | 9.92 | 8.01 | **9.01** | 514 |
| Unified neural | 10.42 | 8.24 | **9.38** | 514 |

The unified models cover fewer player-rounds because their API-to-fantasy crosswalk is incomplete. Their MAEs are therefore informative but not a strict like-for-like win or loss against the 532-row incumbent cohort.

## Six Nations

| Model | MAE | Spearman | Top 10 capture | Top 25 capture | Top 50 capture | Top 100 capture |
|---|---:|---:|---:|---:|---:|---:|
| 6N champion | 7.30 | 0.672 | 60.2% | 65.2% | 70.1% | 72.7% |
| Unified GBDT | 7.23 | 0.696 | 69.7% | 68.8% | 71.5% | 76.3% |
| Unified neural | 7.73 | 0.628 | 67.2% | 69.1% | 72.9% | 74.1% |

## Promotion decision

**Do not promote yet.** The unified neural model fails the frozen non-inferiority gate:

- NCR top-10 capture is more than 2pp below NCR
- NCR top-25 capture is more than 2pp below NCR
- NCR top-50 capture is more than 2pp below NCR
- NCR top-100 capture is more than 2pp below NCR

The GBDT remains a must-beat universal control; it is not blended into the neural output.
