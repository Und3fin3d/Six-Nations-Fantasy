# Unified rugby supermodel v3 benchmark

Six Nations 2025 is the model-selection layer. Six Nations 2026 and NCR GW1–2 are retrospective references because their results influenced development.

Selected architecture on 6N 2025: **baseline**.

## Six Nations 2025 selection

| Competition | Model | MAE | Spearman | Top10 | Top25 | Top50 | Top100 |
|---|---|---:|---:|---:|---:|---:|---:|
| six_nations | baseline | 7.27 | 0.647 | 66.2% | 75.7% | 84.4% | 94.7% |
| six_nations | gbdt_v3 | 7.27 | 0.648 | 68.4% | 74.0% | 84.7% | 94.5% |
| six_nations | incumbent | 7.41 | 0.643 | 67.7% | 76.7% | 85.2% | 94.2% |

## Retrospective references

| Competition | Model | MAE | Spearman | Top10 | Top25 | Top50 | Top100 |
|---|---|---:|---:|---:|---:|---:|---:|
| ncr | baseline | 9.10 | 0.522 | 56.4% | 61.3% | 64.9% | 76.2% |
| ncr | gbdt_v3 | 9.01 | 0.536 | 51.2% | 60.1% | 66.1% | 76.2% |
| ncr | incumbent | 8.90 | 0.581 | 62.7% | 71.5% | 74.2% | 80.4% |
| six_nations | baseline | 7.19 | 0.686 | 70.7% | 78.4% | 87.2% | 95.2% |
| six_nations | gbdt_v3 | 7.14 | 0.689 | 69.4% | 79.2% | 86.2% | 95.0% |
| six_nations | incumbent | 7.30 | 0.666 | 71.5% | 77.1% | 83.5% | 95.2% |

## Fold-level differences versus incumbent

| Phase | Competition | Season | Round | MAE diff | Mean capture diff |
|---|---|---:|---:|---:|---:|
| selection | six_nations | 2025 | 1 | -0.48 | -4.5% |
| selection | six_nations | 2025 | 2 | -0.05 | +0.8% |
| selection | six_nations | 2025 | 3 | -0.53 | +3.5% |
| selection | six_nations | 2025 | 4 | +0.22 | -3.8% |
| selection | six_nations | 2025 | 5 | +0.13 | +0.7% |
| retrospective_reference | six_nations | 2026 | 1 | +0.29 | -4.0% |
| retrospective_reference | six_nations | 2026 | 2 | -0.59 | +5.0% |
| retrospective_reference | six_nations | 2026 | 3 | -0.24 | +4.5% |
| retrospective_reference | six_nations | 2026 | 4 | -0.13 | +2.4% |
| retrospective_reference | six_nations | 2026 | 5 | +0.10 | -2.8% |
| retrospective_reference | ncr | 2026 | 1 | +0.16 | -9.4% |
| retrospective_reference | ncr | 2026 | 2 | +0.23 | -5.6% |

## Paired bootstrap MAE differences

- six_nations: candidate minus incumbent -0.11 (5th–95th percentile -0.30 to +0.08).
- ncr: candidate minus incumbent +0.20 (5th–95th percentile -0.17 to +0.56).

## Decision

- Retrospective safe-overall gate: **FAIL**.
- Competition-balanced pooled utility: -0.0178.
- Promotion: **NO** — GW4–GW7 prospective shadow remains mandatory, and all retrospective gates must also pass.
- Corrective cycle allowed after a miss: **NO**.

- ncr: MAE regression exceeds 2%
- ncr: top-10 capture regression exceeds 2pp
- ncr: top-25 capture regression exceeds 2pp
- ncr: top-50 capture regression exceeds 2pp
- ncr: top-100 capture regression exceeds 2pp
- competition-balanced pooled utility is not positive (-0.0178)
- ncr low_history: MAE regression +0.59 > 0.5
