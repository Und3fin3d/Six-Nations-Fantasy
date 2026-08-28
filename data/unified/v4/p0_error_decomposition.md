# P0.1 — NCR error decomposition (frozen baseline fold artifacts)

Rows: 505 matched NCR label rows (GW1+GW2); played: 503.
Thin-intl (<5 prior intl store rows): 56 rows.

Signed identity per played row: expected - observable = minutes_term + rate_term; DNP rows collapse to appearance_term; official error additionally carries the rubric_term (observable - official, the unobserved-events gap).

## Mean-absolute shares of model error (vs observable points)

| segment | n | official_mae | model_mae_vs_observable | rubric_gap_mae | appearance_share | minutes_share | rate_share |
|---|---|---|---|---|---|---|---|
| all | 505 | 8.864 | 7.578 | 5.398 | 0.003 | 0.310 | 0.853 |
| gw1 | 263 | 9.518 | 7.592 | 6.338 | 0.005 | 0.296 | 0.867 |
| gw2 | 242 | 8.153 | 7.563 | 4.376 | 0.000 | 0.325 | 0.838 |
| thin_intl(<5) | 56 | 7.504 | 6.106 | 4.054 | 0.000 | 0.384 | 0.954 |
| rich_intl(>=5) | 449 | 9.033 | 7.762 | 5.566 | 0.003 | 0.302 | 0.843 |
| forwards | 283 | 8.048 | 7.403 | 5.551 | 0.000 | 0.305 | 0.774 |
| backs | 222 | 9.903 | 7.802 | 5.203 | 0.006 | 0.316 | 0.950 |
| played_only | 503 | 8.878 | 7.588 | 5.419 | 0.000 | 0.311 | 0.856 |

## Verdict: **rate-dominated (proceed)**

## Ledger #23 re-verification

```json
{
  "per_gw_top10_capture": {
    "gw1": {
      "baseline_top10": 0.5261,
      "incumbent_top10": 0.6256
    },
    "gw2": {
      "baseline_top10": 0.6017,
      "incumbent_top10": 0.6286
    }
  },
  "ncr_cohort_intl_history": {
    "median_intl_prior": 23.0,
    "share_le_3": 0.08712871287128712,
    "share_lt_5": 0.11089108910891089,
    "n_thin_lt_5": 56
  }
}
```
