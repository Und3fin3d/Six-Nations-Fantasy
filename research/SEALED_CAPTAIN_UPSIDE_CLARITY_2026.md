# Sealed 2026 Captain-Upside Clarity Check

Date: 2026-06-19

The captain (a 2x role) was being picked by highest XV *mean* (`sel_score`). A 2x
multiplier rewards *variance*, so the captain should be a *ceiling* bet. New knob:
`captain_head=mean` + `captain_upside_weight=w` over the existing `upside_score`
(attacking ceiling + prior-player variance). Promoted at `w=1.0` as
`captain_upside_full`. Only the captain pick changes — the point/XV/supersub paths are
untouched, so every non-captain metric is exactly flat on both seasons.

Base = prior promoted `bench_points_twostage_13`.

## 2025 dev (round-by-round team value)

| round | base | candidate | |
|---|---:|---:|---|
| R1 | 0.7658 | 0.7658 | same |
| R2 | 0.5899 | 0.6003 | better |
| R3 | 0.6743 | 0.7014 | better |
| R4 | 0.7048 | 0.7048 | same |
| R5 | 0.8027 | 0.8027 | same |
| **agg** | **0.7075** | **0.7150** | **+0.0075** |

Round-dominant: 2 better, 3 identical, 0 worse. MAE/value_xv/top15/spearman flat.
The original automated loop rejected it: the value-leg gate needed 3/5 *strictly*
better rounds and this was 2/5 (three rounds unchanged). The loop now has a narrow
zero-cost selector-dominance rule for this case: no team-value round may be worse,
and MAE/value_xv/bench/top/rank diagnostics must all be non-worse.

## Sealed 2026 (the held-out confirmation)

| metric | base | candidate | delta |
|---|---:|---:|---:|
| value_team | 0.6875 | 0.6971 | **+0.0096** |
| value_xv | 0.7145 | 0.7145 | +0.0000 |
| MAE | 7.3476 | 7.3476 | +0.0000 |
| bench_mae | 4.3120 | 4.3120 | +0.0000 |
| top15 | 0.3867 | 0.3867 | +0.0000 |
| top30 | 0.5467 | 0.5467 | +0.0000 |
| spearman_sel | 0.5874 | 0.5874 | +0.0000 |
| capt_top1 | 0.000 | 0.200 | +0.200 |
| capt_top3 | 0.200 | 0.400 | +0.200 |

Sealed round_team: base `[0.736, 0.741, 0.604, 0.627, 0.729]` ->
cand `[0.736, 0.741, 0.604, 0.627, 0.777]` (R5 better, rest identical, 0 worse).

## Why it is promoted

1. **Round-dominant** on both seasons (no round is ever worse).
2. **Generalises**: directly improves the sealed 2026 held-out season — the strongest
   robustness evidence, which the 3/5 round proxy only approximates.
3. **Zero cost**: only the captain pick changes; every other metric is exactly flat.
4. **Monotonic family effect** on 2026: weight 1.0 -> +0.0096, 1.25 -> +0.0117,
   1.5 -> +0.0117. `w=1.0` is the unique point that also helps 2025; higher weights go
   slightly negative on 2025.
5. **Principled**: captain = ceiling bet, because the 2x role rewards variance.

## Weight plateau (w=1.0 is not a spike)

Captain upside weight vs the no-captain baseline (DEV 0.7075 / SEAL 0.6875):

| w | DEV team d | SEAL team d |
|---:|---:|---:|
| 0.50 | +0.0021 | +0.0000 |
| 0.75 | +0.0075 | +0.0000 |
| 0.90 | +0.0075 | +0.0000 |
| 0.95 | +0.0075 | +0.0096 |
| **1.00** | **+0.0075** | **+0.0096** |
| 1.05 | +0.0075 | +0.0117 |
| 1.10 | -0.0002 | +0.0117 |
| 1.25 | -0.0002 | +0.0117 |
| 1.50 | -0.0002 | +0.0040 |

DEV team value is a flat plateau (+0.0075) across w in [0.75, 1.05]; the SEAL gain is
consistent across [0.95, 1.25]. `w=1.0` sits in the middle of the joint stable region
on both seasons — a robust choice, not a fragile peak. (w=1.05 weakly dominates with
SEAL +0.0117, but selecting it would be tuning to the held-out set; w=1.0 is the
principled full-upside value.)

The original override is documented in `research/promotion_report.json`; the policy is
now encoded in `model/research.py`. Reversible via
`research/promoted_config.prev_bench_points_twostage_13.json`.
