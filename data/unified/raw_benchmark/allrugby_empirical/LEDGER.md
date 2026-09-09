# All-rugby hill-climb ledger: NCR empirical engine

Frozen baseline `empirical_event` = **0.939031**, frozen `v1` gate = **0.907918** (lower is better).

Acceptance rule (precommitted, all must hold): (a) stable_score improves on the incumbent; (b) paired-by-fold bootstrap vs incumbent excludes 0; (c) no north/south regression >2% vs v1; (d) no tournament-family regression >2% vs v1; (e) no extended-event loss regression >5% vs v1.

| trial | stable_score | verdict |
| --- | --- | --- |
| `base` | 0.939031 | control, not a candidate |
| `t1_eb_off` | 0.909890 | ACCEPTED |
| `t2_eb_off_ebk` | 0.906896 | ACCEPTED |
| `t3_eb_off_ebk_head` | 0.904538 | ACCEPTED |
| `t4_no_signal_max` | 0.900754 | REJECTED |
| `t5_prior_recency` | 0.903551 | REJECTED |

## Trials

### `base` — control, not a candidate

**Hypothesis.** control: reproduce the frozen empirical_event contract

**stable_score** 0.939031

Reconstructed rubrics: ncr MAE 6.0608 / spearman 0.5343 / capture 0.7279, six_nations MAE 6.1972 / spearman 0.6248 / capture 0.7725

Frozen promotion gate vs v1 (reported, not the acceptance rule): FAIL — competition-balanced stable raw score did not improve over v1; north stable raw score regressed by more than 2%; south stable raw score regressed by more than 2%; tournament-family stable regression >2%: british_irish_lions, nations_championship, rugby_championship, six_nations, summer_internationals; extended-event loss regression >5%: tackle_try_saver, tackle_turnover

### `t1_eb_off` — ACCEPTED

**Hypothesis.** Counts are rate*minutes/80, so the small-sample per-player minutes table multiplies its noise into every event; shrink it toward the position/started mean by empirical Bayes. Also drop the opponent-strength multiplier, which saturates at its clips for any lopsided fixture.

**stable_score** 0.909890

Paired bootstrap vs incumbent: mean -0.02914, 90% CI [-0.03641, -0.02247] over 21 folds.

Reconstructed rubrics: ncr MAE 6.0623 / spearman 0.5361 / capture 0.7209, six_nations MAE 6.1700 / spearman 0.6283 / capture 0.7698

Gated extended-event loss (candidate / incumbent / v1):

- `dominant_tackles`: 0.7677 / 0.7849 / 0.7560
- `tackle_try_saver`: 0.2036 / 0.2023 / 0.1566
- `tackle_turnover`: 0.5045 / 0.5091 / 0.4454

Frozen promotion gate vs v1 (reported, not the acceptance rule): FAIL — competition-balanced stable raw score did not improve over v1; tournament-family stable regression >2%: british_irish_lions, nations_championship; extended-event loss regression >5%: tackle_try_saver, tackle_turnover

### `t2_eb_off_ebk` — ACCEPTED

**Hypothesis.** Replace the global K=220 shrinkage of player history toward the position prior with a per-event empirical-Bayes K fitted from the training fold's within/between player rate variance.

**stable_score** 0.906896

Paired bootstrap vs incumbent: mean -0.00299, 90% CI [-0.00512, -0.00060] over 21 folds.

Reconstructed rubrics: ncr MAE 6.0467 / spearman 0.5367 / capture 0.7218, six_nations MAE 6.1587 / spearman 0.6279 / capture 0.7696

Gated extended-event loss (candidate / incumbent / v1):

- `dominant_tackles`: 0.7547 / 0.7677 / 0.7560
- `tackle_try_saver`: 0.2002 / 0.2036 / 0.1566
- `tackle_turnover`: 0.5012 / 0.5045 / 0.4454

Frozen promotion gate vs v1 (reported, not the acceptance rule): FAIL — tournament-family stable regression >2%: british_irish_lions, nations_championship; extended-event loss regression >5%: tackle_try_saver, tackle_turnover

### `t3_eb_off_ebk_head` — ACCEPTED

**Hypothesis.** The minutes target is MAE-scored (median-optimal) but also scales counts (mean-optimal). Fit a separate median-valued minutes head, leaving the count scaler on E[minutes].

**stable_score** 0.904538

Paired bootstrap vs incumbent: mean -0.00236, 90% CI [-0.00259, -0.00212] over 21 folds.

Reconstructed rubrics: ncr MAE 6.0467 / spearman 0.5367 / capture 0.7218, six_nations MAE 6.1587 / spearman 0.6279 / capture 0.7696

Gated extended-event loss (candidate / incumbent / v1):

- `dominant_tackles`: 0.7547 / 0.7547 / 0.7560
- `tackle_try_saver`: 0.2002 / 0.2002 / 0.1566
- `tackle_turnover`: 0.5012 / 0.5012 / 0.4454

Frozen promotion gate vs v1 (reported, not the acceptance rule): FAIL — tournament-family stable regression >2%: british_irish_lions, nations_championship; extended-event loss regression >5%: tackle_try_saver, tackle_turnover

### `t4_no_signal_max` — REJECTED

**Hypothesis.** Inspecting T2's fitted constants showed red_cards, drop_goals_converted, drop_goal_missed and potm all falling back to the global K=220 because their estimated between-player variance was <= 0 -- precisely the worst remaining targets. A non-positive between-player variance means the data detect no player-level signal, so the empirical-Bayes answer is full shrinkage to the position prior, not moderate shrinkage.

**stable_score** 0.900754

Paired bootstrap vs incumbent: mean -0.00378, 90% CI [-0.00867, +0.00107] over 21 folds.

Rejected because:

- (b) paired bootstrap vs incumbent does not exclude 0 [-0.00867, +0.00107]

Reconstructed rubrics: ncr MAE 6.0464 / spearman 0.5372 / capture 0.7225, six_nations MAE 6.1578 / spearman 0.6282 / capture 0.7697

Gated extended-event loss (candidate / incumbent / v1):

- `dominant_tackles`: 0.7547 / 0.7547 / 0.7560
- `tackle_try_saver`: 0.2006 / 0.2002 / 0.1566
- `tackle_turnover`: 0.4987 / 0.5012 / 0.4454

Frozen promotion gate vs v1 (reported, not the acceptance rule): FAIL — tournament-family stable regression >2%: british_irish_lions, nations_championship; extended-event loss regression >5%: tackle_try_saver, tackle_turnover

### `t5_prior_recency` — REJECTED

**Hypothesis.** The two gated extended events (tackle_turnover, tackle_try_saver) are only recorded from 2021 but are primed from a position prior pooled over all history, while player profiles are already recency-weighted. Weight the position prior by the same half-life.

**stable_score** 0.903551

Paired bootstrap vs incumbent: mean -0.00099, 90% CI [-0.00662, +0.00479] over 21 folds.

Rejected because:

- (b) paired bootstrap vs incumbent does not exclude 0 [-0.00662, +0.00479]
- (d) tournament-family regression >2% vs incumbent: british_irish_lions, pacific_nations_cup

Reconstructed rubrics: ncr MAE 6.0472 / spearman 0.5373 / capture 0.7223, six_nations MAE 6.1625 / spearman 0.6281 / capture 0.7693

Gated extended-event loss (candidate / incumbent / v1):

- `dominant_tackles`: 0.7494 / 0.7547 / 0.7560
- `tackle_try_saver`: 0.2002 / 0.2002 / 0.1566
- `tackle_turnover`: 0.4988 / 0.5012 / 0.4454

Frozen promotion gate vs v1 (reported, not the acceptance rule): FAIL — tournament-family stable regression >2%: british_irish_lions, nations_championship; extended-event loss regression >5%: tackle_try_saver, tackle_turnover

