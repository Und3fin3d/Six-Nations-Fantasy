# Friendly-15 raw-stat test and bounded domain experiment

## Decision

The requested 15-friendly metric is implemented, run and independently verified. Do not open a new universal-champion PR or promote the tested training change. The selected candidate improves the friendly count-stat and metres point estimates, but fails the combined fantasy gates. PR #20 remains the stronger fantasy-team research candidate among the two compared here; this experiment does not establish that either model is definitively best.

The new branch is `codex/friendly-stat-validation-20260919`, based on PR #20 at `73fb3ce9a010480ed2443a6bca1eefc395d55072`. PR #20, main, deployed model routing and the frozen prospective P3 shadow were not changed. The metric and tested source remain on the research branch. No fitted model binaries are committed.

## What Friendly-15 measures

The 15 fixture IDs were frozen before selection/fitting, using the latest completed international friendlies in cache competition 30 before 2026-09-19, excluding fixtures with official fantasy labels in this repository. This is not a claim that no fantasy platform anywhere offered these games. Selection did not use score margins, event values or model errors. The fixture manifest pins kickoffs, names, IDs, statistic names and source-cache hashes. These games comprise 11 November 2025 and four August 2026 friendlies.

Every cached teamsheet player is evaluated: both sides, 46 players per game, 690 player-match rows. Four canonical fixtures initially contained only one tracked side; `COHORT_CORRECTION.md` documents their correction using the permanent full cache, with no fixture replacement. Untracked opponents are explicit cold-start cases, using generic pre-match jersey-position inference and any available canonical/club history. Training data is unchanged. The original incomplete-cohort smoke test and workflow are superseded and are not evidence for this report.

For each game and statistic, take mean absolute error over players with an observed label. Then average each of the 22 stable count-stat MAEs equally across the 15 games. Report metres and minutes separately because their units differ. Per-stat MAEs are always retained. This is an absolute count error, not a fantasy-weighted or standardised loss; larger-volume count events can contribute larger errors.

Observed zeroes and unused substitutes are retained; missing labels are not zeroes. Missing predictions or mismatched player keys fail. Missing game/stat cells invalidate the corresponding aggregate rather than disappearing through a mean. In this run all 24 target columns had 690 observed rows for every model: 22 count statistics plus metres and minutes.

## Friendly-15 result

Lower is better. `empirical_raw` is the raw-event empirical engine, not the complete competition-specific fantasy baseline. `p3_robust_native` is the PR #20 research candidate. `p3_balanced_recent_native` is the single candidate selected on the separate development fixtures.

| engine | fixtures | count_stat_mae | minutes_mae | metres_mae |
| --- | --- | --- | --- | --- |
| empirical_raw | 15 | 0.691210 | 10.466982 | 11.396364 |
| p3_balanced_recent_native | 15 | 0.666641 | 10.361084 | 11.137670 |
| p3_robust_native | 15 | 0.668166 | 10.332761 | 11.153362 |

PR #20 reduces friendly count-stat MAE by **3.33%** against the empirical raw engine. The new candidate reduces it by **3.55%**. Its extra improvement over PR #20 is only **0.23%** and is not statistically resolved in this small sample. It improves 20 of 22 count-stat averages and 10 of 15 game-level count averages against empirical; it is not best on every statistic or game. Its minutes point estimate is slightly worse than PR #20.

Paired 90% whole-fixture bootstrap intervals (10,000 draws, seed 17), candidate minus empirical: count-stat MAE **-0.024569 [-0.042129, -0.007578]**, metres MAE **-0.258694 [-0.398456, -0.130627]**, minutes MAE **-0.105898 [-0.337980, +0.122745]**. Against PR #20 the count-stat interval is **-0.001525 [-0.004413, +0.001411]**. These are descriptive intervals on retrospective fixtures, not selection-adjusted prospective confidence guarantees.

`games.csv` preserves all 15 game-level comparisons; `stat_mae.csv` preserves every statistic, including regressions. `intervals.csv` contains all friendly and fantasy bootstrap results.

## Bounded model iteration

The unchanged canonical store has 153,962 club and 20,546 international player-match rows. The experiment tested whether their imbalance harms international generalisation. Both candidates use the robust empirical component, native categorical v4, unchanged P3 event weights, the same event heads and tree capacity, and a single shared configuration across tournaments.

A uses existing level-balanced training. B uses that same level balancing followed by a 730-day recency half-life. Level balancing equalises total level weights before recency multiplication; the final recency-weighted totals are not forced equal. The tree loss and player-specific post-fit residual aggregation now share the same weights, rather than allowing club-dominated residual sums to undo the weighting change. The natural-weight control is unchanged, verified at prediction and squad level. No competition identifier, fantasy-points prediction head or tournament-specific coefficient was added.

The 12 separate development friendlies all precede 2025 and are disjoint from the test 15. Select lower count-stat MAE than PR #20, with at most 2% minutes/metres regression; choose the lower count MAE among eligible candidates. This protocol was committed before runs, and the full-cohort selection was committed before confirmation jobs started.

| engine | fixtures | count_stat_mae | minutes_mae | metres_mae |
| --- | --- | --- | --- | --- |
| empirical_raw | 12 | 0.675402 | 10.139869 | 10.952262 |
| p3_balanced_native | 12 | 0.659727 | 10.009276 | 10.686419 |
| p3_balanced_recent_native | 12 | 0.658586 | 10.029099 | 10.685127 |
| p3_robust_native | 12 | 0.661657 | 10.069178 | 10.669155 |

Both candidates passed the development guard. B (`p3_balanced_recent_native`) was selected and frozen. Only B, the fixed PR #20 control, and the relevant empirical comparator were evaluated on the final 15 games and 13 fantasy rounds. A was not subsequently substituted based on confirmation results, and no tuning followed these results.

## Fantasy comparison: why B is rejected as the next champion

MAE is the equal-weight mean of per-round player fantasy-point MAE. Team points are realised official outcomes for selected squads, including captain and super-sub multipliers. All models use identical candidate rows, available histories, scoring adapters and optimiser constraints.

| competition | season | engine | mae | team_points | rounds |
| --- | --- | --- | --- | --- | --- |
| ncr | 2026 | empirical_baseline | 8.401189 | 1578.000000 | 3 |
| ncr | 2026 | p3_balanced_recent_native | 8.403139 | 1694.000000 | 3 |
| ncr | 2026 | p3_robust_native | 8.364626 | 1740.000000 | 3 |
| six_nations | 2025 | empirical_baseline | 7.139481 | 2410.000000 | 5 |
| six_nations | 2025 | p3_balanced_recent_native | 7.198233 | 2275.000000 | 5 |
| six_nations | 2025 | p3_robust_native | 7.231897 | 2416.000000 | 5 |
| six_nations | 2026 | empirical_baseline | 7.389271 | 2548.000000 | 5 |
| six_nations | 2026 | p3_balanced_recent_native | 7.250683 | 2519.000000 | 5 |
| six_nations | 2026 | p3_robust_native | 7.254336 | 2550.000000 | 5 |

Against PR #20, B changes team points by **-46 NCR, -141 Six Nations 2025, and -31 Six Nations 2026**. Its Six Nations 2025 MAE improves by 0.033663 but still trails the empirical baseline. Its NCR MAE is worse than both PR #20 and empirical. Six Nations 2026 MAE improves slightly, but squad points fall below empirical. Thus neither the empirical-baseline gate nor the stronger all-metric improvement over PR #20 is met.

This is a useful falsification result: improving friendly stat MAE does not automatically improve the selected fantasy teams. The new metric is an additional requirement, not permission to discard squad points or change acceptance thresholds.

Six Nations squad scores remain **price-free diagnostics** because genuine archived round prices are unavailable. NCR budgets are checked, but GW3 uses retrospectively corrected final lineups/prices; GW1 excludes New Zealand and France for every model. No point estimate here proves universal fantasy superiority. The candidate's squad-point bootstrap intervals versus empirical include zero in all three tournament-seasons; Six Nations 2025 points also regress versus PR #20 with the reported interval excluding zero.

## Point-in-time and validation limits

Every game/round refits on the same earlier completed history for all candidates. The shared availability convention requires exact kickoff plus strictly more than three hours before the lock; date-only histories exclude the cutoff day. Current outcomes are masked. This does not prove original source-publication timestamps: historical source files are unversioned. Friendlies and Six Nations use kickoff lock proxies and oracle teamsheets.

All 40 selected World Rugby ranking inputs were also checked against cached effective-publication timestamps and point values: each was effective before its prediction cutoff, with at least 13.15 hours lead. A cache's query date is not necessarily its publication timestamp. Future ranking rows were not admitted.

The P3 event weights were previously selected using 2022-2025 data, and historical 2025/2026 outcomes were already repository evidence. The disjoint development/test split prevents this experiment from directly tuning to the 15 test outcomes, but it does not turn reused historical data into a new prospective holdout. The small development set also limits selection reliability. NCR GW4-7 was not used and no rugby API request or source-cache refresh was made.

## Verification

- **144 repository tests passed in 44.96 seconds**, including the original frozen LFS P3 artifact. This adds 23 tests to the PR #20 suite.
- All 40 fixture/round jobs completed: 12 development, 15 friendly confirmation, 13 fantasy comparison.
- Independently reconstructed full friendly truth and availability from source caches and recalculated **2,232 model/game/stat MAEs** (1,152 development plus 1,080 confirmation).
- Independently recalculated **39 fantasy MAEs and 39 selected-squad totals**, including all nine NCR budget/hemisphere checks.
- All 26 empirical/control squads, role choices and predictions match the preceding PR #20 evidence; raw rescoring matches saved fantasy expectations within 1e-10.
- All 93 model-source hashes and common store/features/config/package manifests match across jobs. Every confirmation manifest refers to the exact same frozen selection hash.
- Verified all 40 training-history cutoff and key hashes; no current fixture appears in its training cohort. Both complete teamsheets are evaluated on every friendly.
- Current store remains 174,508 player-match rows across 3,841 fixtures. SHA-256: `e87ec4666fcef24de8c6bbf333ba1483e0ed268b965099cace8fe5dfc5b93145`.

Tested source commit: `644c682955075944d45f7a42bae2361bb9c1089e`.
Completed comparison and full-test run: `35475684776`.
Frozen full-cohort selection SHA-256: `ab9022d822638b41d99772bb814ae213f74c3365a9a8e95d37de2d81326a57b1`.
Superseded incomplete-cohort run: `35475086184` (cancelled; not used).

Python 3.11.16, NumPy 1.24.3, pandas 2.1.0, SciPy 1.11.2, scikit-learn 1.3.1, LightGBM 4.6.0 and torch 2.7.1+cpu. No fitted binaries are committed. Temporary execution/stop workflows are removed after completion; complete per-player predictions, squads, original run logs and independent audits are retained in the attached evidence bundle.

## Reproduce

Use Python 3.11 with `requirements-model.txt`, `requirements-unified.txt`, parser-test dependencies and required historical LFS inputs. The checked-in fixture and selection manifests must not be rewritten. Choose a new output directory.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
python -m model.unified.domain_experiment --prepare-only \
  --output data/unified/friendly15_reproduction

# Read exactly the pinned 15 games; do not choose games by result.
for fixture in $(python -c 'import json; print(" ".join(x["fixture_id"] for x in json.load(open("data/unified/friendly15/fixtures.json"))["fixtures"]))'); do
  python -m model.unified.domain_experiment \
    --output data/unified/friendly15_reproduction --job "test:$fixture" \
    --engines p3_robust_native p3_balanced_recent_native
done
```

Each job writes its per-stat `metrics.csv`, full keyed `truth.csv`, raw predictions and immutable source/history manifest. Aggregate the completed fixed cohort with:

```python
import json
from pathlib import Path
import pandas as pd
from model.unified.friendly_eval import summarise_stats

out = Path("data/unified/friendly15_reproduction")
fixed = json.loads(Path("data/unified/friendly15/fixtures.json").read_text())
ids = [row["fixture_id"] for row in fixed["fixtures"]]
metrics = pd.concat([
    pd.read_csv(out / "jobs" / f"test_{fixture}" / "metrics.csv",
                dtype={"fixture_id": str})
    for fixture in ids
], ignore_index=True)
summary, per_stat = summarise_stats(metrics, ids)
summary.to_csv(out / "friendly15_summary.csv", index=False)
per_stat.to_csv(out / "friendly15_stat_mae.csv", index=False)
```

For a fantasy round use, for example, `--job six_nations_2025_r3` with the same engines and a new output directory. For development use the frozen `development_fixtures.json` and `--job dev:<id>`; all three predeclared candidates are the default on development jobs. Do not select again from confirmation outcomes.

## Next evidence needed

Retain the new Friendly-15 test and PR #20 control. A subsequent candidate needs a mechanism that improves fantasy selection as well as raw-stat prediction, assessed on development data before returning to this now-examined historical test. More untouched rounds and budget-valid Six Nations archives remain necessary for a definitive deployment claim. The current evidence supports neither promoting B nor opening a new champion PR.
