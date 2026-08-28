# Phase 1 / Workstream 1 — Failure diagnostician (v5 cycle)

Why did v1 direct totals beat v3 exposure/rates on raw deviance (+6.7%) while
v3 slightly improved official/observable points MAE — and why did no unified
model come close to promotion? Every claim below cites a repo artifact.

## A. Demonstrated facts (file-backed)

**F1 — The aggregate paradox is real but small.** `data/unified/v3/search/baseline_summary.csv`
best raw_deviance 0.22806 vs `gbdt_summary.csv` best 0.24328 (+6.7%);
observable_points_mae 6.031 vs 5.988 (-0.7%). The deviance objective is
`raw_deviance + observable_mae/25` (`model/unified/v3/search.py::_objective`),
so the two terms are weighted roughly equally.

**F2 — The deviance regression is one fold, not a uniform law.** Candidate 0 of
each family (`baseline_candidates.csv`, `gbdt_candidates.csv`), identical
features/weighting (`natural`, no decay, no context):

| fold | baseline raw_dev | gbdt_v3 raw_dev | delta |
|---|---:|---:|---:|
| 2022 | 0.24721 | 0.28953 | **+17.1%** |
| 2023 | 0.22038 | 0.22289 | +1.1% |
| 2024 | 0.21658 | 0.21780 | +0.6% |

**F3 — Inside 2022 the gap is 2–3 ultra-rare events.** Per-event deviances
2022 (v3 vs baseline): `tackle_turnover` 1.978 vs 0.925 (+1.05),
`tackle_try_saver` 0.605 vs 0.271 (+0.33), `drop_goal_missed` 0.069 vs 0.019,
`drop_goals_converted` 0.060 vs 0.028. `metres` is *better* under v3
(1.285 vs 1.389); `tries` 0.464 vs 0.452; `minutes_scaled_mae` ~equal.
In 2023/2024 only the two drop-goal events remain ~2–4× worse under v3;
everything else is within noise.

**F4 — The mechanism is written in `model/unified/v3/gbdt.py`.** For count
events v3 fits `rate = truth / max(minutes,1e-6) * 80` **only on played rows**,
weights rows by `clip(minutes/80, 0.05, 1)`, and caps rates at the **99.5%
quantile of a mostly-zero rate distribution**. For events with <0.5%
prevalence (drop goals, tackle_try_saver, tackle_turnover in some eras) the
quantile can be ~0 (cap kills every prediction: deviance for a y=1 surprise is
`2·log(y/1e-7)` ≈ 30 per positive row) or be inflated by a 1–5 minute cameo
with a positive (rate = 16–80). Prediction then multiplies three separately
fit quantities: `p_play × E[minutes|play] × rate/80`. Poisson deviance is most
sensitive exactly at tiny means where this product is least stable. The
baseline fits totals directly on **all rows including zero-minute rows**
(`model/unified/gbdt.py::fit` — `valid = available & notna`), so appearance
risk and rarity are inside one calibrated target.

**F5 — Selection layer: v3 lost by a hair, not by a rout.** On 6N 2025,
gbdt_v3 had the *higher* selection utility (0.00685 vs 0.00619) but failed
non-inferiority vs the 6N incumbent on top-25 capture (74.0% vs 76.7%,
-2.7pp; `decision.json`, `fold_metrics.csv`). Baseline was selected.

**F6 — The promotion blocker is NCR, and it is ranking, not MAE.** Selected
baseline vs NCR incumbent (retrospective): MAE 9.10 vs 8.90 (+2.2%), top-10
56.4% vs 62.7%, top-25 61.3% vs 71.5%, top-50 64.9% vs 74.2%, top-100 76.2%
vs 80.4%, low_history MAE +0.59 (`benchmark.md`, `decision.json`). Paired
bootstrap MAE diff CI is **[-0.17, +0.56] — includes zero**. On 6N the same
baseline *beats* the 6N incumbent on MAE in both seasons (2025: 7.27 vs 7.41;
2026: 7.19 vs 7.30) with mixed capture. **The unified goal is failing in
exactly one place: NCR ranking/capture, especially low-history players.**

**F7 — Row-level NCR evidence shows position-correlated bias, not diffuse
error** (`benchmark/predictions.csv`, gbdt_v3, GW2): hookers predicted at
~2× the incumbent (Kelleher 30.5 vs 15.8; Grobbelaar 31.2 vs 15.9; Nasser
25.0 vs 16.0); props inflated (Clarkson 16.4 vs 9.1; Heyes 17.5 vs 11.4);
one attacking back exploded (Freeman 48.8 vs 23.0; actual 13). The NCR
deterministic scorer (`model/unified/scoring.py`) credits `lineouts_won +1`
and `2×scrums_won`. The feed attributes essentially all own lineouts to the
hooker (~10–12/80), which is exactly why the NCR incumbent **drops**
`lineouts_won` ("a pure hooker-inflation artifact", `model/ncr_project.py`).
`scrums_won` has **0% coverage in NCR store rows**
(`audit/scoring_event_coverage.csv`) so unified NCR scrums are extrapolated
from 6N-trained patterns. The NCR audit shows potm/lineout_steals/
interceptions also unobservable; `data/ncr/ncr_potm.csv` exists (6 GW2
winners) but is not attached to the canonical store.

**F8 — The scorer contract has a measured reconstruction floor.** Audit:
reconstructed-vs-official weighted MAE **4.16**, NCR GW1 bias **-5.08**, GW2
-3.51, 6N bias -1.7 to -3.9; minimum within-round observable-points Spearman
**0.859 (NCR GW1)** vs 0.95–0.97 (6N). Official NCR points are only ~86%
rank-explained by the observable event set; raw-event modelling has
structurally less ranking leverage in NCR than in 6N.

**F9 — The 2022 fold is a different data era.** Baseline candidates 0 and 2
(`natural` vs `level_balanced`) produce *bit-identical* 2022 rows
(0.24721354224387246) — pre-2022 training data has no club rows, so level
weighting is degenerate. Club data (`ncr_club`) enters the store from ~2023.
The 2022 fold trains on 2019–2021 (internationals only, COVID era); event
availability masks differ by source vintage. The v3 "deviance regression" is
thus partly a **rare-event × era** interaction, not a stable property of the
exposure/rate design.

**F10 — Neural track died on evidence, not on taste.** All 20 configs exceeded
same-fold GBDT-v3 raw deviance AND observable MAE by >3% at rung 1
(`search/neural_track.json`). Blend was ineligible downstream
(`search/blend.json`).

**F11 — v1's MAE disadvantage vs v3 is minutes/appearance, not events.** v3's
minutes_scaled_mae is equal or slightly better every fold; observable-points
MAE is better every fold (5.60/6.21/6.15 vs 5.64/6.25/6.20). v3's hurdle
(appearance classifier × conditional tweedie minutes) helps the points
marginally; its rate layer hurts rare events badly.

**F12 — The incumbent's NCR edge is a prior/shrinkage edge, especially
low-history.** `model/ncr_project.py`: empirical-Bayes shrinkage
(K=220 weighted minutes) toward RugbyPass/position priors, club rates
calibrated to test level (attack ×0.84, confidence 0.55), bench-minute priors
by jersey slot, confirmed status P/B, WR-margin matchup scaling. The unified
features (`model/unified/features.py`) are level-blind ewm per-80 (club and
Test matches pooled) plus a `player_id` categorical — with <5 caps there is no
calibrated prior at all. NCR low_history MAE regression +0.59 (F6) is the
measurable scar.

## B. Causal synthesis (the answer to the v1-vs-v3 question)

1. Direct totals is the better *target parameterisation* for this data: one
   Poisson/tweedie objective per event on all rows matches the deviance metric
   and absorbs appearance risk. (F2, F4)
2. Exposure/rate fails specifically where rates are ultra-rare and eras shift:
   degenerate caps + played-only fitting + triple multiplication. Its 2022
   collapse (F3) is an engineering failure, not a refutation of exposure
   models in principle — but its 2023/2024 parity shows the design buys
   nothing on stable eras either. (F2, F3, F9)
3. v3's small official-MAE edge comes from hurdle minutes/appearance, a
   *modular* improvement that does not require the rate layer. (F11)
4. Neither v1 nor v3 loses to the NCR incumbent because of global model
   quality: they lose on (a) position-correlated scorer-attribution bias
   (lineouts_won→hooker, predicted-but-unobservable scrums_won/potm), and
   (b) low-history players where the incumbent has shrinkage priors and the
   unified models have none. (F6, F7, F12)
5. The NCR MAE gate (+2.2%) is inside bootstrap noise; the capture gaps
   (-4 to -10pp, consistent sign across GWs and all N) are not. (F6)

## C. Unknowns / assumptions

- U1: Exact official-platform attribution of NCR "Own Lineout Won +1" and
  "Scrum Won +2" (thrower vs jumper vs front-row split). The incumbent's
  winning-while-dropping-lineouts is indirect evidence; E0 quantifies.
- U2: Whether the -5.08 NCR GW1 reconstruction bias is mostly scrums/potm/
  steals (model-fixable) or unknown platform bonuses (a hard floor).
- U3: How much of Freeman-type extremes is potm/try overconfidence from
  level-blind ewm form vs legitimate ceiling. Diagnosed in E3 slices.
- U4: Whether 2022-fold rarity is availability-driven (pre-2022 sources lack
  tackle_try_saver/dominant_tackles positives) — needs availability-by-era
  table (E0b).
- U5: The GW3 shadow (`data/unified/v3/shadow/ncr_gw3_baseline.csv`) is frozen
  with the *baseline* engine; GW3 labels are not yet in `labels.py`
  (NCR_ROUNDS only GW1–2). Assumed available at prospective evaluation time.

## D. Ranked falsification experiments (before any full build)

| # | Test | Expected information gain | Cost | Failure signal |
|---|---|---|---|---|
| E0a | Position-level scorer-reconstruction bias on GW1–2 labels (audit data + labels; pure pandas) | Decomposes -5.08 bias by position; decides whether NCR gates measure attribution alignment or skill | hours | Bias ~uniform across positions → attribution theory weak |
| E0b | Event availability × year × source table | Explains 2022 fold; defines era-consistent evaluation | hours | Availability stable → era theory weak |
| E1 | **Scorer-ablation rescore** of existing `predictions.csv` (zero `lineouts_won`/`scrums_won`/`potm` means for NCR rows; recompute MAE/capture) | Directly measures how much of the -6..-10pp capture gap and +2.2% MAE gap is scorer-attribution vs model. No retraining | hours | Gap shrinks <2pp → it's the model, not the scorer |
| E2 | Guarded-rate v3 rerun on the 2022 fold only (minutes floor ≥20 in rate denominator; cap from positives-only quantile; min-positives guard with global-mean fallback) | Decides exposure/rate's fate with one fold | ~1 day | 2022 aggregate gap stays >2% → abandon rate layer permanently |
| E3 | v1 + shrunk empirical-Bayes form features (level-calibrated club→test, position×level priors, slot minutes) on 2022–24 folds; low-history slices | Tests the low-history mechanism on dev data without touching NCR labels | ~2 days | No low-history deviance/MAE gain → EB line inert, stop |
| E4 | Frozen-config v5 on the 6N 2025 selection benchmark only | Go/no-go before the full 12-group benchmark | ~half day | Regression vs v1 >1% MAE or >2pp any capture → revert |

## E. Tempting alternatives explicitly not pursued

- **More neural capacity / reopening the neural track.** 20/20 configs failed
  the >3% material-regression rule at rung 1 (F10); ~1,874 official labels and
  ~200k event rows cannot support shared-embedding multi-task nets beyond
  GBDT-level features here. No new inductive-bias evidence exists.
- **v2-style learned fantasy-points stack.** Prohibited (competition-specific
  points heads/rank stacks) and already failed to beat deterministic stage-1
  under honest OOF caveats (`data/unified/benchmark_v2.md`).
- **Weather/style context.** Weather is not PIT-safe (archived observed
  weather; `audit.json`); style failed feature admission (+0.00014 objective).
- **Market/closing-odds features.** Previously found retrospectively stamped
  (`RESEARCH_GOAL.md` PIT audit, 2026-06-29). Non-starters.
- **A heavier exposure/rate re-engineering as the main line.** E2 is one cheap
  test; even success only restores deviance parity and does nothing for F7/F12.
- **Relaxing the NCR gates or switching capture metrics.** The gates measure
  what deployment needs (safe overall); F8 says interpret MAE gates against
  the reconstruction floor, not weaken them.
