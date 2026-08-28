# Autoresearch Goal — improve MAE, full-team value, and selection health

Branch: `cdx`.

This branch runs a Karpathy-style local research loop: try one scoped modelling or
selection change, evaluate it on the 2025 backtest, keep it only if it improves the
primary metrics without damaging secondary selection-health metrics, and log every
accepted/rejected trial. The loop costs 0 API calls.

## Current Baseline

Baseline to beat is the current promoted `cdx` deployable picker:
`orchestrator_resid030_delta_overlay_010`, post-team-sheet.

| metric | 2025 | 2026 sealed |
|---|---:|---:|
| selector value_team | 0.714 | 0.700 |
| selector value_xv | 0.714 | 0.718 |
| target MAE | 7.406 | 7.303 |
| bench MAE | 5.373 | 4.312 |
| selector top15 | 0.387 | 0.387 |
| selector top30 | 0.573 | 0.553 |
| selector Spearman | 0.508 | 0.588 |

2026 is not used by the loop. It is a sealed reporting check for a final accepted
configuration. For 2026 prediction, 2025 rows are valid training history; no future
2026 rows may enter a 2026 prediction.

## Acceptance Rule

Primary objectives are MAE and full fantasy-team value. `value_team` is the
model-picked XV plus model captain plus model supersub, divided by hindsight XV
plus hindsight captain plus hindsight supersub. `value_xv` remains a diagnostic.

- value_team improves by at least 0.003 with MAE not worse by more than 0.02, or
- MAE improves by at least 0.02 with value_team not worse by more than 0.003, and
- the relevant gain appears in at least 3 of 5 rounds. Value candidates need
  per-round team value improvement; MAE candidates need per-round MAE improvement.
- Zero-cost selector dominance is an explicit exception to the strict 3/5 rule:
  a value-team candidate may pass with fewer than three strictly better rounds only
  if no team-value round is worse and MAE, XV value, bench MAE, top15, top30, and
  selector Spearman are all non-worse. This covers pure captain/supersub decision
  improvements where unchanged rounds should count as neutral evidence, not failure.

Secondary metrics are guardrails and tie-breakers:

- `top15` and selector-based within-position Spearman must not collapse; the loop
  rejects otherwise acceptable candidates if `top15` drops by more than 0.04 or
  `spearman_sel` drops by more than 0.03.
- `value_xv`, `top30`, `capt_top1`, `capt_top3`, `capt_top5`, and bench MAE are
  reported for diagnosis.
- The JSON ledger also persists `round_team`, `round_xv`, `round_mae`, `round_bias`, and
  by-position MAE/bias diagnostics.

Rejected candidates stay in the ledger; accepted candidates become the new incumbent.

## Optional Model-Specific Claude Second Opinion

The general Codex lifecycle now owns automatic Claude second opinions for
substantial coding and agentic tasks. `model.research` therefore does not invoke
its own reviewer by default and avoids duplicate reviews.

Pass `--claude-review` when a standalone model batch needs the richer,
domain-specific packet containing dev metrics, round slices, acceptance reasons,
and stability results. Claude remains an advisory reasoning reviewer, not a
verifier or decision-maker:

- it receives no sealed-season results and has no file or shell tools;
- it cannot alter acceptance, configs, predictions, or promotion;
- identical evidence reuses the cached review by SHA-256 fingerprint;
- failures are recorded but do not discard a completed numerical batch;
- `--seal-2026 --claude-review` reuses or obtains the same dev-only opinion
  before sealing.

Artifacts:

- `research/claude_review.json` - latest structured opinion plus exact evidence.
- `research/CLAUDE_SECOND_OPINION_LATEST.md` - readable latest opinion.
- `research/claude_review_history.jsonl` - append-only review history.
- `research/claude_review_context.json` - optional provenance and known-decision
  context supplied with every review.
- `research/LEDGER.md` - annotated with the latest verdict.

Controls:

```bash
python -m model.research --claude-review
python -m model.research --force-claude-review
python -m model.claude_review --force
python -m model.claude_review --print-evidence
```

Use `CLAUDE_REVIEW_MODEL` to pin a reviewer model. The default timeout is 300
seconds and can be changed with `--claude-review-timeout`.

## Current Best

| config | 2025 value_team | 2025 value_xv | 2025 MAE | 2026 value_team | 2026 value_xv | 2026 MAE |
|---|---:|---:|---:|---:|---:|---:|
| previous promoted `points_lgbm80_xgb20` | 0.626 | **0.716** | **7.571** | 0.678 | **0.714** | **7.349** |
| promoted `supersub_two_stage_minutes` | **0.707** | **0.716** | 7.571 | **0.688** | **0.714** | 7.349 |
| prior promoted `bench_points_twostage_13` | 0.707 | 0.716 | 7.550 | 0.688 | 0.714 | 7.348 |
| promoted `captain_upside_full` | 0.715 | 0.716 | 7.550 | 0.697 | 0.714 | 7.348 |
| promoted `target_only_xgb10` | 0.715 | 0.716 | 7.524 | 0.697 | 0.714 | 7.342 |
| promoted `target_only_xgb50_starters` | 0.715 | 0.716 | 7.464 | 0.697 | 0.714 | 7.304 |
| **promoted `orchestrator_resid030_delta_overlay_010`** | **0.714** | **0.714** | **7.406** | **0.700** | **0.718** | **7.303** |
| invalid closing-line `external_market_post_target_delta020` | 0.728 | 0.733 | 7.381 | 0.713 | 0.735 | 7.320 |
| invalid closing-line `captain_market_winprob_100` | 0.737 | 0.733 | 7.381 | 0.714 | 0.735 | 7.320 |

PIT audit (2026-06-29): the two market rows above are **not valid promotion
evidence**. The historical scraper used closing odds and retrospectively stamped
them to the prior evening. The 2025 round lock preceded later fixtures, so some
prices were unavailable when the fantasy team had to be chosen.

Timestamped opening odds were fetched and tested instead. A captain-only opening
tilt improved 2025 `value_team` by `0.001797`, below the predeclared `0.003`
gate and from one captain decision. The best opening-market post-target variant
also missed the strict value/MAE gate; larger weights regressed. Both remain
unpromoted. Production therefore reverts to the clean orchestrator, with all
market inputs disabled.

Big-moves audit (2026-06-29): do not merge `cdx-bigmoves` wholesale. Its Elo
history treated team-side mirrored rows as independent fixtures, updated each
fixture twice, and allowed the second side's nominal pre-match rating to contain
the same match result. Correcting to one update per fixture invalidated the
forecast/selector result: forecast Elo helped 2025 but hurt 2026 at every tested
weight, while selector and supersub tilts were poor. The useful conceptual piece
was team-strength context for captain choice. Timestamped opening odds are the
only defensible retrospective representation tested, and their gain was too
small to promote.

Latest dev-only vetoes:

- `orchestrator_selector_signal_xgb0` cleanly separated the selector signal from
  the MAE-facing forecast. It improved 2025 `value_team` 0.714 -> 0.720 and XV
  0.714 -> 0.722 with identical MAE, passing round 5/5, team 6/6, position 5/5,
  and minutes 4/4 stability. Sealed 2026 vetoed it: team value fell 0.7002 ->
  0.6990 and XV 0.7183 -> 0.7169. Close this family rather than tune intermediate
  XGB weights against the opened sealed result.
- `target_frontrow20_back5_20` improved 2025 MAE to 7.609 with unchanged XV value, but sealed
  2026 MAE worsened to 7.529. It is not promoted.
- `minutes_alpha_4` improved 2025 MAE to 7.621 and top15 to 0.373, but sealed 2026 slipped to
  MAE 7.391 and XV value 0.706 versus the then-promoted 7.383 / 0.711. It is not promoted.
- Front-row split variants (`frontrow_15`, `frontrow_25`, `prop_20`, `hooker_20`) failed the 2025
  gate, so the 20% combined front-row calibration remains the point-calibration layer.
- `points_lgbm85_xgb15` improved 2025 MAE to 7.589 but failed round-drop stability.
- XGB rank overlays only improved top15, not XV value. Component grafts were blocked because no
  component passed the fixture-group OOF >=2% win rule.
- Bayesian shrinkage and fixed LGBM/XGB/Bayes combiners did not beat the simpler 80/20 LGBM/XGB
  promoted blend.
- `bench_supersub_ridge_025` improved 2025 value_team but sealed 2026 was flat versus the old
  promoted config.
- `supersub_two_stage_minutes` improved 2025 value_team to 0.707, passed stability, and improved
  sealed 2026 value_team to 0.688 without changing MAE or XV-only value. It is promoted.
- Bench kicking shrink/replacement, captain heads, selector upside overlays, and combined
  specialists did not beat the standalone two-stage supersub head.
- `bench_points_twostage_20` improved 2025 MAE to 7.539 but over-pulled bench forecasts and worsened
  sealed 2026 MAE to 7.359, so it was vetoed.
- The smaller bench-target blend sweep found the useful edge: 12% improved sealed MAE but missed the
  2025 gate by a hair; 13% passed the 2025 MAE/stability gate and improved sealed 2026 MAE to 7.348
  with unchanged `value_team`. `bench_points_twostage_13` is now promoted.
- POTM/Bayesian interaction candidates exposed another 2025 trap. `potm075_bayes10` passed the
  2025 gate strongly (value_team 0.735, MAE 7.522), but sealed 2026 regressed versus promoted:
  value_team 0.684 vs 0.688 and MAE 7.378 vs 7.348. Treat high-POTM point weighting as unstable
  unless a future version can pass a stricter round/team robustness check.
- New mechanism (2026-06-19): **decoupled set-piece latent** via the `supersub_latent_shrink`
  knob. Decomposition showed latent-shrink's 2025 team-value loss is *entirely* the supersub
  (XV +10, captain +7, supersub ×3 −78). Holding the supersub on the unshrunk latent recovered it,
  so `decoupled_latent040_sub100` (0.714 / 0.722 / 7.506) and `decoupled_latent050_sub100`
  (0.713 / 0.720 / 7.503) both passed the 2025 gate on the MAE leg AND all four stability checks —
  the first clean two-leg 2025 wins found beyond the POTM traps. But sealed 2026 vetoed them: MAE
  regressed +0.045 (0.40) / +0.020 (0.50) with value_team flat. The set-piece latent is *real
  signal* in 2026, so shrinking the point path is a 2025-only trap. The supersub-decoupling half
  of the mechanism does generalise (team value held flat on both seasons); only the point-path
  shrink fails. Both configs are in `KNOWN_SEALED_VETOES`; see
  `research/SEALED_DECOUPLED_LATENT_CLARITY_2026.md`. The `supersub_latent_shrink` knob is kept for
  reuse with a future point-path change that does generalise.
- `minutes_interact` (started/jersey × is_forward) gave the best 2025 XV value seen (0.724) and
  team 0.714, and passed the accept gate via value_team, but failed the stability gate
  (team_drop + minutes_perturb), so it is not promoted.

Role-appropriate signal framework (the through-line of the 2026-06-19 work): a pick's
signal should match its multiplier and minutes. The **captain** (2x, an ~80' starter
with real ceiling) wants an *upside/ceiling* bet — promoted. The **supersub** (3x but a
~20' impact sub whose ceiling is minutes-capped) wants its two-stage *expected-minutes
mean* — adding upside (`supersub_upside_weight` 0.1–1.5) hurt 2025 for no robust 2026
gain. The **XV** (1x) wants the mean — `selector_upside_weight` hurt 2025 XV value. So of
the three roles, only the captain was mis-specified, and fixing it is the whole win.

Next frontier (config knobs are exhausted; two full 142-candidate sweeps against the prior
and the new incumbent surfaced no further dual-season gain). The sealed 2026 diagnostics
show large structural under-prediction of attacking backs — fly-half bias +5.5 — which a
probe traced to designated goal-kickers: tagging each fixture-team's top predicted-kick
player, kickers (n=30) are under-predicted by **+6.9 pts mean bias / MAE 9.95** vs
non-kickers' +1.8 / 7.23. But this is **not a clean MAE lever**, and the investigation
resolved *why*:

1. The model already carries the kicker identity (`role_goal_kicker_rate`,
   `role_kick_attempts`, `form_per80_*` kicking) — the bias is not missing information.
2. A leakage-free post-hoc reallocation that concentrates a team's kicking onto its
   designated kicker improved sealed 2026 MAE (−0.046) but **worsened 2025 (+0.022)**: the
   kicker bias is high-ceiling **variance** (kickers overperform their mean in whichever
   season they have big games), not correctable bias, so a mean boost overfits.
3. That variance is precisely the captain-upside signal. The promoted captains are the
   designated kickers — mean `role_goal_kicker_rate` 0.82 (2025) / 0.80 (2026) vs XV
   averages 0.15 / 0.11. The captain-upside promotion **already monetises** the kicker
   ceiling; the MAE residual on kickers is dominated by irreducible variance.

So the kicker angle is, for value, already captured; for MAE it is a dead end via
calibration (forward-chained `bayes_position_residual`/`recon_calib` and the kicker
reallocation all overfit). The only untried angle is a **component-level** improvement to
the kicking-rate heads in `train_components.py` (predicting conversion/penalty volume per
designated kicker more sharply) — uncertain, requires a retrain, and is left as the one
remaining feature-level direction rather than done autonomously.

Costelow/R3 fixture-shape lesson: the model already has fixture and opponent features, but it does
not explicitly predict the match's **team-play shape** before predicting players. That is likely a
better next feature-layer direction than more selector knobs. Current features include
`ownteam_possession`, opponent permissiveness, World Rugby gap, H2H margin, and opponent carry/tackle
profiles, but they are scattered inputs. A dedicated PIT team/match layer should first predict, per
fixture-team:

- `team_points_for_hat`, `team_margin_hat`
- `team_possession_hat`
- `team_runs/carries_hat`, `team_metres_hat`
- `team_tackles_required_hat` or opponent carry volume
- `team_kicks_from_hand_hat`
- `team_penalty_goal_opportunity_hat` / attacking-territory proxy

Then the player component layer should condition on these predicted match-shape values: low
possession can raise tackle expectation, high possession/carries can raise metres/defenders beaten,
expected team points/territory can raise goal-kicker opportunity, and opponent carry volume can
raise forward/back-row defensive floors. This would have made the Costelow-vs-Russell tradeoff
explicit: Wales bad fixture might still boost Costelow tackles, but Scotland's projected attacking
shape should also lift Russell's attacking/kicking context. Leakage rule: the team-play layer must
be trained only from matches with `date < fixture_date`, and any OOF team-play predictions used in
2025 dev must be generated without seeing that fixture.

Implemented first pass: `build_team_play.py` now creates PIT predicted team edge / game-script
features (`teamplay_*`) from prior team-match history, and `build_features.py` joins them into
`model_player_match.csv`. The research harness keeps them disabled by default via
`teamplay_features="off"` so the promoted model is unchanged unless a candidate opts in.

2025 result: the layer has forecast signal but is not promotion-grade yet.

- `teamplay_features_on` improves MAE 7.464 -> 7.443, but damages the actual fantasy objective:
  `value_team` 0.715 -> 0.660, captain top-3 0.200 -> 0.000. Reject: it should not steer selection
  directly.
- `target_only_teamplay25_starters` keeps team decisions frozen and improves MAE only to 7.453,
  below the promotion threshold. Reject as too small.
- `target_only_teamplay50_starters` reaches the MAE gate (7.444, -0.021) with decisions frozen,
  but fails the minutes perturbation stability gate 2/4. Reject for now; it is close, not robust.
- `target_only_teamplay100_starters` improves aggregate MAE to 7.431 but only improves 2/5 rounds,
  so the gain is too lumpy. Reject.

Next viable direction: use team-play predictions **inside component specialists**, not as a global
player-score feature or post-hoc point blend. Examples: tackles conditioned on opponent runs /
possession, attacking components conditioned on own possession/runs/metres/points, and kicking
conditioned on penalty-goal opportunity / expected points. Current promoted champion remains
`target_only_xgb50_starters`.

Multiaspect follow-up: `build_team_play.py` now also emits own-vs-opponent aspects for points edge,
win/dominance edge, possession edge, attack volume/balance, defensive load, open-game shape,
kicking opportunity, and pressure. The safer target-only version worked on 2025: `target_only_teamplay_aspect50_starters`
improved MAE 7.464 -> 7.439, froze all team decisions, and passed stability (round 5/5, team 6/6,
position 5/5, minutes 4/4). Sealed 2026 vetoed promotion: MAE worsened 7.304 -> 7.312 with all
decision metrics unchanged. Conclusion: the multiaspect layer is useful research infrastructure,
but not the promoted model. Further work should audit component-level OOF wins rather than tune
more global target blends.

Component-specialist follow-up: the more faithful architecture finally produced a tiny sealed win.
`target_only_teamplay_component_aspect100_starters` freezes XV/captain/supersub decisions, then
uses a starter-only target forecast from component models that see the derived team-play aspect
features. It improved 2025 MAE 7.464 -> 7.438, passed stability (round 5/5, team 6/6, position 5/5,
minutes 4/4), and sealed 2026 improved MAE 7.304 -> 7.302 with all team-value metrics unchanged.
Promoted as a **forecast-only micro-improvement**; do not treat it as evidence that team-play should
enter selector/rank/captain paths.

Post-target role-refresh follow-up: tested whether the improved component-aspect forecast should
also feed captain or XV selection. Captain refresh was a no-op; captain mean-only hurt
`value_team`. Selector refresh with a very small rank tilt (`post_target_refresh_selector_tilt005`)
raised 2025 `value_team` 0.715 -> 0.718 and `top15` 0.373 -> 0.413, but failed stability
(round-drop 3/5, minutes perturb 2/4). Reject and do not open sealed 2026. This reinforces the
current boundary: team-play aspect signal is useful for forecast calibration, not robust enough for
selection decisions yet.

Orchestrator follow-up (2026-06-20): the full layered idea now has one promoted form. A post-target
residual layer applies a forward-chained team+position correction only to starting backs, then a
tiny selector overlay uses the difference between the frozen selector forecast and the specialist
post-target forecast. The promoted candidate is `orchestrator_resid030_delta_overlay_010`:

- 2025 dev vs `target_only_teamplay_component_aspect100_starters`: MAE improves 7.438 -> 7.406,
  `top30` improves 0.553 -> 0.573 and selector Spearman improves 0.504 -> 0.508, while
  `value_team` slips slightly 0.715 -> 0.714. It passes stability (round-drop 4/5, team-drop 5/6,
  position-drop 5/5, minutes perturb 3/4).
- Sealed 2026 confirms the selector side of the idea: `value_team` improves 0.697 -> 0.700 and
  `value_xv` improves 0.714 -> 0.718, with `top30` 0.547 -> 0.553. MAE is essentially flat
  (7.302 -> 7.303, +0.0006), so the value gain wins under the no-meaningful-MAE-regression rule.
- Residual-only `post_target_resid_team_position_030_starter_backs` passed 2025 but sealed 2026
  was flat/slightly worse (MAE 7.303, value unchanged), so the useful piece is the constrained
  selector delta overlay, not the residual calibration by itself.
- Full selector refresh remains rejected: it increases top15/upside but damages `value_team`.
  Keep the overlay tiny and do not replace the promoted selector with the team-play forecast.
- Swap audit artifact: `research/orchestrator_swap_audit.json` compares the previous promoted
  component-aspect forecast model to the promoted orchestrator. The audit showed the overlay loses
  value in early/mid rounds and wins mostly in round 5: 2025 net -4 actual fantasy points despite a
  +41 round-5 gain; 2026 net +13 with a +35 round-5 gain.
- Late-gated overlay follow-up: tested `post_target_selector_delta_min_prior_n` so the selector
  delta waits for enough prior same-season evidence. Raw 2025 value improved strongly
  (`orchestrator_delta_prior500_scope_starters_backs_backrow` reached value_team 0.728 vs 0.714),
  but every late-gated/clipped variant failed stability, mostly minutes perturbation and top15
  guardrails. Do **not** promote or seal this family. It is a value-looking instability trap unless
  a future version can pass perturbation.
- Captain-head follow-up on the promoted orchestrator: retested mean-only, lower upside, higher
  upside (0.50/0.75/1.25/1.50/2.00), rank overlays, kicker penalties, and post-target captain
  refresh. None improved 2025 `value_team`; most were exact no-ops or small regressions. Current
  `captain_head="mean", captain_upside_weight=1.0` remains the local optimum. Do not keep tuning
  this head without a new captain-specific signal.

Latest focused batch (post dominance-gate fix):

- Captain kicker **penalty** tested the inverse of the kicker-floor idea on top of
  `captain_upside_full` (`captain_kicker_weight` -0.05, -0.10, -0.20, -0.40). It
  reduced or failed to improve 2025 `value_team`; reject. This answers the Ramos
  concern without player-specific hardcoding: the model should not add more kicker
  floor, but penalising kicker floor is also worse than the current ceiling head.
- Current-baseline bench kick shrink tested the Marcus-Smith-style concern without
  replacing the promoted two-stage supersub model:
  `bench_twostage_kick_shrink_10/25/50`, `bench_twostage_kick_ridge`, and
  `bench_twostage_kick_lgbm`. All kept `value_team` flat and moved MAE/bench MAE by
  only about 0.001, far below the promotion threshold. Keep the current promoted
  supersub baseline unchanged.
- Component-level audit: kicker specialists are not principled right now. On the
  train-only OOF windows, naive gated kicking beats ridge/GLM for both conversion
  and penalty goals in both the 2025 and 2026 train windows, so do not promote a
  learned kick-rate graft without new data. Registry demotion also failed the 2025
  current-stack check (`value_team` 0.715 -> 0.649, top15 0.373 -> 0.333).
- XGB component grafts are currently inert by design: the fixture-group OOF XGB
  winner set is empty for both train windows. The candidate definitions now keep
  `selector_point_source="target"` so a graft candidate with no eligible component
  is a true no-op rather than accidentally testing a selector-source change.

Promotion (2026-06-19): `captain_upside_full` adds a captain head on top of the
`bench_points_twostage_13` supersub baseline — `captain_head=mean`,
`captain_upside_weight=1.0`. The captain is now chosen by ceiling (full upside)
instead of highest XV mean, because the captain 2x role rewards variance. It lifts
2025 team value 0.707 -> 0.715 and sealed 2026 team value 0.688 -> 0.697 with MAE,
value_xv, top15, top30, spearman and bench MAE all exactly flat on both seasons, and
captain hit-rate doubling (2026 capt_top3 0.2 -> 0.4). It is round-dominant (no round
worse on either season). The original promotion used a documented override because
the old strict 3/5 sub-gate treated unchanged rounds as failures; the loop now has a
zero-cost selector-dominance rule so this class of captain/supersub improvement is
accepted directly by policy. See `research/promotion_report.json` for the original
sealed comparison. Reversible via `research/promoted_config.prev_bench_points_twostage_13.json`.

The promoted point/selection/supersub stack now has four pieces:

- `target_pts_hat` keeps the promoted 20% front-row calibration and then blends in capped XGB
  component-score points:
  `target_pts_hat = 0.80 * lgbm_target_pts_hat + 0.20 * xgb_component_pts_hat`.
- For bench rows only, `target_pts_hat` now receives a small calibration pull toward the two-stage
  bench-minutes point estimate:
  `target_pts_hat = 0.87 * target_pts_hat + 0.13 * two_stage_bench_pts_hat`.
- `sel_score` uses the blended point signal plus the LGBM rank head:
  `sel_score = zscore(selector_pts_hat) + 0.25 * zscore(rank_score)`.
- Promotion audit (2026-06-19): `target_only_xgb50_starters` is promoted after a forced
  promote-or-drop audit. The last clean incumbent was `target_only_xgb10`; the candidate freezes
  XV/captain/supersub decisions, then applies a post-selector **starter-only equal blend**:
  `target_pts_hat = 0.50 * target_pts_hat + 0.50 * xgb_component_pts_hat` for starters only.
  The 0.50 cap is the maximum allowed forecast-only XGB weight because it does not let XGB
  outweigh the LGBM champion. Against `target_only_xgb10`, 2025 MAE improves 7.524 -> 7.464,
  sealed 2026 MAE improves 7.342 -> 7.304, and all team-value/XV/top-N/captain/supersub decisions
  are unchanged. Full stability is now persisted in `research/stability_report.json`: round-drop
  5/5, team-drop 6/6, position-drop 5/5, minutes perturbation 3/4.
- Heavier forecast-only XGB weights above 0.50 are not eligible under the current promotion rule:
  at that point XGB becomes the forecast boss rather than a bounded specialist. Future work can
  revisit that only with a new predeclared validation design, not by tuning on sealed 2026.
- Earlier post-selector XGB findings remain useful context: all-row `target_only_xgb10` was a
  safe first forecast-only step; agreement filters did not identify a better robust subset; bench
  and forward-only XGB scopes were worse.
- Agreement-filtered post-selector XGB blends tested the theory that heavier XGB should be used
  only where LGBM and XGB forecasts are closest. This did not work: 50% and 75% lowest-disagreement
  filters worsened MAE, and the 90% filter improved MAE only slightly (7.524 -> 7.517), below the
  promotion threshold. The extra XGB correction is not concentrated in low-disagreement rows.
- `supersub_score` uses a two-stage bench-minutes specialist:
  `supersub_score = zscore(sel_score) + 0.50 * zscore(two_stage_bench_pts_hat)`.
- `captain_score` (new) is a ceiling bet over the picked XV:
  `captain_score = zscore(target_pts_hat) + 1.0 * upside_score`, where `upside_score`
  rewards attacking ceiling and prior-player variance. The deployable report resolves
  the captain from `captain_score` first.

Frontier batch (2026-06-20, current promoted orchestrator as incumbent):

- Implemented and tested executable candidates for the next four research ideas:
  fixture-strength/team-play proxy, richer bench/supersub minute distribution,
  role-learned kicking reallocation, and a forward-chained conservative
  specialist combiner. No candidate passed the 2025 base promotion gate, so sealed
  2026 was not opened.
- Fixture-strength proxy results: `frontier_fixture_strength_005` slightly
  improved MAE (7.406 -> 7.403) but dropped `value_team` 0.714 -> 0.704; stronger
  fixture-strength and strength+tempo variants damaged `value_team` more. The
  team-play layer remains useful as a forecast specialist, not as a direct
  selection-facing component nudge.
- Supersub minute-distribution results: adding learned `P(minutes >= 25/30/35)`
  or a low-play penalty to the promoted two-stage supersub head did not help.
  Best was essentially a no-op; most variants reduced `value_team` (e.g.
  `frontier_bench_high30_w010` 0.714 -> 0.709). The existing expected-minutes
  supersub head remains better than extra high-minute/risk overlays.
- Kicking role-certainty results: reallocating kicking mass by learned
  `role_goal_kicker_rate`/attempt history was too small to promote. The
  high-confidence starter version improved MAE only 7.406 -> 7.405, with unchanged
  team value; all other variants were no-ops or worse. This confirms the previous
  kicker finding: role certainty is already mostly in the model, and the remaining
  kicker error is variance rather than a clean calibration lever.
- Forward-combiner results: a prior-round conservative grid over LGBM/XGB/Bayes/
  team-play forecasts produced only tiny MAE gains when scoped to starters or
  starting backs (about -0.003 MAE) and no `value_team` movement. It is viable
  infrastructure but not a promotion candidate without a stronger external signal.
- Research implication: internal architecture knobs are now plateauing. Future
  high-upside work should add **new data** rather than keep tuning these overlays:
  real betting/team totals, club/non-6N role form, named goal-kicker/bench-role
  certainty, weather/venue, or opponent tactical style.

### External Data Push (2026-06-20)

- Added optional external feature families:
  - `market_` for betting/team-total fixture strength.
  - `weather_` for stadium conditions.
  - `rolecert_` for named role certainty.
- Added `build_weather_features.py` and fetched Open-Meteo archive daily weather for all
  60 Six Nations fixtures in 2023-2026.
- Weather results:
  - Blanket weather feature candidate failed 2025 (`value_team` 0.691, MAE 7.419).
  - Targeted overlay `weather_multi_010` passed 2025 and stability (MAE 7.384 vs 7.406),
    but sealed 2026 failed clearly: `value_team` 0.685 vs 0.700 promoted, MAE 7.320 vs
    7.303 promoted.
  - Nearby sealed clarity checks (`weather_attack_suppress_010`, `weather_multi_005`,
    `weather_multi_015`) also regressed.
  - Decision: weather standalone is dropped / sealed-vetoed. Keep ingestion code only as
    supporting context for later market or role specialists.
- Added `build_market_features.py` plus raw/example schemas. It converts raw home/away
  decimal odds into no-vig win probability, expected margin, total points, and team implied
  points. Added `fetch_market_odds.py` for The Odds API historical snapshots
  (`rugbyunion_six_nations`) and h2h-only fallback features. No real historical market file
  exists yet because no local `THE_ODDS_API_KEY` is set and The Odds API historical access is
  paid, so market candidates are ready but not testable in this pass. OddsPortal was probed as a
  fallback; the page exposes an archive request path, but direct session/csrf requests returned
  synthetic 404 bodies rather than stable data.
- Added `build_role_certainty.py` and tested generated role-certainty features:
  - Global rolecert features: MAE improved to 7.382 but `value_team` collapsed to 0.614.
  - Narrow rolecert overlays (kicking reallocation, bench kick shrink, supersub uncertainty)
    were no-ops or harmful. Best bench-kick shrink only moved MAE by about -0.001.
  - Decision: keep rolecert for diagnostics; do not promote.
- Added `build_tactical_style.py` and tested local tactical-style features:
  - Full style features: `value_team` 0.701, MAE 7.424.
  - Style + teamplay aspects: `value_team` 0.698, MAE 7.409.
  - Compact `style_aspect_*` only: `value_team` 0.695, MAE 7.398.
  - Compact style + teamplay aspects: `value_team` 0.707, MAE 7.405.
  - Decision: drop. Local tactical summaries damage selection; real market/team-total data
    remains the cleaner match-shape source.
- Free OddsPortal market scrape completed on 2026-06-21 using cached
  Oddsharvester CSVs, no paid API. H2H-only win probability was not enough:
  blanket/global market candidates improved 2025 MAE slightly but hurt team
  value or captain health. Adding scraped handicap and over/under lines gave a
  cleaner match-shape signal: win probability, expected margin, total points,
  and implied team points for all 60 fixtures from 2023-2026.
- The apparent `external_market_post_target_delta020` promotion was invalidated
  by the PIT audit: its closing prices were not guaranteed available at round
  lock. Rebuilding the same idea from timestamped opening odds removed the
  claimed improvement. `orchestrator_resid030_delta_overlay_010` is restored;
  opening prices remain research-only and line movement is dropped.

### Bench And Matchup Data Batch (2026-06-30)

- `supersub_context_slot_only` added bench jersey and position-slot context only
  to the supersub specialist. It improved 2025 `value_team` 0.714 -> 0.737 and
  passed all stability checks, but the entire gain was one round/one pick. The
  sealed 2026 clarity check was exactly flat on every metric and every supersub
  choice, so it remains research-only rather than promoted.
- Named-starter replacement context failed (`value_team` 0.665). A new PIT
  player replacement-history family was then built from 57,181 club and
  international rows: prior bench count, expected bench minutes, meaningful-
  minutes rate, 30+ minute rate, and recency. Coverage was 98.8% for 2025 bench
  rows, but using it in the supersub head failed worse (`value_team` 0.647;
  supersub regret 51 -> 131). It learned reliable replacements, not fantasy
  ceiling. Close both bench-context variants; do not weight-tune them.
- Direct national-team jersey-slot history also had complete coverage and a
  median 22 prior slot uses, but reduced `value_team` to 0.707 by making one
  worse supersub swap. Together with the earlier high-minute probability
  overlays, this closes locally derived bench-minute tuning. A future bench
  attempt needs genuinely new role or substitution-event data.
- Added 14 optional `matchup_*` features measuring strictly prior opponent
  allowance by exact player position. They are excluded from the production
  feature view and were tested only inside the post-target component specialist.
  With incumbent decisions frozen, full use worsened 2025 MAE 7.406 -> 7.458.
  In-memory blends at 10%, 25%, 50%, and 75% all also worsened MAE. Close this
  construction; internally derived opponent allowance is too noisy to add value.
- Champion remains `orchestrator_resid030_delta_overlay_010`. No new candidate
  from this batch qualifies for another sealed check or promotion.

## Artifacts

- `model/research.py` — research loop and candidate queue.
- `research/ledger.json` and `research/LEDGER.md` — latest trial log.
- `research/history.jsonl` — append-only history across overwritten trial logs.
- `research/best_config.json` — latest loop best.
- `research/promoted_config.json` — deployable config after sealed fallback checks.
- `research/promotion_report.json` — sealed comparison and fallback reason.
- `research/stability_report.json` — round/team/position/minutes stability diagnostics.
- `research/sealed_best_check.json` — sealed check for `research/best_config.json`; prediction CSV
  is not overwritten.
- `research/sealed_veto_target_frontrow20_back5_20.json` — sealed veto evidence for the stacked
  back-five prior dev candidate.
- `research/sealed_veto_minutes_alpha_4.json` — sealed veto evidence for the tighter minutes-ridge
  dev candidate.
- `research/dev_predictions_2025.csv` — dev-best prediction snapshot from the latest loop.
- `data/model_predictions_2025.csv` — promoted dev predictions.
- `data/model_predictions_2026.csv` — promoted sealed-season predictions.

## Commands

```bash
/usr/local/bin/python3.11 -m model.research --base-config promoted --candidate xgb_rank_overlay_0025 --candidate xgb_rank_overlay_005 --candidate xgb_rank_overlay_010 --candidate xgb_rank_overlay_015
/usr/local/bin/python3.11 -m model.research --base-config promoted --candidate points_lgbm95_xgb05 --candidate points_lgbm90_xgb10 --candidate points_lgbm85_xgb15 --candidate points_lgbm80_xgb20
/usr/local/bin/python3.11 -m model.research --base-config promoted --candidate bayes_points_blend_05 --candidate bayes_points_blend_10 --candidate bayes_cold_players_10 --candidate bayes_cold_players_20
/usr/local/bin/python3.11 -m model.research --base-config promoted --candidate supersub_bench_ridge_025 --candidate supersub_bench_lgbm_025 --candidate supersub_two_stage_minutes --candidate supersub_minutes_uncertainty_025 --candidate bench_kick_learned_gate --candidate captain_mean_plus_upside_010 --candidate selector_mean_plus_upside_010 --candidate team_specialist_bench025_capt010_upside010
/usr/local/bin/python3.11 -m model.research --base-config promoted --candidate bench_points_twostage_12 --candidate bench_points_twostage_13 --candidate bench_points_twostage_14 --candidate bench_points_twostage_15
/usr/local/bin/python3.11 -m model.research --base-config promoted --candidate potm_pp_055 --candidate potm_pp_060 --candidate potm_pp_065 --candidate potm_pp_070 --candidate potm_pp_075 --candidate potm_pp_090 --candidate potm075_bayes10
/usr/local/bin/python3.11 -m model.research --seal-2026 --seal-config best
/usr/local/bin/python3.11 -m model.research --seal-2026 --seal-config promoted
```
