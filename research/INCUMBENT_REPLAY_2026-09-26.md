# Six Nations incumbent replay

The deployed Six Nations forecast is `orchestrator_resid030_delta_overlay_010` from `research/promoted_config.json`. `gw_update.sh` calls `model.run`, which replaces its diagnostic forecasts with `_predict_config` outputs. The PR #24 `empirical_baseline` is a different model.

The incumbent combines component forecasts, minutes, forward latent calibration, specialist point corrections and separate XV, captain and supersub scores. The replay retains every head. All 1,380 modern candidate keys match the native feature store at `(fixture_id, player_id, team)`. No missing-row forecast or fallback is needed.

## Replay contract

`research.incumbent_replay` uses the exact PR #24 canonical store and archived forecasts. Its store hash must match the archived manifest. Every model uses the same 138-player pool for each slate and the same official labels. Eighteen outcomes remain unknown. An unknown selected outcome makes the squad total unknown. Individual error metrics use observed outcomes and retain their denominator.

Every incumbent invocation excludes later rounds before model fitting and standardisation. Current-round labels are masked before forecasting. Earlier rounds remain available to the configured residual calibration. Components retain the deployed prior-season training policy. This is a replay of the current configuration on previously consulted outcomes, not historically issued forecasts or independent validation.

Two feature variants are retained:

- `native_store` uses the frozen native feature values. These were built relative to each fixture date and are compatibility evidence.
- `lock_rebuilt` rebuilds match-derived form, role, bench, own-team and team-play features using the existing constructors and history available more than three hours before the common slate lock. Team-play training cannot consume earlier results from the same slate.

Both variants preserve the native class, bio, fixture-context and position definitions. Historical publication versions are unavailable. Rebuilding from current source histories combines source-refresh and cutoff effects. It does not identify their separate contributions. The native model trains on prior Six Nations seasons while PR #24 candidates use broader histories. These are model-pipeline differences, recorded in the manifests.

A later audit found that the native captain variance target also consumed globally fitted latent targets from later modern outcomes. Therefore the original `lock_rebuilt` captain output is compatibility evidence. The separate causal captain replay below repairs that path. The point forecasts used by the shared optimiser are unaffected under the promoted configuration.

## Decision policies

`*_common` sends calibrated point forecasts through the existing shared optimiser. It uses the PR #24 roles, 15 positional slots, a captain, a bench supersub and at most four players from a country. Prices remain unavailable, so these are price-free diagnostics.

`*_deployed_policy` preserves the native positional selector and separate captain/supersub heads on the full pool. `*_common_roles_policy` changes only its selection positions to the PR #24 positions. The two role maps differ on 52 appearances. Native policy scores do not enforce the country limit and cannot establish legal-squad superiority. Every output records the number of countries exceeding the common four-player cap.

The report consumer, `report_2026_prediction_xv.py`, previously removed unknown outcomes before selection and used actual points to break tied forecast scores. It now selects from complete finite forecast pools, uses forecast and identity information for ties, and preserves unknown totals. It identifies its output as a position-quota diagnostic without verified country or budget constraints.

## Measured comparison

All totals below use the common optimiser and PR #24 role metadata. Historical prices remain unavailable.

| Forecast | 2025 points | 2026 points |
|---|---:|---:|
| PR #24 empirical baseline | 2,410 | 2,552 |
| Native-store incumbent | 2,250 | 2,529 |
| Lock-rebuilt incumbent | 2,260 | 2,539 |
| PR #24 robust blend | 2,356 | 2,496 |
| PR #24 rolling blend | 2,290 | Unknown |
| PR #24 tournament-frozen blend | 2,361 | Unknown |
| PR #24 weighted blend | 2,236 | 2,513 |

The robust blend gains 96 points over the lock-rebuilt incumbent in 2025 and loses 43 in 2026. Both 2026 unknown totals include one selected unknown outcome. This comparison does not establish a shared-model improvement over the deployed incumbent.

The original native selector produces 2,626/2,686 points with lock-rebuilt features. It exceeds four players per country in 2025 rounds 2, 3 and 5, and all five 2026 rounds. Examples include seven Wales players in 2025 round 2 and seven each from England and France in 2026 round 2. These totals cannot be compared as feasible-squad gains. `policy_legality.csv` lists every violation.

## Causal captain correction

`_prior_player_variance` previously called `_bench_target_points`. For nonmodern or unlabelled history, this preferred `target_pts`, whose latent decomposition was fitted on all modern outcomes. The production variance function now uses a known modern official outcome, otherwise deterministic reconstruction. The other consumer of `_bench_target_points` is the direct learned bench head; the promoted two-stage bench path returns before that call. Selector and supersub upside weights are zero. The correction therefore changes only the promoted captain/upside scores.

`research.incumbent_captain_replay` reconstructs each visible forecast prefix from saved component heads. It reproduces the legacy current-round upside within `1e-10` before correcting it. All point, raw-component, selector and supersub heads remain identical. Original forecast files remain untouched.

Corrected native-role policy totals are 2,600/2,660. Using common role metadata gives 2,611/2,625. Both policies still fail the country constraint in three 2025 rounds and every 2026 round. These are policy diagnostics, not a promotion result. Corrected files are under `/tmp/6n-decision-workflow/incumbent/causal-captain`.

## Reproduction and files

Run with the pinned model environment:

```sh
python -m research.incumbent_replay --store CANONICAL_PLAYER_MATCH_CSV --evidence EXTRACTED_PR24_ARCHIVES --output NEW_OUTPUT_DIRECTORY
```

Each slate writes full forecasts, shared-optimiser squads, native-policy squads and feature/cutoff audits. `forecasts.csv`, `metrics.csv` and `summary.csv` combine the slates. `manifest.json` records sources, effective configuration, data and archived-prediction hashes. The current-round `official_pts`, `has_label` and `is_modern` fields retain their masked inference values; use `actual` for evaluation.

The experiment output is `/tmp/6n-decision-workflow/incumbent/full`. `first-slate` is the initial representative run. The original PR #24 evidence remains unchanged.

`manifest.json` records sources at fit start. `fit_sources` preserves unchanged dependencies and the exact reconstructed fit-time replay module. `source_snapshots.json` identifies files changed during the run. `aggregation_manifest.json` records the later addition of eight frozen-control results stored in the original first-round archives. These aggregation changes did not refit forecasts. The 2025 first-round forecasts were byte-identical across the first-slate and full runs. A later-source 2026 round-5 replay also reproduced both original variants byte-for-byte before the captain correction.

After the production variance correction, a direct 2026 round-5 fit reproduced the repaired point and role heads within `3.56e-15`, with identical selected XV, captain and supersub identities. This verifies one representative final-source slate; it is not a claim that the complete original run used the final source. `final_source_verification.json` records that boundary and confirms the original forecast hash stayed unchanged. Nineteen existing specialist/history tests passed. Complexity checks found no new findings; `model/research.py` retains seven existing findings outside the changed function.

## Operational boundary

Round-prefix prediction prevents future-round normalisation. Reusing `IncumbentFeatureHistory.lock_features` also prevents current-slate match results entering rebuilt features. A general live Six Nations entry point still needs an explicit target teamsheet/timestamp interface: `model.data.load` requires exactly 2,760 historical rows and `model.run` hardcodes 2025/2026. Historical prices, publication snapshots and future teamsheets cannot be inferred from these stores. The replay does not change production routing.
