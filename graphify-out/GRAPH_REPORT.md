# Graph Report - 6n  (2026-06-16)

## Corpus Check
- 27 files · ~29,689 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 332 nodes · 526 edges · 20 communities (19 shown, 1 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 59 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8f4ce95f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 19|Community 19]]

## God Nodes (most connected - your core abstractions)
1. `Six Nations Picker — Model Training Plan` - 15 edges
2. `assemble_predictions()` - 14 edges
3. `RugbyAPI` - 12 edges
4. `main()` - 12 edges
5. `Six Nations Picker — Complete Data Extraction (Once-and-Done)` - 12 edges
6. `ingest()` - 10 edges
7. `predict_rates()` - 10 edges
8. `_selfcheck()` - 10 edges
9. `_surname_pass()` - 9 edges
10. `flatten_team_stats()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `slug_key()` --calls--> `norm_key()`  [INFERRED]
  compare_three_way.py → compare_api_official.py
- `agg_official()` --calls--> `parse_official()`  [INFERRED]
  compare_three_way.py → compare_api_official.py
- `score_recon()` --calls--> `score_components()`  [INFERRED]
  model/baselines.py → build_targets.py
- `fit_latent_asof()` --calls--> `_fit_latent()`  [INFERRED]
  model/assemble.py → build_targets.py
- `QuotaExhausted` --uses--> `RugbyAPI`  [INFERRED]
  ingest_6n.py → rugby_api.py

## Communities (20 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (34): assemble_predictions(), _expected_drives(), _expected_potm(), fit_latent_asof(), rank_scores(), Within-match LightGBM lambdarank score for the test rows.      Trained to rank p, Fit the latent decomposition on the forward-chained modern subset     (date < as, PIT expected SW / LS drives (NO realised team counts). (+26 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (24): add_position_and_recon(), derive_minutes(), _flatten_one_team_stats(), flatten_team_stats(), _in_form_window(), ingest(), main(), _num() (+16 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (29): build(), _consolidate_name_variants(), _fold(), last_token(), load_api(), load_official(), load_rp(), main() (+21 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (27): 2025 backtest (train 2023+2024, post_team_sheet), 2026 deployment (train 2023+2024+2025, post_team_sheet) — sealed, evaluated once, Allowed APIs (cite when used; do not invent), B3 direct-points cross-check (requirement: Strategy C justification), Context, Environment (VERIFIED 2026-06-16), Execution Record / Results (2026-06-16), Final Phase — Verification gate (+19 more)

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (21): build_matrix(), direct_points_oof(), expected_minutes(), numeric_cols(), _position_rate_prior(), PositionMeanImputer, _rate_target(), _rates_linear() (+13 more)

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (22): _selfcheck(), predict_rates(), Per-80 rate predictions (index=test_idx, cols=SCORED) for a baseline engine., build_predictors(), _fit_lgbm_component(), _kick_gate(), _lgbm_frame(), _lgbm_objective() (+14 more)

### Community 6 - "Community 6"
Cohesion: 0.19
Nodes (18): build_bio(), build_compstats(), build_matchlog(), load_json_files(), main(), _num(), _parse_age(), _parse_height_cm() (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.17
Nodes (17): fetch(), fold(), main(), Return (status_code, html_or_None) for a RugbyPass player slug., ASCII-fold + lowercase: 'Sébastien' -> 'sebastien'., Ordered RugbyPass slug guesses for an API player name., API players present in the crosswalk but not linked to RugbyPass., resolve_and_scrape() (+9 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (16): _bool(), build(), class_features(), _decay_weights(), form_role_features(), main(), ownteam_features(), Minutes-weighted per-80 rates over the player's **completed prior**     RugbyPas (+8 more)

### Community 9 - "Community 9"
Cohesion: 0.12
Nodes (16): Future updates (the "months later" upkeep — not part of this once-and-done), Official fantasy labels (manual xlsx), Phase 0 — Allowed APIs / Sources (READ FIRST, do not invent), Phase 1 — Repair & consolidate FREE sources (0 API spend), Phase 2 — Free international widening (Pulselive, 0 paid quota), Phase 3 — `build_crosswalk.py` (new) → `data/player_crosswalk.csv`, Phase 4 — PAID club backfill (the single spend) 🟡, Phase 5 — `build_features.py` (new) → `data/model_player_match.csv` (0 API) (+8 more)

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (14): build(), _decayed_mean(), _load_h2h_meetings(), main(), _merge_wr(), Exponentially weighted mean, weight = 0.5 ** (days_ago / half_life)., Exponentially weighted mean, weight = 0.5 ** (days_ago / half_life)., Exponentially weighted mean, weight = 0.5 ** (days_ago / half_life). (+6 more)

### Community 11 - "Community 11"
Cohesion: 0.23
Nodes (12): build(), main(), parse_official_2023(), _player_nation_map(), Build {norm_key(name): nation} from the 2023 'Player List' sheet., Parse the 2023 split-sheet layout (Rounds 1-4) -> tidy long frame.      Reads ea, compare(), load_api() (+4 more)

### Community 12 - "Community 12"
Cohesion: 0.23
Nodes (11): build(), _fit_latent(), main(), Cross-check: for modern labelled rows, the residual (official − recon)     shoul, Cross-check: for modern labelled rows, the residual (official − recon)     shoul, Fit the latent decomposition on MODERN labelled rows (2025+2026) — the     only, Deterministic fantasy score under the **modern** (2025/2026) official     formul, Deterministic fantasy score under the **modern** (2025/2026) official     formul (+3 more)

### Community 13 - "Community 13"
Cohesion: 0.25
Nodes (10): fetch_chunk(), _get(), main(), month_chunks(), Two mirrored rows (one per side) for a senior men's match where at least     one, 6N-style season label: a match in Jul..Dec belongs to the NEXT calendar     seas, Cached, paginated GET of all completed matches in [start, end].      Caches the, Calendar-month [first, last] chunks covering [start, end]. Month-sized     keeps (+2 more)

### Community 14 - "Community 14"
Cohesion: 0.33
Nodes (8): extract_bio(), extract_comp_stats(), extract_match_log(), main(), make_driver(), Parse #app-competitions → flat list of per-game records., Parse player bio from .player-details., Parse #app-comp-stats → list of per-competition season aggregates.     Returns a

### Community 15 - "Community 15"
Cohesion: 0.33
Nodes (8): _assert_no_leakage(), _feature_cols(), feature_view(), load(), Return the column list to use as model inputs for a given mode.      mode == "pr, The 91 family-prefix feature columns, preserving store order., Read both stores, inner-join 1:1 on (fixture_id, player_id), parse date.      Ta, _selfcheck()

### Community 16 - "Community 16"
Cohesion: 0.6
Nodes (4): fetch_snapshot(), main(), Cached GET of the WR ranking as of `date` (YYYY-MM-DD)., snapshot_rows()

## Knowledge Gaps
- **137 isolated node(s):** `Cached GET of the WR ranking as of `date` (YYYY-MM-DD).`, `Cached, paginated GET of all completed matches in [start, end].      Caches the`, `Calendar-month [first, last] chunks covering [start, end]. Month-sized     keeps`, `Two mirrored rows (one per side) for a senior men's match where at least     one`, `6N-style season label: a match in Jul..Dec belongs to the NEXT calendar     seas` (+132 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `assemble_predictions()` connect `Community 0` to `Community 4`, `Community 5`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Why does `score_recon()` connect `Community 4` to `Community 0`, `Community 12`, `Community 5`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 0` to `Community 5`, `Community 15`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `assemble_predictions()` (e.g. with `run_season()` and `main()`) actually correct?**
  _`assemble_predictions()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `RugbyAPI` (e.g. with `QuotaExhausted` and `ingest()`) actually correct?**
  _`RugbyAPI` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `main()` (e.g. with `load()` and `component_train()`) actually correct?**
  _`main()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Cached GET of the WR ranking as of `date` (YYYY-MM-DD).`, `Cached, paginated GET of all completed matches in [start, end].      Caches the`, `Calendar-month [first, last] chunks covering [start, end]. Month-sized     keeps` to the rest of the system?**
  _137 weakly-connected nodes found - possible documentation gaps or missing edges._