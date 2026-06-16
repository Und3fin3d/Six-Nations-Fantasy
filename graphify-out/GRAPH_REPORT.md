# Graph Report - 6n  (2026-06-16)

## Corpus Check
- 18 files · ~19,200 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 208 nodes · 302 edges · 16 communities
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e391e030`
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

## God Nodes (most connected - your core abstractions)
1. `RugbyAPI` - 12 edges
2. `Six Nations Picker — Complete Data Extraction (Once-and-Done)` - 12 edges
3. `ingest()` - 10 edges
4. `_surname_pass()` - 9 edges
5. `flatten_team_stats()` - 9 edges
6. `main()` - 7 edges
7. `_surname_candidates()` - 7 edges
8. `_consolidate_name_variants()` - 7 edges
9. `slug_key()` - 6 edges
10. `form_role_features()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `QuotaExhausted` --uses--> `RugbyAPI`  [INFERRED]
  ingest_6n.py → rugby_api.py
- `ingest()` --calls--> `RugbyAPI`  [INFERRED]
  ingest_6n.py → rugby_api.py
- `flatten_team_stats()` --calls--> `RugbyAPI`  [INFERRED]
  ingest_6n.py → rugby_api.py
- `slug_key()` --calls--> `norm_key()`  [INFERRED]
  compare_three_way.py → compare_api_official.py
- `agg_official()` --calls--> `parse_official()`  [INFERRED]
  compare_three_way.py → compare_api_official.py

## Communities (16 total, 0 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (29): build(), _consolidate_name_variants(), _fold(), last_token(), load_api(), load_official(), load_rp(), main() (+21 more)

### Community 1 - "Community 1"
Cohesion: 0.16
Nodes (21): add_position_and_recon(), derive_minutes(), _flatten_one_team_stats(), flatten_team_stats(), _in_form_window(), ingest(), main(), _num() (+13 more)

### Community 2 - "Community 2"
Cohesion: 0.19
Nodes (18): build_bio(), build_compstats(), build_matchlog(), load_json_files(), main(), _num(), _parse_age(), _parse_height_cm() (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.17
Nodes (16): _bool(), build(), class_features(), _decay_weights(), form_role_features(), main(), ownteam_features(), Minutes-weighted per-80 rates over the player's **completed prior**     RugbyPas (+8 more)

### Community 4 - "Community 4"
Cohesion: 0.12
Nodes (16): Future updates (the "months later" upkeep — not part of this once-and-done), Official fantasy labels (manual xlsx), Phase 0 — Allowed APIs / Sources (READ FIRST, do not invent), Phase 1 — Repair & consolidate FREE sources (0 API spend), Phase 2 — Free international widening (Pulselive, 0 paid quota), Phase 3 — `build_crosswalk.py` (new) → `data/player_crosswalk.csv`, Phase 4 — PAID club backfill (the single spend) 🟡, Phase 5 — `build_features.py` (new) → `data/model_player_match.csv` (0 API) (+8 more)

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (14): build(), _decayed_mean(), _load_h2h_meetings(), main(), _merge_wr(), Exponentially weighted mean, weight = 0.5 ** (days_ago / half_life)., Exponentially weighted mean, weight = 0.5 ** (days_ago / half_life)., Exponentially weighted mean, weight = 0.5 ** (days_ago / half_life). (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.23
Nodes (12): build(), main(), parse_official_2023(), _player_nation_map(), Build {norm_key(name): nation} from the 2023 'Player List' sheet., Parse the 2023 split-sheet layout (Rounds 1-4) -> tidy long frame.      Reads ea, compare(), load_api() (+4 more)

### Community 7 - "Community 7"
Cohesion: 0.27
Nodes (3): GET an endpoint path (e.g. '/match/123'), with permanent disk cache., RugbyAPI, RuntimeError

### Community 8 - "Community 8"
Cohesion: 0.23
Nodes (11): build(), _fit_latent(), main(), Cross-check: for modern labelled rows, the residual (official − recon)     shoul, Cross-check: for modern labelled rows, the residual (official − recon)     shoul, Fit the latent decomposition on MODERN labelled rows (2025+2026) — the     only, Deterministic fantasy score under the **modern** (2025/2026) official     formul, Deterministic fantasy score under the **modern** (2025/2026) official     formul (+3 more)

### Community 9 - "Community 9"
Cohesion: 0.25
Nodes (10): fetch_chunk(), _get(), main(), month_chunks(), Two mirrored rows (one per side) for a senior men's match where at least     one, 6N-style season label: a match in Jul..Dec belongs to the NEXT calendar     seas, Cached, paginated GET of all completed matches in [start, end].      Caches the, Calendar-month [first, last] chunks covering [start, end]. Month-sized     keeps (+2 more)

### Community 10 - "Community 10"
Cohesion: 0.27
Nodes (10): fetch(), fold(), main(), Return (status_code, html_or_None) for a RugbyPass player slug., ASCII-fold + lowercase: 'Sébastien' -> 'sebastien'., Ordered RugbyPass slug guesses for an API player name., API players present in the crosswalk but not linked to RugbyPass., resolve_and_scrape() (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (8): extract_bio(), extract_comp_stats(), extract_match_log(), main(), make_driver(), Parse #app-competitions → flat list of per-game records., Parse player bio from .player-details., Parse #app-comp-stats → list of per-competition season aggregates.     Returns a

### Community 12 - "Community 12"
Cohesion: 0.46
Nodes (7): extract_bio(), extract_comp_stats(), extract_match_log(), get_unique_players(), is_six_nations(), main(), make_driver()

### Community 13 - "Community 13"
Cohesion: 0.6
Nodes (4): fetch_snapshot(), main(), Cached GET of the WR ranking as of `date` (YYYY-MM-DD)., snapshot_rows()

## Knowledge Gaps
- **81 isolated node(s):** `Cached GET of the WR ranking as of `date` (YYYY-MM-DD).`, `Cached, paginated GET of all completed matches in [start, end].      Caches the`, `Calendar-month [first, last] chunks covering [start, end]. Month-sized     keeps`, `Two mirrored rows (one per side) for a senior men's match where at least     one`, `6N-style season label: a match in Jul..Dec belongs to the NEXT calendar     seas` (+76 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `slug_key()` connect `Community 2` to `Community 6`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `norm_key()` connect `Community 6` to `Community 2`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Why does `RugbyAPI` connect `Community 7` to `Community 1`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `RugbyAPI` (e.g. with `QuotaExhausted` and `ingest()`) actually correct?**
  _`RugbyAPI` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Cached GET of the WR ranking as of `date` (YYYY-MM-DD).`, `Cached, paginated GET of all completed matches in [start, end].      Caches the`, `Calendar-month [first, last] chunks covering [start, end]. Month-sized     keeps` to the rest of the system?**
  _81 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.11 - nodes in this community are weakly interconnected._
- **Should `Community 4` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._