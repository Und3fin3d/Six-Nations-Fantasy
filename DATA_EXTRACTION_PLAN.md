# Six Nations Picker — Complete Data Extraction (Once-and-Done)

**Goal:** produce a *frozen*, model-ready feature + label store for the 6N fantasy picker, with a
single bounded paid-API spend (fits one Pro month). After this runs, no API pulls are needed for
**months** — only an in-season `update` per round (see "Future updates"). Then we train.

**End state (the only two files the model reads):**
- `data/model_player_match.csv` — one row per (player, 6N match, season) for **2023–2026**, with every
  feature family (CLASS, FORM, FIXTURE, BIO, ROLE, OWN-TEAM), all **point-in-time (PIT)**, cold-start handled.
- `data/model_targets.csv` — the measurable components (y) per row + official `Pts` labels where they
  exist (2023 R1–4, 2025, 2026) + the deterministic scorer + the latent-category residual.

**Label strategy = C** (settled): predict measurable components → apply the fantasy formula
deterministically → add a learned position×competition residual for the 5 latent fantasy-only
categories (SW, POTM, 50-22, KR, LS). See `compare_api_official.py:36` (STAT_MAP) and `ingest_6n.py:175`.

---

## Spend & coverage summary (verified 2026-06-15)
| source | access | cost | what it gives |
|---|---|---|---|
| **rugby-live-data** (Opta, RapidAPI) | `rugby_api.py`, key in env/fallback (unchanged after Pro upgrade) | **1,154 `/match` (paid, one-off)** | rich per-PLAYER + per-TEAM match stats for **club** form |
| 6N matches 2021–2026 | already cached (90) | **free, done** | 6N per-player + team stats |
| **WR rankings** | Pulselive `…/rankings/mru?date=` | **free** | PIT team strength (`wr_pts_gap`) |
| **WR match results** | Pulselive `…/match?startDate&endDate&states=C` | **free** | intl results → H2H + autumn FORM/difficulty pool, **0 paid quota** |
| **RugbyPass** | Selenium, no login | **free** | bio (BIO) + season aggregates (CLASS) + thin match_log |
| **Official fantasy** | manual xlsx | **free** | labels `Pts` + fantasy-only cats |

**Total paid spend ≈ 1,154 `/match` ≤ Pro 2,500 cap** (≈1,300 headroom for retries). Everything else is free.

---

## Phase 0 — Allowed APIs / Sources (READ FIRST, do not invent)

### rugby-live-data — `rugby_api.RugbyAPI` (`rugby_api.py`)
- **Only** these methods exist — do NOT invent others:
  `competitions()`, `fixtures(comp:int, season:int)`, `fixtures_by_team(team_id:int)`,
  `match(match_id:int)`, `standings(comp:int, season:int)`.
- Every GET is **permanently disk-cached** (`data/cache/<sanitized-path>.json`); a cached call costs **0 quota**.
  Always go through the client — never raw-`curl` a `/match` (that bypasses the cache and double-spends).
- `self.remaining` = live quota from `x-ratelimit-requests-remaining`. Print it; stop if it nears 0.
- Comp IDs (stable across seasons): **6N=1266, TOP14=1230, URC=1236, Prem=1218, Champions=1464,
  Challenge=1470**, Rugby Champ=1296, Tour=18.
- `/competitions` does NOT enumerate 2024/2025 club seasons — but `fixtures(comp, season)` works regardless.
- Match payload shape (under `results`, the client unwraps to it): `match` (id, comp_id, comp_name,
  season, status∈{Result,Full Time,…}, game_week, home/away_team, home/away_id, date, home/away_score,
  home/away_tries|conversions|penalties|drop_goals), `home`/`away` = {`teamsheet`[player_id,name,position
  (jersey 1–23),substitute,`match_stats`(27 fields)], `team_stats`(8 groups)}, `events`, `referees`.

### WR / Pulselive (free, no key) — VERIFIED working this session
- Rankings (PIT): `GET https://api.wr-rims-prod.pulselive.com/rugby/v3/rankings/mru?date=YYYY-MM-DD&language=en`
  → `{entries:[{team:{id,name,abbreviation},pos,pts,previousPos,previousPts}]}`. `date=` returns the
  table **as it stood that day** (updates Mondays → fixture-date call is pre-match = no leakage).
  Implemented in `build_wr.py` (caches `data/cache/wr_rank_{date}.json`).
- Results: `GET …/rugby/v3/match?startDate=YYYY-MM-DD&endDate=…&sort=asc&pageSize=50&states=C&language=en`
  → `{pageInfo:{numPages},content:[{matchId,teams:[{id,name}],scores:[a,b],status:"C",competition,
  time:{label},rankingsWeight,venue}]}`. Paginate via `page=`. Team ids share the rankings id-space (join by id or name).
- **Anti-pattern:** never use the *current* rankings table for a past fixture — always `date=`.

### RugbyPass — `rugbypass_batch.py` (Selenium, no login)
- Entry: `make_driver()` (`:384`), extractors `extract_bio` (`:413`), `extract_comp_stats` (`:431`),
  `extract_match_log` (`:473`); `main()` (`:525`) iterates `NAME_TO_SLUG` (`:36`), writes
  `players/rugbypass_<slug>.json`, `--force` to re-scrape, skips existing.
- JSON schema (314 files present): `bio`{Nationality,Age,Position,Height,Weight};
  `competition_stats`[27 fields incl. competition, season, games, minutes, tries, try_assists, metres,
  **post_contact_metres**, defenders_beaten, clean_breaks, offloads, tackles, missed_tackles,
  dominant_tackles, **turnovers_won**, turnovers_conceded, penalties_conceded, carries, av_gain,
  passes, bad_passes, kicks_from_hand, handling_errors, points, cards];
  `match_log`[**thin**: competition, match, date("29 Nov 2025"), opposition, mins, tries, conversions,
  yellow/red_cards, win/draw].
- **Anti-pattern:** do NOT treat `match_log` as a rich per-match feed (no metres/tackles/DB there).
  Rich per-match = the paid club backfill. RugbyPass = CLASS (season aggregates) + BIO only.

### Official fantasy labels (manual xlsx)
- `2025/sixnationsmatch.xlsx`, `2026/sixnationmatch.xlsx` — per-round sheets, parsed by
  `compare_api_official.py:parse_official` (`:71`) via dynamic `Min`+`Pts` header detection.
- `2023/Six_Nations_Data_2023.xlsx` — **different shape**: separate `Round N Data` (components) and
  `Round N Points` (labels) sheets, **Rounds 1–4 only (no R5)**. Needs a dedicated parser path.
- `STAT_MAP` (`:36`) = official→API column map; `API_BLIND` (`:53`) = ['50-22','KR','LS','POTM','SW'].
- **Anti-pattern:** do not expect 2023 R5 labels; do not reuse the 2025/26 parser unchanged for 2023.

---

## Phase 1 — Repair & consolidate FREE sources (0 API spend)

**1a. RugbyPass slug repair.** Fix `NAME_TO_SLUG` in `rugbypass_batch.py:36` — 8 wrong slugs (404s,
e.g. `dino-aldegheri`, `tom-obrien`) + 14 unmapped names. Re-scrape only those: `python rugbypass_batch.py --force` after editing (skips existing). 
- *Verify:* failed-player list at end of run is empty / <5; `ls players/rugbypass_*.json` grows by ~22.

**1b. `build_rugbypass_tables.py`** (new) → `data/rp_compstats.csv`, `rp_matchlog.csv`, `rp_bio.csv`.
- Reuse the JSON walk + `slug_key`/`SLUG_TO_NAME` from `compare_three_way.py:50` and `norm_key`
  from `compare_api_official.py:56`. Key every row by `norm_key`.
- *Verify:* row counts (~314 bio, ~4.7k compstats, ~28k matchlog); `norm_key` non-null 100%.

**1c. `build_official.py`** (new) → `data/official_player_match.csv` (label `Pts` + fantasy-only cats).
- Add `2023: BASE/"2023"/"Six_Nations_Data_2023.xlsx"` to `OFFICIAL` (`compare_api_official.py:30`).
- Reuse `parse_official` for 2025/26; add a `parse_official_2023()` for the Data/Points split (R1–4).
- *Verify:* seasons present {2023,2025,2026}; 2023 rounds == {1,2,3,4}; join-rate sanity vs `6n_player_match.csv`.

**1d. Generalize `ingest_6n.py` → `data/api_player_match.csv`.** Parameterize `ingest()` by a comp list
(currently hard-coded `SIX_NATIONS`); add `--comps`. Keep `derive_minutes`, `add_position_and_recon`.
Run 6N-only first (free, all cached) to prove parity with `6n_player_match.csv`.
- *Verify:* 6N run reproduces `6n_player_match.csv` row count (4,142) exactly; no new quota spent.
- *Anti-pattern:* don't drop the existing `--team` mode or `flatten_team_stats` (already DONE).

## Phase 2 — Free international widening (Pulselive, 0 paid quota)

**2a. `build_intl_results.py`** (new) → `data/intl_results.csv`. Pull Pulselive `…/match` for
2020-07 → today (paginate), filter `competition` to internationals + `teams` to the 6 nations (by id).
Columns: date, comp, team, opponent, team_score, opp_score, margin, result, rankingsWeight.
- *Verify:* covers 6N + autumn + summer; ≥ ~30 intl matches/season; 6 nations all present.

**2b. Extend fixture difficulty with the widened pool.** Feed `intl_results.csv` (team-level) into the
opponent profile so autumn meetings count; re-run `build_wr.py --dates <new intl dates>` then
`build_team_form.py`. (H2H already in `fixture_difficulty.csv`; widening just deepens it.)
- *Verify:* cold-start rows drop below 3%; `wr_pts_gap` coverage stays 100%.
- *Note:* national-team opponent **per-component** vector still uses rugby-live-data `team_stats`
  (richer than Pulselive scores) for 6N + any intl we also have via the paid feed; Pulselive adds
  result-level H2H/strength only. Keep the two layers distinct.

## Phase 3 — `build_crosswalk.py` (new) → `data/player_crosswalk.csv`
One table: `norm_key` ↔ API `player_id` ↔ RugbyPass slug ↔ official name. Resolve initial mismatches
(e.g. *Ignacio* vs *J.* Brex). Reuse `norm_key`, `slug_key`.
- *Verify:* <1% of API players unmatched to RugbyPass; spot-check 10 cross-source joins.

## Phase 4 — PAID club backfill (the single spend) 🟡
Generalize the Phase-1d ingest over the 5 club comps in the **PIT form windows** (Oct(prev yr) → 6N
start), appending club rows to `api_player_match.csv` **and** club rows to `api_team_match.csv`.
- Exact budget (from cached fixtures lists): **1,154 `/match`** = 2023:309, 2024:286, 2025:269, 2026:290
  across TOP14/URC/Prem/Champions/Challenge. Fits Pro (2,500).
- Driver: iterate `api.fixtures(comp, season)` (lists already cached, free), keep `status∈{Result,
  Full Time}` AND `date ∈ [season-1-10-01, 6N-start)`, call `api.match(id)` (cached → 0 re-spend).
- Print `api.remaining` each N matches; abort if < 50.
- *Verify:* `api.remaining` decreased by ≤ ~1,160; every targeted `/match` now cached; club rows have
  non-empty `match_stats` + `team_stats`; `derive_minutes` yields sane minutes (events present in club matches).
- *Anti-pattern:* do NOT pull full club seasons (out-of-window matches waste quota and add nothing under
  the 90-day decay); do NOT re-pull the 90 cached 6N matches.

## Phase 5 — `build_features.py` (new) → `data/model_player_match.csv` (0 API)
For each (player, 6N match, season) row in `api_player_match.csv` (6N subset), assemble — all **strictly
`date < fixture_date`** (FORM) or `date == fixture_date` (WR), cold-start = shrink to position×comp baseline:
1. **CLASS** — per-80 component rates per competition from `rp_compstats.csv`, prior completed seasons only.
2. **FORM** — decayed (≈90d half-life) recent per-match rates from `api_player_match.csv` (rich: 6N +
   club backfill) + thin `rp_matchlog`/`intl_results` for coverage; minutes trend; hot/cold vs class.
3. **FIXTURE** — left-join `fixture_difficulty.csv` on (fixture_id, team_id) — the 22 features (DONE).
4. **BIO** — age/position/height/weight from `rp_bio.csv` via crosswalk.
5. **ROLE** — goal-kicker flag (PIT from conv+pen history), starter prob (recent starts), set-piece
   role (lineouts_won/position).
6. **OWN-TEAM** — own side's `api_team_match` attack strength + own `team_wr_pts` + expected margin.
- *Verify:* row count == 6N player-matches 2023–2026; per-family non-null coverage report; **leakage test**
  — assert max(form source date) < fixture_date for every row; WR snapshot date == fixture_date.

## Phase 6 — `build_targets.py` (new) → `data/model_targets.csv` (0 API)
- y = measurable components per row (tries, assists, conv, pen, DG, metres, DB, tackles, offloads,
  pens-conceded, cards) from `api_player_match.csv`.
- Deterministic scorer = `fantasy_pts_recon` formula (`ingest_6n.py:175`) as a standalone function.
- **Latent residual:** fit position×competition baseline = mean(official `Pts` − recon) over labeled
  rows (2023 R1–4, 2025, 2026) from `official_player_match.csv`; this is the SW/POTM/50-22/KR/LS top-up.
- *Verify:* recon+residual reconstructs official `Pts` to within small MAE per position; forwards no
  longer systematically undervalued (the SW gap, obs 880, is closed by the residual).

## Phase 7 — Verification (final, gate before training)
- **Quota:** `api.remaining` accounts for ≈1,154 spend only; ≤ Pro cap. No raw-curl `/match` anywhere
  (`grep -rn "curl.*match" .` → none outside Pulselive).
- **Leakage:** automated assertion across `model_player_match.csv` (Phase 5 test) passes for all rows.
- **Crosswalk:** <1% unmatched (Phase 3).
- **Coverage:** every 2023–2026 6N player-match has CLASS+FIXTURE (FORM/BIO may be cold-start, flagged).
- **Reproducibility:** `python build_features.py && python build_targets.py` regenerate both stores
  from cache with **0 new quota**.

---

## Future updates (the "months later" upkeep — not part of this once-and-done)
- In 6N season, per round: `python ingest_6n.py --seasons <yr>` (new 6N matches), `build_wr.py`
  (new fixture dates), `build_team_form.py`, `build_features.py`. ~15 `/match`/round → free Basic tier.
- Add a new official xlsx to `OFFICIAL` as labels arrive.
- Club form refresh only if modelling a later season → another bounded PIT-window pull.

## Reuse map
`norm_key`,`parse_official`,`OFFICIAL`,`STAT_MAP`,`API_BLIND` ← `compare_api_official.py` •
`slug_key`,`SLUG_TO_NAME`,JSON walk ← `compare_three_way.py` • `RugbyAPI` ← `rugby_api.py` •
`ingest`,`derive_minutes`,`add_position_and_recon`,`flatten_team_stats`,recon formula ← `ingest_6n.py` •
`fetch_snapshot` ← `build_wr.py` • permissiveness/H2H/WR merge ← `build_team_form.py`.
```
