# Non-fantasy rugby stat sources: probe report (2026-10-01)

Scope: official match-centre and stat feeds, lineups and team news, timestamped betting
markets, and other unusual sources. The test for each one was whether it gives real
per-player per-match records (or per-match pre-kickoff records) that we can join to
`player_match.csv` and that reduce a known error source:

- official-only stats: BS (breakdown steals), MC (metres carried), POTM, KR, LS;
- bench minutes and replacement timing;
- goal-kicker identity;
- tries.

No RapidAPI calls were made. `data/cache` was not touched.

## Environment caveat (read first)

In this cloud session the egress proxy **denies every rugby, odds and reference host**,
from `curl`/`requests` and from WebFetch alike: `CONNECT 403`, "blocked by environment".
The denied hosts are:

- Rugby feeds and official sites: `api.wr-rims-prod.pulselive.com`, `rugby-union-feeds.incrowdsports.com`,
  `site.api.espn.com`, `site.web.api.espn.com`, `sports.core.api.espn.com`,
  `stats.sixnationsrugby.com`, `www.sixnationsrugby.com`, `unitedrugby.com`,
  `premiershiprugby.com`, `top14.lnr.fr`, `super.rugby`, `epcrugby.com`, `englandrugby.com`,
  the other union sites, and `rugbylivecenter.yormedia.com` (Planet Rugby).
- Score aggregators: `api.sofascore.com`, `flashscore.com`, `webws.365scores.com`.
- Odds: `oddsportal.com`, `api.the-odds-api.com`, `historicdata.betfair.com`.
- Reference: `en.wikipedia.org`, `wikidata.org`, `web.archive.org`, `rugbypass.com`.

None of these is a "doesn't exist". Pulselive worked in earlier sessions (`build_wr.py`,
`build_intl_results.py`, and the `wr_match_*` cache). To fetch from them, a person must
add the hosts to the environment's allowed domains.

What *is* reachable is `raw.githubusercontent.com` (plus GitHub code search). So
viability was proven with **real records that open-source projects downloaded from these
same feeds and committed to GitHub**:

- `bpcsaund/rugby-analytics`: InCrowd feed flattened to CSV, Opta widget scrapes, and
  Pulselive-derived minutes.
- `transientlunatic/Rugby-Data`: InCrowd lineups and substitution minutes.
- `godver3/mediastorm`: one ESPN summary payload.

Those records were joined to our store and to the official Six Nations labels. The live
downloaders in `research/stat_sources/` are written against the documented payload shapes
and unit-tested offline. Their live `fetch` mode is untested here because egress is blocked.

## Headline findings

1. **Our BS proxy is the wrong Opta stat.** The unified model scores turnovers from the
   RapidAPI field `tackle_turnover` (`model/unified/scoring.py`). Compared with official
   Six Nations BS on the same 572 player-matches:
   - API `tackle_turnover` correlates **0.11** with BS.
   - Opta **"Turnovers won"**, the public match-centre stat, correlates **0.75**, with 88.5%
     exact agreement.
   - When BS > 0, Opta turnovers won is > 0 in **92%** of cases. API `tackle_turnover` is
     > 0 in only 15%.

   Source: `results/opta_mc.json`. RapidAPI does not expose per-player "turnovers won" at all.
2. **The club form of the right stat predicts official BS.** Two pre-tournament club rates
   were tested against official Six Nations BS per 80 minutes in 2025 and 2026 (274
   players). Both use point-in-time club windows.
   - InCrowd `turnoverWon` per 80: r = **0.52**.
   - API `tackle_turnover` per 80: r = **0.085**.
   - Out of sample (fit on 2025, score 2026): BS/80 MAE is **0.192** with InCrowd, against
     0.236 with the API rate and 0.239 with a constant.
   - RugbyPass's prior-season `turnovers_won`, **already in `data/rp_compstats.csv`**,
     reaches r = 0.516 on the common set.
   - The mean of InCrowd and RugbyPass reaches r = **0.576**.

   Source: `results/bs_value.json`. This is the clearest data-driven fix available. The
   existing RugbyPass column can be wired in today, and in-season InCrowd form adds to it.
3. **InCrowd and RapidAPI are the same Opta feed for base stats.** Over 58,351 joined club
   player-matches, metres, tackles, passes, offloads, tries and goals agree exactly
   (exact = 1.000). InCrowd therefore adds no new metres or tackles, but it does add:
   - `turnoverWon` (3.0× `tackle_turnover`);
   - `lineoutSteals`, which our store lacks for clubs;
   - `kicksFromHand`. Opta kicks in play correlate 0.78 per match with official KR in 2026;
   - `lineoutsWon` credited to the **jumper**. Our store credits the hooker with about 8.3
     per match, which is why `data/ncr/README.md` drops the NCR "Own Lineout Won" category;
   - the feed's own `minutesPlayedTotal` and Sub On/Off events.
4. **Official MC is not Opta metres.** Opta-widget metres equal API metres (r = 0.997), and
   both run about 0.68× official MC. No public source probed reproduces MC. The 1.45×
   ratio is a definition gap, not a data gap. Keep the learned scale.
5. **Bench minutes: no better source exists.**
   - Pulselive summary+timeline minutes (13.5k joined rows) match our event-derived minutes
     with r = 0.947.
   - On Six Nations bench players against official `Min`, Pulselive agrees exactly more
     often (70% vs 49%), but its MAE is no better (4.71 vs 4.52 for ours) and its
     correlation is lower (0.48 vs 0.57). Source: `results/pulselive_minutes.json`.
   - The replacement-timing error is a forecasting problem, not a labelling problem.
6. **POTM and tries have no new signal source.**
   - Opta turnovers won vs POTM: r = 0.01.
   - Historical try-scorer odds: no archive found. The Odds API has no rugby player props,
     and historical odds there are paid.
   - POTM flags exist in ESPN's summary payload (per open-source clients) but were not
     verified with a sample.

## Ranked source table

Verdicts: **NOW** = pursue now, **LATER** = when egress or another condition allows,
**NO** = not worth it.

| # | Source | Verdict | Evidence (real records) | Historical availability | Access / terms | Joinability with store | Effort | Value by error source |
|---|---|---|---|---|---|---|---|---|
| 1 | **RugbyPass `turnovers_won` (already in repo)**, `data/rp_compstats.csv` | **NOW** | Prior completed club season, per 80: r = 0.516 with official 6N BS/80 (265 players, 2025+2026) | Club seasons 2022/23 to 2026/27 in repo | Already collected (Selenium, `rugbypass_batch.py`) | `key` = initial\|surname, same as `official_player_match.csv` | Hours: feature wiring only | **BS: high.** Replace the `tackle_turnover` prior for the BS/turnover component. KR/POTM: none |
| 2 | **InCrowd `rugby-union-feeds`** (RugbyViz/Opta; powers premiershiprugby.com, unitedrugby.com, LNR/EPCR/League One centres). `research/stat_sources/incrowd.py` | **NOW**, once host allowed | Mirror: 64,584 player rows, 1,404 matches, Prem/PremCup/URC/Top14/Champions/Challenge/Japan, 2024-25 and 2025-26. 1,294/1,294 non-PremCup matches matched to our fixtures; 58,351 player rows joined (98%). Exact agreement on base stats. Rugby-Data proves lineups and sub minutes are served for **2022-23** | Lineups/events back to 2005-06 for URC (Rugby-Data cutoff config). Per-player `stats` confirmed 2024-25 onward; older seasons unverified | No key, no login. Public JSON used by the clubs' own sites; no published API terms (robots unreadable here). Keep requests low and spaced (≤1 req/s), research use only. Comp ids: 1011 Prem (`clientId=PRL`), 1297 PremCup, 1002 Top14, 1013 ProD2, 1008/1026 EPCR, 1068 URC, 2074 JRLO, 1051 Championship | Date ±1 day + surname overlap fixture matcher (`common.match_fixtures`): 100% of covered matches. `turnoverWon`→ new column; base stats are a QA cross-check | Low. Fixtures + ~300 match calls per comp-season; **0 RapidAPI quota** | **BS: high** (in-season form, r 0.52, adds to RugbyPass → 0.58). **KR: medium** (kicks from hand). **LS + NCR lineout-won/steal: medium** (fills categories NCR currently omits). Bench minutes: low (sub events duplicate ours). Goal-kicker: low (scoring events duplicate). Tries/POTM: none |
| 3 | **Opta public widgets on union match centres** (englandrugby.com; `opta-widget` comp 209 = 6N). `research/stat_sources/opta_matchcentre.py` | **LATER / limited** | Mirror: 7,401 player rows, 165 England matches 2013–2026 (70 Six Nations matches, both teams; the 30 from 2021–26 joined to store). Against official labels on 14 matches / 572 rows: Turnovers Won ~ BS r = 0.747; Kicks in play ~ KR r = 0.78; Lineout Steals ~ LS r = 0.80 | England tests since 2013 | Requires a browser and the site's embedded widget key. The open-source scraper uses headless "stealth" to evade bot detection; **do not replicate that** (anti-bot circumvention). Opta widget terms restrict reuse | Surname within (season, round, team) → 95% of rows joined to store, 572 to official | Medium-high, and England-only | Calibration evidence (done here). Little extra modelling value because 6N rows already have official BS |
| 4 | **World Rugby / Pulselive** `/rugby/v3/match/{altId}/summary`, `/timeline`, `/stats`. `research/stat_sources/pulselive.py` | **LATER** (blocked now; repo used it before) | Mirror: 28,177 player-minute rows, 9 nations, 377 test dates 2015–2026, built from summary+timeline (HIA/IR/blood toggles). Joined 13,514 rows to store | Internationals back to the 2000s; summary includes officials (referee) | No key; undocumented but already used by this repo | Date + team + surname (unique) | Low | Bench minutes: none measurable (see finding 5). Referee/venue: low. Team-news timestamps: **prospective only.** Poll `summary` after team announcement to timestamp named XV/bench (5-3/6-2 split, named FH on bench). Per-player stats: not verified (open-source parsers read team stats only) |
| 5 | **transientlunatic/Rugby-Data** (GitHub; InCrowd-derived lineups with on/off minutes, cards, scorers incl. missed conversions; Wikipedia for internationals) | **LATER** (reachable now) | Downloaded `celtic-2022-2023` (151 matches), `celtic-2023-2024`, `premiership-2023-2024`, `top14-2023-2024`, `euro-champions-2023-2024` | URC/Prem/Top14/EPCR **2006-07 → present** (some gaps) | Public GitHub repo; weekly auto-update | Name-based within date | Low | Pre-2022 replacement timing per coach (bench-usage priors) and kicker identity history. Modest value because our store already has 2022+ |
| 6 | **ESPN** `site.api.espn.com/apis/site/v2/sports/rugby/{league}/summary?event=` (6N = 180659) | **NO** (unless POTM needed) | 1 real payload (6N 2025 ITA v IRE): boxscore team stats with Opta names. Totals ≈ ours (metres 301 vs 308, 313 vs 310). Open-source clients read `rosters[].roster[].stats` per player and a `playerOfMatch` flag | Many leagues, years back | Undocumented, no key; ESPN terms prohibit automated reuse | ESPN ids; name join | Medium | Duplicates Opta stats we already hold. Possible POTM label for non-6N internationals (unverified) |
| 7 | **Six Nations official STATS web service** `stats.sixnationsrugby.com` (`api/RU/matchStats/{matchId}`, `playerStatsV2`; also `rugbyunion-api.stats.com`) | **LATER: check access** | Search-indexed API docs only (XML "GameStats" feed v2.0.2.1). No sample obtainable | Six Nations, presumably multi-season | Blocked here. STATS web services normally require credentials. Do not attempt without an issued key | Six Nations match ids | Unknown | Potentially the only source of **official-definition MC/BS** for non-labelled seasons. Worth one access check if egress opens |
| 8 | **Betfair historical data** (BASIC, 1-min, free with account) | **LATER** | None (login required, blocked) | Exchange markets since ~2015: match odds, handicap, total points | Requires a Betfair login; no credential use from agents | Fixture/date | Medium | Team totals or handicap → team-try expectation (tries). Prior repo work: closing lines mis-timestamped; opening odds gave only a tiny gain |
| 9 | **The Odds API** (`rugbyunion_six_nations`) | **NO** | Docs: rugby has h2h/spreads/totals only; player props are US-only; historical snapshots (5–10 min, since 2020) are **paid** | 2020+ (paid) | API key | Event id/date | Low | Tries: no try-scorer props. Team totals duplicate removed market features |
| 10 | **OddsPortal** | **NO** | Search shows 6N archive back to 2009 (1x2, AH, O/U) | Closing/opening only, weak timestamps | Terms prohibit scraping; blocked | Date + teams | Medium | Same family as removed odds features |
| 11 | **api-sports.io rugby** (used by bpcsaund `fetch_odds.py`) | **NO** | None (key) | Recent seasons | Key, 100 req/day free | — | Low | Odds/handicap only |
| 12 | **SofaScore / FlashScore / 365Scores** | **NO** | Open-source SofaScore client hits `/event/{id}/lineups`, `/statistics` (team-level for rugby) | Recent | Terms prohibit scraping; all blocked | — | — | Lineups duplicate ours; no official-definition stats |
| 13 | **Super Rugby** (super.rugby match centre) | **NO** (for now) | No feed found in public code. RapidAPI and RugbyPass already cover Super Rugby | — | Blocked | — | — | Turnovers won for SRP only via RugbyPass season totals (already held) |
| 14 | **LNR Top 14** (top14.lnr.fr stats / feuilles de match) | **NO** | Top 14 per-player stats come via InCrowd comp 1002 (16,744 rows in the mirror) | — | Blocked | — | — | Covered by #2 |
| 15 | **EPCR** (epcrugby.com) | **NO** (use #2) | Champions/Challenge via InCrowd 1008/1026 (10,350 rows) | — | Blocked | — | — | Covered by #2 |
| 16 | **Union team announcements / press releases** (irishrugby.ie, englandrugby.com, wru.wales, …) | **NO** as a scrape target | Blocked | Article dates are post-hoc and unreliable | Varies | Name join | High | Use #4/#2 lineups captured prospectively with a collection timestamp instead. Prior `rolecert` experiments were rejected |
| 17 | **Wikipedia / Wikidata** (referee, venue, POTM in match boxes) | **NO** | Blocked | Long history | Open licence | Date + teams | Medium | Referee available from Pulselive/InCrowd officials. POTM per Wikipedia match box is inconsistent |
| 18 | **Wayback Machine** | **NO** | Blocked (WebFetch refuses `web.archive.org`) | — | — | — | — | Only useful to backfill historic team-sheet timestamps; low |
| 19 | **Planet Rugby live centre** (`rugbylivecenter.yormedia.com/api/match-lineups`, `match-h2h`) | **NO** | Endpoint seen in open-source client; blocked | Recent | Undocumented | — | — | Lineups and team stats duplicate #2/#4 |

## How to reproduce the evidence

The mirror CSVs live under
`$S/agentD/samples/{incrowd,opta_matchcentre,pulselive,rugbydata_lineups,espn}/`, with
`S=/tmp/claude-0/-home-user-Six-Nations-Fantasy/f11ffa4a-ba0b-5432-952a-6accc17d3a46/scratchpad`.
They can be re-downloaded from raw GitHub, which is reachable here.

```bash
export OMP_NUM_THREADS=1
python research/stat_sources/incrowd.py compare --mirror "$S/agentD/bp/[ptucjp]*_player_stats.csv" --json research/stat_sources/results/incrowd_compare.json
python research/stat_sources/opta_matchcentre.py --rfu $S/agentD/bp/rfu_player_stats.csv --json research/stat_sources/results/opta_mc.json
python research/stat_sources/bs_value.py --mirror "$S/agentD/bp/[ptucj]*_player_stats.csv" --json research/stat_sources/results/bs_value.json
python research/stat_sources/pulselive.py compare --minutes-glob "$S/agentD/wr/*_minutes.csv" --json research/stat_sources/results/pulselive_minutes.json
python -m pytest -q tests/test_stat_sources.py
```

Live collection, once `rugby-union-feeds.incrowdsports.com` and
`api.wr-rims-prod.pulselive.com` are allowed:

```bash
python research/stat_sources/incrowd.py fetch --comp urc --season 202501 --limit 1 --out /tmp/incrowd
python research/stat_sources/pulselive.py fetch --alt-id <matchAltId from data/cache/wr_match_*.json> --out /tmp/pl
```

`incrowd.parse_match` keeps **every** scalar stat key in the payload, not only the subset
the mirror flattened, plus on/off minutes from the event log. The first live run should
therefore also reveal any extra fields (e.g. POTM or kick metres).

Committed excerpts, all under 200 KB, are in `research/stat_sources/samples/`:

- two InCrowd matches;
- France v England 2026, with Opta-widget stats joined to official MC/Ta/BS/LS/KR;
- one Pulselive-minutes match;
- one Rugby-Data 2022-23 InCrowd lineup;
- the ESPN boxscore payload.

The `bpcsaund/rugby-analytics` mirror has no licence file. Use it only as evidence and
re-collect from the source rather than redistributing it.

## Recommended next steps (in order)

1. **Model change, no new data needed.** Feed the RugbyPass prior-season `turnovers_won`
   per 80 (club competitions) into the BS / turnover component. In the official-label
   residual, stop treating API `tackle_turnover` as the BS proxy: its correlation with
   official BS is 0.11. Gate it with the normal 2025 → sealed-2026 protocol.
2. **Allow the host `rugby-union-feeds.incrowdsports.com`.** Then backfill `turnoverWon`,
   `lineoutSteals`, `kicksFromHand` and jumper-credited `lineoutsWon` for the club seasons
   in the point-in-time form windows (2022-23 → 2025-26). Each season is about 1.2k match
   requests at ≤1 req/s. Store the data as a separate tidy table keyed by our
   `(fixture_id, player_id, team)` via the fixture matcher; do not merge it into
   `data/cache`. First check that 2022-23 payloads carry `stats`.
3. For NCR, use InCrowd club `lineoutsWon` (jumper) and `lineoutSteals` rates to restore
   the "Own Lineout Won" (+1) and "Lineout Steal" (+5) categories, which are currently
   dropped.
4. Only if egress opens: make one access check on `stats.sixnationsrugby.com` for
   official-definition MC/BS. Take a prospective Pulselive `summary` snapshot after each
   team announcement to get team-sheet timestamps.
5. Do not revisit odds, aggregators or Wikipedia for these error sources.
