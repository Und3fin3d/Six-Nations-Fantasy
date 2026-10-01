# Fantasy-game data sources for rugby: probe results (2026-10-01)

**Question.** Which fantasy-game data sources can supply real player-level records
(fantasy points per player per round, stat breakdowns, prices, positions, lineup
status)? The goals are:

- (a) many more labelled evaluation rounds than the 13 official ones;
- (b) official-definition stats (POTM, breakdown steals, official metres);
- (c) prices, so evaluation squads can be built within the budget.

**How.** Every candidate was probed with real requests. A source counts as "proven"
only when real player-round records were downloaded and parsed. Tidy tables were
joined to the store (`runs/base/inputs/player_match.csv`, read only). No RapidAPI
calls were made. `data/cache` was not touched.

## Constraint that shaped this probe: egress

This session's network policy denied every fantasy and stats host with a 403 on
CONNECT. Both curl/python and WebFetch were blocked. Denied hosts:

- Fantasy and stats sites: www.superbru.com, www.playfantasyrugby.com,
  fantasy.sixnationsrugby.com, sixnationsrugby.com, cdn.sixnationsrugby.com,
  fantasy.nationschampionshiprugby.com, lagrandemelee.midi-olympique.fr,
  duiuhak4urjo2.cloudfront.net (vendor CDN), www.fantasytop14-pmu.fr, top14.lnr.fr,
  oval3.game, api.oval3.game, fantasyrugbygeek.com, fantasyrugbygeek.gumroad.com,
  www.draftxv.com, api.wr-rims-prod.pulselive.com.
- Datasets and archives: www.kaggle.com, web.archive.org, archive.org,
  index.commoncrawl.org, data.commoncrawl.org. Anonymous reads of
  commoncrawl.s3 also returned 403.

Only package registries, Docker Hub, raw.githubusercontent.com and GitHub code
search were reachable.

**Verdicts below separate two cases.**

- "Blocked by environment" means the source is known to exist, from search results
  and third-party code. It could not be fetched from here.
- "Doesn't exist / dead" is a separate verdict.

**Where the proof comes from.** Real records for the important sources came from
public GitHub repositories where other fantasy players committed pulls of the
official feeds (fetched via raw.githubusercontent.com). The downloaders in
`research/fantasy_sources/` are written for the live endpoints, so they should run
from any unrestricted runner, such as the weekly refresh job. To run them here, add
the hosts above to the environment's allowed domains.

## Ranked verdicts

EV = expected value for the three goals: (a) evaluation rounds, (b) official-definition stats, (c) prices / budget-legal squads.

| # | Source | Verdict | Records proven (evidence) | Historical? | Access / terms | Join to store | Effort | EV (a) / (b) / (c) |
|---|---|---|---|---|---|---|---|---|
| 1 | **PlayFantasyRugby platform** (NZR/SANZAAR/World Rugby official games: Super Rugby Pacific 2025-, Rugby Championship 2025-, RWC 2023 at fantasy.rugbyworldcup.com, WRWC 2025 at play.world.rugby) | **Pursue now** | RWC 2023 full-season `players.json` (mirror `fnb-software/rugby-fantasy/2023/data/players.js`): **681 players, 2,195 scored player-rounds over 8 rounds**, ownership % for all 5,392 player-round listings, cost, position. SRP 2026 official season aggregates (mirror `ihumble/code7-data/data.json`): 409 players, price 2.5–10.9M, games played, total, high and average. Field layout confirmed by 3 independent scrapers (`alexmgl/super-rugby`, `mushola/fantasy-rugby`, `altierispeixoto/laboratory`). | The season-end `players.json` keeps **every round** (`stats.scores`, `selected`, and `priceHistory` from 2026). Past seasons only via Wayback or mirrors (Wayback blocked here; not verified). | Public, unauthenticated JSON: `<site>/json/fantasy/{players,rounds,squads,player_stats}.json`. That is 4 requests per game. The site's T&Cs were not readable from here; the game's own front-end serves these files to anonymous visitors. Host blocked by environment. | RWC 2023: **99.1 % of store player-rows in the fantasy windows get a fantasy score**. 91.3 % of scored fantasy rounds find a store row (4 of 48 RWC fixtures are absent from the store). 92.2 % of players matched (621 full name, 2 initial, 5 surname). SRP 2026: **99.7 % of players with ≥1 game matched** (365/366); games played ≤ store appearances for 99.5 %; corr 0.974. | Low: `pfr.py` is done and tested. | **High** / Low (no per-round stat breakdown; season totals only) / **High** (cost, per-round price path from 2026, per-round ownership) |
| 2 | **Six Nations vendor platform, private per-round detail**, incl. **La Grande Mêlée (Top 14, the game listed on LNR's site)** | **Pursue now for Top 14 2026-27, if an account token and the terms allow** | 6N 2026 full pull of `/v1/private/statsjoueur` detail (mirror `fnb-software/rugby-fantasy/2026/6nations/data/players.js`): **238 players × 5 rounds = 1,190 rows (653 played)**. Each row has minutes, starter/sub flag, points, price before/after and 17 official stats. Against our official 2026 workbook: 99.1 % join; exact match for **points 97.8 %, minutes 99.85 %, tackles 98.0 %, metres carried 98.8 %, breakdown steals 100 %, POTM 100 %, tries 100 %**. Top 14: the same vendor (`X-Access-Key 740@<ver>@…` versus 6N `600@<ver>@`). Real LGM fixture files are mirrored (`2026/top14/data/rounds.js` 30 journées, `2027/top14/data/rounds.js` 26). The LGM stat list comes from mirrored scoring code: tries, kicks, missed kicks, conceded penalties, yellow/orange/red cards, tackles, missed tackles, line-breaks, tackle breaks, carries, forward passes, turnovers won, interceptions, offloads. | Current season only. The 2025-26 LGM season is probably gone now that 2026-27 is live; not verified. | The private API needs a logged-in **account token** (free account). The public `/v1/public/sportifs` catalogue (price, ownership, availability) needs no login and is already used for 6N by `snapshot_fantasy_market.py`. LGM rules PDF exists on the vendor CDN (blocked here). **Read the game's terms before automating with a personal token.** | Measured on the 6N analogue: 99.1 % join to official rows. The store has 35,256 TOP 14 rows to join by name, team and date; not measured because there is no LGM player sample. | Medium: parser done (`sixn_platform.py`); about 700 POSTs per pull; needs a token and a terms check. | **High** (about 26 Top 14 rounds/season, about 320 player rows/round) / **High** (official vendor-defined stats per player-round) / **High** (price before/after every round) |
| 3 | **6N per-round prices rebuilt from public mirrors** (`sixn_prices.py`) | **Pursue now (works today)** | 2023: rounds 1–5 points and prices for 262 players (`david-sykes/fantasy-rugby-streamlit`). 2026: full price path for 238 players × 5 rounds. 2025: GW1 (138 players) and round-4 catalogue (230) only (`alexmgl/six_nations_solver`). | Fixed; already complete. | raw.githubusercontent.com, 4 files. Third-party copies with no licence stated: download at run time, don't commit. | Official rows that get a price: 2023 R1–R4 98.5–99.3 %; 2026 R1–R5 94.7–100 %; 2025 R1 100 %, R4 97.8 %, R2/R3/R5 0 %. | Done | Low (+1 round: 2023 R5 points, which the repo lacks) / – / **High** (budget-legal squads for 2023 and 2026 official rounds) |
| 4 | **Nations Championship official** (Sportz Interactive feeds) | **Keep (already ingested)** | Local `data/ncr/feeds/players_gw1..4.json`: 445–479 players per GW, about 260 non-zero `cur_gd_points` per GW, P/B lineup status, price, ownership, metres gained, try-saving tackles. | Per-gameday files `players_1_en_<gd>.json` remain retrievable. | Public JSON with the site's static `Entity` header. | Already in the store. | – | Medium / Medium / High |
| 5 | **6N and LGM public catalogue** (`/v1/public/sportifs`) | **Pursue now (cheap)**: point `snapshot_fantasy_market.py` at the LGM base URL as well | Same response shape as 6N (repo code); not fetched here (blocked). | Prospective only | Public | Names and clubs | Low | – / – / Medium (point-in-time price and ownership) |
| 6 | **World Rugby pulselive match stats** (not a fantasy game; context only) | Pursue later (likely in another agent's scope) | RWC 2023, 48 matches (mirror `fnb-software/.../2023/data/matches.js`): **2,305 player-match rows, 149 official WR stats**. 87 % join to the store (unmatched rows include unused replacements). Against the store: tackles corr 0.992 (80 % exact); carry metres corr 0.992 (76 % exact). The `isPlayerOfTheMatch` flag is empty in this endpoint. | Event archive (all WR events) | Public JSON; host blocked here | Good | Medium | – / Medium / – |
| 7 | **Superbru** (URC, PREM, Investec Champions Cup, Six Nations, Rugby Championship) | **Not worth it unless Superbru gives written consent** | None: no public JSON, and no scraper or data on GitHub (code search for superbru plus fantasy returned only predictor scrapers). Game pages are login-scoped. | Unknown | **ToS prohibits "any systematic or automated data collection activities (including … scraping, data mining, data extraction …) … without our express written consent"** (superbru.com/policies/terms-of-use). Host blocked here. | – | – | High if licensed (URC/PREM official games) / – / Medium |
| 8 | Wayback Machine snapshots of the feeds above | Pursue later (from an unrestricted runner) | Not verifiable: web.archive.org denied. | Would unlock SRP 2025 and TRC 2025 season-end `players.json` if captured. | Public | Same as #1 | Low | Medium / – / Medium |
| 9 | Fantasy TOP 14 PMU (Feeling Sports) | Not worth it | Search shows it was the LNR game in the 2010s. LNR now lists La Grande Mêlée. | Dead / superseded | Blocked | – | – | – |
| 10 | LNR stats (top14.lnr.fr/statistiques) | Not worth it for fantasy (official stats are better obtained through #2) | Season aggregates and match sheets (search results); blocked | Season pages | Blocked | Names | Medium | – / Medium / – |
| 11 | Oval3 (web3 fantasy rugby) | Not worth it | Score is an Opta-based 0–100 attribute composite (impact, attack, defence, skills, strength), not per-stat points. Token/NFT economy; community data in Dune or Google Sheets only. | – | Blocked | – | – | Low / – / – |
| 12 | DraftXV (URC/PREM draft, 2025-), Dream XV (PREM) | Not worth it | Draft leagues only, app-first, no public feed found | – | Blocked | – | – | – |
| 13 | ESPN Fantasy Rugby; Dream Team | Dead | Only 2017–19 references | – | – | – | – | – |
| 14 | Japan League One fantasy | Doesn't exist | No official game found | – | – | – | – | – |
| 15 | fantasyrugbygeek.com and its Gumroad | Not worth it | Gumroad sells a 2018 eBook only; the site publishes game overviews, not datasets | – | Blocked | – | – | – |
| 16 | Kaggle ("World_Rugby_Master" odds; "Six Nations Historical Data" squads/KPIs) | Not worth it | No fantasy points or prices; kaggle.com blocked | – | – | – | – | – |
| 17 | Other GitHub repos | Catalogued; nothing further usable | `alexmgl/super-rugby`, `mushola/fantasy-rugby` and `altierispeixoto/laboratory` all gitignore their PFR data; their code confirms the endpoints. `Moloshow/lagrandemelee-optimizer` needs a token and its outputs are not committed. `juliankom/fantasy-rugby` claims 69,782 performances, but no data files were found. `LuiHol/FantasyRugby` is RugbyPass-derived. `suzygebbett` is a 6N scraper stub. `multipitch/six` and `ThomasVirgo/*` have no data found. | – | – | – | – | – |
| 18 | Common Crawl | Blocked by environment | index.commoncrawl.org denied; anonymous S3 403 | – | – | – | – | – |

## What the joins show

| Sample (real records) | Rows | Join result |
|---|---|---|
| PFR RWC 2023 `players.json` → store `Rugby World Cup` 2023 (date windows per fantasy round) | 681 players, 2,195 scored player-rounds | Players 92.2 %; scored rounds → store row 91.3 %; store rows labelled 2,005/2,024 = **99.1 %**; corr(points, tries) 0.69, corr(points, minutes) 0.46 |
| PFR SRP 2026 season aggregates → store `Super Rugby Pacific` 2026 | 409 players | 89.7 % of all listed players; **99.7 %** of those with ≥1 game |
| 6N vendor `statsjoueur` detail 2026 → `data/official_player_match.csv` 2026 | 653 played player-rounds | 99.1 % joined; points 97.8 % exact (R1–R4 100 %, R5 89.6 %); breakdown steals, POTM and tries 100 % exact |
| 6N per-round prices (`sixn_prices.py`) → official rows | 2,868 price rows | 2023 and 2026: 94.7–100 % of official player-rounds priced; 2025: R1 and R4 only |
| WR pulselive RWC 2023 → store | 2,305 player-match rows | 87 % (all listed players incl. unused replacements); tackles and metres corr 0.992 |

## Side finding: the official 2023 Round 1 rows look shifted by one row

`data/official_player_match.csv` season 2023, round 1, comes from the
`Round 1 Data` sheet of `2023/Six_Nations_Data_2023.xlsx`. Its rows do not agree
with the game-API pull (`david-sykes` mirror), but rounds 2–4 agree 99.3 %.

- In R1, only Wales rows agree with the game-API values.
- For **107 of 134 rows**, the official `Pts` of row *i+1* equals the game-API
  value for the player in row *i*. The stats are offset as well. For example, the
  official "C. Doris" row has 20 minutes, but the game API gives Doris 63.0 points,
  consistent with a full match.
- The workbook's own `Round 1 Points` sheet disagrees with its `Round 1 Data` sheet
  for the same players (e.g. Aki 80 vs 20 minutes; Sexton 12 vs 68).

**Recommendation:** treat 2023 R1 labels as suspect for non-Wales teams and
re-derive them before the next evaluation that uses 2023. Two options:

- fix the sheet;
- take R1 points from the mirror.

## Expected value and recommendation

1. **PlayFantasyRugby (pursue now).** This is the cheapest large gain in labelled
   rounds with real prices and ownership. Steps:
   - Run `pfr.py fetch` for `https://www.playfantasyrugby.com` from an unrestricted
     runner, once per season end and ideally weekly in-season. Weekly runs are
     needed because `status` (starting/uncertain/injured) and ownership are
     point-in-time.
   - Probe the Rugby Championship game path, `/rugby-championship/...`. Whether it
     has its own JSON root was not verifiable here.
   - Recover SRP 2025 and TRC 2025 via a Wayback CDX query on
     `playfantasyrugby.com/json/fantasy/players.json`.

   What it gives: about 16–19 SRP rounds/season × about 200 scored players, plus
   TRC 6 rounds and RWC 2023 8 rounds, which are already in hand. All carry real
   budget constraints ($100M, 15-man XV, team caps).

   Caveat: PFR uses its own scoring table. Evaluate decision quality by applying
   PFR scoring to the model's component forecasts, not by comparing against 6N
   points.
2. **La Grande Mêlée / Top 14 (pursue now, conditional).** This is the only
   source with official vendor-defined stats per player-round at volume, and it
   includes price before/after each round. It is the same vendor and stat engine
   as the 6N game, whose detail matched our official workbook almost exactly.
   - It needs the owner's own account token, and LGM's terms must be read first;
     the rules PDF is on the vendor CDN, which is blocked here.
   - The season resets yearly, so start pulling during 2026-27, which is in
     progress now. The 2025-26 season has probably been lost.
3. **`sixn_prices.py` (use now).** It makes 2023 and 2026 official rounds
   evaluable as budget-legal squads, and it adds 2023 R5 points. 2025 prices
   remain partial (GW1 and R4). The 2025 gap could be closed from the repo's own
   `snapshot_fantasy_market.py` once future snapshots exist; past rounds cannot be
   recovered that way.
4. **Superbru.** Skip unless written consent is obtained. The ToS explicitly
   forbids scraping.

## Files

- `research/fantasy_sources/fantasy_common.py`: name keys, store loader, tiered
  (team, name) matcher, round windows.
- `research/fantasy_sources/pfr.py`: PFR `fetch` (4 GETs), `tidy` (per player-round:
  points, ownership, price) and `join` (store join rates).
- `research/fantasy_sources/sixn_platform.py`: 6N/LGM `statsjoueur` detail parser
  (17 official stats, minutes, starter flag, price path); comparison with the
  official workbook; token-gated `fetch-detail` (never logs in, reads
  `FANTASY_TOKEN`).
- `research/fantasy_sources/sixn_prices.py`: per-round 6N price table from the
  public mirrors, plus coverage of official rows.
- `research/fantasy_sources/samples/`: tiny real excerpts used by tests (72 KB):
  - PFR RWC 2023 France and New Zealand players, plus squads;
  - 6N 2026 `statsjoueur` detail for 4 players.
- `tests/test_fantasy_sources.py`: offline tests (5).

Larger real samples were kept out of git, in the session scratchpad
`agentC/samples/`. These are:

- `pfr_rwc2023/` (players, squads, tidy player-rounds);
- `pfr_srp2026/`;
- `sixn_2026_detail/`;
- `sixn_2023_round_points/`;
- `sixn_2025_gw1/`;
- `sixn_price_mirrors/` (incl. the `sixn_round_prices.csv` output);
- `lgm_top14_rounds/`;
- `wr_pulselive_rwc2023/`;
- `ncr_sportz/`.

Commands used for the join numbers above:

```bash
python research/fantasy_sources/pfr.py tidy --players <players.json> --squads <squads.json> --out pr.csv
python research/fantasy_sources/pfr.py join --tidy pr.csv --store <player_match.csv> \
    --competition "Rugby World Cup" --season 2023 --windows rwc2023
python research/fantasy_sources/sixn_platform.py tidy --players <players.json> --out d.csv
python research/fantasy_sources/sixn_platform.py compare --tidy d.csv --official data/official_player_match.csv --season 2026
python research/fantasy_sources/sixn_prices.py --cache <dir> --out prices.csv --official data/official_player_match.csv
```
