# Gameweek runbook (NCR + Six Nations)

## The one command

```bash
./gw_update.sh ncr                              # snapshot → ingest → WR → pick the team
./gw_update.sh 6n                               # snapshot → ingest → WR → predictions
./gw_update.sh ncr --exclude "New Zealand" "Italy"   # drop already-kicked-off matches
./gw_update.sh ncr --dry-run                    # print the plan, touch nothing
./gw_update.sh ncr --no-fetch                   # cache only, spends zero API quota
./gw_update.sh 6n  --rebuild-features           # also regenerate the model stores
```

It rebuilds the pinned venv if `/tmp` wiped it, checks the API quota (and auto-switches
to `--no-fetch` below 50 calls), and picks the right interpreter per step. Everything
below is what it runs, in case you need a step on its own.

> **6n `--rebuild-features` warning.** `ingest_6n.py` rewrites `api_player_match.csv` /
> `api_team_match.csv`; the feature store is then stale (the script says so). Rebuilding
> it regenerates `model_player_match.csv`, which `model/data.py` asserts is exactly 2,760
> rows. If that assertion trips, the store genuinely changed — investigate, don't relax it.

---

Everything needed to go from "a gameweek is coming" to "a team is picked", plus the
occasional maintenance chores. Nothing here is inline Python any more — each step is a
script in the repo root.

## Interpreters (read this first)

The system Python does not contain pandas. Set the data interpreter once:

```bash
PY_DATA="$HOME/.venvs/main/bin/python"
```

| For | Interpreter |
|---|---|
| Data scripts (pandas + urllib only) | `$PY_DATA` |
| Anything under `model/` (sklearn, lightgbm, xgboost) | `/tmp/6n-model-pinned/bin/python` |

The pinned venv lives in `/tmp` and **gets wiped periodically**. Rebuild it with:

```bash
~/.local/bin/python3.11 -m venv /tmp/6n-model-pinned
/tmp/6n-model-pinned/bin/pip install -r requirements-model.txt
```

---

## Each gameweek

### 1. Snapshot the fantasy feed — free, no API quota

```bash
"$PY_DATA" ncr_snapshot.py
```

Refreshes `ncr_players.csv` (prices, positions, injury flags, **team sheets**) and
`ncr_fixtures.csv` (gameday, lock time, `iscurrent`), and archives the raw feed under
`data/ncr/feeds/` with a UTC stamp. The player-feed path is selected from the current
fixture gameweek (`players_1_en_<GW>.json`). The current catalogue is stored as
`players_cur.json`, while `players_latest.json` points to the latest fully completed
round and retains that round's start/bench statuses for multiplier scoring.
`players_gw<GW>.json` is updated only after the payload's `gameday_id` is validated.

It prints whether the team sheets are **CURRENT** or **CARRYOVER**. The feed's
`player_status` (P = starts, B = bench) only flips to the new gameweek roughly 48h
before kickoff. **If it says CARRYOVER, any team you generate is provisional** — the
minutes model, captain eligibility (P only) and super-sub eligibility (B only, for the
3×) are all keyed off it. Re-run once sheets post.

> Each numbered feed path exposes that gameweek. Snapshot **twice**: once before the
> lock (sheets + prices), once after the matches (`--label post_gw2`) to preserve that
> week's realised fantasy points. Use `--gameday N` only as an explicit diagnostic or
> historical override; normal runs follow the fixtures feed automatically.

### 2. Ingest last gameweek's results — ~6–7 API calls

```bash
"$PY_DATA" ncr_ingest.py                 # fetch new results, then rebuild both tables
"$PY_DATA" ncr_ingest.py --rebuild       # cache-only re-assembly (0 API calls)
```

Rebuilds `ncr_player_match.csv` (internationals → FORM + minutes) and
`club_player_match.csv` (club form → the calibrated club blend). Run `--rebuild` on its
own after any backfill, to pull newly-cached matches into the tables.

### 3. Refresh World Rugby ratings — free

```bash
"$PY_DATA" build_wr.py --dates 2026-07-11
```

Rankings update the Monday after each round. Safe to run repeatedly: it now **merges**
into `wr_rankings.csv` rather than replacing it.

### 4. Pick the team

```bash
/tmp/6n-model-pinned/bin/python -m model.ncr_project
/tmp/6n-model-pinned/bin/python -m model.ncr_project --exclude "New Zealand" "Italy"
```

`--exclude` drops teams whose match has already kicked off. Writes
`ncr_gw{N}_squad.csv/.md` and the projections CSV.

`ASOF` is **derived** from the current gameday's kickoff — never hardcode it. Anything
dated on or after `ASOF` is excluded from the model's history, so a stale `ASOF` would
both mis-weight recency and leak a played gameweek.

### 5. After the matches — score it

Snapshot the feed again for labels, then evaluate with `model/ncr_eval.py`. Its metric
bundle **always includes MAE (raw and calibrated)** alongside rho / XV / team / top-15.
Beware the team metric on a single gameweek: it's dominated by the captain 2× and the
super-sub 3×, and has repeatedly swung ±60 points on one coin-flip.

---

## Maintenance chores

### Club run-in backfill — after each quota reset

The 6N pipeline only ever cached the Oct→Feb form window, so the club **run-ins**
(Mar–Jun) needed a separate pull.

```bash
"$PY_DATA" backfill_club_runs.py --dry-run                       # what's missing, 0 match calls
"$PY_DATA" backfill_club_runs.py --floor 60                      # everything
"$PY_DATA" backfill_club_runs.py --comps 1230 --seasons 2026 --floor 50
```

Resumable, idempotent, serial (concurrent fetchers trip the API's per-minute cap).
Then `ncr_ingest.py --rebuild` to fold the new matches into the tables.

**Competition ids are easy to transpose — do not guess them:**

| id | competition |
|---|---|
| 1236 | United Rugby Championship |
| **1218** | **Gallagher Premiership** |
| **1230** | **TOP 14** |
| 1464 | Champions Cup |
| 1470 | Challenge Cup |
| 1242 | Super Rugby Pacific |
| 2538 | Japan Rugby League One D1 |
| 696 | Nations Championship |

### RugbyPass — new call-ups only

```bash
"$PY_DATA" rugbypass_backfill.py
"$PY_DATA" build_rugbypass_tables.py
```

Free (plain HTTP). `model/rp_rates.py` raises if `rp_compstats.csv` drops below 600
distinct slugs — a guard against a git restore silently reverting the tables to the
6N-only versions, which once buried every Southern-hemisphere player.

### API quota

The target Basic plan allows 250 requests each month. Keep 50 requests as a hard reserve.
Check the current quota once before a paid run. This check costs one request:

```bash
"$PY_DATA" -c "import sys; sys.path.insert(0,'.'); \
from rugby_api import RugbyAPI; api=RugbyAPI(verbose=False); \
api.get('/competitions', refresh=True); print('remaining', api.remaining)"
```

Steady state is ~7–19 calls per gameweek. Everything else the model eats is free:
the fantasy feeds, RugbyPass, World Rugby rankings. All history is cached permanently
in `data/cache/` and never re-fetched.

See `FREE_API_MAINTENANCE_PLAN.md` for the monthly budget and annual schedule.

---

## Betting odds: removed

Bookmaker odds and every feature derived from them were deleted (2026-07-10). The model
predicts the game itself: the NCR matchup multiplier now comes from World Rugby ratings
(~2 points of margin per rating point, +3 for home advantage). The 6N champion was
unaffected — its predictions were bit-identical after the purge, confirming the research
ledger's verdict that the family never earned its place.

`weather_` features remain collected but their multiplier stays off by default: GW1
replicated the ledger's rejection (−0.01 rho, a flipped captain). Weather is retained as
a physical observation, unlike odds, which are a third party's forecast of the result.
