# Rugby data maintenance on the free API plan

Updated: 2026-08-30. Repository evidence is current through 2026-08-28.

## Current baseline

The repository is configured for the 250-call Basic plan. The committed
`data/cache/refresh_run_2026-08-28.json` manifest records five calls and 224
remaining after the 2026-08-28 refresh. This is dated evidence, not a live quota
check. The current live quota is unverified.

The local cache contains 3,824 match payloads. The cache includes all three July
2026 NCR rounds and the two match payloads added by the 2026-08-28 refresh.

The main data stores now have this coverage:

| Dataset | Current coverage |
|---|---|
| `data/ncr/ncr_player_match.csv` | 20,546 rows and 499 matches through 2026-08-25 |
| `data/ncr/club_player_match.csv` | 152,214 rows and 3,304 matches through 2026-06-20 |
| `data/intl_results.csv` | 433 team-side rows through 2026-07-18 |
| `data/ncr/ncr_intl_results.csv` | 830 team-side rows through 2026-08-25 |
| `data/ncr/feeds/players_gw3.json` | Complete GW3 fantasy points for 470 players |
| `data/ncr/ncr_players.csv` | Current 479-player GW4 catalogue |

## Source policy

Use RapidAPI only for rich player and team match statistics. One match request returns both teams.

Use free sources for all other data:

- Use the fantasy feed for prices, team sheets, and official fantasy points.
- Use World Rugby Pulselive for results and ranking snapshots.
- Use RugbyPass for biographies and season aggregates.
- Use Open-Meteo for historical weather and forecasts.

Do not replace a rich paid match payload with a free result-only row. Keep both sources in separate fields or tables.

## Monthly quota rules

The Basic plan allows 250 requests each month. The scheduled process must use no more than 180 requests.

Keep 50 requests as a hard reserve. Keep the other 20 requests for fixture discovery and retries.

Use this priority order when a monthly backlog exceeds 180 requests:

1. Fetch completed Six Nations and NCR matches.
2. Fetch current-season matches that affect the active player pool.
3. Fetch other current-season club matches.
4. Defer historical backfill until the next quota reset.

Every paid request must use `rugby_api.RugbyAPI`. The client writes each successful response to `data/cache/`.

Never call a match endpoint with raw `curl`, `requests`, or `urllib`. These calls can bypass the permanent cache.

Stop every paid run when the remaining quota reaches 50. Do not assume that RapidAPI will prevent overage charges.

## Planned scheduled task

`Refresh rugby data within free API quota` is planned for every Tuesday morning.
The repository does not contain evidence that its first scheduled automation
execution has completed. Treat that execution as unverified until a run manifest
identifies it as a scheduled run.

Each run can make no more than 40 live paid requests. The 40 requests include discovery and match calls.

The scheduled task keeps the 50-call monthly reserve. The scheduled task carries a backlog into the next run.

The selected competitions currently contain about 889 rich matches each year. Weekly discovery adds about 468 calls.

The expected annual use is about 1,357 calls. The free plan provides 3,000 calls each year.

The monthly reserve leaves 2,400 planned calls each year. The expected headroom is about 1,043 calls.

## Annual schedule

| Period | Paid work | Expected calls | Free work |
|---|---|---:|---|
| January | Finish the October-to-February form window | Up to 180 | RugbyPass and World Rugby refresh |
| February to March | Fetch each completed Six Nations round | About 20 total | Fantasy labels, rankings, weather |
| April to June | Fetch current club and southern competition matches monthly | Up to 180 per month | RugbyPass and World Rugby refresh |
| July | Fetch each completed NCR or international round | About 21 for three rounds | Fantasy labels, rankings, weather |
| August to October | Continue monthly current-season club refresh | Up to 180 per month | World Rugby results and rankings |
| November | Fetch NCR rounds before club backlog | About 21 plus backlog | Fantasy labels, rankings, weather |
| December | Resume the club backlog after the quota reset | Up to 180 | RugbyPass and World Rugby refresh |

The exact tournament dates can change. Use completed fixture status rather than fixed calendar dates.

## Monthly procedure

1. Check the quota once.
2. Refresh each active competition fixture list once.
3. Build a list of completed match IDs that are absent from `data/cache/`.
4. Sort the list by the priority rules above.
5. Use no more than 40 live paid requests in one weekly task.
6. Rebuild all derived CSV files from the cache.
7. Refresh the free fantasy, World Rugby, RugbyPass, and weather sources.
8. Run the quality gates below.
9. Back up new cache files and the refreshed source tables.

Re-run the same procedure with fetching disabled. The second run must use zero paid match requests.

## Quality gates

The monthly run passes only when all applicable checks pass:

- Every expected completed international fixture has one cached match payload.
- The player-match grain `(fixture_id, player_id, team)` has no duplicates.
- Required identifiers and dates have no missing values.
- The latest data date equals the latest completed target fixture date.
- Every completed tournament round has the expected fixture count.
- The official fantasy feed contains the requested gameweek ID.
- The second cache-only rebuild changes no source data.
- The remaining paid quota is at least 50.

## Required one-time work before the next season

1. Change the RapidAPI plan from the paid plan to Basic. Do not remove API access completely.
2. Move the API key out of `rugby_api.py`. Read the key only from `RUGBY_API_KEY`.
3. Rotate the exposed fallback key after the environment variable works.
4. Generalize `ncr_ingest.py` to accept the competition ID and season as arguments.
5. Add a generic year-aware refresh script for the scheduled task.
6. Make the data interpreter configurable. The current `/usr/bin/python3` does not contain pandas.
7. Add an automated cache manifest with file size, checksum, fixture ID, and fetch date.

## Backup policy

Git now tracks the raw paid cache. A cloud clone can reuse every committed response without spending quota again.

After each monthly refresh, commit the changed cache files and derived tables. Also create an incremental backup of these paths:

- `data/cache/`
- `data/ncr/feeds/`
- `data/ncr/ncr_player_match.csv`
- `data/ncr/club_player_match.csv`
- `data/intl_results.csv`
- `data/wr_rankings.csv`
- `players/rugbypass_*.json`

Keep one local backup and one encrypted off-device backup. Verify the cache manifest after each restore test.

## Annual rollover

Before the first fixture of each calendar year, update the active season mapping.

Run fixture discovery in dry-run mode first. Confirm competition IDs from the API response.

Do not assume that a season number equals the match year. Some competitions use the end year as the season label.

Archive the prior year quality report. Record final row counts, match counts, date coverage, quota use, and missing fixtures.

## Pricing references

- [Rugby Live Data Complete pricing](https://rapidapi.com/belchiorarkad-FqvHs2EDOtP/api/rugby-live-data-complete/pricing)
- [RapidAPI plan changes and overages](https://docs.rapidapi.com/v2.0/docs/api-pricing)
