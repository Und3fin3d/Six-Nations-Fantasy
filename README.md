# Rugby fantasy models

This repository picks fantasy teams for two games:

- the Six Nations fantasy game;
- the Nations Championship (NCR) fantasy game, with 12 nations from both hemispheres.

`gw_update.sh` runs the full flow for one gameweek. The flow refreshes the data, runs the model and writes the squad.

```bash
./gw_update.sh ncr                  # NCR: snapshot, ingest, score last GW, rankings, squad, P3 shadow
./gw_update.sh 6n                   # Six Nations: snapshot, ingest, rankings, predictions
./gw_update.sh ncr --dry-run        # print the plan only
./gw_update.sh ncr --no-fetch       # use the cache only; spend no API quota
```

`data/ncr/RUNBOOK.md` gives the step-by-step NCR procedure.

## Models

| Model | Code | Use |
|---|---|---|
| Six Nations picker | `model/sixn/` | Selects the Six Nations squad. `model/sixn/promoted_config.json` holds the deployed configuration. |
| NCR empirical projection | `model/ncr/project.py` | Selects the NCR squad with a MILP over expected points. |
| P3 event model | `model/unified/` | Predicts raw match events for any international. It freezes write-once NCR predictions for GW4–7 as a prospective check. See `model/unified/README.md`. |

## Layout

| Path | Contents |
|---|---|
| `pipeline/sources/` | RapidAPI client, RugbyPass scraper, World Rugby rankings, fantasy market snapshots |
| `pipeline/sixn/` | Six Nations ingest, official labels, crosswalk and feature builders |
| `pipeline/ncr/` | NCR feed snapshots, ingest, international results and club backfill |
| `model/` | The three models and the shared `history.py` cutoff helpers |
| `data/cache/` | Permanent RapidAPI responses |
| `players/` | Raw RugbyPass profiles |
| `data/official/` | Official Six Nations workbooks |
| `data/ncr/` | NCR feeds, stores, projections and squads |
| `tests/` | Pytest suite |

Run every script from the repository root with `python -m`, for example `python -m pipeline.ncr.ncr_ingest --rebuild`.

## Environments

| Use | Interpreter | Requirements |
|---|---|---|
| Data scripts | `$HOME/.venvs/main/bin/python` | pandas, requests, BeautifulSoup |
| Models | `$HOME/.venvs/6n-model/bin/python` (Python 3.12) | `requirements-model.txt` (exact pins) |
| Scheduled cloud refresh | any Python 3.12 | `requirements-refresh.txt` |

`gw_update.sh` rebuilds the model environment when `model_env_preflight.py` fails.

## Data refresh rules

Set `RUGBY_API_KEY` in the environment. The code has no fallback key.

- Use RapidAPI only for rich player and team match statistics. One match request returns both teams.
- Use free sources for everything else. The fantasy feed gives prices, team sheets and official points. World Rugby gives results and rankings. RugbyPass gives biographies and season aggregates. Open-Meteo gives weather.
- Send every paid request through `pipeline.sources.rugby_api.RugbyAPI`. The client writes each response to `data/cache/`.
- Use at most 40 paid requests in one weekly run. The count includes discovery calls and retries.
- Stop paid requests when the remaining monthly quota reaches 50.
- When the backlog exceeds the budget, fetch completed Six Nations and NCR matches first. Then fetch current-season matches for the active player pool.
- Refresh every stored RugbyPass profile each week with `python -m pipeline.sources.rugbypass_backfill --refresh-existing`. Then run `python -m pipeline.sources.build_rugbypass_tables`.

After a refresh, run the same rebuild with fetching disabled. The cache-only rebuild must change no source data. Then run the focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_ncr_player_matching.py tests/test_ncr_score_gw.py tests/test_ncr_snapshot.py
```

## NCR game rules

A squad has 16 players: a starting XV (2 props, 1 hooker, 2 locks, 3 loose forwards, 1 scrum-half, 1 fly-half, 2 centres, 3 back three) and 1 super sub. The budget is 100M, with at most 3 players per nation and 10 per hemisphere. The captain scores 2x. The super sub scores 3x off the bench, 0.5x if they start and 0 if they do not play.

Scoring: try 12, try assist 5, conversion 2 (miss −1), penalty 3 (miss −1), drop goal 5, defender beaten 2, offload 2, line break 3, tackle 1, missed tackle −1, turnover won 4, interception 5, own lineout won 1, lineout steal 5, scrum won 2 (front row only), player of the match 15, penalty conceded −1, handling error −1, lineout error −2, yellow card −5, red card −10.
