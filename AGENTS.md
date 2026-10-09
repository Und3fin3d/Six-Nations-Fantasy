# Cloud agent instructions

- Treat `data/cache` as the permanent API response store. Never delete or refresh a cached match without a specific reason.
- Read the RapidAPI key only from `RUGBY_API_KEY`. Never print, commit, or copy the value into a file.
- Count fixture discovery, retries, and match requests against the run budget.
- Stop RapidAPI requests when the remaining quota reaches 50.
- Use the repository Python environment. Install `requirements-model.txt` when model dependencies are required.
- Preserve the `(fixture_id, player_id, team)` grain in player-match tables.
- Run focused tests and cache-only rebuild checks before committing generated data.
- Do not retrain models during a routine data refresh.
- In the weekly refresh, refresh every stored RugbyPass profile with `rugbypass_backfill.py --refresh-existing`, then fetch newly unmatched players. Players who played that week need updated statistics.
- Report each source that the refresh skipped or that added no rows, with the reason.
