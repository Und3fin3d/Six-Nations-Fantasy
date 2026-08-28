# External Data Sources

## Betting markets — REMOVED (2026-07-10)

Bookmaker odds, and every script and feature derived from them, were deleted from this
repo. A model that reads the closing line is laundering the market's forecast rather
than making its own; the research ledger had already rejected the family on merit
(`external_market_*` candidates all failed, and the promoted config ran with
`market_features=False`).

Verified behaviour-preserving for the champion: after the purge `model_predictions_2025/2026.csv`
were bit-identical across all 55 shared columns (only the unused `market_win_prob`
passthrough column disappeared), and the 2025 backtest verdict was unchanged
(lgbm_only value_xv 0.680 / MAE 7.680, both PASS).

The NCR model *was* consuming the line for its matchup multiplier; it now predicts the
expected margin itself from World Rugby ratings (~2 points of margin per rating point,
+3 rating points for home advantage). Weather is retained — physical venue conditions
are observations, not somebody else's prediction of the result.

Purpose: add new point-in-time match context without changing the current champion
unless an explicit research candidate proves the feature family helps.

## Weather / Venue

Input file: `data/external_fixture_weather.csv`

Primary source option:
- Open-Meteo forecast/archive fields for stadium coordinates.

Recommended columns:

```csv
season,round,fixture_id,date,venue,weather_source_timestamp,weather_source_kind,weather_temp_c,weather_rain_mm,weather_precip_probability,weather_wind_kph,weather_wind_gust_kph
```

Rules:
- One row per fixture.
- For historical training, actual weather can be used as a proxy for what a pre-match forecast would approximate.
- For future prediction, use the latest forecast available before kickoff.
- Missing files or rows fall back to no weather features.

Useful model signal:
- wet / windy games suppressing metres, tries, handling-heavy upside
- cold or windy kicking conditions

## Named Role Certainty

Input file: `data/external_player_roles.csv`

Primary source options:
- Official team announcements, Six Nations match-centre lineups, trusted liveblogs,
  or curated manual notes from post-team-sheet information.

Recommended columns:

```csv
season,round,fixture_id,team_id,team,player_id,player_name,player_key,rolecert_source_timestamp,rolecert_goal_kicker_prob,rolecert_named_captain,rolecert_lineout_role_prob,rolecert_bench_fh_kick_share,rolecert_replacement_role_uncertainty
```

Rules:
- One row per fixture-player where a role signal is known.
- Prefer `player_id`; `player_key`/`player_name` are fallback joins.
- Use only information known after team sheets and before kickoff.
- Missing files or rows fall back to no role-certainty features.

Useful model signal:
- goal-kicker certainty
- captain/POTM prior
- bench fly-half kicking share
- lineout/set-piece role
- replacement-role uncertainty for supersub choices

Implemented local generator: `build_role_certainty.py`.

It creates `data/external_player_roles.csv` from PIT role/form features and the
named team sheet. This is not player hardcoding; it computes team-relative
signals such as likely primary kicker and replacement-role uncertainty.

2025 promoted-base checks:

| candidate | value_team | MAE | verdict |
|---|---:|---:|---|
| `external_rolecert_features_on` | 0.614 | 7.382 | reject; MAE improves but team value collapses |
| `rolecert_kick_realloc_starters_025` | 0.714 | 7.406 | reject; no meaningful movement |
| `rolecert_bench_kick_shrink_050` | 0.714 | 7.405 | reject; tiny bench MAE gain only |
| `rolecert_supersub_uncertainty_025` | 0.702 | 7.406 | reject; hurts team value |

Read: rolecert is useful diagnostic infrastructure and it correctly identifies
bench-kicker uncertainty, but the current model already handles most of this
through existing role/form and bench heads. Do not promote standalone rolecert
features or overlays.

## Tactical Style

Input file: `data/external_team_style.csv`

Implemented local generator: `build_tactical_style.py`.

It creates PIT tactical priors from `data/api_team_match.csv`: attack, tempo,
kicking volume, tackle demand, discipline risk, and set-piece strength.

2025 promoted-base checks:

| candidate | value_team | value_xv | MAE | top15 | verdict |
|---|---:|---:|---:|---:|---|
| `external_style_features_on` | 0.701 | 0.689 | 7.424 | 0.347 | reject |
| `external_style_teamplay_aspects` | 0.698 | 0.684 | 7.409 | 0.320 | reject |
| `external_style_aspects_on` | 0.695 | 0.691 | 7.398 | 0.347 | reject |
| `external_style_aspects_teamplay_aspects` | 0.707 | 0.707 | 7.405 | 0.360 | reject |

Read: local tactical summaries are too noisy/wide for this tiny dataset. They
sometimes smooth MAE slightly, but they damage selection. This strengthens the
case for real external team-total data rather than more derived internal
match-shape summaries.

## Research Candidates

The feature families are off by default. Use these candidates after populating
the files and rerunning `build_features.py`:

- `external_weather_features_on`
- `external_rolecert_features_on`
- `external_style_features_on`
- `external_style_aspects_on`
- `external_all_data_on`

If no external data file is present, these candidates are intentionally inert.

## Initial Weather Checks

Built `data/external_fixture_weather.csv` from Open-Meteo archive daily weather
for all 60 Six Nations fixtures in 2023-2026.

2025 promoted-base blanket-feature check:

| candidate | value_team | value_xv | MAE | bench MAE | top15 | verdict |
|---|---:|---:|---:|---:|---:|---|
| `orchestrator_resid030_delta_overlay_010` | 0.714 | 0.714 | 7.406 | 5.373 | 0.387 | incumbent |
| `external_weather_features_on` | 0.691 | 0.727 | 7.419 | 5.389 | 0.387 | reject |

Read: raw weather as a blanket model feature may contain some XV-selection signal,
but it currently damages the full fantasy objective and point calibration. If it
is revisited, use it as a targeted component/selection specialist rather than
globally feeding every head.

Targeted component overlay check:

| candidate | 2025 value_team | 2025 MAE | 2025 gate | sealed 2026 value_team | sealed 2026 MAE | verdict |
|---|---:|---:|---|---:|---:|---|
| promoted | 0.714 | 7.406 | incumbent | 0.700 | 7.303 | keep |
| `weather_attack_suppress_010` | 0.715 | 7.389 | fail, sub-threshold | 0.685 | 7.321 | drop |
| `weather_multi_005` | 0.714 | 7.398 | fail, sub-threshold | 0.685 | 7.313 | drop |
| `weather_multi_010` | 0.715 | 7.384 | pass 2025 + stability | 0.685 | 7.320 | sealed veto |
| `weather_multi_015` | 0.715 | 7.380 | not champion after 0.010 | 0.687 | 7.319 | drop |

Read: weather is a classic 2025 trap here. The model can use it to smooth 2025
MAE, but it does not generalise to 2026 and it hurts the full fantasy objective.
Keep the weather ingestion code because it may become useful with betting/team
totals or a narrower forecast-only specialist, but do not promote weather as a
standalone overlay.

## Official Fantasy Market Snapshots

The official fantasy frontend exposes a public current-player catalogue while a
game is active. It is not a historical API: after the competition closes the
catalogue is empty or unavailable, so ownership and price cannot be recovered
retrospectively from the live endpoint.

Prospective collector:

```bash
/tmp/6n-model-pinned/bin/python snapshot_fantasy_market.py --game m6n
```

Run it after each team-sheet release and again shortly before the round locks.
It discovers the current app identity/version from `card-game.js`, calls the
public catalogue, and preserves:

- the complete raw JSON response, including fields not yet understood;
- a flat CSV containing every scalar player field;
- the UTC collection time and app version needed for PIT auditing.

The closed-season probe on 2026-06-29 correctly returns unavailable/empty. Use
`--allow-unavailable` for monitoring jobs that should not fail outside the
tournament.

Research use once enough snapshots exist:

- ownership as a crowd prior or disagreement signal, not a target;
- price and price movement as a compressed public expectation;
- availability/status changes as role-certainty evidence;
- ownership/price changes between team sheets and lock as late information.

Do not train on a snapshot taken after kickoff. Keep raw snapshots immutable and
join each player only to the latest snapshot strictly before that fixture.

## Data Frontier Audit (2026-06-29)

| idea | status | next decision |
|---|---|---|
| club/non-6N form | already used | FORM already backfills club matches; CLASS uses completed prior club seasons |
| World Rugby strength | already used | ranking/rating gaps are existing fixture features |
| bookmaker dispersion / line movement | tested, dropped | movement changed relationship across seasons and did not improve captain selection robustly |
| fantasy ownership / price | prospective only | snapshot every future round with `snapshot_fantasy_market.py` |
| role/bench minutes | highest unresolved player edge | seek free team-sheet, substitution, injury-return, and replacement-role evidence |
| weather / derived tactical style | tested and rejected alone | retain as secondary context, not a standalone research priority |

More internal model sweeps are now lower expected value than collecting genuinely
new PIT observations.
