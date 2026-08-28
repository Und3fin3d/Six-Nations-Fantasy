# Nations Championship Rugby (NCR) fantasy — data & model

Southern-hemisphere-inclusive port of the 6n pipeline for the new game at
`fantasy.nationschampionshiprugby.com` (World Rugby Nations Championship 2026).

## The game (scraped from the site's Sportz Interactive feeds, gameworld id 1)
- **12 nations, 6 per hemisphere.**
  - North (hemi 1): England, France, Ireland, Italy, Scotland, Wales
  - South (hemi 2): Argentina, Australia, Fiji, Japan, New Zealand, South Africa
- **445 players**, 8 positions (Prop, Hooker, Lock, Loose Forward, Scrum-Half,
  Fly-Half, Centre, Back Three), price `value` 2.0–8.5M.
- **42 fixtures over 7 gamedays.** GW1 (2026-07-04, all SH teams at home):
  NZ v France · Japan v Italy · Australia v Ireland · Fiji v Wales ·
  South Africa v England · Argentina v Scotland.

### Squad rules
16 players = XV (2 Prop, 1 Hooker, 2 Lock, 3 Loose Forward, 1 Scrum-Half,
1 Fly-Half, 2 Centre, 3 Back Three) + 1 **Super Sub** (any position).
Budget **100M**, **max 3 per nation**, **max 10 per hemisphere**, unlimited free transfers.
Captain **2x**. Super Sub: **3x if it comes off the bench**, 0.5x if it starts, 0 if it does not play.
Boosters (one-shot): Co-captain (captain + vice both 2x), Triple Captain (3x).

### Scoring rubric
Try +12 · Try Assist +5 · Conversion +2 / miss −1 · Penalty +3 / miss −1 ·
Drop Goal +5 · Defender Beaten +2 · Offload +2 · Line Break +3 ·
Tackle +1 · Missed Tackle −1 · Turnover Won +4 · Interception +5 ·
Own Lineout Won +1 · Lineout Steal +5 · Scrum Won +2 (front row) · POTM +15 ·
Penalty Conceded −1 · Error (knock-on/fwd pass) −1 · Lineout Error −2 ·
Yellow −5 · Red −10. (No appearance/minutes or metres points.)

## Data files
- `ncr_players.csv` — 445-player catalogue (price, position, team, hemisphere, selection %,
  and `player_status`: **P = confirmed starter, B = bench, blank/NIS = not in matchday 23** —
  the green/yellow lineup icons; re-scrape from the live players feed before each lock).
- `ncr_teams.csv`, `ncr_fixtures.csv`, `feeds/` — raw scraped game feeds.
- `ncr_player_match.csv` — **8,050 player-match rows, 199 tests, 903 players, all 12 nations,
  2023-07 → 2026-03.** Built by re-parsing the rugby-live-data API (`rugby_api.py`) over the
  competitions where these nations play: Six Nations (1266), The Rugby Championship (1296),
  Pacific Nations Cup (1326), and July/November Internationals (30). Same per-player event
  schema as `../api_player_match.csv`; this is the SH-inclusive foundation dataset.
- `ncr_stadiums.csv` — 24 tournament venues with city, lat/lon, timezone, and host country
  (captures relocated "home" games, e.g. Fiji hosting Wales at Cardiff City Stadium).
- `ncr_fixture_weather.csv` — Open-Meteo forecast per fixture (temp, rain, precip prob, wind, gust).
  Free, no API key. Wet/windy venues (e.g. Wellington GW1: 43 kph wind) suppress attacking output.
- `ncr_fixture_markets.csv` — per fixture-team market strength: win prob, expected margin, total
  points, implied team points. **Source `wr_rating_model`** — calibrated from World Rugby ratings on
  376 real internationals (margin ≈ 1.98·rating_gap; win-prob logistic scale 5.5; home edge assigned
  by venue country). Real bookmaker odds need The Odds API key / an account; drop them into this same
  schema to override the proxy.
- `ncr_gw1_projections.csv` — model output: expected GW1 points per player.
- `ncr_gw1_squad.csv` / `.md` — the optimised 16-man squad (`_ex_*` variants exclude started matches).

## GW1 verdict (scored vs actuals — see memory/ncr-fantasy-pipeline)
Empirical+RP with full club data: team 588, Wainiqolo (actual GW top scorer) ranked #1 and captained.
Market/weather multipliers and the research.py transplant both measured negative; the original bad run
was caused by a git revert of `data/rp_*.csv` wiping SH club data (guard added in `model/rp_rates.py`).
**GW2 default: `model/ncr_project.py` (empirical), market/weather multipliers OFF.**

## Predictions from the real research.py model (`model/ncr_predict_model.py`)
The deployed path runs the trained 6n **component-rate heads** (`model/baselines.predict_rates`,
GLM engine) on the NCR players and scores the predicted per-80 rates under the NCR rubric — the
model predicts 13 competition-agnostic rates (tries, tackles, try_assists, conversion/penalty/drop
goals, defenders_beaten, offload, tackle_turnover, penalties_conceded, cards, metres) which are
exactly the raw ingredients of NCR scoring. NCR feature rows (FORM/CLASS/ROLE/BIO/MATCHUP) are built
in the model's schema from `ncr_player_match.csv` + `rp_*` and reindexed to the 6n feature columns
(unbuilt families → NaN → PositionMeanImputer). The rates then flow into the same matchup / minutes /
scoring / MILP as below. **Run with the pinned env** (has sklearn/lightgbm):
`/tmp/6n-model-pinned/bin/python -m model.ncr_predict_model [--exclude "New Zealand" "France"]`.

## Empirical baseline model (`model/ncr_project.py`)
Projection, not a full retrain of the 6n deep model (the NCR scoring rubric differs and SH
players are new). Per player: recency-weighted (420-day half-life) per-80 rates for each
scoring event → NCR points. **Small samples shrink toward a RugbyPass-derived, player-specific
prior** (`model/rp_rates.py`, from `rp_compstats.csv` — club + international form, now scraped for
all 12 nations; calibrated to test level on the ~600 players in both sources, then to a position
baseline), adjusted by a
**market + weather matchup model**: attacking output scales with the market expected margin and
game total, tackle volume rises for the defending underdog, and wet/windy venues suppress attack
(`ncr_fixture_markets.csv`, `ncr_fixture_weather.csv`); plus expected minutes. **Expected minutes come from the confirmed team sheets**
(`player_status`): starters get their historical start-minutes, bench players get capped impact
minutes; only matchday-23 players are selectable. The 16-man squad is solved exactly as a MILP
(`scipy.optimize.milp`) maximising expected points incl. captain 2x and the Super Sub's
3x-off-the-bench multiplier, subject to all squad constraints.

Known approximations: Interception / Lineout Steal are not in the per-player API feed (omitted);
`Own Lineout Won` is dropped because the API credits the hooker with every won lineout (a pure
inflation artifact); Scrum Won and POTM are modelled as small position/quality-scaled expectations.

Run: `python model/ncr_project.py`
