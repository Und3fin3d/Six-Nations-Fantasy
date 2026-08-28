# Nations Championship Fantasy — GW2: three teams compared

**xPts1** = NCR empirical model (recency-weighted per-80 rates + calibrated club form, matchup from World Rugby ratings). Predicts NCR fantasy points.
**xPts2** = 6N champion, frozen and transplanted (LGBM rate heads + XGB point specialist + latent/POTM block). **Raw, unrescaled** — these are its own numbers, on the 6n points scale it was trained against, which runs higher than the NCR scale. It is also **matchup-blind** on NCR: its FIXTURE and OWNTEAM feature families are entirely missing, so it cannot see who a player is up against.
**avg** = 0.50·xPts1 + 0.50·xPts2.

All points are per-player at **1x**; the captain 2x and super-sub 3x multipliers are applied in the totals, not in the columns.

> **The two columns are not on the same scale, and are not averaged evenly.** Because xPts2 has the wider spread, a nominal 0.50/0.50 split gives an effective **45% empirical / 55% champion**. Pass `--w` to compensate. Comparing a player's xPts1 against their xPts2 is meaningful only as a *ranking*, not as a difference in points.

## Cross-valuation — each team scored by each model

| Team | by xPts1 (NCR) | by xPts2 (6N) | by avg |
|---|---:|---:|---:|
| NCR empirical | 406.9 | 405.6 | 406.2 |
| 6N champion | 317.4 | 459.7 | 388.5 |
| Blend (0.50/0.50) | 396.1 | 430.1 | 413.1 |

> Each team is best under the model that picked it — that is the optimiser working, not evidence. The informative number is how badly a team degrades under the *other* model.

## 1. NCR empirical

**£99.5M / 100M**  ·  **North 9 / South 7**  ·  **10 nations**  ·  **Captain:** Cam Roigard

| Pos | Player | Nation | £ | xMin | xPts1 (NCR) | xPts2 (6N champ) | avg | Role |
|---|---|---|---:|---:|---:|---:|---:|---|
| Prop | Toms Rapetti | Argentina | 3.0 | 72 | 15.9 | 19.9 | 17.9 |  |
| Prop | Pierre Schoeman | Scotland | 6.0 | 60 | 15.7 | 14.9 | 15.3 |  |
| Hooker | Ewan Ashman | Scotland | 5.5 | 56 | 20.2 | 16.6 | 18.4 |  |
| Lock | Warner Dearns | Japan | 6.5 | 78 | 17.5 | 21.3 | 19.4 |  |
| Lock | Alex Coles | England | 5.0 | 72 | 16.9 | 18.3 | 17.6 |  |
| Loose Forward | Ardie Savea | New Zealand | 8.5 | 79 | 27.4 | 29.1 | 28.3 |  |
| Loose Forward | Joaquin Oviedo | Argentina | 5.5 | 70 | 24.4 | 26.5 | 25.5 |  |
| Loose Forward | Jac Morgan | Wales | 7.0 | 73 | 25.2 | 24.7 | 24.9 |  |
| Scrum Half | **Cam Roigard** | New Zealand | 8.0 | 66 | 31.0 | 25.9 | 28.4 | **CAPTAIN** 2x |
| Fly Half | Matthieu Jalibert | France | 8.0 | 76 | 30.9 | 27.8 | 29.4 |  |
| Centre | Fabien Brau-Boirie | France | 6.0 | 71 | 23.3 | 31.0 | 27.2 |  |
| Centre | Stuart McCloskey | Ireland | 6.5 | 71 | 22.1 | 25.3 | 23.7 |  |
| Back Three | Theo Attissogbe | France | 6.5 | 80 | 25.4 | 30.5 | 27.9 |  |
| Back Three | Will Jordan | New Zealand | 8.5 | 79 | 29.6 | 25.0 | 27.3 |  |
| Back Three | Jiuta Wainiqolo | Fiji | 5.5 | 74 | 24.8 | 21.8 | 23.3 |  |
| Hooker | **Gianmarco Lucchesi** | Italy | 3.5 | 28 | 8.6 | 7.0 | 7.8 | **SUPER SUB** 3x off bench |

## 2. 6N champion

**£88.0M / 100M**  ·  **North 10 / South 6**  ·  **9 nations**  ·  **Captain:** Bautista Delguy

| Pos | Player | Nation | £ | xMin | xPts1 (NCR) | xPts2 (6N champ) | avg | Role |
|---|---|---|---:|---:|---:|---:|---:|---|
| Prop | Toms Rapetti | Argentina | 3.0 | 72 | 15.9 | 19.9 | 17.9 |  |
| Prop | Boris Wenger | Argentina | 4.0 | 72 | 15.6 | 17.7 | 16.6 |  |
| Hooker | Ronan Kelleher | Ireland | 6.0 | 57 | 15.8 | 21.4 | 18.6 |  |
| Lock | Gregor Brown | Scotland | 4.5 | 58 | 15.8 | 23.6 | 19.7 |  |
| Lock | Tadhg Beirne | Ireland | 7.5 | 74 | 12.8 | 21.5 | 17.1 |  |
| Loose Forward | Wallace Sititi | New Zealand | 6.5 | 71 | 23.8 | 29.6 | 26.7 |  |
| Loose Forward | Lenni Nouchi | France | 5.0 | 80 | 21.7 | 30.6 | 26.1 |  |
| Loose Forward | Nick Timoney | Ireland | 5.0 | 63 | 18.8 | 30.1 | 24.5 |  |
| Scrum Half | Cam Roigard | New Zealand | 8.0 | 66 | 31.0 | 25.9 | 28.4 |  |
| Fly Half | Sam Costelow | Wales | 4.0 | 60 | 13.4 | 37.2 | 25.3 |  |
| Centre | Fabien Brau-Boirie | France | 6.0 | 71 | 23.3 | 31.0 | 27.2 |  |
| Centre | Tommaso Menoncello | Italy | 8.0 | 78 | 18.6 | 26.8 | 22.7 |  |
| Back Three | Theo Attissogbe | France | 6.5 | 80 | 25.4 | 30.5 | 27.9 |  |
| Back Three | **Bautista Delguy** | Argentina | 5.0 | 66 | 16.8 | 29.1 | 23.0 | **CAPTAIN** 2x |
| Back Three | Takuro Matsunaga | Japan | 3.5 | 65 | 8.8 | 32.3 | 20.5 |  |
| Loose Forward | **Henry Pollock** | England | 5.5 | 21 | 7.7 | 7.8 | 7.8 | **SUPER SUB** 3x off bench |

## 3. Blend (0.50/0.50)

**£96.0M / 100M**  ·  **North 9 / South 7**  ·  **8 nations**  ·  **Captain:** Matthieu Jalibert

| Pos | Player | Nation | £ | xMin | xPts1 (NCR) | xPts2 (6N champ) | avg | Role |
|---|---|---|---:|---:|---:|---:|---:|---|
| Prop | Toms Rapetti | Argentina | 3.0 | 72 | 15.9 | 19.9 | 17.9 |  |
| Prop | Boris Wenger | Argentina | 4.0 | 72 | 15.6 | 17.7 | 16.6 |  |
| Hooker | Ronan Kelleher | Ireland | 6.0 | 57 | 15.8 | 21.4 | 18.6 |  |
| Lock | Gregor Brown | Scotland | 4.5 | 58 | 15.8 | 23.6 | 19.7 |  |
| Lock | Warner Dearns | Japan | 6.5 | 78 | 17.5 | 21.3 | 19.4 |  |
| Loose Forward | Ardie Savea | New Zealand | 8.5 | 79 | 27.4 | 29.1 | 28.3 |  |
| Loose Forward | Ross Vintcent | Italy | 4.5 | 70 | 22.8 | 28.1 | 25.5 |  |
| Loose Forward | Joaquin Oviedo | Argentina | 5.5 | 70 | 24.4 | 26.5 | 25.5 |  |
| Scrum Half | Cam Roigard | New Zealand | 8.0 | 66 | 31.0 | 25.9 | 28.4 |  |
| Fly Half | **Matthieu Jalibert** | France | 8.0 | 76 | 30.9 | 27.8 | 29.4 | **CAPTAIN** 2x |
| Centre | Fabien Brau-Boirie | France | 6.0 | 71 | 23.3 | 31.0 | 27.2 |  |
| Centre | Stuart McCloskey | Ireland | 6.5 | 71 | 22.1 | 25.3 | 23.7 |  |
| Back Three | Theo Attissogbe | France | 6.5 | 80 | 25.4 | 30.5 | 27.9 |  |
| Back Three | Will Jordan | New Zealand | 8.5 | 79 | 29.6 | 25.0 | 27.3 |  |
| Back Three | Immanuel Feyi-Waboso | England | 6.5 | 75 | 22.2 | 28.0 | 25.1 |  |
| Hooker | **Gianmarco Lucchesi** | Italy | 3.5 | 28 | 8.6 | 7.0 | 7.8 | **SUPER SUB** 3x off bench |

## Where they agree

Only **4/16** players appear in both single-model teams: Cam Roigard, Fabien Brau-Boirie, Theo Attissogbe, Toms Rapetti.

Low overlap means the models are making genuinely different bets — which is when a blend is worth more than either, and also when the captain choice carries the most variance.

<!-- GW_RESULT_START -->
## GW2 scored result

**Blend: 589 points**

| Model team | Points |
|---|---:|
| Bayes / NCR empirical | **560** |
| 6N champion | **624** |
| Blend | **589** |

### Blend player contributions

| Player | Nation | Base points | Multiplier | Contribution | Role |
|---|---|---:|---:|---:|---|
| Cam Roigard | New Zealand | 30 | 1x | **30** |  |
| Theo Attissogbe | France | 55 | 1x | **55** |  |
| Fabien Brau-Boirie | France | 22 | 1x | **22** |  |
| Matthieu Jalibert | France | 31 | 2x | **62** | Captain |
| Will Jordan | New Zealand | 92 | 1x | **92** |  |
| Ardie Savea | New Zealand | 36 | 1x | **36** |  |
| Joaquin Oviedo | Argentina | 74 | 1x | **74** |  |
| Stuart McCloskey | Ireland | 13 | 1x | **13** |  |
| Warner Dearns | Japan | 11 | 1x | **11** |  |
| Toms Rapetti | Argentina | 15 | 1x | **15** |  |
| Immanuel Feyi-Waboso | England | 49 | 1x | **49** |  |
| Gregor Brown | Scotland | 12 | 1x | **12** |  |
| Ronan Kelleher | Ireland | 22 | 1x | **22** |  |
| Ross Vintcent | Italy | 29 | 1x | **29** |  |
| Gianmarco Lucchesi | Italy | 15 | 3x | **45** | Super sub (bench) |
| Boris Wenger | Argentina | 22 | 1x | **22** |  |

Platform-confirmed result from the gameweek-versioned official GW2 fantasy feed, covering the fixtures on 2026-07-11.
<!-- GW_RESULT_END -->
