# Historical Raw Rugby Benchmark v1

All figures are retrospective, point-in-time tournament holdouts using oracle teamsheets.
Fantasy values are reconstructed/observable, never official points.

## Stable raw score

| engine | stable_score |
| --- | --- |
| p3_event_50 | 0.8865 |
| v1 | 0.9079 |
| v5_t | 0.9089 |
| v4 | 0.9133 |
| empirical_event | 0.9390 |
| gbdt_v3 | 1.0575 |

## Reconstructed stable rubrics

| engine | rubric | mae | spearman | mean_capture | slates |
| --- | --- | --- | --- | --- | --- |
| empirical_event | ncr | 6.0608 | 0.5343 | 0.7279 | 98 |
| empirical_event | six_nations | 6.1972 | 0.6248 | 0.7725 | 98 |
| gbdt_v3 | ncr | 6.3249 | 0.5227 | 0.6957 | 98 |
| gbdt_v3 | six_nations | 6.4246 | 0.6185 | 0.7451 | 98 |
| p3_event_50 | ncr | 6.1727 | 0.5367 | 0.7187 | 98 |
| p3_event_50 | six_nations | 6.2767 | 0.6287 | 0.7689 | 98 |
| v1 | ncr | 6.3863 | 0.5205 | 0.6995 | 98 |
| v1 | six_nations | 6.4873 | 0.6135 | 0.7475 | 98 |
| v4 | ncr | 6.5178 | 0.5182 | 0.7023 | 98 |
| v4 | six_nations | 6.6093 | 0.6127 | 0.7487 | 98 |
| v5_t | ncr | 6.3863 | 0.5205 | 0.6995 | 98 |
| v5_t | six_nations | 6.4873 | 0.6135 | 0.7475 | 98 |

## Event coverage

| event | tier | player_rows | player_coverage | fixtures | fixture_coverage | first_year | last_year |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tries | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| try_assists | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| conversion_goals | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| missed_conversion_goals | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| penalty_goals | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| missed_penalty_goals | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| drop_goals_converted | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| drop_goal_missed | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| defenders_beaten | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| clean_breaks | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| offload | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| runs | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| tackles | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| missed_tackles | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| rucks_won | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| rucks_lost | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| passes | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| bad_passes | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| turnovers_conceded | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| penalties_conceded | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| yellow_cards | stable | 18660 | 1.000 | 458 | 1.000 | 2019 | 2026 |
| red_cards | stable | 18660 | 1.000 | 458 | 1.000 | 2019 | 2026 |
| metres | stable | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |
| dominant_tackles | extended | 18052 | 0.967 | 443 | 0.967 | 2020 | 2026 |
| tackle_turnover | extended | 14006 | 0.751 | 343 | 0.749 | 2021 | 2026 |
| tackle_try_saver | extended | 14006 | 0.751 | 343 | 0.749 | 2021 | 2026 |
| fifty_22 | extended | 1881 | 0.101 | 42 | 0.092 | 2023 | 2026 |
| lineout_steals | extended | 1881 | 0.101 | 42 | 0.092 | 2023 | 2026 |
| scrums_won | extended | 1342 | 0.072 | 30 | 0.066 | 2025 | 2026 |
| kicks_retained | extended | 668 | 0.036 | 15 | 0.033 | 2026 | 2026 |
| potm | extended | 1881 | 0.101 | 42 | 0.092 | 2023 | 2026 |
| lineouts_won | diagnostic_only | 18627 | 0.998 | 458 | 1.000 | 2019 | 2026 |

## Historical selection

| engine | stable_score | improvement_vs_v1 | passed | reasons |
| --- | --- | --- | --- | --- |
| p3_event_50 | 0.8865 | 0.0236 | True |  |
| v1 | 0.9079 | 0.0000 | True |  |
| v5_t | 0.9089 | -0.0010 | False | competition-balanced stable raw score did not improve over v1 |
| v4 | 0.9133 | -0.0059 | False | competition-balanced stable raw score did not improve over v1 |
| empirical_event | 0.9390 | -0.0343 | False | competition-balanced stable raw score did not improve over v1; north stable raw score regressed by more than 2%; south stable raw score regressed by more than 2%; tournament-family stable regression >2%: british_irish_lions, nations_championship, rugby_championship, six_nations, summer_internationals; extended-event loss regression >5%: tackle_try_saver, tackle_turnover |
| gbdt_v3 | 1.0575 | -0.1648 | False | competition-balanced stable raw score did not improve over v1; north stable raw score regressed by more than 2%; south stable raw score regressed by more than 2%; tournament-family stable regression >2%: autumn_internationals, rugby_championship, rugby_world_cup, six_nations, summer_internationals |

Selected candidate: **p3_event_50**

NCR GW4–7 remains a confirmatory severe-failure veto, not the primary gate.
