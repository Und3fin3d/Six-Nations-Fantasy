# Historical Raw Rugby Benchmark v1 — challengers

The original v1 benchmark artifacts remain frozen. Raw-eligible challengers are trained on the same 21 cutoff manifests and compared with the frozen core engines.

## Raw-contract compatibility

| model | raw_contract | result |
| --- | --- | --- |
| v1_neural | eligible | evaluated on all 21 folds |
| v2_rank_stack | ineligible | fantasy-points-only: learns rank and official-points heads over v1 raw forecasts; it does not emit independent raw-event predictions |
| six_nations_champion | ineligible | specialist/partial contract: promoted output includes latent and points-specialist heads, its native store covers only Six Nations 2023-26, and its internal component set omits stable benchmark events |

## Stable raw score

| engine | stable_score |
| --- | --- |
| p3_event_50 | 0.8865 |
| v1 | 0.9079 |
| v5_t | 0.9089 |
| v4 | 0.9133 |
| empirical_event | 0.9390 |
| gbdt_v3 | 1.0575 |
| v1_neural | 1.1989 |

## Stable raw score by year

| calendar_year | p3_event_50 | v1 | v1_neural |
| --- | --- | --- | --- |
| 2022 | 0.8729 | 0.8990 | 1.9407 |
| 2023 | 0.8727 | 0.8837 | 1.0602 |
| 2024 | 0.9080 | 0.9387 | 1.0300 |
| 2025 | 0.8982 | 0.9151 | 1.0079 |
| 2026 | 0.8523 | 0.8757 | 0.9884 |

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
| v1_neural | ncr | 6.3942 | 0.4639 | 0.6698 | 98 |
| v1_neural | six_nations | 6.6128 | 0.5667 | 0.7206 | 98 |
| v4 | ncr | 6.5178 | 0.5182 | 0.7023 | 98 |
| v4 | six_nations | 6.6093 | 0.6127 | 0.7487 | 98 |
| v5_t | ncr | 6.3863 | 0.5205 | 0.6995 | 98 |
| v5_t | six_nations | 6.4873 | 0.6135 | 0.7475 | 98 |

## Six Nations champion partial component diagnostic

This is not promotion-eligible: it covers only Six Nations 2025-26 and 12 of the 23 stable events plus minutes.

| engine | partial_stable_score | folds | targets |
| --- | --- | --- | --- |
| p3_event_50 | 0.8479 | 2 | 13 |
| v1 | 0.8693 | 2 | 13 |
| six_nations_champion_components | 1.1144 | 2 | 13 |

### Component breakdown

| target | p3_event_50 | six_nations_champion_components | v1 | champion_vs_v1_pct |
| --- | --- | --- | --- | --- |
| metres | 0.8710 | 0.7616 | 0.9428 | -19.2203 |
| conversion_goals | 0.4341 | 0.4610 | 0.5392 | -14.4996 |
| tackles | 0.8734 | 0.9283 | 0.9028 | 2.8241 |
| yellow_cards | 0.9990 | 1.0315 | 0.9802 | 5.2336 |
| defenders_beaten | 0.8851 | 0.9636 | 0.9156 | 5.2482 |
| minutes | 0.9295 | 1.0609 | 0.9619 | 10.2960 |
| penalties_conceded | 0.9929 | 1.1075 | 0.9947 | 11.3415 |
| tries | 0.9335 | 1.1166 | 0.9868 | 13.1600 |
| penalty_goals | 0.5775 | 0.6654 | 0.5665 | 17.4531 |
| offload | 0.9007 | 1.1112 | 0.9389 | 18.3574 |
| try_assists | 0.9101 | 1.1253 | 0.9373 | 20.0630 |
| red_cards | 0.8606 | 1.8341 | 0.7776 | 135.8605 |
| drop_goals_converted | 0.8557 | 2.3202 | 0.8566 | 170.8555 |

## Gate result

| engine | stable_score | improvement_vs_v1 | passed | reasons |
| --- | --- | --- | --- | --- |
| p3_event_50 | 0.8865 | 0.0236 | True |  |
| v1 | 0.9079 | 0.0000 | True |  |
| v5_t | 0.9089 | -0.0010 | False | competition-balanced stable raw score did not improve over v1 |
| v4 | 0.9133 | -0.0059 | False | competition-balanced stable raw score did not improve over v1 |
| empirical_event | 0.9390 | -0.0343 | False | competition-balanced stable raw score did not improve over v1; north stable raw score regressed by more than 2%; south stable raw score regressed by more than 2%; tournament-family stable regression >2%: british_irish_lions, nations_championship, rugby_championship, six_nations, summer_internationals; extended-event loss regression >5%: tackle_try_saver, tackle_turnover |
| gbdt_v3 | 1.0575 | -0.1648 | False | competition-balanced stable raw score did not improve over v1; north stable raw score regressed by more than 2%; south stable raw score regressed by more than 2%; tournament-family stable regression >2%: autumn_internationals, rugby_championship, rugby_world_cup, six_nations, summer_internationals |
| v1_neural | 1.1989 | -0.3205 | False | competition-balanced stable raw score did not improve over v1; north stable raw score regressed by more than 2%; south stable raw score regressed by more than 2%; tournament-family stable regression >2%: autumn_internationals, british_irish_lions, nations_championship, pacific_nations_cup, rugby_championship, rugby_world_cup, six_nations, summer_internationals |

Selected model after adding challengers: **p3_event_50**
