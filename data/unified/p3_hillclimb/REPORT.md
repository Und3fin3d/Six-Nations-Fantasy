# P3 local raw-event hill climb

The search optimises raw rugby events. Tournament fantasy metrics are secondary diagnostics.
The search uses 2022-2024 development folds and 2025 selection folds.
The selector does not use 2026. The repository already contained the 2026 results.

## Result

The search accepts 20 event coordinates. The retrospective 2026 confirmation also improves.
The all-fold raw score improves by 0.96% against P3 50/50.
This result defines a research candidate. It does not change the active P3 shadow artifact.

## Raw-event objective

| split | folds | baseline_score | candidate_score | delta | delta_p05 | delta_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| development | 13 | 0.886332 | 0.879703 | -0.006629 | -0.011939 | -0.002052 |
| selection | 6 | 0.898153 | 0.885919 | -0.012234 | -0.018967 | -0.007317 |
| confirmation | 2 | 0.852323 | 0.842593 | -0.009730 | -0.009754 | -0.009705 |
| all | 21 | 0.886470 | 0.877945 | -0.008526 | -0.012344 | -0.005111 |

Lower scores are better. Each score averages training-naive-relative loss across folds and stable event heads.
The final two columns give the paired 90% fold-bootstrap interval for the score delta.

## Accepted coordinates

| target | selected_weight_v4 | development_gain | selection_delta |
| --- | --- | --- | --- |
| tries | 0.100000 | 0.016899 | -0.015451 |
| try_assists | 0.200000 | 0.012430 | -0.017606 |
| conversion_goals | 0.200000 | 0.012968 | -0.028942 |
| missed_conversion_goals | 0.250000 | 0.007089 | -0.027312 |
| drop_goals_converted | 0.750000 | 0.017370 | -0.082616 |
| drop_goal_missed | 0.650000 | 0.004145 | -0.003217 |
| defenders_beaten | 0.250000 | 0.009846 | -0.004181 |
| clean_breaks | 0.200000 | 0.009438 | -0.008397 |
| offload | 0.250000 | 0.004232 | -0.009821 |
| runs | 0.600000 | 0.003317 | -0.005623 |
| tackles | 0.650000 | 0.003594 | 0.000181 |
| missed_tackles | 0.400000 | 0.000535 | 0.000258 |
| rucks_won | 0.600000 | 0.002467 | 0.000117 |
| rucks_lost | 0.650000 | 0.001583 | -0.002435 |
| passes | 0.650000 | 0.003765 | -0.007881 |
| bad_passes | 0.550000 | 0.000397 | -0.000507 |
| turnovers_conceded | 0.300000 | 0.001147 | 0.000980 |
| penalties_conceded | 0.350000 | 0.000832 | 0.000200 |
| red_cards | 0.900000 | 0.025814 | -0.056133 |
| metres | 0.100000 | 0.021218 | -0.025231 |

All unspecified event weights remain at 0.5. Extended events remain at 0.5 because their support is limited.

## Fantasy-adapter diagnostics

| rubric | model | mae | spearman | mean_capture | slates |
| --- | --- | --- | --- | --- | --- |
| ncr | P3 50/50 | 6.172660 | 0.536744 | 0.718740 | 98 |
| ncr | weighted P3 | 6.053627 | 0.541317 | 0.728723 | 98 |
| six_nations | P3 50/50 | 6.276726 | 0.628702 | 0.768913 | 98 |
| six_nations | weighted P3 | 6.168857 | 0.629705 | 0.772390 | 98 |

Fantasy metrics do not select or reject the candidate. They show intended-application behaviour.

## Official NCR GW1-3 soft diagnostic

| model | mae | spearman | mean_capture |
| --- | --- | --- | --- |
| P3 50/50 | 8.732456 | 0.526647 | 0.621957 |
| weighted P3 | 8.548991 | 0.556713 | 0.684344 |

| model | team_points |
| --- | --- |
| NCR incumbent | 1595.000000 |
| P3 50/50 | 1291.000000 |
| weighted P3 | 1473.000000 |

The official NCR diagnostic runs only after the event weights are frozen.
It is retrospective evidence and does not alter the selected weights.

## Interpretation limits

The search adds one competition-independent weight for each accepted raw event head.
The 2025 results guide every coordinate and are not independent validation.
The 2026 blocks do not enter selection, but they are existing repository evidence.
Historical fantasy diagnostics reweight expected means and retain baseline dispersion.
The official NCR diagnostic exercises the deployable weighted distribution blend.
Official NCR GW1-3 is retrospective evidence. Future NCR GW4-7 and a new raw tournament remain prospective tests.

## Fold evidence

| split | fold | baseline_score | candidate_score | delta |
| --- | --- | --- | --- | --- |
| confirmation | nations_championship_2026 | 0.835679 | 0.825924 | -0.009754 |
| confirmation | six_nations_2026 | 0.868967 | 0.859262 | -0.009705 |
| development | autumn_internationals_2022 | 0.852348 | 0.852981 | 0.000632 |
| development | autumn_internationals_2024 | 0.911890 | 0.901836 | -0.010054 |
| development | pacific_nations_cup_2024 | 0.918835 | 0.914997 | -0.003839 |
| development | rugby_championship_2022 | 0.898196 | 0.901721 | 0.003525 |
| development | rugby_championship_2023 | 0.873105 | 0.840045 | -0.033061 |
| development | rugby_championship_2024 | 0.898974 | 0.873874 | -0.025100 |
| development | rugby_world_cup_2023 | 0.860349 | 0.847656 | -0.012693 |
| development | six_nations_2022 | 0.884193 | 0.887810 | 0.003618 |
| development | six_nations_2023 | 0.844963 | 0.844934 | -0.000029 |
| development | six_nations_2024 | 0.883728 | 0.883811 | 0.000084 |
| development | summer_internationals_2022 | 0.856827 | 0.859643 | 0.002816 |
| development | summer_internationals_2023 | 0.912288 | 0.901894 | -0.010394 |
| development | summer_internationals_2024 | 0.926618 | 0.924942 | -0.001676 |
| selection | autumn_internationals_2025 | 0.900826 | 0.892780 | -0.008047 |
| selection | british_irish_lions_2025 | 0.854932 | 0.843461 | -0.011471 |
| selection | pacific_nations_cup_2025 | 0.961540 | 0.956094 | -0.005446 |
| selection | rugby_championship_2025 | 0.930575 | 0.899820 | -0.030755 |
| selection | six_nations_2025 | 0.846087 | 0.833686 | -0.012401 |
| selection | summer_internationals_2025 | 0.894959 | 0.889674 | -0.005285 |

## Reproduce

```bash
/tmp/6n-model-pinned/bin/python -m model.unified.cli raw-benchmark p3-hillclimb
```

The command reads frozen cache-derived data and fitted P3 artifacts. It makes no network request.
