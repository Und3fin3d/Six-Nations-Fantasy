# NCR-GW1-3-retrospective-optimized

This checkpoint is a retrospective reachability result.
Official NCR GW1-3 labels enter the weight search.
The raw fold guard also uses available 2026 event labels.
The checkpoint does not change an active model or projection artifact.

## Result

The global event-weight vector reaches 1595 points.
The verified total is 1630 points.
The round totals are 495, 592, and 543.
The search used 276 official MILP evaluations.

## Raw guards

The development raw score is 0.884331657028.
The 2025 raw score is 0.889732171226.
The all-fold raw score is 0.882312734688.
The all-fold score uses the primary stable raw-event target set.
The maximum fold regression is +0.009663092557.
The limiting fold is `summer_internationals_2022`.
The fold guard includes every optimized extended event with raw evidence.

## Files

`config.json` contains the complete merged vector.
`trials.jsonl` contains every attempted trial.
`raw_grid.csv` contains the precomputed 0.01 raw-loss grid.
`official_direct_verification.json` records the direct model check.
`manifest.json` records source and result SHA-256 hashes.
