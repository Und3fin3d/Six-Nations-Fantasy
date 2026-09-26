# Methodology review evidence

These artifacts support [the methodology review](../../../research/METHODOLOGY_REVIEW_2026-09-26.md). They derive from merged source `93f9670f38ed1699c0029192ac38227392b329fb` and the existing complete-comparison archives. They do not change fitted models, production routing or original results.

## Retained calculations

| Files | Calculation |
|---|---|
| `season_differences.csv`, `round_differences.csv`, `squad_components.csv`, `selected_players.csv` | Exact robust-versus-empirical XV, captain and super-sub contributions. All 26 archived totals reproduce. |
| `squad_swaps.csv`, `xv_position_differences.csv` | Player and position contributions to changes in selected teams. |
| `season_error_by_status.csv`, `season_error_by_position.csv`, `round_error.csv`, `round_calibration_bins.csv` | Descriptive mean bias, MAE, RMSE and rank correlation on known outcomes. These are not fitted calibrators. |
| `interpolation_rounds.csv`, `interpolation_seasons.csv` | Five fixed empirical/robust forecast mixtures through the unchanged optimiser. All 26 endpoints reproduce original selections. No mixture is selected for deployment. |
| `front_row_rounds.csv`, `front_row_summary.csv` | Remove only non-front-row scrum points from saved NCR P3 forecasts and re-optimise. All nine original selections reproduce first. |
| `raw_scoring_relevance.csv`, `raw_event_differences.csv`, `raw_minutes_summary.csv` | Equal-block arithmetic on original raw metrics. Mean calibration slopes in the minutes file are averages of block fits, not a pooled calibration estimate. |
| `workflow_runtime.csv` | Archived job and comparison-step durations. Elapsed workflow time is 1,493 seconds; summed job time is 10,474 seconds. |
| `archive_replay.json` | Three outer and 35 inner archive hashes verified; all 11 original summaries reproduced byte for byte. |
| `diagnostic_manifest.json`, `front_row_manifest.json`, `rule_sources.json`, `verification.json` | Source, runtime, rules evidence and verification records. |

All point deltas are robust minus empirical unless the file names another comparator. Positive team-point differences favour robust. Negative error/loss differences favour robust. The interpolation uses the archived NCR scoring convention; the front-row intervention is a separate analysis. Unknown outcomes remain unknown. None of the 26 decomposed or 65 interpolated squads selected an unknown outcome.

The scripts contain direct analyses and consistency checks. They are not new test suites. `analyse.py` validates identity mappings and reproduces source totals. `interpolate.py` verifies original endpoint selections before reporting intermediate ones. `front_row.py` performs the scoring intervention. `raw_diagnostics.py` calculates raw-metric relevance and workflow durations.

The original source manifests record their extraction paths. Those paths identify the calculation locations; the underlying files remain available in the committed PR #24 bundles. Larger intermediate candidate and interpolation tables are regenerated in scratch storage rather than duplicated here.

## Reproduce from the repository root

Use the repository's Python 3.11 numerical environment with `requirements-model.txt` and `requirements-unified.txt`. This review used `/tmp/6n-pr-decision/venv/bin/python`, Python 3.11.15, NumPy 1.24.3, pandas 2.1.0 and SciPy 1.11.2. The original workflow used Python 3.11.16. No fitting is required for these commands.

```bash
REVIEW_PY=/tmp/6n-pr-decision/venv/bin/python
REVIEW_REPO="$PWD"
REVIEW_CODE="$REVIEW_REPO/data/unified/methodology_review_2026-09-26"
REVIEW_RUN=/tmp/6n-methodology-reproduction
mkdir -p "$REVIEW_RUN/replay" "$REVIEW_RUN/squad" "$REVIEW_RUN/scoring" "$REVIEW_RUN/raw"
cp "$REVIEW_CODE/analyse.py" "$REVIEW_CODE/interpolate.py" "$REVIEW_RUN/squad/"
cp "$REVIEW_CODE/front_row.py" "$REVIEW_RUN/scoring/"
export PYTHONPATH="$REVIEW_REPO:$REVIEW_RUN/squad"
"$REVIEW_PY" - "$REVIEW_REPO" "$REVIEW_RUN" <<'PY'
import hashlib
import json
import os
import sys
import tarfile

repo, output = sys.argv[1:]
source = f'{repo}/data/unified/complete_comparison/evidence'
with open(f'{source}/manifest.json') as stream:
    manifest = json.load(stream)
for bundle in manifest['bundles']:
    path = f'{source}/{bundle["path"]}'
    if hashlib.sha256(open(path, 'rb').read()).hexdigest() != bundle['sha256']:
        raise ValueError(path)
    with tarfile.open(path) as archive:
        archive.extractall(f'{output}/replay')
for job in manifest['jobs']:
    directory = f'{output}/replay/{job["name"]}'
    path = f'{directory}/evidence.tar.gz'
    if hashlib.sha256(open(path, 'rb').read()).hexdigest() != job['sha256']:
        raise ValueError(path)
    with tarfile.open(path) as archive:
        archive.extractall(directory)
os.symlink(f'{output}/replay', f'{output}/squad/evidence')
PY
"$REVIEW_PY" -m research.summarise_complete_comparison --evidence "$REVIEW_RUN/replay" --output "$REVIEW_RUN/replayed-results"
"$REVIEW_PY" "$REVIEW_RUN/squad/analyse.py" "$REVIEW_REPO" "$REVIEW_RUN/squad"
"$REVIEW_PY" "$REVIEW_RUN/squad/interpolate.py" "$REVIEW_REPO" "$REVIEW_RUN/squad"
"$REVIEW_PY" "$REVIEW_RUN/scoring/front_row.py" "$REVIEW_REPO" "$REVIEW_RUN"
"$REVIEW_PY" "$REVIEW_CODE/raw_diagnostics.py" "$REVIEW_REPO" "$REVIEW_RUN/raw"
```

Use a new scratch directory on a subsequent replay. Compare the regenerated CSVs with the matching retained files. Paths and runtime strings in newly generated manifests will reflect the new execution location.

## Existing checks

The following five existing tests passed in the review:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_THREAD_LIMIT=2 /tmp/6n-pr-decision/venv/bin/python -m pytest -q \
  tests/test_rolling_history.py::test_expected_six_nations_points_accounts_for_metres_distribution \
  tests/test_rolling_history.py::test_robust_minutes_use_international_role_not_club_minutes \
  tests/test_rolling_history.py::test_robust_position_rate_uses_observed_exposure \
  tests/test_unified_ncr_gw_eval.py
```

They cover existing minutes/rate behaviour and scorer arithmetic. They did not detect the cross-competition scrum-role mismatch. The direct forecast intervention supplies that evidence. No full-suite pass is claimed for this review. The pre-existing fixed-label-count failure remains documented in the original complete comparison.
