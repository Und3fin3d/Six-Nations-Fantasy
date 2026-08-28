# Cloud setup

This repository contains the source code, cached API responses, derived datasets, model files, reports, and tests required by cloud agents.

## Required secret

Set `RUGBY_API_KEY` in the cloud environment. The code does not contain a fallback key.

The previous key appeared in public Git history. Revoke that key and create a replacement before adding the cloud secret.

## Data refresh

Use the repository Python environment for data scripts.

```bash
python ncr_ingest.py --fetch --floor 50
python ncr_ingest.py --rebuild
python build_intl_results.py
```

Each successful API response enters `data/cache`. Commit new cache files and rebuilt tables after validation.

## Validation

```bash
python -m py_compile rugby_api.py ncr_ingest.py build_intl_results.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_ncr_player_matching.py \
  tests/test_ncr_rank_eval.py \
  tests/test_ncr_score_gw.py \
  tests/test_ncr_snapshot.py
```

The scheduled refresh must make at most 40 RapidAPI requests per weekly run. This cap includes discovery calls and retries.
