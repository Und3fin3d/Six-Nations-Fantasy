# Unified rugby event model: P3

P3 predicts raw match events and minutes for any international fixture.
Deterministic scoring adapters in `scoring.py` convert the same `RawPrediction`
into Nations Championship or Six Nations fantasy points.

P3 (`p3_event_50`) is a fixed 50/50 blend of two components:

- the empirical event engine, `raw_benchmark/empirical.py`;
- the v4 hierarchical GBDT, `v4/gbdt.py`.

Each serialized P3 artifact contains both fitted components.
`raw_benchmark/robust_empirical.py` holds the robust empirical variant used by
`p3_robust_native`.

## Layout

| Path | Purpose |
|---|---|
| `data.py`, `features.py`, `labels.py`, `schema.py` | Canonical store, point-in-time features and labels |
| `contracts.py`, `scoring.py`, `evaluation.py` | Prediction contract and fantasy scoring |
| `raw_benchmark/` | Leak-free historical folds, P3 fitting and raw-event metrics |
| `v4/` | v4 GBDT component |
| `v3/` | Write-once NCR shadow predictions at the fantasy lock |

## Commands

```bash
PY="$HOME/.venvs/6n-model/bin/python"

# Build the corrected store, coverage ledger and event catalog.
$PY -m model.unified.raw_benchmark.cli audit

# Fit P3 and its components on the historical tournament folds.
$PY -m model.unified.raw_benchmark.cli run

# Freeze a write-once NCR prediction before lock (normally run by gw_update.sh).
$PY -m model.unified.v3.cli shadow --gw 4 --engine p3_event_50 \
  --model data/unified/raw_benchmark/v1/models/p3_event_50/nations_championship_2026.pkl

# After GW7 labels arrive, apply the prospective gates.
$PY -m model.unified.v3.cli shadow-evaluate --engine p3_event_50
```

Do not retune the 0.5 blend weight on NCR GW4–7. Those rounds are the
predefined prospective check, not another development set.
