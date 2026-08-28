# Unified model retirement record

Date: 2026-07-19

## Active candidate

- `p3_event_50`: fixed 50/50 raw-event empirical/v4 blend.
- Historical fold artifacts under `raw_benchmark/v1/models/p3_event_50/` are
  self-contained and include both fitted components.

## Rollback models retained

- v1 GBDT artifacts under `models/unified_gbdt*`.
- Competition-specific NCR and Six Nations incumbents outside `data/unified/`.
- Frozen P3 benchmark reports, metrics and all fold manifests.

## Fitted artifacts retired

- Historical per-fold v1 duplicates.
- GBDT-v3.
- v1 neural.
- v5 terminal.
- Standalone empirical-event and v4 fold artifacts duplicated inside P3.
- Legacy unified neural and fantasy-points rank-stack artifacts.
- v3 benchmark fit artifacts and its superseded GW3 baseline model.

Non-model reports, source code, manifests and aggregate metrics remain for
auditability. No scoring converters, canonical stores, source data, official
points outputs, or competition-specific incumbents were removed.
